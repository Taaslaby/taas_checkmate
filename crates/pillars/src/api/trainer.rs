//! -------------------------------------------------------------------------
//! Filename:    src/trainer.rs
//! Class:       train_suite
//! Purpose:
//! Description: Batched E-Prop Izhikevich Multi trainer (Hogwild! parallelism)
//!
//! Each worker has its **own copy** of weights. Workers process images
//! in parallel with zero sharing. After a micro-batch completes, all
//! weight copies are averaged back into the master copy and redistributed.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::api::neurons::eprop::EPropIzhikevichMulti;

// ═══════════════════════════════════════════════════════════
//  Per-worker state
// ═══════════════════════════════════════════════════════════

pub struct WorkerState {
    eprop: EPropIzhikevichMulti,
    rng: StdRng,
    /// Own copy of input weights [ni × nh]
    weights: Vec<f32>,
    /// Own copy of output weights [nh × no]
    w_out: Vec<f32>,
    /// Own copy of output bias [no]
    bias_out: Vec<f32>,

    z_pre: Vec<i32>,
    spikes: Vec<i32>,
    hidden_count: Vec<i32>,
    rates: Vec<f32>,
    hc_f32: Vec<f32>,
    logits: Vec<f32>,
    exps: Vec<f32>,
    probs: Vec<f32>,
    error: Vec<f32>,
    learning_signals: Vec<f32>,
}

impl WorkerState {
    fn new(
        ni: usize,
        nh: usize,
        no: usize,
        dt: f32,
        tau_e: f32,
        weight_scale: f32,
        seed: u64,
        worker_id: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights: Vec<f32> = (0..ni * nh)
            .map(|_| rng.gen_range(-weight_scale..weight_scale))
            .collect();
        let w_out: Vec<f32> = (0..nh * no).map(|_| rng.gen_range(-0.1..0.1)).collect();
        // Unique seed per worker for Poisson noise — ensures Hogwild! workers
        // see independent stochasticity, not identical noise patterns.
        let rng = StdRng::seed_from_u64(seed.wrapping_add(worker_id as u64 * 7919));
        Self {
            eprop: EPropIzhikevichMulti::new(ni, nh, dt, tau_e),
            rng,
            weights,
            w_out,
            bias_out: vec![0.0f32; no],
            z_pre: vec![0i32; ni],
            spikes: vec![0i32; nh],
            hidden_count: vec![0i32; nh],
            rates: vec![0.0f32; ni],
            hc_f32: vec![0.0f32; nh],
            logits: vec![0.0f32; no],
            exps: vec![0.0f32; no],
            probs: vec![0.0f32; no],
            error: vec![0.0f32; no],
            learning_signals: vec![0.0f32; nh],
        }
    }

    fn poisson_step(&mut self, dt_sec: f32) {
        for (z, &rate) in self.z_pre.iter_mut().zip(self.rates.iter()) {
            let r: f32 = self.rng.gen();
            *z = if r < (rate * dt_sec).clamp(0.0, 1.0) {
                1
            } else {
                0
            };
        }
    }

    fn reset_image(&mut self) {
        self.eprop.reset();
        self.hidden_count.fill(0);
    }
}

// ═══════════════════════════════════════════════════════════
//  EPropTrainer
// ═══════════════════════════════════════════════════════════

/// Multi-threaded batch trainer using Hogwild! parallelism.
///
/// Each worker has its own weight copies and processes images
/// independently. Weights are averaged across workers after each batch.
pub struct EPropTrainer {
    /// Master copy of averaged input weights [ni × nh]
    pub weights: Vec<f32>,
    /// Output weights [nh × no]
    pub w_out: Vec<f32>,
    /// Hidden bias `[nh]`
    pub bias_hidden: Vec<f32>,
    /// Output bias `[no]`
    pub bias_out: Vec<f32>,
    /// Fixed feedback weights `[no × nh]`
    pub w_fb: Vec<f32>,

    pub ni: usize,
    pub nh: usize,
    pub no: usize,
    pub n_steps: usize,
    pub poisson_max_hz: f32,
    pub dt_sec: f32,
    pub lr_in: f32,
    pub lr_out: f32,

    pub workers: Vec<WorkerState>,
}

impl EPropTrainer {
    /// Create a new trainer.
    ///
    /// All workers start with **identical** weights (same seed).
    /// `num_workers` — number of parallel workers (default CPU count).
    pub fn new(
        ni: usize,
        nh: usize,
        no: usize,
        n_steps: usize,
        poisson_max_hz: f32,
        dt: f32,
        tau_e: f32,
        lr_in: f32,
        lr_out: f32,
        bias: f32,
        weight_scale: f32,
        seed: u64,
        num_workers: usize,
    ) -> Self {
        assert!(ni > 0, "EPropTrainer: ni must be > 0");
        assert!(nh > 0, "EPropTrainer: nh must be > 0");
        assert!(no > 0, "EPropTrainer: no must be > 0");
        let nw = num_workers.max(1);

        // Workers: identical init via same seed
        let workers: Vec<WorkerState> = (0..nw)
            .map(|wi| WorkerState::new(ni, nh, no, dt, tau_e, weight_scale, seed, wi))
            .collect();

        // Master copy (copy from first worker since all are identical)
        let weights = workers[0].weights.clone();
        let w_out = workers[0].w_out.clone();
        let w_fb: Vec<f32> = {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(9999));
            (0..no * nh)
                .map(|_| rng.gen_range(-1.0 / (no as f32).sqrt()..1.0 / (no as f32).sqrt()))
                .collect()
        };

        Self {
            weights,
            w_out,
            bias_hidden: vec![bias; nh],
            bias_out: vec![0.0f32; no],
            w_fb,
            ni,
            nh,
            no,
            n_steps,
            poisson_max_hz,
            dt_sec: dt / 1000.0,
            lr_in,
            lr_out,
            workers,
        }
    }

    /// Validate a batch before processing: `images` must be exactly
    /// `targets.len() * ni` elements and every target must index an output
    /// class in `[0, no)`. Rejects caller shape errors up front instead of
    /// panicking deep inside `train_one` (or slicing the wrong span).
    fn validate_batch(&self, images: &[f32], targets: &[i32]) {
        let expected = self
            .ni
            .checked_mul(targets.len())
            .expect("EPropTrainer: batch size overflows");
        assert_eq!(
            images.len(),
            expected,
            "EPropTrainer: images.len() = {} does not match targets.len() * ni = {} * {} = {}",
            images.len(),
            targets.len(),
            self.ni,
            expected
        );
        for (idx, &t) in targets.iter().enumerate() {
            assert!(
                t >= 0 && (t as usize) < self.no,
                "EPropTrainer: target {idx} = {t} out of range [0, {})",
                self.no
            );
        }
    }

    // ─── Process one image ─────────────────────────────

    /// Full forward + backward on one worker's own weight copy.
    fn train_one(
        worker: &mut WorkerState,
        image: &[f32],
        target: i32,
        bias_hidden: &[f32],
        w_fb: &[f32],
        _ni: usize,
        nh: usize,
        no: usize,
        n_steps: usize,
        poisson_max_hz: f32,
        dt_sec: f32,
        lr_in: f32,
        lr_out: f32,
    ) -> (f32, u32) {
        worker.reset_image();
        for (r, &pix) in worker.rates.iter_mut().zip(image.iter()) {
            *r = pix * poisson_max_hz;
        }

        for _ in 0..n_steps {
            worker.poisson_step(dt_sec);
            let sc = worker.eprop.step(
                &worker.weights,
                &worker.z_pre,
                bias_hidden,
                &mut worker.spikes,
            );
            if sc > 0 {
                for (hc, &sp) in worker.hidden_count.iter_mut().zip(worker.spikes.iter()) {
                    *hc += sp;
                }
            }
        }

        for j in 0..nh {
            worker.hc_f32[j] = worker.hidden_count[j] as f32;
        }
        for k in 0..no {
            let mut sum = worker.bias_out[k];
            for j in 0..nh {
                sum += worker.hc_f32[j] * worker.w_out[j * no + k];
            }
            worker.logits[k] = sum;
        }

        let max_l = worker
            .logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for (e, &l) in worker.exps.iter_mut().zip(worker.logits.iter()) {
            *e = (l - max_l).exp();
            exp_sum += *e;
        }
        let inv = 1.0 / (exp_sum + 1e-30);
        for (p, &e) in worker.probs.iter_mut().zip(worker.exps.iter()) {
            *p = e * inv;
        }

        let loss = -(worker.probs[target as usize].max(1e-30)).ln();
        let predicted = worker
            .probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
            .map(|(i, _)| i)
            .unwrap_or(0);

        for k in 0..no {
            worker.error[k] = -worker.probs[k];
        }
        worker.error[target as usize] += 1.0;

        // BUGFIX: gradient clipping for output weights
        const GRAD_CLIP: f32 = 1.0;
        for k in 0..no {
            let ek = worker.error[k].clamp(-GRAD_CLIP, GRAD_CLIP);
            for j in 0..nh {
                worker.w_out[j * no + k] += lr_out * worker.hc_f32[j] * ek;
            }
            worker.bias_out[k] += lr_out * ek;
        }

        for j in 0..nh {
            let mut sum = 0.0f32;
            for k in 0..no {
                sum += w_fb[k * nh + j] * worker.error[k];
            }
            worker.learning_signals[j] = sum;
        }
        worker
            .eprop
            .update_weights_multi(&mut worker.weights, &worker.learning_signals, lr_in);

        (loss, if predicted == target as usize { 1 } else { 0 })
    }

    /// Train a chunk of the batch on one worker.
    #[allow(clippy::too_many_arguments)]
    fn worker_batch(
        worker: &mut WorkerState,
        start: usize,
        end: usize,
        images: &[f32],
        targets: &[i32],
        bias_hidden: &[f32],
        w_fb: &[f32],
        ni: usize,
        nh: usize,
        no: usize,
        ns: usize,
        pmhz: f32,
        dt: f32,
        lri: f32,
        lro: f32,
    ) -> (f32, u32) {
        if start >= end {
            return (0.0, 0);
        }
        let mut loss_sum = 0.0f32;
        let mut corr_sum = 0u32;
        for img_idx in start..end {
            let img = &images[img_idx * ni..(img_idx + 1) * ni];
            let (l, c) = Self::train_one(
                worker,
                img,
                targets[img_idx],
                bias_hidden,
                w_fb,
                ni,
                nh,
                no,
                ns,
                pmhz,
                dt,
                lri,
                lro,
            );
            loss_sum += l;
            corr_sum += c;
        }
        (loss_sum, corr_sum)
    }

    // ─── Sync: average all workers into master ─────────

    fn sync_weights(&mut self) {
        let nw = self.workers.len() as f32;
        for i in 0..self.weights.len() {
            self.weights[i] = self.workers.iter().map(|w| w.weights[i]).sum::<f32>() / nw;
        }
        for i in 0..self.w_out.len() {
            self.w_out[i] = self.workers.iter().map(|w| w.w_out[i]).sum::<f32>() / nw;
        }
        for k in 0..self.no {
            self.bias_out[k] = self.workers.iter().map(|w| w.bias_out[k]).sum::<f32>() / nw;
        }
        for w in &mut self.workers {
            w.weights.copy_from_slice(&self.weights);
            w.w_out.copy_from_slice(&self.w_out);
            w.bias_out.copy_from_slice(&self.bias_out);
        }
    }

    // ─── Batch training ───────────────────────────────

    /// Train on a batch of images with full Hogwild! parallelism.
    ///
    /// Each worker processes a chunk of the batch independently with its
    /// own weight copy. After the batch, weights are averaged and re-synced.
    ///
    /// `images`: flat `[batch_size × ni]` f32
    /// `targets`: `[batch_size]` i32
    /// Returns (avg_loss, correct_count).
    pub fn train_batch(&mut self, images: &[f32], targets: &[i32]) -> (f32, u32) {
        let n = targets.len();
        if n == 0 {
            return (0.0, 0);
        }
        self.validate_batch(images, targets);
        let nw = self.workers.len();
        let chunk_size = n.div_ceil(nw);

        let bias_hidden = &*self.bias_hidden;
        let w_fb = &*self.w_fb;
        let ni = self.ni;
        let nh = self.nh;
        let no = self.no;
        let ns = self.n_steps;
        let pmhz = self.poisson_max_hz;
        let dt = self.dt_sec;
        let lri = self.lr_in;
        let lro = self.lr_out;

        #[cfg(feature = "rayon")]
        let results: Vec<(f32, u32)> = self
            .workers
            .par_iter_mut()
            .enumerate()
            .map(|(wi, worker)| {
                let start = wi * chunk_size;
                let end = (start + chunk_size).min(n);
                Self::worker_batch(
                    worker,
                    start,
                    end,
                    images,
                    targets,
                    bias_hidden,
                    w_fb,
                    ni,
                    nh,
                    no,
                    ns,
                    pmhz,
                    dt,
                    lri,
                    lro,
                )
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let results: Vec<(f32, u32)> = {
            let mut results = Vec::with_capacity(nw);
            for (wi, worker) in self.workers.iter_mut().enumerate() {
                let start = wi * chunk_size;
                let end = (start + chunk_size).min(n);
                results.push(Self::worker_batch(
                    worker,
                    start,
                    end,
                    images,
                    targets,
                    bias_hidden,
                    w_fb,
                    ni,
                    nh,
                    no,
                    ns,
                    pmhz,
                    dt,
                    lri,
                    lro,
                ));
            }
            results
        };

        // Serial: average diverged worker weights
        self.sync_weights();

        let total_loss: f32 = results.iter().map(|r| r.0).sum();
        let total_correct: u32 = results.iter().map(|r| r.1).sum();
        (total_loss / n as f32, total_correct)
    }

    // ─── Sequential baseline ──────────────────────────

    /// Single-worker sequential training for comparison.
    pub fn train_sequential(&mut self, images: &[f32], targets: &[i32]) -> (f32, u32) {
        let n = targets.len();
        if n == 0 {
            return (0.0, 0);
        }
        self.validate_batch(images, targets);
        let worker = &mut self.workers[0];
        let mut tl = 0.0f32;
        let mut tc = 0u32;
        for img_idx in 0..n {
            let img = &images[img_idx * self.ni..(img_idx + 1) * self.ni];
            let (l, c) = Self::train_one(
                worker,
                img,
                targets[img_idx],
                &self.bias_hidden,
                &self.w_fb,
                self.ni,
                self.nh,
                self.no,
                self.n_steps,
                self.poisson_max_hz,
                self.dt_sec,
                self.lr_in,
                self.lr_out,
            );
            tl += l;
            tc += c;
        }
        self.weights.copy_from_slice(&worker.weights);
        self.w_out.copy_from_slice(&worker.w_out);
        self.bias_out.copy_from_slice(&worker.bias_out);
        (tl / n as f32, tc)
    }

    // ─── Inference ────────────────────────────────────

    /// Number of parallel workers.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    pub fn predict(&mut self, image: &[f32]) -> i32 {
        let w = &mut self.workers[0];
        w.reset_image();
        for (r, &pix) in w.rates.iter_mut().zip(image.iter()) {
            *r = pix * self.poisson_max_hz;
        }
        for _ in 0..self.n_steps {
            w.poisson_step(self.dt_sec);
            let sc = w
                .eprop
                .step(&w.weights, &w.z_pre, &self.bias_hidden, &mut w.spikes);
            if sc > 0 {
                for (hc, &sp) in w.hidden_count.iter_mut().zip(w.spikes.iter()) {
                    *hc += sp;
                }
            }
        }
        for j in 0..self.nh {
            w.hc_f32[j] = w.hidden_count[j] as f32;
        }
        for k in 0..self.no {
            let mut sum = w.bias_out[k];
            for j in 0..self.nh {
                sum += w.hc_f32[j] * w.w_out[j * self.no + k];
            }
            w.logits[k] = sum;
        }
        w.logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let t = EPropTrainer::new(
            100, 32, 10, 10, 200.0, 1.0, 30.0, 0.05, 0.02, 3.0, 0.5, 42, 4,
        );
        assert_eq!(t.workers.len(), 4);
        assert_eq!(t.bias_hidden[0], 3.0);
    }

    #[test]
    fn test_workers_identical_init() {
        let t = EPropTrainer::new(50, 16, 5, 5, 200.0, 1.0, 30.0, 0.05, 0.02, 3.0, 0.5, 42, 3);
        let w0 = &t.workers[0].weights;
        for w in &t.workers[1..] {
            assert_eq!(w.weights, *w0, "workers must start with identical weights");
        }
    }

    #[test]
    fn test_sequential_training() {
        let mut t = EPropTrainer::new(
            784, 32, 10, 5, 200.0, 1.0, 30.0, 0.05, 0.02, 3.0, 0.357, 42, 2,
        );
        let mut imgs = vec![0.0f32; 16 * 784];
        let mut tgts = vec![0i32; 16];
        for i in 0..16 {
            for j in 0..784 {
                imgs[i * 784 + j] = rand::random::<f32>() * 0.3;
            }
            tgts[i] = (i % 10) as i32;
        }
        let (loss, correct) = t.train_sequential(&imgs, &tgts);
        assert!(loss > 0.0, "loss={loss}");
        assert!(correct <= 16);
    }

    #[test]
    fn test_parallel_training() {
        let mut t = EPropTrainer::new(
            784, 32, 10, 5, 200.0, 1.0, 30.0, 0.05, 0.02, 3.0, 0.357, 42, 4,
        );
        let mut imgs = vec![0.0f32; 32 * 784];
        let mut tgts = vec![0i32; 32];
        for i in 0..32 {
            for j in 0..784 {
                imgs[i * 784 + j] = rand::random::<f32>() * 0.3;
            }
            tgts[i] = (i % 10) as i32;
        }
        let (loss, correct) = t.train_batch(&imgs, &tgts);
        assert!(loss > 0.0, "loss={loss}");
        assert!(correct <= 32);
    }

    #[test]
    fn test_empty_batch() {
        let mut t = EPropTrainer::new(
            100, 32, 10, 5, 200.0, 1.0, 30.0, 0.05, 0.02, 3.0, 0.5, 42, 2,
        );
        let (loss, corr) = t.train_batch(&[], &[]);
        assert_eq!(loss, 0.0);
        assert_eq!(corr, 0);
    }

    /// Малый тренер для negative-тестов форм (ni=4, nh=4, no=3).
    fn small_trainer() -> EPropTrainer {
        EPropTrainer::new(4, 4, 3, 2, 100.0, 1.0, 20.0, 0.05, 0.02, 3.0, 0.5, 42, 2)
    }

    #[test]
    #[should_panic(expected = "does not match targets.len() * ni")]
    fn train_batch_rejects_short_images() {
        let mut t = small_trainer();
        let _ = t.train_batch(&[0.0f32; 7], &[0i32, 1]); // 7 != 2*4=8
    }

    #[test]
    #[should_panic(expected = "does not match targets.len() * ni")]
    fn train_batch_rejects_extra_images() {
        let mut t = small_trainer();
        let _ = t.train_batch(&[0.0f32; 9], &[0i32, 1]); // 9 != 8
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn train_batch_rejects_negative_label() {
        let mut t = small_trainer();
        let _ = t.train_batch(&[0.0f32; 8], &[-1i32, 0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn train_batch_rejects_label_at_or_above_no() {
        let mut t = small_trainer();
        let _ = t.train_batch(&[0.0f32; 8], &[0i32, 3]); // no=3 → 3 вне диапазона
    }

    #[test]
    #[should_panic(expected = "does not match targets.len() * ni")]
    fn train_sequential_rejects_mismatched_images() {
        let mut t = small_trainer();
        let _ = t.train_sequential(&[0.0f32; 4], &[0i32, 1]); // 4 != 8
    }

    #[test]
    #[should_panic(expected = "ni must be > 0")]
    fn trainer_rejects_zero_ni() {
        let _ = EPropTrainer::new(0, 4, 3, 2, 100.0, 1.0, 20.0, 0.05, 0.02, 3.0, 0.5, 42, 2);
    }

    #[test]
    #[should_panic(expected = "nh must be > 0")]
    fn trainer_rejects_zero_nh() {
        let _ = EPropTrainer::new(4, 0, 3, 2, 100.0, 1.0, 20.0, 0.05, 0.02, 3.0, 0.5, 42, 2);
    }

    #[test]
    #[should_panic(expected = "no must be > 0")]
    fn trainer_rejects_zero_no() {
        let _ = EPropTrainer::new(4, 4, 0, 2, 100.0, 1.0, 20.0, 0.05, 0.02, 3.0, 0.5, 42, 2);
    }
}
