//! MNIST digit classification via T3 — HybridDecoder (InverseKuramoto + PST + NLMS).
//!
//! Zero-shot: первый predict = чистая физика Stuart-Landau (IK weight = 1.0).
//! После обучения: NLMS комбинирует PST фичи + IK оценку.
//!
//! Usage:
//!   python3 examples/python/download_mnist.py   # download data first
//!   cargo build --no-default-features && cargo run --example mnist_t3
//!
//! Architecture:
//!   Input:    784 pixels → random projection → 64 drives
//!   T3:       3 populations × 64 Kuramoto-ST oscillators
//!   Features: PST (1164) + IK drive (1) = 1165 dim
//!   Decoder:  shared IK + 10×NLMS → softmax → digit
//!   Init:     IK weight = 1.0 → zero-shot physics prediction

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_checkmate::t3::{
    TriOscillator, Avx2Backend,
    pst_linear_features_into, pst_linear_dim,
    InverseKuramotoDecoder,
    NlmsDecoder, Decoder,
};

// ─── Constants ────────────────────────────────────────────

const N_OSCILLATORS: usize = 64;   // per population
const N_INPUT: usize = 784;
const N_ACTIONS: usize = 10;
const N_T3_STEPS: usize = 15;
const DT: f32 = 0.05;
const FREQ_MIN: f32 = 0.3;
const FREQ_MAX: f32 = 2.5;
const SEED: u32 = 137;

const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;

const BATCH_SIZE: usize = 500;
const DATA_DIR: &str = "examples/data";
const LR: f32 = 0.02;

// ─── Random projection: 784 → 64, fixed seed ─────────────

struct RandomProjection {
    mat: Vec<f32>,  // [64 × 784]
    n_out: usize,
    n_in: usize,
}

impl RandomProjection {
    fn new(n_out: usize, n_in: usize, seed: u32) -> Self {
        let n = n_out * n_in;
        let mut mat = vec![0.0f32; n];
        let mut state = seed;
        for i in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = (state >> 8) as f32 / 16777216.0;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (state >> 8) as f32 / 16777216.0;
            let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
            mat[i] = r * (2.0 * std::f32::consts::PI * u2).cos() * 0.5;
        }
        Self { mat, n_out, n_in }
    }

    fn project(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_out];
        for o in 0..self.n_out {
            let row_off = o * self.n_in;
            let mut sum = 0.0;
            for i in 0..self.n_in {
                sum += self.mat[row_off + i] * input[i];
            }
            out[o] = sum;
        }
        out
    }
}

// ─── IDX parser ──────────────────────────────────────────

fn load_images(path: &str) -> Vec<Vec<f32>> {
    let data = fs::read(path).expect("failed to read images file");
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let pixel_count = rows * cols;
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let offset = 16 + i * pixel_count;
        let mut img = Vec::with_capacity(pixel_count);
        for p in 0..pixel_count {
            img.push(data[offset + p] as f32 / 255.0);
        }
        images.push(img);
    }
    images
}

fn load_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).expect("failed to read labels file");
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    data[8..8 + n].to_vec()
}

fn check_data_file(path: &str) {
    if Path::new(path).exists() { return; }
    println!("  ✗ {path} not found!");
    println!("  Run: python3 examples/download_mnist.py");
    panic!("missing MNIST data file: {path}");
}

// ─── Fast exp (portable, no crate dep) ───────────────────

#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let log2e = 1.4426950408889634f32;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;
    let c1 = 0.6930321207;
    let c2 = 0.2413797629;
    let c3 = 0.0520972777;
    let c4 = 0.0093553389;
    let poly = ((c4 * f + c3) * f + c2) * f + c1;
    let exp2f = poly * f + 1.0;
    let biased = n.wrapping_add(127);
    let exp2n = if biased <= 0 { 0.0 }
        else if biased >= 255 { f32::INFINITY }
        else { f32::from_bits((biased as u32) << 23) };
    exp2n * exp2f
}

// ─── Softmax helper ──────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| {
        fast_exp((x - max_v).max(-50.0).min(50.0))
    }).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 1e-20 {
        exps.iter_mut().for_each(|e| *e /= sum);
    }
    exps
}

// ═══════════════════════════════════════════════════════════
//  MultiHybridDecoder — shared IK + 10×NLMS + zero-shot physics
// ═══════════════════════════════════════════════════════════
//
// Feature vector: [pst_linear_features(N) | ik_drive]
//   pst_dim = 3·N·6 + 12
//   total_dim = pst_dim + 1
//
// Zero-shot init:
//   IK weight = 1.0  → predict = чистая физика (InverseKuramoto)
//   PST weights = 0.0 → без обучения
//
// После NLMS training:
//   predict = Σ(w_i·pst_i) + w_ik·ik_drive
//
// Example learning curve:
//   Step 1:  accuracy = IK physics (zero-shot, ~10-15%)
//   Step N:  accuracy → 70%+ (learned PST + IK)

struct MultiHybridDecoder {
    n: usize,
    pst_dim: usize,
    /// 10 NLMS decoders, one per class
    nlms: Vec<NlmsDecoder>,
    /// Shared InverseKuramoto decoder (tracks amplitude changes)
    ik: InverseKuramotoDecoder,
    /// Reusable amplitude buffer [3 × N]
    amps_buf: Vec<f32>,
    /// Reusable feature vector [total_dim]
    fv: Vec<f32>,
}

impl MultiHybridDecoder {
    fn new(n: usize, n_classes: usize, lr: f32, eps: f32,
           alpha: f32, drive_gain: f32, dt: f32) -> Self {
        let pst_dim = pst_linear_dim(n);
        let total_dim = pst_dim + 1; // +1 for IK drive

        // 10 NLMS decoders, each with IK weight = 1.0 (zero-shot)
        let mut nlms = Vec::with_capacity(n_classes);
        for _ in 0..n_classes {
            let mut dec = NlmsDecoder::new(total_dim, lr, eps);
            // Zero-shot: init with pure IK physics
            dec.weights[pst_dim] = 1.0;
            nlms.push(dec);
        }

        // Shared IK decoder on concatenated amplitudes (3 populations × N)
        let mut ik = InverseKuramotoDecoder::new(3 * n, alpha, drive_gain);
        ik.set_dt(dt);

        Self {
            n, pst_dim,
            nlms, ik,
            amps_buf: vec![0.0f32; 3 * n],
            fv: vec![0.0f32; total_dim],
        }
    }

    /// Build concatenated amplitude buffer [r1 | r2 | r3] from TriOscillator.
    #[inline]
    fn build_amps_buf<B: taas_checkmate::t3::T3Backend>(&mut self, tri: &TriOscillator<B>) {
        let n = self.n;
        self.amps_buf[..n].copy_from_slice(&tri.r1[..n]);
        self.amps_buf[n..2 * n].copy_from_slice(&tri.r2[..n]);
        self.amps_buf[2 * n..3 * n].copy_from_slice(&tri.r3[..n]);
    }

    /// Build feature vector: fv = [pst_features | ik_drive]
    fn build_fv<B: taas_checkmate::t3::T3Backend>(&mut self, tri: &TriOscillator<B>) {
        // Amplitudes → IK drive
        self.build_amps_buf(tri);
        let ik_drive = self.ik.decode(&self.amps_buf);

        // PST linear features (0 alloc — пишем напрямую в self.fv)
        pst_linear_features_into(tri, &mut self.fv[..self.pst_dim]);
        debug_assert_eq!(self.pst_dim, taas_checkmate::t3::pst_linear_dim(self.n));
        // fv[..self.pst_dim] уже заполнен
        self.fv[self.pst_dim] = ik_drive;
    }

    /// Predict logits for all classes from TriOscillator state.
    /// Zero-shot (before training): predict = pure IK physics.
    fn predict_all<B: taas_checkmate::t3::T3Backend>(&mut self, tri: &TriOscillator<B>) -> Vec<f32> {
        self.build_fv(tri);
        let mut out = Vec::with_capacity(self.nlms.len());
        for d in &self.nlms {
            out.push(d.predict(&self.fv));
        }
        out
    }

    /// Train all decoders from TriOscillator state.
    /// Returns RMS error across all classes.
    fn train_all<B: taas_checkmate::t3::T3Backend>(&mut self, tri: &TriOscillator<B>, target_class: usize) -> f32 {
        self.build_fv(tri);
        let mut total_err = 0.0;
        for (c, d) in self.nlms.iter_mut().enumerate() {
            let target = if c == target_class { 1.0 } else { -1.0 };
            let err = d.train(&self.fv, target);
            total_err += err * err;
        }
        (total_err / self.nlms.len() as f32).sqrt()
    }

    /// Reset all decoders to zero-shot state.
    fn reset_all(&mut self) {
        for d in &mut self.nlms {
            d.reset();
            d.weights[self.pst_dim] = 1.0; // restore IK init
        }
        self.ik.reset();
        self.fv.fill(0.0);
    }
}

// ─── Main ─────────────────────────────────────────────────

fn main() {
    let pst_dim = pst_linear_dim(N_OSCILLATORS);
    let total_dim = pst_dim + 1;

    println!("═══════════════════════════════════════════════════");
    println!("  MNIST via T3 — HybridDecoder");
    println!("  (InverseKuramoto + PST + NLMS, zero-shot physics)");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Architecture:");
    println!("  T3 oscillators: 3 populations × {N_OSCILLATORS}");
    println!("  PST features:   {pst_dim} dim");
    println!("  IK drive:       +1 dim → total {total_dim} dim");
    println!("  Decoder:        shared IK + 10×NLMS → softmax → digit");
    println!("  Init:           IK weight = 1.0 → zero-shot physics");
    println!("  T3 steps:       {N_T3_STEPS} @ dt = {DT}");
    println!("  Coupling:       w12=0.3, w13=0.15, w23=0.08");
    println!("  Encoder:        784 pixels → random projection → {N_OSCILLATORS} drives");
    println!("  Learning rate:  {LR}");
    println!();

    // 1. Load data
    println!("─── Loading MNIST data ───");
    fs::create_dir_all(DATA_DIR).ok();
    for f in &["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"] {
        check_data_file(&format!("{DATA_DIR}/{f}"));
    }

    let train_images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let train_labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let test_images = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let test_labels = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));

    println!("  Training:   {} images", train_images.len());
    println!("  Test:       {} images", test_images.len());
    println!();

    // 2. Random projection: 784 → 64
    let proj = RandomProjection::new(N_OSCILLATORS, N_INPUT, SEED);

    // 3. Create T3 oscillator system
    println!("─── Initialising T3 ───");
    let mut tri = TriOscillator::new(
        N_OSCILLATORS, FREQ_MIN, FREQ_MAX, Avx2Backend,
    );
    tri.set_uniform_coupling(0.3, 0.15, 0.08);
    tri.drive_gain = 8.0;
    tri.alpha = 1.0;
    tri.epsilon = 0.01;
    tri.gamma1 = 0.5;
    tri.gamma2 = 0.8;
    tri.gamma3 = 1.0;
    tri.phase_gate_gain = 6.0;

    // 4. Create MultiHybridDecoder
    let mut cls = MultiHybridDecoder::new(
        N_OSCILLATORS, N_ACTIONS, LR, 1e-8, 1.0, 8.0, DT,
    );
    println!("  Zero-shot init: IK weight = 1.0 → first predict = pure physics");

    // 5. Zero-shot accuracy (pure IK physics, NO training)
    // Per-image: init IK on step 1, measure on step N → internal derivative
    println!();
    println!("─── Zero-shot test (pure IK physics, NO training) ───");
    let mut zs_correct = 0usize;
    for (i, img) in test_images.iter().enumerate().take(1000) {
        let drives = proj.project(img);
        for j in 0..N_OSCILLATORS {
            tri.drive1[j] = drives[j];
            tri.drive2[j] = 0.0;
            tri.drive3[j] = 0.0;
        }
        // Step 1: init IK prev_amps for this image (returns 0)
        cls.ik.reset();
        tri.step(DT);
        let _ = cls.predict_all(&tri);  // IK init: copies amps, returns 0
        // Steps 2..N: burn-in — IK measures step(N-1)→step(N)
        for _ in 1..N_T3_STEPS { tri.step(DT); }

        let logits = cls.predict_all(&tri);
        let probs = softmax(&logits);
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == test_labels[i] as usize { zs_correct += 1; }
    }
    let zs_acc = zs_correct as f64 / 1000.0 * 100.0;
    println!("  Zero-shot accuracy (first 1000 test): {zs_acc:.1}%");
    println!("  (InverseKuramoto — internal derivative within same image)");

    // 6. Train
    println!();
    println!("─── Training ───");
    let total_start = Instant::now();

    let mut total_correct = 0usize;
    for img_idx in 0..N_TRAIN {
        let drives = proj.project(&train_images[img_idx]);

        for i in 0..N_OSCILLATORS {
            tri.drive1[i] = drives[i];
            tri.drive2[i] = 0.0;
            tri.drive3[i] = 0.0;
        }
        for _ in 0..N_T3_STEPS { tri.step(DT); }

        let label = train_labels[img_idx] as usize;
        let _rmse = cls.train_all(&tri, label);

        // Predict immediately (online learning)
        let logits = cls.predict_all(&tri);
        let probs = softmax(&logits);
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == label { total_correct += 1; }

        if (img_idx + 1) % BATCH_SIZE == 0 {
            let accuracy = total_correct as f64 / (img_idx + 1) as f64 * 100.0;
            let elapsed = total_start.elapsed();
            let img_per_sec = (img_idx + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "  [{:5}/{}] acc={:.1}% | {:.0} img/s | {:.1}s",
                img_idx + 1, N_TRAIN, accuracy,
                img_per_sec, elapsed.as_secs_f64(),
            );
        }
    }

    let total_elapsed = total_start.elapsed();
    let final_accuracy = total_correct as f64 / N_TRAIN as f64 * 100.0;
    println!();
    println!("─── Training Done ───");
    println!("  Final accuracy:  {final_accuracy:.1}% ({total_correct}/{N_TRAIN})");
    println!("  Total time:      {:.1}s ({:.0} img/s)",
        total_elapsed.as_secs_f64(),
        N_TRAIN as f64 / total_elapsed.as_secs_f64(),
    );

    // 7. Test
    println!();
    println!("─── Testing on {N_TEST} held-out images ───");
    let test_start = Instant::now();
    let mut test_correct = 0usize;
    for (i, img) in test_images.iter().enumerate() {
        let drives = proj.project(img);
        for j in 0..N_OSCILLATORS {
            tri.drive1[j] = drives[j];
            tri.drive2[j] = 0.0;
            tri.drive3[j] = 0.0;
        }
        for _ in 0..N_T3_STEPS { tri.step(DT); }

        let logits = cls.predict_all(&tri);
        let probs = softmax(&logits);
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == test_labels[i] as usize { test_correct += 1; }
    }
    let test_elapsed = test_start.elapsed();
    let test_accuracy = test_correct as f64 / N_TEST as f64 * 100.0;
    println!("  Test accuracy:   {test_accuracy:.1}% ({test_correct}/{N_TEST})");
    println!("  Test time:       {:.1}s ({:.0} img/s)",
        test_elapsed.as_secs_f64(),
        N_TEST as f64 / test_elapsed.as_secs_f64(),
    );

    // 8. Summary
    println!();
    println!("─── Summary ───");
    println!("  Zero-shot (IK physics):     {zs_acc:.1}%");
    println!("  After training:              {final_accuracy:.1}% (train) → {test_accuracy:.1}% (test)");
    println!("  IK improvement:              {:.1}×", final_accuracy / zs_acc.max(1.0));
    println!();
    println!("═══ Done ═══");
}
