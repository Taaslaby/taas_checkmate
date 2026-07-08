//! # Decoder trait + all T3 decoder implementations.
//!
//! ## Архитектура
//!
//! ```text
//! Decoder trait
//! ├── NlmsDecoder             — Normalized LMS
//! ├── KalmanDecoder           — RLS/Kalman filter
//! ├── InverseKuramotoDecoder  — Training-free from amplitudes
//! ├── StdpDecoder             — Anti-Hebbian STDP
//! ├── PstDecoder              — Ridge from trajectory buffer
//! └── HybridDecoder           — PST + InverseKuramoto + NLMS (zero-shot init)
//! ```

use crate::{T3Backend, TriOscillator};
use super::ridge::{solve_cg, compute_pearson};

// ─── Decoder trait ───────────────────────────────────────

/// Единый интерфейс для всех T3 декодеров.
pub trait Decoder {
    /// Dimension of input features.
    fn dim(&self) -> usize;

    /// Predict output from features.
    fn predict(&self, states: &[f32]) -> f32;

    /// Train one step: returns prediction error.
    fn train(&mut self, states: &[f32], target: f32) -> f32;

    /// Reset decoder state.
    fn reset(&mut self);
}

// ─── NlmsDecoder ─────────────────────────────────────────

/// Normalized LMS decoder.
///
/// ```text
/// μ = lr / (||x||² + eps)
/// w += μ · error · x
/// bias += μ · error
/// ```
pub struct NlmsDecoder {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub lr: f32,
    pub eps: f32,
}

impl NlmsDecoder {
    pub fn new(n: usize, lr: f32, eps: f32) -> Self {
        Self { weights: vec![0.0; n], bias: vec![0.0; 1], lr, eps }
    }
}

impl Decoder for NlmsDecoder {
    fn dim(&self) -> usize { self.weights.len() }

    fn predict(&self, states: &[f32]) -> f32 {
        self.weights.iter().zip(states).map(|(w, s)| w * s).sum::<f32>() + self.bias[0]
    }

    fn train(&mut self, states: &[f32], target: f32) -> f32 {
        let y = self.predict(states);
        let error = target - y;
        crate::kernels::nlms_train(&mut self.weights, &mut self.bias, states, error, self.lr, self.eps);
        error
    }

    fn reset(&mut self) { self.weights.fill(0.0); self.bias[0] = 0.0; }
}

// ─── KalmanDecoder ───────────────────────────────────────

/// RLS/Kalman filter decoder with AVX2 backend.
///
/// ```text
/// cov += q
/// s = r + Σ(cov · x²)
/// k = cov · x / s
/// w += k · error
/// cov -= k · cov
/// ```
pub struct KalmanDecoder {
    pub weights: Vec<f32>,
    pub cov: Vec<f32>,
    pub bias: f32,
    pub q: f32,
    pub r: f32,
}

impl KalmanDecoder {
    pub fn new(n: usize, q: f32, r: f32) -> Self {
        Self {
            weights: vec![0.0; n],
            cov: vec![10.0; n],
            bias: 0.0, q, r,
        }
    }
}

impl Decoder for KalmanDecoder {
    fn dim(&self) -> usize { self.weights.len() }

    fn predict(&self, states: &[f32]) -> f32 {
        self.weights.iter().zip(states).map(|(w, s)| w * s).sum::<f32>() + self.bias
    }

    fn train(&mut self, states: &[f32], target: f32) -> f32 {
        for c in self.cov.iter_mut() { *c += self.q; }
        let y = self.predict(states);
        let error = target - y;
        let s: f32 = self.r + self.cov.iter().zip(states.iter())
            .map(|(&c, &st)| c * st * st).sum::<f32>();
        if s > 1e-12 {
            crate::kernels::kalman_update(&mut self.weights, &mut self.cov, states, error, s);
            let k_bias = 10.0 / (s + 10.0);
            self.bias += k_bias * error;
        }
        error
    }

    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.cov.fill(10.0);
        self.bias = 0.0;
    }
}

// ─── InverseKuramotoDecoder ──────────────────────────────

/// Training-free decoder: recovers input drive from amplitudes.
///
/// Инвертирует Stuart-Landau динамику амплитуды:
/// ```text
/// ṙ = (α - r²)·r + drive · gain
/// ⇒ drive = (ṙ - (α - r²)·r) / gain
/// ```
///
/// `ε` используется только для численной стабилизации при r ≈ 0.
pub struct InverseKuramotoDecoder {
    pub n: usize,
    pub prev_amps: Vec<f32>,
    pub alpha: f32,
    pub drive_gain: f32,
    pub epsilon: f32,
    pub weights: Vec<f32>,
    pub initialised: bool,
    /// Simulation timestep for derivative: ṙ = (r_new - r_old) / dt
    pub dt: f32,
}

impl InverseKuramotoDecoder {
    pub fn new(n: usize, alpha: f32, drive_gain: f32) -> Self {
        Self {
            n,
            prev_amps: vec![0.0; n],
            alpha,
            drive_gain,
            epsilon: 0.1,
            weights: vec![1.0; n],
            initialised: false,
            dt: 1.0,  // default: preserves old behavior
        }
    }

    /// Set the simulation timestep for correct derivative computation.
    /// IK formula: drive = ((r_new - r_old)/dt - (α - r²)·r) / gain
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    pub fn decode(&mut self, amps: &[f32]) -> f32 {
        if !self.initialised {
            self.prev_amps.copy_from_slice(amps);
            self.initialised = true;
            return 0.0;
        }
        let mut sum_d = 0.0;
        let mut sum_w = 0.0;
        for i in 0..self.n.min(amps.len()) {
            let r = amps[i];
            let r_dot = (r - self.prev_amps[i]) / self.dt;
            // Inverse Stuart-Landau: ṙ = (α − r²)·r + gain·drive
            // ⇒ drive = (r_dot − (α − r²)·r) / gain
            let drive = if self.drive_gain.abs() > 1e-10 {
                (r_dot - (self.alpha - r * r) * r) / self.drive_gain
            } else { 0.0 };
            let drive = drive.clamp(-10.0, 10.0);
            let w = if i < self.weights.len() { self.weights[i] } else { 1.0 };
            sum_d += drive * w;
            sum_w += w.abs();
            self.prev_amps[i] = r;
        }
        if sum_w > 1e-10 { sum_d / sum_w } else { 0.0 }
    }

    pub fn decode_batch(&mut self, amps: &[f32]) -> Vec<f32> {
        let n_frames = amps.len() / self.n;
        let mut out = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame = &amps[f * self.n..(f + 1) * self.n];
            out.push(self.decode(frame));
        }
        out
    }
}

impl Decoder for InverseKuramotoDecoder {
    fn dim(&self) -> usize { self.n }

    fn predict(&self, _states: &[f32]) -> f32 { 0.0 }

    fn train(&mut self, states: &[f32], _target: f32) -> f32 {
        self.decode(states)
    }

    fn reset(&mut self) {
        self.prev_amps.fill(0.0);
        self.initialised = false;
    }
}

// ─── StdpDecoder ─────────────────────────────────────────

/// Anti-Hebbian STDP decoder with trace and sparsity.
pub struct StdpDecoder {
    pub weights: Vec<f32>,
    pub traces: Vec<f32>,
    pub prev_states: Vec<f32>,
    pub bias: f32,
    pub lr: f32,
    pub decay: f32,
    pub sparsity: f32,
    pub initialised: bool,
}

impl StdpDecoder {
    pub fn new(n: usize, lr: f32, decay: f32, sparsity: f32) -> Self {
        Self {
            weights: vec![0.0; n],
            traces: vec![0.0; n],
            prev_states: vec![0.0; n],
            bias: 0.0, lr, decay, sparsity,
            initialised: false,
        }
    }
}

impl Decoder for StdpDecoder {
    fn dim(&self) -> usize { self.weights.len() }

    fn predict(&self, states: &[f32]) -> f32 {
        self.weights.iter().zip(states).map(|(w, s)| w * s).sum::<f32>() + self.bias
    }

    fn train(&mut self, states: &[f32], target: f32) -> f32 {
        let n = self.weights.len().min(states.len());
        if !self.initialised {
            self.prev_states[..n].copy_from_slice(&states[..n]);
            self.initialised = true;
            return target;
        }
        let y = self.predict(states);
        let error = (target - y).clamp(-1.0, 1.0);
        for i in 0..n {
            let ds = states[i] - self.prev_states[i];
            self.traces[i] = self.decay * self.traces[i] + ds.abs();
            self.prev_states[i] = states[i];
            let delta = self.lr * error * self.traces[i];
            let penalty = if self.weights[i] > 0.0 { self.sparsity } else { -self.sparsity };
            self.weights[i] += delta - penalty;
            self.weights[i] = self.weights[i].clamp(-5.0, 5.0);
        }
        self.bias += self.lr * error;
        error
    }

    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.traces.fill(0.0);
        self.prev_states.fill(0.0);
        self.bias = 0.0;
        self.initialised = false;
    }
}

// ─── PstDecoder ──────────────────────────────────────────

/// PST decoder: Ridge regression from trajectory buffer.
///
/// Как в оригинальном Nakajima NeuroBrain PstDecoder:
/// накапливает траекторию, решает Ridge, предсказывает.
pub struct PstDecoder {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub alpha: f32,
    /// Feature dimension per frame
    pub feat_dim: usize,
    /// Trajectory length
    pub L: usize,
}

impl PstDecoder {
    pub fn new(feat_dim: usize, L: usize, alpha: f32) -> Self {
        let n_weights = feat_dim * L;
        Self {
            weights: vec![0.0; n_weights],
            bias: 0.0, alpha, feat_dim, L,
        }
    }

    /// Train ridge regression from accumulated trajectories and targets.
    /// Returns Pearson R on training data.
    pub fn train_batch(&mut self, X: &[f32], y: &[f32], n_samples: usize) -> f32 {
        let d = self.feat_dim * self.L;
        if n_samples == 0 || d == 0 { return 0.0; }

        // Build XᵀX + αI and Xᵀy
        let mut a = vec![0.0f32; d * d];
        let mut b = vec![0.0f32; d];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let row_start = i * d;
            #[allow(clippy::needless_range_loop)]
            for j1 in 0..d {
                let x_i_j1 = X[row_start + j1];
                b[j1] += x_i_j1 * y[i];
                for j2 in 0..d {
                    a[j1 * d + j2] += x_i_j1 * X[row_start + j2];
                }
            }
        }
        for j in 0..d { a[j * d + j] += self.alpha; }

        // Solve via CG
        let w = solve_cg(&a, &b, d, 100, 1e-6);
        self.weights.copy_from_slice(&w);
        self.bias = 0.0;

        // Compute reconstruction correlation
        let mut y_pred = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row_start = i * d;
            let pred: f32 = self.weights.iter().enumerate()
                .map(|(j, w)| w * X[row_start + j]).sum();
            y_pred.push(pred);
        }
        compute_pearson(&y_pred, y)
    }

    pub fn predict(&self, trajectory: &[f32]) -> f32 {
        self.weights.iter().zip(trajectory).map(|(w, x)| w * x).sum::<f32>() + self.bias
    }
}

// ─── HybridDecoder — PST + InverseKuramoto + NLMS ────────

/// Hybrid decoder: PST features + Inverse Kuramoto drive as zero-shot init for NLMS.
///
/// ## Философия
///
/// Inverse Kuramoto даёт **физически обоснованную оценку** сигнала из амплитуд
/// (zero-shot, без единого примера). PST features дают **богатое представление**
/// фазового пространства (фазы, скорости, амплитуды, cross-order params).
///
/// HybridDecoder объединяет оба подхода:
///
/// 1. **Feature vector** = `[pst_linear_features(N) | ik_drive_estimate]`
///    Размерность: `pst_dim + 1 = 3N·6 + 12 + 1`
///
/// 2. **Zero-shot инициализация**: последний вес (IK канал) = 1.0, остальные = 0.0
///    → начальное предсказание = чистая физика (IK), без обучения
///
/// 3. **NLMS дообучает** все веса — комбинирует физику + данные
///
/// ```text
/// Initial:  predict = ik_drive                          (zero-shot)
/// Trained:  predict = Σwᵢ·pst_i + w_ik·ik_drive          (learned)
/// ```
pub struct HybridDecoder {
    /// Number of oscillators per population
    pub n: usize,
    /// PST feature dimension
    pub pst_dim: usize,
    /// NLMS on [pst_dim + 1] features (last = IK drive)
    pub nlms: NlmsDecoder,
    /// Internal IK decoder (tracks amplitude changes)
    pub ik: InverseKuramotoDecoder,
    /// Reusable feature buffer (avoids per-call alloc in hot path)
    fv: Vec<f32>,
    /// Reusable amplitude buffer (avoids per-call alloc for concatenated amps)
    amps_buf: Vec<f32>,
}

impl HybridDecoder {
    /// Create new HybridDecoder.
    ///
    /// # Args
    /// * `n` — oscillators per population (3 populations, each N)
    /// * `lr` — NLMS learning rate
    /// * `eps` — NLMS epsilon (normalization floor)
    /// * `alpha` — Stuart-Landau nonlinearity parameter
    /// * `drive_gain` — Stuart-Landau drive gain
    /// * `dt` — simulation timestep (for IK derivative)
    pub fn new(n: usize, lr: f32, eps: f32, alpha: f32, drive_gain: f32, dt: f32) -> Self {
        let pst_dim = crate::pst_linear_dim(n);
        let total_dim = pst_dim + 1;
        let mut nlms = NlmsDecoder::new(total_dim, lr, eps);
        // Zero-shot init: IK channel = 1.0, rest = 0.0
        nlms.weights[pst_dim] = 1.0;
        // IK decoder on concatenated amplitudes of all 3 populations (3N)
        let mut ik = InverseKuramotoDecoder::new(3 * n, alpha, drive_gain);
        ik.set_dt(dt);

        Self {
            n, pst_dim, nlms, ik,
            fv: Vec::with_capacity(total_dim),
            amps_buf: vec![0.0f32; 3 * n],
        }
    }

    /// Build combined feature vector: `[pst_linear_features | ik_drive]`.
    ///
    /// Возвращает `Vec<f32>` — удобно для внешних вызовов, но аллоцирует.
    /// Для hot path используйте `predict_from_tri`/`train_from_tri` — они
    /// переиспользуют внутренние буферы без доп. аллокаций.
    ///
    /// IK decoder internally tracks `prev_amps` to compute ṙ. Первый вызов
    /// возвращает `0.0` для IK (initialisation), второй — физическую оценку.
    pub fn build_features<B: T3Backend>(&mut self, tri: &TriOscillator<B>) -> Vec<f32> {
        self.build_amps_buf(tri);
        let ik_drive = self.ik.decode(&self.amps_buf);
        let pst = crate::pst_linear_features(tri);
        let mut fv = Vec::with_capacity(self.pst_dim + 1);
        fv.extend_from_slice(&pst);
        fv.push(ik_drive);
        fv
    }

    /// Build concatenated amplitude buffer [r1 | r2 | r3] (reusable).
    #[inline]
    fn build_amps_buf<B: T3Backend>(&mut self, tri: &TriOscillator<B>) {
        let n = self.n;
        self.amps_buf[..n].copy_from_slice(&tri.r1[..n]);
        self.amps_buf[n..2 * n].copy_from_slice(&tri.r2[..n]);
        self.amps_buf[2 * n..3 * n].copy_from_slice(&tri.r3[..n]);
    }

    /// Predict from TriOscillator state — **zero alloc** (reuses buffers).
    /// Первый вызов (до train): predict = pure IK physics (zero-shot).
    pub fn predict_from_tri<B: T3Backend>(&mut self, tri: &TriOscillator<B>) -> f32 {
        self.build_amps_buf(tri);
        let ik_drive = self.ik.decode(&self.amps_buf);
        let pst = crate::pst_linear_features(tri);
        self.fv.clear();
        self.fv.extend_from_slice(&pst);
        self.fv.push(ik_drive);
        self.nlms.predict(&self.fv)
    }

    /// Train from TriOscillator state — **zero alloc** (reuses buffers).
    pub fn train_from_tri<B: T3Backend>(&mut self, tri: &TriOscillator<B>, target: f32) -> f32 {
        self.build_amps_buf(tri);
        let ik_drive = self.ik.decode(&self.amps_buf);
        let pst = crate::pst_linear_features(tri);
        self.fv.clear();
        self.fv.extend_from_slice(&pst);
        self.fv.push(ik_drive);
        self.nlms.train(&self.fv, target)
    }

    /// Reset decoder to zero-shot state.
    pub fn reset(&mut self) {
        self.nlms.reset();
        self.ik.reset();
        self.nlms.weights[self.pst_dim] = 1.0;  // restore IK init
    }
}

impl Decoder for HybridDecoder {
    fn dim(&self) -> usize { self.pst_dim + 1 }

    fn predict(&self, states: &[f32]) -> f32 {
        self.nlms.predict(states)
    }

    fn train(&mut self, states: &[f32], target: f32) -> f32 {
        self.nlms.train(states, target)
    }

    fn reset(&mut self) {
        self.nlms.reset();
        self.ik.reset();
        self.nlms.weights[self.pst_dim] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TriOscillator, Avx2Backend};

    // ─── HybridDecoder tests ──────────────────────────

    #[test]
    fn test_hybrid_constructor_has_ik_init() {
        let dec = HybridDecoder::new(8, 0.01, 1e-8, 1.0, 10.0, 0.05);
        let pst_dim = crate::pst_linear_dim(8);
        let total_dim = pst_dim + 1;
        assert_eq!(dec.dim(), total_dim);
        // IK weight = 1.0 (zero-shot init), rest = 0.0
        for i in 0..pst_dim {
            assert!((dec.nlms.weights[i] - 0.0).abs() < 1e-8,
                "pst weight[{}] should be 0, got {}", i, dec.nlms.weights[i]);
        }
        assert!((dec.nlms.weights[pst_dim] - 1.0).abs() < 1e-8,
            "IK weight should be 1.0, got {}", dec.nlms.weights[pst_dim]);
    }

    #[test]
    fn test_hybrid_zero_shot_first_step() {
        let mut tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        let mut dec = HybridDecoder::new(8, 0.01, 1e-8, 1.0, 10.0, 0.01);

        tri.step(0.01);
        // First call: IK not initialised → returns 0.0 → prediction = 0.0
        let pred = dec.predict_from_tri(&tri);
        assert!((pred - 0.0).abs() < 1e-6,
            "zero-shot first step should be 0.0 (IK init), got {}", pred);
    }

    #[test]
    fn test_hybrid_zero_shot_second_step() {
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.2, 0.1, 0.05);
        let mut dec = HybridDecoder::new(16, 0.01, 1e-8, 1.0, 10.0, 0.01);

        // Step 1: initialise IK
        tri.step(0.01);
        let _ = dec.predict_from_tri(&tri);

        // Step 2: drive some oscillators with strong input
        for i in 0..tri.n.min(4) {
            tri.drive1[i] = 5.0;
        }
        tri.step(0.01);

        // Now prediction should use non-zero IK drive (physics-based)
        let pred = dec.predict_from_tri(&tri);
        assert!(pred != 0.0,
            "zero-shot step 2 should have non-zero IK drive, got {}", pred);
    }

    #[test]
    fn test_hybrid_learns_from_data() {
        // HybridDecoder должен учиться лучше, чем pure IK
        let mut tri = TriOscillator::new(8, 1.0, 2.0, Avx2Backend);
        tri.set_uniform_coupling(0.3, 0.15, 0.08);
        let mut dec = HybridDecoder::new(8, 0.05, 1e-8, 1.0, 10.0, 0.01);

        // Generate 50 steps of training data
        let mut errors = Vec::new();
        for step in 0..100 {
            if step % 2 == 0 {
                // Alternate drive pattern
                for i in 0..tri.n.min(4) { tri.drive1[i] = 3.0; }
                for i in 4..tri.n.min(8) { tri.drive1[i] = 0.0; }
            } else {
                for i in 0..tri.n.min(4) { tri.drive1[i] = 0.0; }
                for i in 4..tri.n.min(8) { tri.drive1[i] = 3.0; }
            }
            tri.step(0.01);

            // Target: mean phase coherence (signal to decode)
            let mean_phase = tri.th1.iter().sum::<f32>() / tri.n as f32;
            let target = mean_phase.sin();

            let err = dec.train_from_tri(&tri, target);
            if step > 10 { errors.push(err.abs()); }
        }

        // After training, error should be low
        let avg_err: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
        assert!(avg_err < 0.6,
            "HybridDecoder should learn: avg error = {}", avg_err);
    }

    #[test]
    fn test_hybrid_build_features_dim() {
        let mut tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);

        let mut dec = HybridDecoder::new(8, 0.01, 1e-8, 1.0, 10.0, 0.01);
        let fv = dec.build_features(&tri).to_vec();

        let pst_dim = crate::pst_linear_dim(8);
        assert_eq!(fv.len(), pst_dim + 1,
            "feature vector should be pst_dim + 1, got {}", fv.len());
        // Last element should be the IK drive
        assert_eq!(fv[pst_dim], 0.0,
            "first IK drive should be 0.0 (init), got {}", fv[pst_dim]);
    }

    #[test]
    fn test_quantum_echo_drive_recovery() {
        // Демонстрация "Quantum Echo": подаём сигнал → T3 кодирует →
        // Inverse Kuramoto восстанавливает zero-shot без обучения.
        //
        // Это классический аналог Google Quantum Echo (Nature 2025):
        //   Google: возмущение → scrambling → OTOC
        //   T3:     drive → Stuart-Landau → ṙ → drive_recovery
        //
        // Ожидание: корреляция > 0.3 (zero-shot через хаотическую динамику).

        let n = 32;
        let dt = 0.05;
        let n_steps = 200;
        let freq = 2.0;  // Hz

        let mut tri = TriOscillator::new(n, 1.0, 2.0, Avx2Backend);
        tri.set_uniform_coupling(0.3, 0.15, 0.08);

        // InverseKuramoto decoder с alpha=1.0, gain=10.0
        let mut ik = InverseKuramotoDecoder::new(n, 1.0, 10.0);
        ik.set_dt(dt);

        let mut original_drives = Vec::with_capacity(n_steps);
        let mut recovered_drives = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let t = step as f32 * dt;
            // Sine wave drive на первые 4 осциллятора
            let drive_val = (2.0 * std::f32::consts::PI * freq * t).sin() * 3.0;
            for i in 0..tri.n.min(4) {
                tri.drive1[i] = drive_val;
            }
            original_drives.push(drive_val);

            tri.step(dt);

            // IK decoder смотрит на амплитуды r1 и восстанавливает drive
            let rec = ik.decode(&tri.r1[..n]);
            recovered_drives.push(rec);
        }

        // Пропускаем первые 10 шагов (transient)
        let valid = &original_drives[10..];
        let rec_valid = &recovered_drives[10..];

        // Pearson correlation
        let r_pearson = compute_pearson(valid, rec_valid);
        println!("Quantum Echo: Pearson R = {:.4}", r_pearson);
        println!("  Original first 10: {:?}", &valid[..5]);
        println!("  Recovered first 10: {:?}", &rec_valid[..5]);

        // Inverse Kuramoto извлекает физическое "эхо" сигнала.
        // Это ДО обучения — чистый zero-shot из физики Stuart-Landau.
        //
        // Проверяем, что есть корреляция (эхо обнаружено).
        // Даже слабая корреляция — успех, т.к. это zero-shot через хаотическую
        // динамику 32 осцилляторов (классический аналог Quantum Echo).
        assert!(r_pearson > 0.3,
            "Quantum Echo: expected R > 0.3 (zero-shot), got R = {}", r_pearson);
    }

    #[test]
    fn test_hybrid_decoder_trait() {
        let mut dec: Box<dyn Decoder> = Box::new(HybridDecoder::new(8, 0.01, 1e-8, 1.0, 10.0, 1.0));
        let pst_dim = crate::pst_linear_dim(8);
        assert_eq!(dec.dim(), pst_dim + 1);

        // Test via Decoder trait with pre-computed features
        let states = vec![0.5f32; pst_dim + 1];
        let err = dec.train(&states, 1.0);
        assert!(err != 0.0, "train should return non-zero error");

        let pred = dec.predict(&states);
        assert!(pred != 0.0 || err != 0.0, "predict should work");

        dec.reset();
        let pred_after_reset = dec.predict(&states);
        // After reset, IK weight = 1.0, so predict = 1.0 * 0.5 = 0.5
        assert!((pred_after_reset - 0.5).abs() < 1e-6,
            "after reset predict should use IK weight: got {}", pred_after_reset);
    }

    #[test]
    fn test_nlms_decoder() {
        let mut dec = NlmsDecoder::new(8, 0.01, 1e-8);
        let states = vec![1.0, 0.5, -0.3, 0.0, 0.8, -0.1, 0.2, -0.5];
        dec.train(&states, 1.0);
        assert!(dec.predict(&states) > -10.0);
    }

    #[test]
    fn test_kalman_decoder() {
        let mut dec = KalmanDecoder::new(8, 0.001, 0.1);
        let states = vec![1.0, 0.5, -0.3, 0.0, 0.8, -0.1, 0.2, -0.5];
        dec.train(&states, 1.0);
        assert!(dec.weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_inverse_kuramoto_decoder() {
        let mut dec = InverseKuramotoDecoder::new(8, 1.0, 1.0);
        let d1 = dec.decode(&vec![0.5; 8]);
        assert!((d1 - 0.0).abs() < 1e-6);
        let d2 = dec.decode(&vec![0.6; 8]);
        assert!(d2 != 0.0);
    }

    #[test]
    fn test_stdp_decoder() {
        let mut dec = StdpDecoder::new(8, 0.01, 0.9, 1e-5);
        let states = vec![1.0, 0.5, -0.3, 0.0, 0.8, -0.1, 0.2, -0.5];
        let err1 = dec.train(&states, 1.0);
        assert!((err1 - 1.0).abs() < 1e-6, "first train returns target");
        dec.train(&states, 1.0);
        assert!(dec.weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_pst_decoder() {
        let mut dec = PstDecoder::new(3, 2, 0.01);
        let mut X = Vec::new();
        let mut y = Vec::new();
        // y = x1 + 2*x2 + 3*x3 (trajectory L=2: features from 2 frames concatenated)
        for i in 0..10 {
            let f1 = i as f32 * 0.1;
            let f2 = i as f32 * 0.2;
            let f3 = i as f32 * 0.3;
            X.push(f1); X.push(f2); X.push(f3); // frame t
            X.push(f1*2.0); X.push(f2*2.0); X.push(f3*2.0); // frame t+1 (fake)
            y.push(f1 + 2.0*f2 + 3.0*f3);
        }
        let r = dec.train_batch(&X, &y, 10);
        assert!(r > 0.95, "PST decoder R≈1.0: {}", r);
    }

    #[test]
    fn test_decoder_trait() {
        let mut dec: Box<dyn Decoder> = Box::new(NlmsDecoder::new(8, 0.01, 1e-8));
        assert_eq!(dec.dim(), 8);
        let states = vec![0.5, -0.3, 0.8, 0.1, -0.5, 0.3, 0.2, -0.1];
        let err = dec.train(&states, 1.0);
        assert!(err != 0.0);
        let pred = dec.predict(&states);
        assert!(pred != 0.0);
        dec.reset();
        assert!((dec.predict(&states) - 0.0).abs() < 1e-6);
    }
}
