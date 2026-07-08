//! # Спарковка — голографический док между спайками и T³.
//!
//! Pipeline:
//!   SPIKES (бинарные, sparse)
//!     → SpikeTrace (эксп. скользящее среднее, dense)
//!     → HologramEncoder (сжатие N → 3 через cos-проекции)
//!     → 3 числа для FM-драйва T³
//!
//! Варианты:
//! - [`Sparker`] — базовый: спайки → trace → hologram(3)
//! - [`NormalizedSparker`] — с online нормализацией (DC + scale)
//! - [`HybridSparker`] — с адаптивным FM gain по статистике спайков

use std::f32::consts::PI;

// ─── SpikeTrace ─────────────────────────────────────────────

/// Экспоненциальное сглаживание бинарных спайков в плотный сигнал.
///
/// trace[n][t] = α · trace[n][t-1] + (1-α) · spike[n][t]
/// где α = exp(-dt/τ), τ — постоянная времени (ms)
pub struct SpikeTrace {
    trace: Vec<f32>,
    alpha: f32,
}

impl SpikeTrace {
    /// Create exponential trace with time constant `tau_ms`, simulation step `dt_ms`.
    pub fn new(n_neurons: usize, tau_ms: f32, dt_ms: f32) -> Self {
        let alpha = (-dt_ms / tau_ms).exp();
        Self { trace: vec![0.0; n_neurons], alpha }
    }

    /// Update trace with one step of spikes (in-place EMA).
    pub fn update(&mut self, spikes: &[f32]) {
        let a = self.alpha;
        let b = 1.0 - a;
        for (t, &s) in self.trace.iter_mut().zip(spikes) {
            *t = a * *t + b * s;
        }
    }

    /// Current trace values (read-only).
    pub fn get(&self) -> &[f32] { &self.trace }
}

// ─── HologramEncoder ───────────────────────────────────────

/// Сжимает N нейронов → K чисел через косо-проекцию.
///
/// По умолчанию (K=3) использует PCA-оптимальный базис:
///   h_k = Σ trace[i] · cos(position[i] · direction_k)
///
/// direction_k — k-я главная компонента распределения позиций,
/// что гарантирует декоррелированные каналы.
pub struct HologramEncoder {
    proj: Vec<Vec<f32>>,  // [N][K]
    n: usize,
    k: usize,
}

impl HologramEncoder {
    /// Создать с PCA-базисом (K=3).
    pub fn new(positions: &[[f32; 2]]) -> Self {
        Self::with_k(positions, 3)
    }

    /// Создать с PCA-базисом и произвольным K.
    pub fn with_k(positions: &[[f32; 2]], k: usize) -> Self {
        let n = positions.len();
        if n == 0 || k == 0 {
            return Self { proj: vec![vec![]; n], n, k };
        }
        let dirs = Self::pca_directions(positions, k);
        let mut proj = Vec::with_capacity(n);
        for &pos in positions {
            let mut h = Vec::with_capacity(k);
            for &d in &dirs {
                let p = pos[0] * d[0] + pos[1] * d[1];
                h.push(p.cos());
            }
            proj.push(h);
        }
        Self { proj, n, k }
    }

    /// Создать с произвольными направлениями проекции.
    pub fn with_directions(positions: &[[f32; 2]], directions: &[[f32; 2]]) -> Self {
        let n = positions.len();
        let k = directions.len();
        if n == 0 || k == 0 {
            return Self { proj: vec![vec![]; n], n, k };
        }
        let mut proj = Vec::with_capacity(n);
        for &pos in positions {
            let mut h = Vec::with_capacity(k);
            for &d in directions {
                h.push((pos[0] * d[0] + pos[1] * d[1]).cos());
            }
            proj.push(h);
        }
        Self { proj, n, k }
    }

    /// PCA directions: до K главных компонент 2D позиций.
    fn pca_directions(positions: &[[f32; 2]], k: usize) -> Vec<[f32; 2]> {
        let n = positions.len();
        if n == 0 { return vec![]; }
        let mut mx = 0.0f64; let mut my = 0.0f64;
        for &p in positions { mx += p[0] as f64; my += p[1] as f64; }
        mx /= n as f64; my /= n as f64;
        let mut cxx = 0.0f64; let mut cxy = 0.0f64; let mut cyy = 0.0f64;
        for &p in positions {
            let dx = p[0] as f64 - mx; let dy = p[1] as f64 - my;
            cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
        }
        cxx /= n as f64; cxy /= n as f64; cyy /= n as f64;

        let trace = cxx + cyy;
        let delta = ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt();
        let l1 = (trace + delta) / 2.0;
        let _l2 = (trace - delta) / 2.0;

        let mut dirs: Vec<[f32; 2]> = Vec::with_capacity(k);
        let (e1x, e1y) = if cxy.abs() > 1e-12 { (cxy, l1 - cxx) } else { (1.0, 0.0) };
        let n1 = (e1x * e1x + e1y * e1y).sqrt();
        if n1 > 1e-12 { dirs.push([(e1x / n1) as f32, (e1y / n1) as f32]); }
        if k > 1 {
            let (e2x, e2y) = if cxy.abs() > 1e-12 { (l1 - cyy, cxy) } else { (0.0, 1.0) };
            let n2 = (e2x * e2x + e2y * e2y).sqrt();
            if n2 > 1e-12 { dirs.push([(e2x / n2) as f32, (e2y / n2) as f32]); }
        }
        while dirs.len() < k {
            let angle = (dirs.len() as f64) * PI as f64 / k as f64;
            dirs.push([angle.cos() as f32, angle.sin() as f32]);
        }
        if dirs.is_empty() {
            let angles = [0.0, PI / 3.0, 2.0 * PI / 3.0];
            dirs.extend(angles.iter().map(|&a| [a.cos(), a.sin()]));
        }
        dirs.truncate(k);
        dirs
    }

    /// Encode trace into K hologram channels.
    pub fn encode(&self, trace: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0; self.k];
        for i in 0..self.n {
            let t = trace[i];
            for j in 0..self.k {
                h[j] += t * self.proj[i][j];
            }
        }
        h
    }

    /// Плоская проекционная матрица K×N для GPU.
    pub fn projection_matrix(&self) -> Vec<f32> {
        let mut mat = Vec::with_capacity(self.k * self.n);
        for j in 0..self.k {
            for i in 0..self.n {
                mat.push(self.proj[i][j]);
            }
        }
        mat
    }

    /// Number of hologram channels.
    #[inline]
    pub fn k(&self) -> usize { self.k }
    /// Number of input neurons.
    #[inline]
    pub fn n(&self) -> usize { self.n }
}

// ─── Sparker — полный pipeline ─────────────────────────────

/// Спарковка: спайки → trace → hologram.
///
/// ```
/// use taas_t3::Sparker;
///
/// let positions = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]];
/// let mut sparker = Sparker::new(&positions, 50.0, 1.0);
/// sparker.feed(&[1.0, 0.0, 0.0, 0.0]);
/// let h = sparker.hologram();
/// assert_eq!(h.len(), 3);
/// ```
pub struct Sparker {
    trace: SpikeTrace,
    encoder: HologramEncoder,
}

impl Sparker {
    /// Create sparker with 2D neuron positions, EMA time constant `tau_ms`, step `dt_ms`.
    pub fn new(positions: &[[f32; 2]], tau_ms: f32, dt_ms: f32) -> Self {
        let n = positions.len();
        Self {
            trace: SpikeTrace::new(n, tau_ms, dt_ms),
            encoder: HologramEncoder::new(positions),
        }
    }

    /// Feed one step of spikes into the EMA trace.
    pub fn feed(&mut self, spikes: &[f32]) {
        self.trace.update(spikes);
    }

    /// Compute hologram from current trace (returns K = 3 channels).
    pub fn hologram(&self) -> Vec<f32> {
        self.encoder.encode(self.trace.get())
    }

    /// Current trace values (read-only).
    pub fn trace(&self) -> &[f32] {
        self.trace.get()
    }
}

// ─── SpikeStats — статистика спайков ───────────────────────

/// Статистика фрейма спайков для адаптивного gain.
#[derive(Default)]
pub struct SpikeStats {
    pub mean_rate: f32,
    pub burstiness: f32,
    pub synchrony: f32,
}

impl SpikeStats {
    /// Compute per-frame statistics: mean rate, burstiness, synchrony.
    pub fn compute_frame(spikes: &[f32], sample_rate: f32) -> Self {
        let n = spikes.len() as f32;
        let active = spikes.iter().filter(|&&s| s > 0.5).count() as f32;
        let mean_rate = active / n * sample_rate;

        let mut diffs = 0.0;
        let mut prev = 0.0f32;
        for &s in spikes {
            if (s - prev).abs() > 0.5 { diffs += 1.0; }
            prev = s;
        }
        let burstiness = if n > 1.0 { diffs / (n - 1.0) } else { 0.0 };

        let mean_spikes: f32 = spikes.iter().sum::<f32>() / n;
        let mut var = 0.0;
        for &s in spikes {
            let d = s - mean_spikes;
            var += d * d;
        }
        let synchrony = (var / n).sqrt();

        Self { mean_rate, burstiness, synchrony }
    }

    /// Scale base gain down when synchrony or firing rate is high.
    pub fn adaptive_gain(&self, base_gain: f32) -> f32 {
        let syn_factor = 1.0 / (1.0 + self.synchrony * 2.0);
        let rate_factor = 1.0 / (1.0 + self.mean_rate * 0.1);
        base_gain * syn_factor * rate_factor
    }
}

// ─── NormalizedSparker ─────────────────────────────────────

/// Спарковка с online нормализацией: DC subtraction + scale.
pub struct NormalizedSparker {
    inner: Sparker,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    target_std: f32,
    n_updates: u64,
}

impl NormalizedSparker {
    /// Create normalised sparker with online DC/scale estimation.
    pub fn new(positions: &[[f32; 2]], tau_ms: f32, dt_ms: f32, target_std: f32) -> Self {
        let k = 3;
        Self {
            inner: Sparker::new(positions, tau_ms, dt_ms),
            running_mean: vec![0.0; k],
            running_var: vec![1.0; k],
            target_std,
            n_updates: 0,
        }
    }

    /// Feed one step of spikes.
    pub fn feed(&mut self, spikes: &[f32]) {
        self.inner.feed(spikes);
    }

    /// Compute normalised hologram (DC-free, unit variance scaled to `target_std`).
    pub fn hologram(&mut self) -> Vec<f32> {
        let h = self.inner.hologram();
        let k = h.len();

        self.n_updates += 1;
        let n = self.n_updates as f32;
        for j in 0..k {
            let delta = h[j] - self.running_mean[j];
            self.running_mean[j] += delta / n;
            let delta2 = h[j] - self.running_mean[j];
            self.running_var[j] += delta * delta2;
        }

        let mut h_norm = vec![0.0; k];
        for j in 0..k {
            let std = (self.running_var[j] / n).sqrt() + 1e-8;
            let dc_free = h[j] - self.running_mean[j];
            h_norm[j] = dc_free * self.target_std / std;
        }
        h_norm
    }

    /// Current raw trace (read-only).
    pub fn trace(&self) -> &[f32] { self.inner.trace() }

    /// Reset online statistics (running mean/var).
    pub fn reset_stats(&mut self) {
        let k = self.running_mean.len();
        self.running_mean = vec![0.0; k];
        self.running_var = vec![1.0; k];
        self.n_updates = 0;
    }
}

// ─── HybridSparker — адаптивный FM gain ────────────────────

/// Спарковка с автоматическим расчётом FM gain по статистике.
pub struct HybridSparker {
    inner: NormalizedSparker,
    sample_rate: f32,
    base_gain: f32,
}

impl HybridSparker {
    /// Create hybrid sparker with adaptive gain from spike statistics.
    pub fn new(positions: &[[f32; 2]], tau_ms: f32, dt_ms: f32, sample_rate: f32, base_gain: f32) -> Self {
        Self {
            inner: NormalizedSparker::new(positions, tau_ms, dt_ms, 1.0),
            sample_rate,
            base_gain,
        }
    }

    /// Feed one step of spikes.
    pub fn feed(&mut self, spikes: &[f32]) {
        self.inner.feed(spikes);
    }

    /// Compute hologram + adaptive FM gain (lower gain when synchrony/burstiness high).
    pub fn hologram_with_gain(&mut self, spikes: &[f32]) -> (Vec<f32>, f32) {
        let stats = SpikeStats::compute_frame(spikes, self.sample_rate);
        let gain = stats.adaptive_gain(self.base_gain);
        let h = self.inner.hologram();
        (h, gain)
    }

    /// Current normalised trace (read-only).
    pub fn trace(&self) -> &[f32] { self.inner.trace() }
}

// ─── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_trace_decay() {
        let mut st = SpikeTrace::new(1, 10.0, 1.0);
        st.update(&[1.0]);
        assert!((st.get()[0] - 0.095).abs() < 0.01);
        st.update(&[0.0]);
        assert!((st.get()[0] - 0.086).abs() < 0.01);
    }

    #[test]
    fn test_hologram_single_neuron() {
        let pos = [[1.0, 0.0]];
        let enc = HologramEncoder::new(&pos);
        let h = enc.encode(&[1.0]);
        assert!(h[0].abs() > 0.0);
        assert!(h[1].abs() > 0.0);
    }

    #[test]
    fn test_sparker_pipeline() {
        let positions = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]];
        let mut sparker = Sparker::new(&positions, 50.0, 1.0);

        sparker.feed(&[1.0, 0.0, 0.0, 0.0]);
        let h = sparker.hologram();
        assert!(h[0].abs() > 0.0 || h[1].abs() > 0.0 || h[2].abs() > 0.0);

        for _ in 0..1000 {
            sparker.feed(&[0.0, 0.0, 0.0, 0.0]);
        }
        let h_decayed = sparker.hologram();
        for k in 0..3 {
            assert!(h_decayed[k].abs() < 0.1);
        }
    }

    #[test]
    fn test_encode_projection_matrix_equivalence() {
        let pos = [[0.5, 0.5], [-0.5, 0.5]];
        let enc = HologramEncoder::new(&pos);
        let trace = [0.3, 0.7];
        let h1 = enc.encode(&trace);
        // Manual matmul via projection_matrix
        let mat = enc.projection_matrix();
        let k = enc.k();
        let n = enc.n();
        let mut h2 = vec![0.0; k];
        for j in 0..k {
            for i in 0..n {
                h2[j] += mat[j * n + i] * trace[i];
            }
        }
        for j in 0..k {
            assert!((h1[j] - h2[j]).abs() < 1e-6, "mismatch at index {}", j);
        }
    }

    #[test]
    fn test_spike_stats() {
        let spikes = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let stats = SpikeStats::compute_frame(&spikes, 1000.0);
        assert!(stats.mean_rate > 0.0);
        assert!(stats.burstiness > 0.0);
    }

    #[test]
    fn test_adaptive_gain_decreases_with_synchrony() {
        let high_sync = SpikeStats { mean_rate: 10.0, burstiness: 0.5, synchrony: 0.8 };
        let low_sync = SpikeStats { mean_rate: 10.0, burstiness: 0.5, synchrony: 0.1 };
        let g_high = high_sync.adaptive_gain(1.0);
        let g_low = low_sync.adaptive_gain(1.0);
        assert!(g_high < g_low, "high synchrony → lower gain");
    }

    #[test]
    fn test_normalized_sparker() {
        let positions = [[0.5, 0.5], [-0.5, 0.5]];
        let mut ns = NormalizedSparker::new(&positions, 50.0, 1.0, 1.0);
        ns.feed(&[1.0, 0.0]);
        let h = ns.hologram();
        assert_eq!(h.len(), 3);
    }
}
