//! # PST Control — forward + inverse Kuramoto control loop.
//!
//! Forward:   drive → T³ → PST → Ridge → class
//! Inverse:   target_class → target_PST → drive (via optimization)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use taas_t3::{TriOscillator, Avx2Backend, PstControl, PstControlConfig};
//!
//! let config = PstControlConfig::default();
//! let mut ctrl = PstControl::new(config);
//!
//! // Train: collect (PST, one-hot class) pairs, then fit
//! let x = vec![/* n_samples × n_pst flattened */];
//! let y = vec![/* n_samples × n_classes one-hot flattened */];
//! let acc = ctrl.fit_ridge(&x, &y, n, 10.0);
//!
//! // Inverse: what PST features produce class 0?
//! let target = ctrl.target_pst(0);
//!
//! // Find drive for that PST
//! let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
//! let (drive, pred_class) = ctrl.drive_for(&mut tri, 0, "gradient");
//! ```

use super::{
    pst::{pst_linear_dim, pst_linear_features},
    ridge::RidgeSolver,
    T3Backend, TriOscillator,
};
use serde::{Deserialize, Serialize};

// ===================================================================
// PstControlConfig
// ===================================================================

/// Конфигурация PST Control.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PstControlConfig {
    /// Число осцилляторов в каждой популяции
    pub n: usize,
    /// Длина траектории (L шагов PST)
    pub L: usize,
    /// Количество классов
    pub n_classes: usize,
    /// Регуляризация Ridge (α)
    pub ridge_alpha: f32,
    /// Коэффициент смешивания для inverse:
    ///   0.0 = чистый псевдоинверс
    ///   1.0 = чистый средний PST класса
    pub inverse_blend: f32,
    /// Диапазон драйва
    pub drive_min: f32,
    pub drive_max: f32,
    /// Число шагов прогрева тора
    pub n_warmup: usize,
    /// Шаг симуляции
    pub dt: f32,
}

impl Default for PstControlConfig {
    fn default() -> Self {
        Self {
            n: 16,
            L: 2,
            n_classes: 3,
            ridge_alpha: 10.0,
            inverse_blend: 0.3,
            drive_min: 0.0,
            drive_max: 1.5,
            n_warmup: 70,
            dt: 0.01,
        }
    }
}

// ===================================================================
// PstControl
// ===================================================================

/// PST Control: forward (PST → class) + inverse (class → PST → drive).
pub struct PstControl {
    pub config: PstControlConfig,

    /// Ridge веса: W[c][f] = weights[c * n_pst + f]
    weights: Vec<f32>,
    /// Смещение на класс
    bias: Vec<f32>,
    /// Средний PST на класс
    mean_pst: Vec<f32>,
    /// (W^T W)^{-1} — n_classes × n_classes для inverse
    wtw_inv: Vec<f32>,
    /// Флаг fitted
    fitted: bool,

    /// Буфер траектории PST (L × base_dim)
    pst_buf: Vec<f32>,
    pst_pos: usize,
    pst_filled: bool,

    /// Размерность PST-траектории
    n_pst: usize,
    /// Базовая размерность (1 шаг)
    base_dim: usize,
}

impl PstControl {
    /// Create PST controller with given config.
    pub fn new(config: PstControlConfig) -> Self {
        let base_dim = pst_linear_dim(config.n);
        let n_pst = config.L * base_dim;
        Self {
            weights: vec![0.0; config.n_classes * n_pst],
            bias: vec![0.0; config.n_classes],
            mean_pst: vec![0.0; config.n_classes * n_pst],
            wtw_inv: vec![0.0; config.n_classes * config.n_classes],
            pst_buf: vec![0.0; n_pst],
            pst_pos: 0,
            pst_filled: false,
            fitted: false,
            n_pst,
            base_dim,
            config,
        }
    }

    /// PST feature dimension (base_dim × L).
    pub fn feat_dim(&self) -> usize {
        self.n_pst
    }
    /// Per-frame feature dimension (3N·6 + 12).
    pub fn base_feat_dim(&self) -> usize {
        self.base_dim
    }
    /// Number of output classes.
    pub fn n_classes(&self) -> usize {
        self.config.n_classes
    }
    /// Whether ridge weights have been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    // ─── PST features ──────────────────────────────────────────

    /// Compute PST features from oscillator state (delegates to `pst_linear_features`).
    pub fn compute_pst_features<B: T3Backend>(tri: &TriOscillator<B>) -> Vec<f32> {
        pst_linear_features(tri)
    }

    /// Get circular trajectory buffer as contiguous `feat_dim`-length Vec.
    pub fn get_trajectory(&self) -> Vec<f32> {
        if !self.pst_filled {
            return vec![0.0; self.n_pst];
        }
        let mut out = vec![0.0; self.n_pst];
        for t in 0..self.config.L {
            let src_idx = (self.pst_pos + t) % self.config.L;
            let dst_idx = t * self.base_dim;
            let src_start = src_idx * self.base_dim;
            out[dst_idx..dst_idx + self.base_dim]
                .copy_from_slice(&self.pst_buf[src_start..src_start + self.base_dim]);
        }
        out
    }

    /// Push one PST frame into the circular trajectory buffer.
    pub fn push_pst<B: T3Backend>(&mut self, tri: &TriOscillator<B>) {
        let start = self.pst_pos * self.base_dim;
        let feats = Self::compute_pst_features(tri);
        self.pst_buf[start..start + self.base_dim].copy_from_slice(&feats);
        self.pst_pos += 1;
        if self.pst_pos >= self.config.L {
            self.pst_pos = 0;
            self.pst_filled = true;
        }
    }

    /// Reset circular trajectory buffer (clears accumulated history).
    pub fn reset_buffer(&mut self) {
        self.pst_buf.fill(0.0);
        self.pst_pos = 0;
        self.pst_filled = false;
    }

    // ─── Forward ───────────────────────────────────────────────

    /// Прямой ход: PST → class scores.
    pub fn predict_from_pst(&self, pst: &[f32]) -> (usize, Vec<f32>) {
        let c = self.config.n_classes;
        let mut scores = vec![0.0; c];
        for class in 0..c {
            let mut s = self.bias[class];
            let base = class * self.n_pst;
            for f in 0..self.n_pst {
                s += self.weights[base + f] * pst[f];
            }
            scores[class] = s;
        }
        let mut best = 0;
        for i in 1..c {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        (best, scores)
    }

    /// Прямой ход из TriOscillator (накапливает траекторию).
    pub fn predict<B: T3Backend>(&mut self, tri: &TriOscillator<B>) -> (usize, Vec<f32>) {
        self.push_pst(tri);
        if !self.pst_filled {
            return (0, vec![0.0; self.config.n_classes]);
        }
        let traj = self.get_trajectory();
        self.predict_from_pst(&traj)
    }

    // ─── Обучение ──────────────────────────────────────────────

    /// Обучить multi-output Ridge.
    pub fn fit_ridge(&mut self, x: &[f32], y: &[f32], n: usize, alpha: f32) -> f32 {
        let d = self.n_pst;
        let c = self.config.n_classes;
        let ridge = RidgeSolver::new(alpha);

        self.weights = Vec::with_capacity(c * d);
        for class in 0..c {
            let mut y_class = Vec::with_capacity(n);
            for i in 0..n {
                y_class.push(y[i * c + class]);
            }
            let w = ridge.solve(x, &y_class, n, d);
            for &wi in &w {
                self.weights.push(wi);
            }
        }

        self.bias = vec![0.0; c];
        let mut sum_y = vec![0.0f64; c];
        let mut sum_pred = vec![0.0f64; c];
        for i in 0..n {
            for class in 0..c {
                sum_y[class] += y[i * c + class] as f64;
                let mut pred = 0.0f64;
                for f in 0..d {
                    pred += x[i * d + f] as f64 * self.weights[class * d + f] as f64;
                }
                sum_pred[class] += pred;
            }
        }
        for class in 0..c {
            self.bias[class] = ((sum_y[class] - sum_pred[class]) / n as f64) as f32;
            if !self.bias[class].is_finite() {
                self.bias[class] = 0.0;
            }
        }

        self.mean_pst = vec![0.0; c * d];
        let mut count = vec![0usize; c];
        for i in 0..n {
            let mut best_class = 0;
            for class in 1..c {
                if y[i * c + class] > y[i * c + best_class] {
                    best_class = class;
                }
            }
            for f in 0..d {
                self.mean_pst[best_class * d + f] += x[i * d + f];
            }
            count[best_class] += 1;
        }
        for class in 0..c {
            if count[class] > 0 {
                let inv = 1.0 / count[class] as f32;
                for f in 0..d {
                    self.mean_pst[class * d + f] *= inv;
                }
            }
        }

        let mut correct = 0;
        for i in 0..n {
            let (pred_class, _) = self.predict_from_pst(&x[i * d..(i + 1) * d]);
            let true_class = (0..c)
                .max_by(|&a, &b| {
                    y[i * c + a]
                        .partial_cmp(&y[i * c + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if pred_class == true_class {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / n as f32;

        self.compute_pinv();
        self.fitted = true;
        accuracy
    }

    // ─── Inverse ───────────────────────────────────────────────

    /// Cholesky (W^T W + εI) = L L^T, затем invert.
    fn compute_pinv(&mut self) {
        let c = self.config.n_classes;
        let d = self.n_pst;

        let mut wtw = vec![0.0f64; c * c];
        for i in 0..c {
            for j in 0..=i {
                let mut sum = 0.0f64;
                for f in 0..d {
                    sum += self.weights[i * d + f] as f64 * self.weights[j * d + f] as f64;
                }
                wtw[i * c + j] = sum;
                wtw[j * c + i] = sum;
            }
        }

        let eps = 1e-6;
        for i in 0..c {
            wtw[i * c + i] += eps;
        }

        let mut l = vec![0.0f64; c * c];
        for j in 0..c {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[j * c + k] * l[j * c + k];
            }
            let val = wtw[j * c + j] - sum;
            if val <= 0.0 {
                self.wtw_inv = vec![0.0; c * c];
                for i in 0..c {
                    self.wtw_inv[i * c + i] = 1.0 / eps as f32;
                }
                return;
            }
            l[j * c + j] = val.sqrt();
            for i in (j + 1)..c {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * c + k] * l[j * c + k];
                }
                l[i * c + j] = (wtw[i * c + j] - sum) / l[j * c + j];
            }
        }

        let mut inv = vec![0.0f64; c * c];
        for col in 0..c {
            let mut z = vec![0.0f64; c];
            z[col] = 1.0;
            for i in 0..c {
                let mut sum = 0.0;
                for k in 0..i {
                    sum += l[i * c + k] * z[k];
                }
                z[i] = (z[i] - sum) / l[i * c + i];
            }
            for i in (0..c).rev() {
                let mut sum = 0.0;
                for k in (i + 1)..c {
                    sum += l[k * c + i] * inv[k * c + col];
                }
                inv[i * c + col] = (z[i] - sum) / l[i * c + i];
            }
        }

        self.wtw_inv = Vec::with_capacity(c * c);
        for i in 0..c {
            for j in 0..c {
                self.wtw_inv.push(inv[i * c + j] as f32);
            }
        }
    }

    /// Целевой PST для класса.
    pub fn target_pst(&self, target_class: usize) -> Vec<f32> {
        assert!(self.fitted, "PstControl not fitted");
        let c = self.config.n_classes;
        let d = self.n_pst;

        let mut y_centered = vec![0.0f64; c];
        for i in 0..c {
            let target = if i == target_class { 1.0 } else { 0.0 };
            y_centered[i] = target - self.bias[i] as f64;
        }

        let mut z = vec![0.0f64; c];
        for i in 0..c {
            for j in 0..c {
                z[i] += self.wtw_inv[i * c + j] as f64 * y_centered[j];
            }
        }

        let mut pst_pinv = vec![0.0f64; d];
        for f in 0..d {
            for class in 0..c {
                pst_pinv[f] += self.weights[class * d + f] as f64 * z[class];
            }
        }

        let blend = self.config.inverse_blend;
        let mut result = vec![0.0; d];
        let base = target_class * d;
        for f in 0..d {
            let from_pinv = pst_pinv[f] as f32;
            let from_mean = self.mean_pst[base + f];
            result[f] = (1.0 - blend) * from_pinv + blend * from_mean;
        }

        result
    }

    /// Check how well the target PST reconstructs: returns (predicted_class, margin).
    pub fn check_target_pst(&self, target_class: usize) -> (usize, f32) {
        let pst = self.target_pst(target_class);
        let (pred, scores) = self.predict_from_pst(&pst);
        let score_target = scores[target_class];
        let score_best = scores[pred];
        (pred, score_target - score_best)
    }

    // ─── Drive optimization ────────────────────────────────────

    /// Измерить PST для заданного драйва.
    pub fn measure_drive<B: T3Backend>(&self, tri: &mut TriOscillator<B>, drive: f32) -> Vec<f32> {
        let L = self.config.L;
        let dt = self.config.dt;
        let base = self.base_dim;

        tri.reset();
        tri.set_scalar_drive(drive);

        for _ in 0..self.config.n_warmup {
            tri.step(dt);
        }

        let mut traj = vec![0.0; self.n_pst];
        for t in 0..L {
            tri.step(dt);
            let feats = Self::compute_pst_features(tri);
            let start = t * base;
            traj[start..start + base].copy_from_slice(&feats);
        }

        traj
    }

    /// Random search: драйв под целевой класс.
    pub fn find_drive<B: T3Backend>(
        &self,
        tri: &mut TriOscillator<B>,
        target_class: usize,
        n_trials: usize,
    ) -> (f32, f32) {
        use rand::Rng;
        let target = self.target_pst(target_class);
        let mut best_drive = 0.0;
        let mut best_loss = f32::MAX;
        let mut rng = rand::thread_rng();

        let range = self.config.drive_max - self.config.drive_min;
        for _ in 0..n_trials {
            let d = self.config.drive_min + rng.gen::<f32>() * range;
            let pst = self.measure_drive(tri, d);
            let loss: f32 = pst
                .iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if loss < best_loss {
                best_loss = loss;
                best_drive = d;
            }
        }

        (best_drive, best_loss)
    }

    /// Градиентная оптимизация драйва (finite-difference).
    pub fn optimize_drive<B: T3Backend>(
        &self,
        tri: &mut TriOscillator<B>,
        target_class: usize,
        steps: usize,
        lr: f32,
    ) -> (f32, usize) {
        use rand::Rng;
        let target = self.target_pst(target_class);
        let range = self.config.drive_max - self.config.drive_min;
        let mut d = self.config.drive_min + rand::thread_rng().gen::<f32>() * range;
        let eps = 1e-4;

        for _ in 0..steps {
            let pst = self.measure_drive(tri, d);
            let loss: f32 = pst
                .iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();

            let pst_eps = self.measure_drive(tri, d + eps);
            let loss_eps: f32 = pst_eps
                .iter()
                .zip(target.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let grad = (loss_eps - loss) / eps;

            d -= lr * grad;
            d = d.clamp(self.config.drive_min, self.config.drive_max);
        }

        let pst_final = self.measure_drive(tri, d);
        let (pred, _) = self.predict_from_pst(&pst_final);
        (d, pred)
    }

    /// Универсальный интерфейс: drive для целевого класса.
    pub fn drive_for<B: T3Backend>(
        &self,
        tri: &mut TriOscillator<B>,
        target_class: usize,
        method: &str,
    ) -> (f32, usize) {
        match method {
            "gradient" => self.optimize_drive(tri, target_class, 60, 0.02),
            _ => {
                let (d, _) = self.find_drive(tri, target_class, 300);
                let pst = self.measure_drive(tri, d);
                let (pred, _) = self.predict_from_pst(&pst);
                (d, pred)
            }
        }
    }

    // ─── Сериализация ──────────────────────────────────────────

    /// Serialise controller weights + config to JSON file.
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        let data = SaveDataRef {
            config: &self.config,
            weights: &self.weights,
            bias: &self.bias,
            mean_pst: &self.mean_pst,
            wtw_inv: &self.wtw_inv,
            n_pst: self.n_pst,
        };
        let f = File::create(path)?;
        serde_json::to_writer(BufWriter::new(f), &data)?;
        Ok(())
    }

    /// Deserialise controller from JSON file.
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::BufReader;
        let f = File::open(path)?;
        let data: SaveData = serde_json::from_reader(BufReader::new(f))?;
        let n_pst = data.n_pst;
        let n = data.config.n;
        Ok(Self {
            config: data.config,
            weights: data.weights,
            bias: data.bias,
            mean_pst: data.mean_pst,
            wtw_inv: data.wtw_inv,
            n_pst,
            pst_buf: vec![0.0; n_pst],
            pst_pos: 0,
            pst_filled: false,
            fitted: true,
            base_dim: pst_linear_dim(n),
        })
    }

    /// Fitted ridge weights (n_classes × feat_dim).
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    /// Fitted bias per class.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
    /// Mean PST template per class (n_classes × feat_dim).
    pub fn mean_pst(&self) -> &[f32] {
        &self.mean_pst
    }
}

#[derive(Serialize)]
struct SaveDataRef<'a> {
    config: &'a PstControlConfig,
    weights: &'a [f32],
    bias: &'a [f32],
    mean_pst: &'a [f32],
    wtw_inv: &'a [f32],
    n_pst: usize,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
    config: PstControlConfig,
    weights: Vec<f32>,
    bias: Vec<f32>,
    mean_pst: Vec<f32>,
    wtw_inv: Vec<f32>,
    n_pst: usize,
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Avx2Backend, TriOscillator};
    use rand::Rng;

    fn make_test_data(n: usize, n_pst: usize, n_classes: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::thread_rng();
        let mut x = Vec::with_capacity(n * n_pst);
        let mut y = Vec::with_capacity(n * n_classes);
        for i in 0..n {
            let cls = i % n_classes;
            for _ in 0..n_pst {
                x.push(rng.gen::<f32>() * 0.5 + cls as f32 * 0.3);
            }
            for c in 0..n_classes {
                y.push(if c == cls { 1.0 } else { 0.0 });
            }
        }
        (x, y)
    }

    #[test]
    fn test_pst_control_config_default() {
        let cfg = PstControlConfig::default();
        assert_eq!(cfg.n, 16);
    }

    #[test]
    fn test_pst_control_feat_dim() {
        let cfg = PstControlConfig {
            n: 16,
            L: 1,
            ..Default::default()
        };
        let ctrl = PstControl::new(cfg);
        let base = pst_linear_dim(16);
        assert_eq!(ctrl.base_feat_dim(), base);
        assert_eq!(ctrl.feat_dim(), base);
    }

    #[test]
    fn test_pst_control_feat_dim_l2() {
        let cfg = PstControlConfig {
            n: 16,
            L: 2,
            ..Default::default()
        };
        let ctrl = PstControl::new(cfg);
        let base = pst_linear_dim(16);
        assert_eq!(ctrl.feat_dim(), 2 * base);
    }

    #[test]
    fn test_pst_control_fit_ridge() {
        let cfg = PstControlConfig {
            n: 8,
            L: 1,
            n_classes: 2,
            ridge_alpha: 1.0,
            ..Default::default()
        };
        let mut ctrl = PstControl::new(cfg);
        let n_pst = ctrl.feat_dim();
        let n = 200;
        let (x, y) = make_test_data(n, n_pst, 2);
        let acc = ctrl.fit_ridge(&x, &y, n, 0.01);
        assert!(acc > 0.75, "forward accuracy on small = {}", acc);
    }

    #[test]
    fn test_pst_control_forward_backward() {
        let cfg = PstControlConfig {
            n: 8,
            L: 1,
            n_classes: 2,
            ridge_alpha: 1.0,
            inverse_blend: 0.0,
            ..Default::default()
        };
        let mut ctrl = PstControl::new(cfg);
        let n_pst = ctrl.feat_dim();
        let n = 500;
        let (x, y) = make_test_data(n, n_pst, 2);
        let acc = ctrl.fit_ridge(&x, &y, n, 0.01);
        assert!(acc > 0.75, "forward accuracy = {}", acc);

        let (pred, _) = ctrl.check_target_pst(0);
        assert_eq!(pred, 0, "target_pst(0) → class {}", pred);
        let (pred, _) = ctrl.check_target_pst(1);
        assert_eq!(pred, 1, "target_pst(1) → class {}", pred);
    }

    #[test]
    fn test_pst_control_serialize_roundtrip() {
        let cfg = PstControlConfig {
            n: 4,
            L: 1,
            n_classes: 2,
            ..Default::default()
        };
        let mut ctrl = PstControl::new(cfg);
        let n_pst = ctrl.feat_dim();
        let n = 50;
        let (x, y) = make_test_data(n, n_pst, 2);
        ctrl.fit_ridge(&x, &y, n, 0.1);

        let path = "test_pst_control.json";
        ctrl.save_to_file(path).unwrap();
        let loaded = PstControl::load_from_file(path).unwrap();

        assert_eq!(loaded.config.n, ctrl.config.n);
        assert_eq!(loaded.feat_dim(), ctrl.feat_dim());
        assert_eq!(loaded.weights(), ctrl.weights());
        assert_eq!(loaded.bias(), ctrl.bias());

        let (pred1, _) = ctrl.predict_from_pst(&x[..n_pst]);
        let (pred2, _) = loaded.predict_from_pst(&x[..n_pst]);
        assert_eq!(pred1, pred2);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_pst_control_drive_on_synthetic_torus() {
        let cfg = PstControlConfig {
            n: 8,
            L: 2,
            n_classes: 3,
            ridge_alpha: 10.0,
            inverse_blend: 0.0,
            drive_min: 0.0,
            drive_max: 1.5,
            n_warmup: 80,
            dt: 0.01,
        };
        let n_pst = pst_linear_dim(cfg.n) * cfg.L;
        let mut ctrl = PstControl::new(cfg.clone());
        let n_per_class = 30;
        let n_total = n_per_class * 3;
        let mut x = Vec::with_capacity(n_total * n_pst);
        let mut y = Vec::with_capacity(n_total * 3);

        let mut tri = TriOscillator::new(cfg.n, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.3, 0.15, 0.08);

        for (cls, dval) in [0.0f32, 0.5, 1.0].iter().enumerate() {
            for _ in 0..n_per_class {
                let d = (dval + rand::thread_rng().gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.5);
                let pst = ctrl.measure_drive(&mut tri, d);
                x.extend_from_slice(&pst);
                for c in 0..3 {
                    y.push(if c == cls { 1.0 } else { 0.0 });
                }
            }
        }

        let acc = ctrl.fit_ridge(&x, &y, n_total, 10.0);
        assert!(acc > 0.3, "forward accuracy on synthetic = {}", acc);

        for cls in 0..3 {
            let (pred, _) = ctrl.check_target_pst(cls);
            assert_eq!(pred, cls, "inverse: target_pst({}) → class {}", cls, pred);
        }

        let (d, _) = ctrl.find_drive(&mut tri, 0, 30);
        assert!((0.0..=1.5).contains(&d), "drive out of range: {}", d);
    }
}
