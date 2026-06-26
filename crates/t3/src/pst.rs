//! # PST (Phase Space Topology) features.
//!
//! Два типа фич:
//!
//! - [`pst_features`] — pairwise O(N²) sin/cos между двумя популяциями
//! - [`pst_linear_features`] — линейная O(N) комбинация признаков
//!
//! ## Pairwise PST
//!
//! ```text
//! out_sin[p,q] = sin(θ₁ₚ - θ₂_q) = sin_p·cos_q - cos_p·sin_q
//! out_cos[p,q] = cos(θ₁ₚ - θ₂_q) = cos_p·cos_q + sin_p·sin_q
//! ```
//!
//! ## Linear PST (optimized O(N))
//!
//! Состав feature vector (pre-computes sin/cos один раз, без лишних trig call):
//! 1. sin(θ), cos(θ) — 3×N×2
//! 2. dθ/dt — 3×N
//! 3. Амплитуды r — 3×N
//! 4. cos²(θ), sin²(θ) — из pre-computed sin/cos — 3×N×2
//! 5. Cross order parameters R₁₂, R₂₃, R₁₃ — из pre-computed sin/cos — 3
//! 6. Mean sin/cos + mean amplitude per population — 3×3

use crate::{PI, TWO_PI, TriOscillator, T3Backend};

#[cfg(test)]
use crate::backend::Avx2Backend;

/// PST sin/cos features between two phase populations.
/// Returns (sin_matrix, cos_matrix) each of size n1 × n2.
/// O(N²) — pairwise, используйте для малых N или особых случаев.
pub fn pst_features(ph1: &[f32], ph2: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n1 = ph1.len();
    let n2 = ph2.len();
    let mut out_sin = vec![0.0f32; n1 * n2];
    let mut out_cos = vec![0.0f32; n1 * n2];
    for p in 0..n1 {
        let sin_p = ph1[p].sin();
        let cos_p = ph1[p].cos();
        for q in 0..n2 {
            let sin_q = ph2[q].sin();
            let cos_q = ph2[q].cos();
            out_sin[p * n2 + q] = sin_p * cos_q - cos_p * sin_q;
            out_cos[p * n2 + q] = cos_p * cos_q + sin_p * sin_q;
        }
    }
    (out_sin, out_cos)
}

/// Linear (O(N)) feature vector — per-oscillator, without O(N²) pairwise.
///
/// ## Оптимизация
///
/// Pre-computes sin/cos для всех 3 популяций ОДИН раз (6N trig calls),
/// затем переиспользует для:
/// - шага 1 (sin/cos фичи)
/// - шага 4 (sin²/cos² — квадратичные, без trig)
/// - шага 5 (cross order params — без trig, только мультипликация)
/// - шага 6 (mean sin/cos — без trig)
///
/// Всего: **6N trig calls** вместо **18N** в старой версии.
pub fn pst_linear_features<B: T3Backend>(tri: &TriOscillator<B>) -> Vec<f32> {
    let n = tri.n;
    let mut fv = Vec::with_capacity(3 * n * 6 + 12);

    // ── Pre-compute sin/cos for ALL 3 populations ONCE ──────────
    // (no padding needed — TriOscillator stores exact n elements)

    let mut sin1 = vec![0.0f32; n];
    let mut cos1 = vec![0.0f32; n];
    let mut sin2 = vec![0.0f32; n];
    let mut cos2 = vec![0.0f32; n];
    let mut sin3 = vec![0.0f32; n];
    let mut cos3 = vec![0.0f32; n];

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        sin1[i] = tri.th1[i].sin();
        cos1[i] = tri.th1[i].cos();
        sin2[i] = tri.th2[i].sin();
        cos2[i] = tri.th2[i].cos();
        sin3[i] = tri.th3[i].sin();
        cos3[i] = tri.th3[i].cos();
    }

    // 1. sin(θ), cos(θ) for each oscillator
    for i in 0..n { fv.push(sin1[i]); fv.push(cos1[i]); }
    for i in 0..n { fv.push(sin2[i]); fv.push(cos2[i]); }
    for i in 0..n { fv.push(sin3[i]); fv.push(cos3[i]); }

    // 2. dθ/dt — real phase velocity from prev
    for (ph, prev) in [(&tri.th1, &tri.prev_th1), (&tri.th2, &tri.prev_th2), (&tri.th3, &tri.prev_th3)] {
        for i in 0..n {
            let mut d = ph[i] - prev[i];
            if d > PI { d -= TWO_PI; } else if d < -PI { d += TWO_PI; }
            fv.push(d);
        }
    }

    // 3. Amplitudes r (direct copy — no trig)
    fv.extend_from_slice(&tri.r1[..n]);
    fv.extend_from_slice(&tri.r2[..n]);
    fv.extend_from_slice(&tri.r3[..n]);

    // 4. Squared features — from PRE-COMPUTED sin/cos (NO trig!)
    for i in 0..n { fv.push(sin1[i] * sin1[i]); fv.push(cos1[i] * cos1[i]); }
    for i in 0..n { fv.push(sin2[i] * sin2[i]); fv.push(cos2[i] * cos2[i]); }
    for i in 0..n { fv.push(sin3[i] * sin3[i]); fv.push(cos3[i] * cos3[i]); }

    // 5. Cross order parameters — from PRE-COMPUTED sin/cos (NO trig!)
    fv.push(cross_order_from_sincos(&sin1, &cos1, &sin2, &cos2, n));
    fv.push(cross_order_from_sincos(&sin2, &cos2, &sin3, &cos3, n));
    fv.push(cross_order_from_sincos(&sin1, &cos1, &sin3, &cos3, n));

    // 6. Mean sin/cos + mean amplitude per population
    let nf = n as f32;
    // Pop 1
    let mut ms = 0.0; let mut mc = 0.0; let mut mr = 0.0;
    for i in 0..n { ms += sin1[i]; mc += cos1[i]; mr += tri.r1[i]; }
    fv.push(ms / nf); fv.push(mc / nf); fv.push(mr / nf);
    // Pop 2
    ms = 0.0; mc = 0.0; mr = 0.0;
    for i in 0..n { ms += sin2[i]; mc += cos2[i]; mr += tri.r2[i]; }
    fv.push(ms / nf); fv.push(mc / nf); fv.push(mr / nf);
    // Pop 3
    ms = 0.0; mc = 0.0; mr = 0.0;
    for i in 0..n { ms += sin3[i]; mc += cos3[i]; mr += tri.r3[i]; }
    fv.push(ms / nf); fv.push(mc / nf); fv.push(mr / nf);

    fv
}

/// Compute cross order parameter R = |mean(exp(i·(θ₁ − θ₂)))| from
/// PRE-COMPUTED sin/cos arrays.
///
/// Математика:
/// ```text
/// sin(θ₁−θ₂) = sin₁·cos₂ − cos₁·sin₂
/// cos(θ₁−θ₂) = cos₁·cos₂ + sin₁·sin₂
/// R = √((Σ sin(θ₁−θ₂))² + (Σ cos(θ₁−θ₂))²) / N
/// ```
///
/// **0 trig calls** — только умножения и сложения.
#[inline]
fn cross_order_from_sincos(sin1: &[f32], cos1: &[f32], sin2: &[f32], cos2: &[f32], n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let (sum_sin, sum_cos): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
        let s1 = sin1[i] as f64;
        let c1 = cos1[i] as f64;
        let s2 = sin2[i] as f64;
        let c2 = cos2[i] as f64;
        (s + s1.mul_add(c2, -(c1 * s2)), c + c1.mul_add(c2, s1 * s2))
    });
    ((sum_sin * sum_sin + sum_cos * sum_cos).sqrt() / n as f64) as f32
}

/// Dimension of `pst_linear_features` for given N.
///
/// `3N · 6 + 12` — не меняется от оптимизации, только реализация O(N).
pub fn pst_linear_dim(n: usize) -> usize {
    3 * n * 6 + 12
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TriOscillator;

    #[test]
    fn test_pst_features_size() {
        let ph1 = vec![0.0, 0.5, 1.0];
        let ph2 = vec![0.0, 0.5];
        let (sin_m, cos_m) = pst_features(&ph1, &ph2);
        assert_eq!(sin_m.len(), 3 * 2);
        assert_eq!(cos_m.len(), 3 * 2);
    }

    #[test]
    fn test_pst_linear_features_shape() {
        let mut tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);
        let fv = pst_linear_features(&tri);
        assert!(!fv.is_empty());
        assert_eq!(fv.len(), pst_linear_dim(8));
    }

    #[test]
    fn test_pst_linear_features_cross_order_match() {
        // Проверка: cross_order_from_sincos даёт тот же результат,
        // что и TriOscillator::order_parameter_cross (с sin/cos разности)
        let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);

        // Pre-compute sin/cos
        let sin1: Vec<f32> = tri.th1.iter().map(|p| p.sin()).collect();
        let cos1: Vec<f32> = tri.th1.iter().map(|p| p.cos()).collect();
        let sin2: Vec<f32> = tri.th2.iter().map(|p| p.sin()).collect();
        let cos2: Vec<f32> = tri.th2.iter().map(|p| p.cos()).collect();

        let fast_r = cross_order_from_sincos(&sin1, &cos1, &sin2, &cos2, tri.n);
        let ref_r = tri.order_parameter_cross(&tri.th1, &tri.th2);

        assert!((fast_r - ref_r).abs() < 1e-6,
            "cross_order_from_sincos({}) != order_parameter_cross({})", fast_r, ref_r);
    }

    #[test]
    fn test_pst_linear_features_sin_cos_values() {
        // Проверка: sin/cos в feature vector матчатся pre-computed значениям
        let mut tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);
        let fv = pst_linear_features(&tri);

        let n = tri.n;
        // Pop 1 sin/cos: первые 2N элементов
        for i in 0..n {
            let ref_sin = tri.th1[i].sin();
            let ref_cos = tri.th1[i].cos();
            assert!((fv[2 * i] - ref_sin).abs() < 1e-6, "sin1[{}]: {} != {}", i, fv[2 * i], ref_sin);
            assert!((fv[2 * i + 1] - ref_cos).abs() < 1e-6, "cos1[{}]: {} != {}", i, fv[2 * i + 1], ref_cos);
        }
    }

    #[test]
    fn test_pst_linear_dim() {
        assert_eq!(pst_linear_dim(0), 12);
        assert_eq!(pst_linear_dim(8), 3 * 8 * 6 + 12);
    }
}
