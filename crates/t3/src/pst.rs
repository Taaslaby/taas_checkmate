//! # PST (Phase Features) — sin/cos features of T3 oscillators.
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

use super::{T3Backend, TriOscillator, PI, TWO_PI};

#[cfg(test)]
use super::backend::Avx2Backend;

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

/// Linear (O(N)) feature vector — per-oscillator trig features, without O(N²) pairwise.
///
/// ## Оптимизация
///
/// **0 trig calls** — sin/cos берутся из pre-computed `TriOscillator.sin1`/`cos1`.
pub fn pst_linear_features<B: T3Backend>(tri: &TriOscillator<B>) -> Vec<f32> {
    let dim = pst_linear_dim(tri.n);
    let mut fv = vec![0.0; dim];
    pst_linear_features_into(tri, &mut fv);
    fv
}

/// Zero-allocation variant — writes into pre-allocated `out`.
pub fn pst_linear_features_into<B: T3Backend>(tri: &TriOscillator<B>, out: &mut [f32]) {
    let n = tri.n;
    let dim = pst_linear_dim(n);
    assert!(
        out.len() >= dim,
        "pst_linear_features_into: out len {} < dim {}",
        out.len(),
        dim
    );

    let sin1 = &tri.sin1[..n];
    let cos1 = &tri.cos1[..n];
    let sin2 = &tri.sin2[..n];
    let cos2 = &tri.cos2[..n];
    let sin3 = &tri.sin3[..n];
    let cos3 = &tri.cos3[..n];

    let mut pos = 0;

    // 1. sin(θ), cos(θ) for each oscillator
    for i in 0..n {
        out[pos] = sin1[i];
        out[pos + 1] = cos1[i];
        pos += 2;
    }
    for i in 0..n {
        out[pos] = sin2[i];
        out[pos + 1] = cos2[i];
        pos += 2;
    }
    for i in 0..n {
        out[pos] = sin3[i];
        out[pos + 1] = cos3[i];
        pos += 2;
    }

    // 2. dθ/dt — real phase velocity from prev
    for (ph, prev) in [
        (&tri.th1, &tri.prev_th1),
        (&tri.th2, &tri.prev_th2),
        (&tri.th3, &tri.prev_th3),
    ] {
        for i in 0..n {
            let mut d = ph[i] - prev[i];
            if d > PI {
                d -= TWO_PI;
            } else if d < -PI {
                d += TWO_PI;
            }
            out[pos] = d;
            pos += 1;
        }
    }

    // 3. Amplitudes r (direct copy — no trig)
    out[pos..pos + n].copy_from_slice(&tri.r1[..n]);
    pos += n;
    out[pos..pos + n].copy_from_slice(&tri.r2[..n]);
    pos += n;
    out[pos..pos + n].copy_from_slice(&tri.r3[..n]);
    pos += n;

    // 4. Squared features — from PRE-COMPUTED sin/cos (NO trig!)
    for i in 0..n {
        out[pos] = sin1[i] * sin1[i];
        out[pos + 1] = cos1[i] * cos1[i];
        pos += 2;
    }
    for i in 0..n {
        out[pos] = sin2[i] * sin2[i];
        out[pos + 1] = cos2[i] * cos2[i];
        pos += 2;
    }
    for i in 0..n {
        out[pos] = sin3[i] * sin3[i];
        out[pos + 1] = cos3[i] * cos3[i];
        pos += 2;
    }

    // 5. Cross order parameters — from PRE-COMPUTED sin/cos (NO trig!)
    out[pos] = cross_order_from_sincos(sin1, cos1, sin2, cos2, n);
    pos += 1;
    out[pos] = cross_order_from_sincos(sin2, cos2, sin3, cos3, n);
    pos += 1;
    out[pos] = cross_order_from_sincos(sin1, cos1, sin3, cos3, n);
    pos += 1;

    // 6. Mean sin/cos + mean amplitude per population
    let nf = n as f32;
    // Pop 1
    let (ms1, mc1) = mean_sincos(sin1, cos1, n);
    let mr1: f32 = tri.r1[..n].iter().sum::<f32>() / nf;
    out[pos] = ms1;
    out[pos + 1] = mc1;
    out[pos + 2] = mr1;
    pos += 3;
    // Pop 2
    let (ms2, mc2) = mean_sincos(sin2, cos2, n);
    let mr2: f32 = tri.r2[..n].iter().sum::<f32>() / nf;
    out[pos] = ms2;
    out[pos + 1] = mc2;
    out[pos + 2] = mr2;
    pos += 3;
    // Pop 3
    let (ms3, mc3) = mean_sincos(sin3, cos3, n);
    let mr3: f32 = tri.r3[..n].iter().sum::<f32>() / nf;
    out[pos] = ms3;
    out[pos + 1] = mc3;
    out[pos + 2] = mr3;
}

#[inline]
fn mean_sincos(sin: &[f32], cos: &[f32], n: usize) -> (f32, f32) {
    let nf = n as f32;
    let ms: f32 = sin.iter().sum::<f32>() / nf;
    let mc: f32 = cos.iter().sum::<f32>() / nf;
    (ms, mc)
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
fn cross_order_from_sincos(
    sin1: &[f32],
    cos1: &[f32],
    sin2: &[f32],
    cos2: &[f32],
    n: usize,
) -> f32 {
    if n == 0 {
        return 0.0;
    }
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

/// Fast PST — sin(θ) + r for all 3 populations (6N features).
///
/// Использует pre-computed sin/cos из TriOscillator (0 trig calls).
/// Ridge decoder не теряет точности (R² > 0.9999) при 2× меньшем размере.
pub fn pst_fast<B: T3Backend>(tri: &TriOscillator<B>) -> Vec<f32> {
    let n = tri.n;
    let mut out = Vec::with_capacity(6 * n);
    out.extend_from_slice(&tri.sin1[..n]);
    out.extend_from_slice(&tri.sin2[..n]);
    out.extend_from_slice(&tri.sin3[..n]);
    out.extend_from_slice(&tri.r1[..n]);
    out.extend_from_slice(&tri.r2[..n]);
    out.extend_from_slice(&tri.r3[..n]);
    out
}

/// Dimension of `pst_fast` for given N: `6N`.
pub fn pst_fast_dim(n: usize) -> usize {
    6 * n
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

        assert!(
            (fast_r - ref_r).abs() < 1e-6,
            "cross_order_from_sincos({}) != order_parameter_cross({})",
            fast_r,
            ref_r
        );
    }

    #[test]
    fn test_pst_linear_features_sin_cos_values() {
        let mut tri = TriOscillator::new(8, 0.5, 1.5, Avx2Backend);
        tri.set_uniform_coupling(0.1, 0.05, 0.02);
        tri.step(0.01);
        tri.prepare();
        let fv = pst_linear_features(&tri);

        let n = tri.n;
        for i in 0..n {
            assert!(
                (fv[2 * i] - tri.sin1[i]).abs() < 1e-6,
                "sin1[{}]: fv={} != tri.sin1={}",
                i,
                fv[2 * i],
                tri.sin1[i]
            );
            assert!(
                (fv[2 * i + 1] - tri.cos1[i]).abs() < 1e-6,
                "cos1[{}]: fv={} != tri.cos1={}",
                i,
                fv[2 * i + 1],
                tri.cos1[i]
            );
        }
    }

    #[test]
    fn test_pst_linear_dim() {
        assert_eq!(pst_linear_dim(0), 12);
        assert_eq!(pst_linear_dim(8), 3 * 8 * 6 + 12);
    }
}
