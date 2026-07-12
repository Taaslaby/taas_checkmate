//! PST Jacobian-vector product — **JIT** (no matrix materialization).
//!
//! Вместо построения плотной матрицы `J[pst_dim × n]` (6 × 108n² + 72n флоатов)
//! и последующего умножения `v = J^T · grad`, считаем v напрямую за O(pst_dim + n):
//!
//! - Каждая PST-фича зависит от 1-2 state-переменных (кроме cross-R, который зависит от всех)
//! - Для каждой фичи: v[state_idx] += grad_pst[feat] · d(feat)/d(state)
//! - Нет промежуточной матрицы — экономия ~108n² флоатов для n=1000 это 108M → 12K
//!
//! # Usage
//!
//! ```ignore
//! let mut v = PstGradients::new(n);
//! pst_jacobian_v_jit(th1, r1, th2, r2, th3, r3, n, grad_pst, &mut v);
//! ```

// ─── PstGradients (6 × n) ────────────────────────────────

/// Gradient-vector buffers (6 × n) — единственный тип для JIT Jacobian.
pub struct PstGradients {
    pub v_th1: Vec<f32>, pub v_th2: Vec<f32>, pub v_th3: Vec<f32>,
    pub v_r1: Vec<f32>, pub v_r2: Vec<f32>, pub v_r3: Vec<f32>,
}

impl PstGradients {
    pub fn new(n: usize) -> Self {
        Self {
            v_th1: vec![0.0; n], v_th2: vec![0.0; n], v_th3: vec![0.0; n],
            v_r1: vec![0.0; n], v_r2: vec![0.0; n], v_r3: vec![0.0; n],
        }
    }

    pub fn resize(&mut self, n: usize) {
        let grow = |v: &mut Vec<f32>, len: usize| { if v.len() < len { v.resize(len, 0.0); }};
        grow(&mut self.v_th1, n); grow(&mut self.v_th2, n); grow(&mut self.v_th3, n);
        grow(&mut self.v_r1, n); grow(&mut self.v_r2, n); grow(&mut self.v_r3, n);
    }

    /// Zero all buffers.
    pub fn zero(&mut self) {
        self.v_th1.fill(0.0);
        self.v_th2.fill(0.0);
        self.v_th3.fill(0.0);
        self.v_r1.fill(0.0);
        self.v_r2.fill(0.0);
        self.v_r3.fill(0.0);
    }

    /// Add another result in-place.
    pub fn add_into(&mut self, other: &PstGradients) {
        for (a, b) in self.v_th1.iter_mut().zip(&other.v_th1) { *a += b; }
        for (a, b) in self.v_th2.iter_mut().zip(&other.v_th2) { *a += b; }
        for (a, b) in self.v_th3.iter_mut().zip(&other.v_th3) { *a += b; }
        for (a, b) in self.v_r1.iter_mut().zip(&other.v_r1) { *a += b; }
        for (a, b) in self.v_r2.iter_mut().zip(&other.v_r2) { *a += b; }
        for (a, b) in self.v_r3.iter_mut().zip(&other.v_r3) { *a += b; }
    }

    /// Copy into BpttWorkspace slices.
    pub fn copy_to_slices(&self, th1: &mut [f32], th2: &mut [f32], th3: &mut [f32],
                          r1: &mut [f32], r2: &mut [f32], r3: &mut [f32]) {
        th1.copy_from_slice(&self.v_th1);
        th2.copy_from_slice(&self.v_th2);
        th3.copy_from_slice(&self.v_th3);
        r1.copy_from_slice(&self.v_r1);
        r2.copy_from_slice(&self.v_r2);
        r3.copy_from_slice(&self.v_r3);
    }
}

// ═══════════════════════════════════════════════════════════
//  JIT Jacobian-vector product: v = J^T · grad_pst
// ═══════════════════════════════════════════════════════════
//
//  Вместо построения J[pst_dim × n] → v = J^T · g, вычисляем v напрямую.
//  Каждая PST-фича вносит вклад в 1-2 state-переменных (θ или r одного осциллятора),
//  кроме cross-R (зависит от всех θ обеих популяций).
//
//  PST layout (pst_linear_dim = 18n + 12):
//    0..2n:     sin(θ₁), cos(θ₁) — 2 per osc, interleaved
//    2n..4n:    sin(θ₂), cos(θ₂)
//    4n..6n:    sin(θ₃), cos(θ₃)
//    6n..9n:    velocity dθ₁/dt, dθ₂/dt, dθ₃/dt
//    9n..12n:   amplitude r₁, r₂, r₃
//    12n..14n:  sin²(θ₁), cos²(θ₁)
//    14n..16n:  sin²(θ₂), cos²(θ₂)
//    16n..18n:  sin²(θ₃), cos²(θ₃)
//    18n+0..+2: R₁₂, R₂₃, R₁₃ (cross order params)
//    18n+3..+11: mean sin/cos/r per population (9 floats)

/// JIT Jacobian-vector product: v = J^T · grad_pst
///
/// **Ноль матриц.** Один проход по PST-фичам за O(pst_dim + n).
///
/// - th1..th3, r1..r3: post-update state (final state, или step+1 для SPIGRAD)
/// - grad_pst: [pst_dim] градиент потерь по PST
/// - out: [PstGradients] результат, будет занулён и заполнен
pub fn pst_jacobian_v_jit(
    th1: &[f32], _r1: &[f32],
    th2: &[f32], _r2: &[f32],
    th3: &[f32], _r3: &[f32],
    n: usize,
    grad_pst: &[f32],
    out: &mut PstGradients,
) {
    out.zero();
    let nf = n as f32;
    let inv_nf = 1.0 / nf;

    // ── Pre-compute sin/cos (O(n)) ──
    let mut sin1 = vec![0.0f32; n]; let mut cos1 = vec![0.0f32; n];
    let mut sin2 = vec![0.0f32; n]; let mut cos2 = vec![0.0f32; n];
    let mut sin3 = vec![0.0f32; n]; let mut cos3 = vec![0.0f32; n];
    for i in 0..n {
        sin1[i] = th1[i].sin(); cos1[i] = th1[i].cos();
        sin2[i] = th2[i].sin(); cos2[i] = th2[i].cos();
        sin3[i] = th3[i].sin(); cos3[i] = th3[i].cos();
    }

    let mut pos = 0usize;

    // 1. sin(θ₁ᵢ), cos(θ₁ᵢ) [0..2n]: d(sin)/dθ = cos, d(cos)/dθ = -sin
    for i in 0..n {
        out.v_th1[i] += grad_pst[pos] * cos1[i] - grad_pst[pos + 1] * sin1[i];
        pos += 2;
    }

    // 2. sin(θ₂ᵢ), cos(θ₂ᵢ) [2n..4n]
    for i in 0..n {
        out.v_th2[i] += grad_pst[pos] * cos2[i] - grad_pst[pos + 1] * sin2[i];
        pos += 2;
    }

    // 3. sin(θ₃ᵢ), cos(θ₃ᵢ) [4n..6n]
    for i in 0..n {
        out.v_th3[i] += grad_pst[pos] * cos3[i] - grad_pst[pos + 1] * sin3[i];
        pos += 2;
    }

    // 4. Velocity [6n..9n]: d(vel)/d(th_post) = 1
    for i in 0..n { out.v_th1[i] += grad_pst[pos]; pos += 1; }
    for i in 0..n { out.v_th2[i] += grad_pst[pos]; pos += 1; }
    for i in 0..n { out.v_th3[i] += grad_pst[pos]; pos += 1; }

    // 5. Amplitude [9n..12n]: d(r)/dr = 1
    for i in 0..n { out.v_r1[i] += grad_pst[pos]; pos += 1; }
    for i in 0..n { out.v_r2[i] += grad_pst[pos]; pos += 1; }
    for i in 0..n { out.v_r3[i] += grad_pst[pos]; pos += 1; }

    // 6. sin²(θ₁ᵢ), cos²(θ₁ᵢ) [12n..14n]: d(sin²)/dθ = 2·sin·cos, d(cos²)/dθ = -2·sin·cos
    for i in 0..n {
        let d_sc = 2.0 * sin1[i] * cos1[i];
        out.v_th1[i] += (grad_pst[pos] - grad_pst[pos + 1]) * d_sc;
        pos += 2;
    }

    // 7. sin²(θ₂ᵢ), cos²(θ₂ᵢ) [14n..16n]
    for i in 0..n {
        let d_sc = 2.0 * sin2[i] * cos2[i];
        out.v_th2[i] += (grad_pst[pos] - grad_pst[pos + 1]) * d_sc;
        pos += 2;
    }

    // 8. sin²(θ₃ᵢ), cos²(θ₃ᵢ) [16n..18n]
    for i in 0..n {
        let d_sc = 2.0 * sin3[i] * cos3[i];
        out.v_th3[i] += (grad_pst[pos] - grad_pst[pos + 1]) * d_sc;
        pos += 2;
    }

    // pos is now 18n

    // 9. Cross order parameters R₁₂, R₂₃, R₁₃ [18n..18n+3]
    //
    // R = sqrt(ss² + sc²) / n, где:
    //   ss = Σ (sin_a·cos_b - cos_a·sin_b)
    //   sc = Σ (cos_a·cos_b + sin_a·sin_b)
    // dR/dθ_aᵢ = (ss·dss_da + sc·dsc_da) / (n²·R)
    // dR/dθ_bᵢ = (ss·dss_db + sc·dsc_db) / (n²·R)

    // Pre-compute per-pair sums
    let cross_sums = |sa: &[f32], ca: &[f32], sb: &[f32], cb: &[f32]| -> (f64, f64) {
        let (ss, sc) = (0..n).fold((0.0f64, 0.0f64), |(s, c), i| {
            let a = sa[i] as f64; let b = ca[i] as f64;
            let c2 = sb[i] as f64; let d = cb[i] as f64;
            (s + a * d - b * c2, c + b * d + a * c2)
        });
        (ss, sc)
    };

    let (ss12, sc12) = cross_sums(&sin1, &cos1, &sin2, &cos2);
    let (ss23, sc23) = cross_sums(&sin2, &cos2, &sin3, &cos3);
    let (ss13, sc13) = cross_sums(&sin1, &cos1, &sin3, &cos3);

    let norm_n = |ss: f64, sc: f64| -> f64 {
        let nrm = (ss * ss + sc * sc).sqrt();
        if nrm > 1e-12 { 1.0 / (nrm * nf as f64) } else { 0.0 }
    };
    let inv12 = norm_n(ss12, sc12);
    let inv23 = norm_n(ss23, sc23);
    let inv13 = norm_n(ss13, sc13);

    let g_r12 = grad_pst[pos] as f64;
    let g_r23 = grad_pst[pos + 1] as f64;
    let g_r13 = grad_pst[pos + 2] as f64;

    for i in 0..n {
        let (s1, c1) = (sin1[i] as f64, cos1[i] as f64);
        let (s2, c2) = (sin2[i] as f64, cos2[i] as f64);
        let (s3, c3) = (sin3[i] as f64, cos3[i] as f64);

        // R₁₂ derivatives: dR/dθ₁ᵢ and dR/dθ₂ᵢ
        let dss_1_12 = c1 * c2 + s1 * s2;
        let dsc_1_12 = c1 * s2 - s1 * c2;
        let dss_2_12 = -(c1 * c2 + s1 * s2);
        let dsc_2_12 = -s1 * c2 + c1 * s2;
        let d12_1 = g_r12 * (ss12 * dss_1_12 + sc12 * dsc_1_12) * inv12;
        let d12_2 = g_r12 * (ss12 * dss_2_12 + sc12 * dsc_2_12) * inv12;

        // R₂₃ derivatives
        let dss_2_23 = c2 * c3 + s2 * s3;
        let dsc_2_23 = c2 * s3 - s2 * c3;
        let dss_3_23 = -(c2 * c3 + s2 * s3);
        let dsc_3_23 = -s2 * c3 + c2 * s3;
        let d23_2 = g_r23 * (ss23 * dss_2_23 + sc23 * dsc_2_23) * inv23;
        let d23_3 = g_r23 * (ss23 * dss_3_23 + sc23 * dsc_3_23) * inv23;

        // R₁₃ derivatives
        let dss_1_13 = c1 * c3 + s1 * s3;
        let dsc_1_13 = c1 * s3 - s1 * c3;
        let dss_3_13 = -(c1 * c3 + s1 * s3);
        let dsc_3_13 = -s1 * c3 + c1 * s3;
        let d13_1 = g_r13 * (ss13 * dss_1_13 + sc13 * dsc_1_13) * inv13;
        let d13_3 = g_r13 * (ss13 * dss_3_13 + sc13 * dsc_3_13) * inv13;

        out.v_th1[i] += (d12_1 + d13_1) as f32;
        out.v_th2[i] += (d12_2 + d23_2) as f32;
        out.v_th3[i] += (d23_3 + d13_3) as f32;
    }
    pos += 3;

    // 10. Mean sin/cos/r [18n+3..18n+12]
    // d(mean_sin)/dθᵢ = cosᵢ/n, d(mean_cos)/dθᵢ = -sinᵢ/n, d(mean_r)/drᵢ = 1/n
    let ms1_g = grad_pst[pos] as f64;
    let mc1_g = grad_pst[pos + 1] as f64;
    let mr1_g = grad_pst[pos + 2] as f64;
    for i in 0..n {
        out.v_th1[i] += (ms1_g * cos1[i] as f64 - mc1_g * sin1[i] as f64) as f32 * inv_nf;
        out.v_r1[i] += (mr1_g * inv_nf as f64) as f32;
    }
    pos += 3;

    let ms2_g = grad_pst[pos] as f64;
    let mc2_g = grad_pst[pos + 1] as f64;
    let mr2_g = grad_pst[pos + 2] as f64;
    for i in 0..n {
        out.v_th2[i] += (ms2_g * cos2[i] as f64 - mc2_g * sin2[i] as f64) as f32 * inv_nf;
        out.v_r2[i] += (mr2_g * inv_nf as f64) as f32;
    }
    pos += 3;

    let ms3_g = grad_pst[pos] as f64;
    let mc3_g = grad_pst[pos + 1] as f64;
    let mr3_g = grad_pst[pos + 2] as f64;
    for i in 0..n {
        out.v_th3[i] += (ms3_g * cos3[i] as f64 - mc3_g * sin3[i] as f64) as f32 * inv_nf;
        out.v_r3[i] += (mr3_g * inv_nf as f64) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t3::pst_linear_dim;

    #[test]
    fn test_pst_jacobian_v_jit_zero_grad() {
        // Zero grad_pst → zero output
        let n = 8;
        let pst_dim = pst_linear_dim(n);
        let th1 = vec![0.0; n]; let th2 = vec![0.5; n]; let th3 = vec![1.0; n];
        let r1 = vec![1.0; n]; let r2 = vec![0.5; n]; let r3 = vec![0.8; n];
        let grad_pst = vec![0.0; pst_dim];
        let mut v = PstGradients::new(n);
        pst_jacobian_v_jit(&th1, &r1, &th2, &r2, &th3, &r3, n, &grad_pst, &mut v);
        for x in v.v_th1.iter().chain(v.v_th2.iter()).chain(v.v_th3.iter())
            .chain(v.v_r1.iter()).chain(v.v_r2.iter()).chain(v.v_r3.iter()) {
            assert!((*x).abs() < 1e-10, "zero grad → zero out: {}", x);
        }
    }

    #[test]
    fn test_pst_jacobian_v_jit_single_feat() {
        // Single non-zero grad: sin(θ₁₀)
        let n = 8;
        let pst_dim = pst_linear_dim(n);
        let th1: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7)).collect();
        let th2 = vec![0.0; n]; let th3 = vec![0.0; n];
        let r1 = vec![1.0; n]; let r2 = vec![1.0; n]; let r3 = vec![1.0; n];

        // grad_pst[0] = 1.0 → dL/d(sin(θ₁₀)) = 1.0
        let mut grad_pst = vec![0.0; pst_dim];
        grad_pst[0] = 1.0;  // sin(θ₁₀)

        let mut v = PstGradients::new(n);
        pst_jacobian_v_jit(&th1, &r1, &th2, &r2, &th3, &r3, n, &grad_pst, &mut v);

        // Only v_th1[0] should be non-zero: d(sin(θ₁₀))/dθ₁₀ = cos(θ₁₀)
        let expected = th1[0].cos();
        assert!((v.v_th1[0] - expected).abs() < 1e-6,
            "sin(θ₁₀): v_th1[0]={:.10} expected={:.10}", v.v_th1[0], expected);
        for i in 1..n { assert!(v.v_th1[i].abs() < 1e-10, "v_th1[{}] should be 0", i); }
        for x in v.v_th2.iter().chain(v.v_th3.iter()).chain(v.v_r1.iter()).chain(v.v_r2.iter()).chain(v.v_r3.iter()) {
            assert!(x.abs() < 1e-10, "other channels should be 0");
        }
    }
}
