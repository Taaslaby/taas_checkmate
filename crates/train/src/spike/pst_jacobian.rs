//! PST analytical Jacobian + Jacobian-vector product.

use crate::t3::pst_linear_dim;

/// Result of PST Jacobian-vector product.
pub struct PstJacobianV {
    pub th1: Vec<f32>,
    pub th2: Vec<f32>,
    pub th3: Vec<f32>,
    pub r1: Vec<f32>,
    pub r2: Vec<f32>,
    pub r3: Vec<f32>,
}

/// Compute analytical PST Jacobian at final state.
pub fn pst_jacobian(
    th1: &[f32], _r1: &[f32],
    th2: &[f32], _r2: &[f32],
    th3: &[f32], _r3: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let pst_dim = pst_linear_dim(n);
    let mut d_th1 = vec![0.0; pst_dim * n];
    let mut d_th2 = vec![0.0; pst_dim * n];
    let mut d_th3 = vec![0.0; pst_dim * n];
    let mut d_r1 = vec![0.0; pst_dim * n];
    let mut d_r2 = vec![0.0; pst_dim * n];
    let mut d_r3 = vec![0.0; pst_dim * n];

    let mut sin1 = vec![0.0; n]; let mut cos1 = vec![0.0; n];
    let mut sin2 = vec![0.0; n]; let mut cos2 = vec![0.0; n];
    let mut sin3 = vec![0.0; n]; let mut cos3 = vec![0.0; n];
    for i in 0..n {
        sin1[i] = th1[i].sin(); cos1[i] = th1[i].cos();
        sin2[i] = th2[i].sin(); cos2[i] = th2[i].cos();
        sin3[i] = th3[i].sin(); cos3[i] = th3[i].cos();
    }

    let mut pos = 0usize;

    for i in 0..n {
        d_th1[(pos + 2 * i) * n + i] = cos1[i];
        d_th1[(pos + 2 * i + 1) * n + i] = -sin1[i];
    }
    pos += 2 * n;
    for i in 0..n {
        d_th2[(pos + 2 * i) * n + i] = cos2[i];
        d_th2[(pos + 2 * i + 1) * n + i] = -sin2[i];
    }
    pos += 2 * n;
    for i in 0..n {
        d_th3[(pos + 2 * i) * n + i] = cos3[i];
        d_th3[(pos + 2 * i + 1) * n + i] = -sin3[i];
    }
    pos += 2 * n;

    for i in 0..n { d_th1[(pos + i) * n + i] = 1.0; }
    pos += n;
    for i in 0..n { d_th2[(pos + i) * n + i] = 1.0; }
    pos += n;
    for i in 0..n { d_th3[(pos + i) * n + i] = 1.0; }
    pos += n;

    for i in 0..n { d_r1[(pos + i) * n + i] = 1.0; }
    pos += n;
    for i in 0..n { d_r2[(pos + i) * n + i] = 1.0; }
    pos += n;
    for i in 0..n { d_r3[(pos + i) * n + i] = 1.0; }
    pos += n;

    for i in 0..n {
        let d_sc = 2.0 * sin1[i] * cos1[i];
        d_th1[(pos + 2 * i) * n + i] = d_sc;
        d_th1[(pos + 2 * i + 1) * n + i] = -d_sc;
    }
    pos += 2 * n;
    for i in 0..n {
        let d_sc = 2.0 * sin2[i] * cos2[i];
        d_th2[(pos + 2 * i) * n + i] = d_sc;
        d_th2[(pos + 2 * i + 1) * n + i] = -d_sc;
    }
    pos += 2 * n;
    for i in 0..n {
        let d_sc = 2.0 * sin3[i] * cos3[i];
        d_th3[(pos + 2 * i) * n + i] = d_sc;
        d_th3[(pos + 2 * i + 1) * n + i] = -d_sc;
    }
    pos += 2 * n;

    cross_order_jacobian(&sin1, &cos1, &sin2, &cos2, n, &mut d_th1, &mut d_th2, pos);
    pos += 1;
    cross_order_jacobian(&sin2, &cos2, &sin3, &cos3, n, &mut d_th2, &mut d_th3, pos);
    pos += 1;
    cross_order_jacobian(&sin1, &cos1, &sin3, &cos3, n, &mut d_th1, &mut d_th3, pos);
    pos += 1;

    let nf = n as f32;
    for i in 0..n {
        d_th1[(pos) * n + i]     = cos1[i] / nf;
        d_th1[(pos + 1) * n + i] = -sin1[i] / nf;
        d_r1[(pos + 2) * n + i]  = 1.0 / nf;
    }
    pos += 3;
    for i in 0..n {
        d_th2[(pos) * n + i]     = cos2[i] / nf;
        d_th2[(pos + 1) * n + i] = -sin2[i] / nf;
        d_r2[(pos + 2) * n + i]  = 1.0 / nf;
    }
    pos += 3;
    for i in 0..n {
        d_th3[(pos) * n + i]     = cos3[i] / nf;
        d_th3[(pos + 1) * n + i] = -sin3[i] / nf;
        d_r3[(pos + 2) * n + i]  = 1.0 / nf;
    }

    (d_th1, d_th2, d_th3, d_r1, d_r2, d_r3)
}

fn cross_order_jacobian(
    sin1: &[f32], cos1: &[f32],
    sin2: &[f32], cos2: &[f32],
    n: usize,
    jac_theta1: &mut [f32],
    jac_theta2: &mut [f32],
    pos: usize,
) {
    let mut sum_s = 0.0f64;
    let mut sum_c = 0.0f64;
    for i in 0..n {
        let s1 = sin1[i] as f64; let c1 = cos1[i] as f64;
        let s2 = sin2[i] as f64; let c2 = cos2[i] as f64;
        sum_s += (s1 * c2 - c1 * s2) as f64;
        sum_c += (c1 * c2 + s1 * s2) as f64;
    }
    let norm = (sum_s * sum_s + sum_c * sum_c).sqrt();
    if norm < 1e-12 { return; }
    let inv_norm_n = (1.0 / norm) / n as f64;

    for i in 0..n {
        let dss_dth1 = cos1[i] as f64 * cos2[i] as f64 + sin1[i] as f64 * sin2[i] as f64;
        let dsc_dth1 = -sin1[i] as f64 * cos2[i] as f64 + cos1[i] as f64 * sin2[i] as f64;
        jac_theta1[pos * n + i] = ((sum_s * dss_dth1 + sum_c * dsc_dth1) * inv_norm_n) as f32;

        let dss_dth2 = -(cos1[i] as f64 * cos2[i] as f64 + sin1[i] as f64 * sin2[i] as f64);
        let dsc_dth2 = -cos1[i] as f64 * sin2[i] as f64 + sin1[i] as f64 * cos2[i] as f64;
        jac_theta2[pos * n + i] = ((sum_s * dss_dth2 + sum_c * dsc_dth2) * inv_norm_n) as f32;
    }
}

/// Compute v = J^T · grad_pst from pre-computed Jacobian matrices.
pub fn compute_pst_jacobian_v(
    grad_pst: &[f32],
    d_th1_j: &[f32], d_th2_j: &[f32], d_th3_j: &[f32],
    d_r1_j: &[f32], d_r2_j: &[f32], d_r3_j: &[f32],
    n: usize,
    pst_dim: usize,
) -> PstJacobianV {
    let mut v_th1 = vec![0.0; n];
    let mut v_th2 = vec![0.0; n];
    let mut v_th3 = vec![0.0; n];
    let mut v_r1 = vec![0.0; n];
    let mut v_r2 = vec![0.0; n];
    let mut v_r3 = vec![0.0; n];

    for pst_idx in 0..pst_dim {
        let gp = grad_pst[pst_idx];
        if gp == 0.0 { continue; }
        let base = pst_idx * n;
        for i in 0..n {
            v_th1[i] += d_th1_j[base + i] * gp;
            v_th2[i] += d_th2_j[base + i] * gp;
            v_th3[i] += d_th3_j[base + i] * gp;
            v_r1[i] += d_r1_j[base + i] * gp;
            v_r2[i] += d_r2_j[base + i] * gp;
            v_r3[i] += d_r3_j[base + i] * gp;
        }
    }

    PstJacobianV { th1: v_th1, th2: v_th2, th3: v_th3, r1: v_r1, r2: v_r2, r3: v_r3 }
}
