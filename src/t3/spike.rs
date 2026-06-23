//! # T2-SPIGRAD — T3-SPIKE GRADient in Rust.
//!
//! Forward: drives → T3 (N steps) → PST features + trajectory
//! Backward: grad_PST → BPTT through T3 dynamics → grad_drives
//!
//! Вся математика на f32, аналитические Jacobian.
//! Diagonal approximation для coupling Jacobian (cross terms O(1/N)).

use crate::t3::{
    T3Backend, Avx2Backend, TriOscillator,
    pst_linear_features, pst_linear_dim,
};

// ═══════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════

const DEFAULT_EPSILON: f32 = 0.1;
const DEFAULT_PHASE_GATE_GAIN: f32 = 4.0;
const DEFAULT_GAMMA: [f32; 3] = [0.5, 1.0, 1.12];

// ═══════════════════════════════════════════════════════════
//  Trajectory
// ═══════════════════════════════════════════════════════════

/// Записанная траектория T3 для backward pass.
pub struct T3SpikeTrajectory {
    pub n_osc: usize,
    pub n_steps: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,

    // Frequencies [n_osc]
    pub omega1: Vec<f32>,
    pub omega2: Vec<f32>,
    pub omega3: Vec<f32>,

    // Per-step PRE-state [n_steps × n_osc] — state BEFORE step t
    pub th1_pre: Vec<f32>,
    pub th2_pre: Vec<f32>,
    pub th3_pre: Vec<f32>,
    pub r1_pre: Vec<f32>,
    pub r2_pre: Vec<f32>,
    pub r3_pre: Vec<f32>,

    // Final state AFTER last step [n_osc] — state_N for PST Jacobian
    pub th1_final: Vec<f32>,
    pub th2_final: Vec<f32>,
    pub th3_final: Vec<f32>,
    pub r1_final: Vec<f32>,
    pub r2_final: Vec<f32>,
    pub r3_final: Vec<f32>,

    // Per-step coupling [n_steps × n_osc] — coupling at PRE-state
    pub coupling1: Vec<f32>,
    pub coupling2: Vec<f32>,
    pub coupling3: Vec<f32>,

    // Per-step drive [n_steps × n_osc]
    pub drive1: Vec<f32>,
    pub drive2: Vec<f32>,
    pub drive3: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════
//  Forward
// ═══════════════════════════════════════════════════════════

/// Forward pass: drives → T3 (N steps) → PST features + trajectory.
///
/// Generic over any [`T3Backend`]: Avx2Backend (CPU SIMD), ScalarBackend (debug), CudaBackend (GPU).
///
/// `backend`: the compute backend (e.g., `Avx2Backend`, `CudaBackend`)
/// `drives`: [n_steps × 3*n_osc] flattened.
/// Returns (pst_features [pst_dim], trajectory).
pub fn t3spike_forward_with_backend<B: T3Backend>(
    backend: B,
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: &[f32],
) -> (Vec<f32>, T3SpikeTrajectory) {
    let mut tri = TriOscillator::new(n_osc, freq_min, freq_max, backend);
    tri.drive_gain = drive_gain;
    tri.alpha = alpha;
    tri.epsilon = DEFAULT_EPSILON;
    tri.set_uniform_coupling(coupling_w12, coupling_w13, coupling_w23);

    let step_size = 3 * n_osc;

    let mut traj = T3SpikeTrajectory {
        n_osc,
        n_steps,
        dt,
        drive_gain,
        alpha,
        epsilon: DEFAULT_EPSILON,
        freq_min,
        freq_max,
        w12: coupling_w12,
        w13: coupling_w13,
        w23: coupling_w23,
        omega1: tri.omega1.clone(),
        omega2: tri.omega2.clone(),
        omega3: tri.omega3.clone(),
        th1_pre: vec![0.0; n_steps * n_osc],
        th2_pre: vec![0.0; n_steps * n_osc],
        th3_pre: vec![0.0; n_steps * n_osc],
        r1_pre: vec![0.0; n_steps * n_osc],
        r2_pre: vec![0.0; n_steps * n_osc],
        r3_pre: vec![0.0; n_steps * n_osc],
        th1_final: vec![0.0; n_osc],
        th2_final: vec![0.0; n_osc],
        th3_final: vec![0.0; n_osc],
        r1_final: vec![0.0; n_osc],
        r2_final: vec![0.0; n_osc],
        r3_final: vec![0.0; n_osc],
        coupling1: vec![0.0; n_steps * n_osc],
        coupling2: vec![0.0; n_steps * n_osc],
        coupling3: vec![0.0; n_steps * n_osc],
        drive1: vec![0.0; n_steps * n_osc],
        drive2: vec![0.0; n_steps * n_osc],
        drive3: vec![0.0; n_steps * n_osc],
    };

    for step in 0..n_steps {
        let off = step * n_osc;
        let d_off = step * step_size;

        // ── Save PRE-step state ──
        copy_slice(&mut traj.th1_pre[off..off + n_osc], &tri.th1[..n_osc]);
        copy_slice(&mut traj.th2_pre[off..off + n_osc], &tri.th2[..n_osc]);
        copy_slice(&mut traj.th3_pre[off..off + n_osc], &tri.th3[..n_osc]);
        copy_slice(&mut traj.r1_pre[off..off + n_osc], &tri.r1[..n_osc]);
        copy_slice(&mut traj.r2_pre[off..off + n_osc], &tri.r2[..n_osc]);
        copy_slice(&mut traj.r3_pre[off..off + n_osc], &tri.r3[..n_osc]);

        // ── Set drive ──
        let d1 = &drives[d_off..d_off + n_osc];
        let d2 = &drives[d_off + n_osc..d_off + 2 * n_osc];
        let d3 = &drives[d_off + 2 * n_osc..d_off + 3 * n_osc];
        tri.drive1[..n_osc].copy_from_slice(d1);
        tri.drive2[..n_osc].copy_from_slice(d2);
        tri.drive3[..n_osc].copy_from_slice(d3);

        // ── Save drive ──
        copy_slice(&mut traj.drive1[off..off + n_osc], d1);
        copy_slice(&mut traj.drive2[off..off + n_osc], d2);
        copy_slice(&mut traj.drive3[off..off + n_osc], d3);

        // ── Step ──
        tri.step(dt);

        // ── Save coupling (computed from PRE-state during step) ──
        copy_slice(&mut traj.coupling1[off..off + n_osc], &tri.coupling1[..n_osc]);
        copy_slice(&mut traj.coupling2[off..off + n_osc], &tri.coupling2[..n_osc]);
        copy_slice(&mut traj.coupling3[off..off + n_osc], &tri.coupling3[..n_osc]);
    }

    // ── Save final state ──
    copy_slice(&mut traj.th1_final, &tri.th1[..n_osc]);
    copy_slice(&mut traj.th2_final, &tri.th2[..n_osc]);
    copy_slice(&mut traj.th3_final, &tri.th3[..n_osc]);
    copy_slice(&mut traj.r1_final, &tri.r1[..n_osc]);
    copy_slice(&mut traj.r2_final, &tri.r2[..n_osc]);
    copy_slice(&mut traj.r3_final, &tri.r3[..n_osc]);

    // ── PST features from final state ──
    let pst = pst_linear_features(&tri);

    (pst, traj)
}

/// Forward pass with default Avx2Backend (CPU SIMD).
/// Convenience wrapper around [`t3spike_forward_with_backend`].
pub fn t3spike_forward(
    n_osc: usize,
    n_steps: usize,
    dt: f32,
    drive_gain: f32,
    alpha: f32,
    freq_min: f32,
    freq_max: f32,
    coupling_w12: f32,
    coupling_w13: f32,
    coupling_w23: f32,
    drives: &[f32],
) -> (Vec<f32>, T3SpikeTrajectory) {
    t3spike_forward_with_backend(
        Avx2Backend,
        n_osc, n_steps, dt, drive_gain, alpha,
        freq_min, freq_max,
        coupling_w12, coupling_w13, coupling_w23,
        drives,
    )
}

#[inline]
fn copy_slice(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
}

// ═══════════════════════════════════════════════════════════
//  PST Jacobian
// ═══════════════════════════════════════════════════════════

/// Compute analytical PST Jacobian at final state.
///
/// Returns (d_th1, d_th2, d_th3, d_r1, d_r2, d_r3) —
/// each is (pst_dim, n_osc) flattened column-major: [pst_dim × n_osc].
pub fn pst_jacobian(
    th1: &[f32],
    _r1: &[f32],
    th2: &[f32],
    _r2: &[f32],
    th3: &[f32],
    _r3: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let pst_dim = pst_linear_dim(n);
    let mut d_th1 = vec![0.0; pst_dim * n];
    let mut d_th2 = vec![0.0; pst_dim * n];
    let mut d_th3 = vec![0.0; pst_dim * n];
    let mut d_r1 = vec![0.0; pst_dim * n];
    let mut d_r2 = vec![0.0; pst_dim * n];
    let mut d_r3 = vec![0.0; pst_dim * n];

    // Pre-compute sin/cos
    let mut sin1 = vec![0.0; n];
    let mut cos1 = vec![0.0; n];
    let mut sin2 = vec![0.0; n];
    let mut cos2 = vec![0.0; n];
    let mut sin3 = vec![0.0; n];
    let mut cos3 = vec![0.0; n];
    for i in 0..n {
        sin1[i] = th1[i].sin();
        cos1[i] = th1[i].cos();
        sin2[i] = th2[i].sin();
        cos2[i] = th2[i].cos();
        sin3[i] = th3[i].sin();
        cos3[i] = th3[i].cos();
    }

    let mut pos = 0usize;

    // ── 1. sin(θ), cos(θ) — 6N ──
    // FEATURES: sin1[0], cos1[0], sin1[1], cos1[1], ... (interleaved per oscillator)
    // Pop 1: row = pos + 2*i (sin), row = pos + 2*i + 1 (cos)
    for i in 0..n {
        // d(sin1[i])/dθ1[i] = cos1[i]  → column i, row pos+2*i
        d_th1[(pos + 2 * i) * n + i] = cos1[i];
        // d(cos1[i])/dθ1[i] = -sin1[i] → column i, row pos+2*i+1
        d_th1[(pos + 2 * i + 1) * n + i] = -sin1[i];
    }
    pos += 2 * n;
    // Pop 2
    for i in 0..n {
        d_th2[(pos + 2 * i) * n + i] = cos2[i];
        d_th2[(pos + 2 * i + 1) * n + i] = -sin2[i];
    }
    pos += 2 * n;
    // Pop 3
    for i in 0..n {
        d_th3[(pos + 2 * i) * n + i] = cos3[i];
        d_th3[(pos + 2 * i + 1) * n + i] = -sin3[i];
    }
    pos += 2 * n;

    // ── 2. dθ/dt (velocity) — 3N ──
    // FEATURES: vel1[0], vel1[1], ..., vel1[n-1] (sequential per population)
    // d(vel1_i)/d(θ1_i) = 1 → row pos+i, column i
    for i in 0..n {
        d_th1[(pos + i) * n + i] = 1.0;
    }
    pos += n;
    for i in 0..n {
        d_th2[(pos + i) * n + i] = 1.0;
    }
    pos += n;
    for i in 0..n {
        d_th3[(pos + i) * n + i] = 1.0;
    }
    pos += n;

    // ── 3. Amplitudes r — 3N ──
    // FEATURES: r1[0], r1[1], ..., r1[n-1] (sequential)
    for i in 0..n {
        d_r1[(pos + i) * n + i] = 1.0;
    }
    pos += n;
    for i in 0..n {
        d_r2[(pos + i) * n + i] = 1.0;
    }
    pos += n;
    for i in 0..n {
        d_r3[(pos + i) * n + i] = 1.0;
    }
    pos += n;

    // ── 4. Squared features sin²(θ), cos²(θ) — 6N ──
    // FEATURES: sin²1[0], cos²1[0], sin²1[1], cos²1[1], ... (interleaved)
    // d(sin²)/dθ = 2·sin·cos, d(cos²)/dθ = -2·sin·cos
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

    // ── 5. Cross order parameters R₁₂, R₂₃, R₁₃ — 3 ──
    // EACH is one PST feature. dR/dθ₁[i] for ALL i goes to row `pos`, column `i`.
    cross_order_jacobian(&sin1, &cos1, &sin2, &cos2, n,
                         &mut d_th1, &mut d_th2, pos);
    pos += 1;
    cross_order_jacobian(&sin2, &cos2, &sin3, &cos3, n,
                         &mut d_th2, &mut d_th3, pos);
    pos += 1;
    cross_order_jacobian(&sin1, &cos1, &sin3, &cos3, n,
                         &mut d_th1, &mut d_th3, pos);
    pos += 1;

    // ── 6. Mean sin/cos/r — 9 ──
    // FEATURES: mean_sin1, mean_cos1, mean_r1, mean_sin2, mean_cos2, mean_r2, ...
    // EACH mean is ONE PST feature. d(mean)/d(state[i]) for ALL i goes to the SAME row.
    let nf = n as f32;
    // Pop 1: mean_sin at pos, mean_cos at pos+1, mean_r at pos+2
    for i in 0..n {
        d_th1[(pos) * n + i]     = cos1[i] / nf;    // d(mean_sin1)/dθ1[i]
        d_th1[(pos + 1) * n + i] = -sin1[i] / nf;   // d(mean_cos1)/dθ1[i]
        d_r1[(pos + 2) * n + i]  = 1.0 / nf;        // d(mean_r1)/dr1[i]
    }
    pos += 3;
    // Pop 2
    for i in 0..n {
        d_th2[(pos) * n + i]     = cos2[i] / nf;
        d_th2[(pos + 1) * n + i] = -sin2[i] / nf;
        d_r2[(pos + 2) * n + i]  = 1.0 / nf;
    }
    pos += 3;
    // Pop 3
    for i in 0..n {
        d_th3[(pos) * n + i]     = cos3[i] / nf;
        d_th3[(pos + 1) * n + i] = -sin3[i] / nf;
        d_r3[(pos + 2) * n + i]  = 1.0 / nf;
    }

    (d_th1, d_th2, d_th3, d_r1, d_r2, d_r3)
}

/// Fill Jacobian of cross order parameter R = |mean(exp(i(θ₁-θ₂)))|.
///
/// dR/dθ₁[i] and dR/dθ₂[i] at position `pos` in the PST Jacobian.
fn cross_order_jacobian(
    sin1: &[f32], cos1: &[f32],
    sin2: &[f32], cos2: &[f32],
    n: usize,
    jac_theta1: &mut [f32],
    jac_theta2: &mut [f32],
    pos: usize,
) {
    // sum_s = Σ(sin1·cos2 - cos1·sin2)
    // sum_c = Σ(cos1·cos2 + sin1·sin2)
    let mut sum_s = 0.0f64;
    let mut sum_c = 0.0f64;
    for i in 0..n {
        let s1 = sin1[i] as f64;
        let c1 = cos1[i] as f64;
        let s2 = sin2[i] as f64;
        let c2 = cos2[i] as f64;
        sum_s += (s1 * c2 - c1 * s2) as f64;
        sum_c += (c1 * c2 + s1 * s2) as f64;
    }
    let norm = (sum_s * sum_s + sum_c * sum_c).sqrt();
    if norm < 1e-12 {
        return;
    }
    let inv_norm_n = (1.0 / norm) / n as f64;

    for i in 0..n {
        // dR/dθ₁[i]
        // d(sum_s)/dθ₁[i] = cos1·cos2 + sin1·sin2 = cos(θ₁-θ₂)
        // d(sum_c)/dθ₁[i] = -sin1·cos2 + cos1·sin2 = -sin(θ₁-θ₂)
        let dss_dth1 = cos1[i] as f64 * cos2[i] as f64 + sin1[i] as f64 * sin2[i] as f64;
        let dsc_dth1 = -sin1[i] as f64 * cos2[i] as f64 + cos1[i] as f64 * sin2[i] as f64;
        jac_theta1[pos * n + i] = ((sum_s * dss_dth1 + sum_c * dsc_dth1) * inv_norm_n) as f32;

        // dR/dθ₂[i]
        // d(sum_s)/dθ₂[i] = -(cos1·cos2 + sin1·sin2) = -cos(θ₁-θ₂)
        // d(sum_c)/dθ₂[i] = -cos1·sin2 + sin1·cos2 = sin(θ₁-θ₂)
        let dss_dth2 = -(cos1[i] as f64 * cos2[i] as f64 + sin1[i] as f64 * sin2[i] as f64);
        let dsc_dth2 = -cos1[i] as f64 * sin2[i] as f64 + sin1[i] as f64 * cos2[i] as f64;
        jac_theta2[pos * n + i] = ((sum_s * dss_dth2 + sum_c * dsc_dth2) * inv_norm_n) as f32;
    }
}

// ═══════════════════════════════════════════════════════════
//  Backward (BPTT)
// ═══════════════════════════════════════════════════════════

/// Backward pass: BPTT through T3 dynamics.
///
/// `grad_pst`: [pst_dim] — d(L)/d(PST)
/// `traj`: trajectory from `t3spike_forward`
///
/// Returns `grad_drives` [n_steps × 3*n_osc] flattened.
pub fn t3spike_backward(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
) -> Vec<f32> {
    let n = traj.n_osc;
    let pst_dim = pst_linear_dim(n);

    // ── Step 1: d(L)/d(state_N) from PST Jacobian ──
    let (d_th1_j, d_th2_j, d_th3_j, d_r1_j, d_r2_j, d_r3_j) = pst_jacobian(
        &traj.th1_final,
        &traj.r1_final,
        &traj.th2_final,
        &traj.r2_final,
        &traj.th3_final,
        &traj.r3_final,
        n,
    );

    // v = J^T · grad_pst → (n,) vectors
    let mut v = compute_pst_jacobian_v(grad_pst, &d_th1_j, &d_th2_j, &d_th3_j,
                                        &d_r1_j, &d_r2_j, &d_r3_j, n, pst_dim);

    // ── Step 2: BPTT from N-1 to 0 ──
    bptt_loop(grad_pst, traj, &mut v.th1, &mut v.th2, &mut v.th3,
              &mut v.r1, &mut v.r2, &mut v.r3)
}

/// Result of PST Jacobian-vector product.
pub struct PstJacobianV {
    pub th1: Vec<f32>,
    pub th2: Vec<f32>,
    pub th3: Vec<f32>,
    pub r1: Vec<f32>,
    pub r2: Vec<f32>,
    pub r3: Vec<f32>,
}

/// Compute v = J^T · grad_pst from pre-computed Jacobian matrices.
#[inline]
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

/// BPTT loop: backprop through T3 dynamics given initial v = J^T · grad_pst.
///
/// Shared between CPU and GPU backward passes.
pub fn bptt_loop(
    grad_pst: &[f32],
    traj: &T3SpikeTrajectory,
    v_th1: &mut Vec<f32>,
    v_th2: &mut Vec<f32>,
    v_th3: &mut Vec<f32>,
    v_r1: &mut Vec<f32>,
    v_r2: &mut Vec<f32>,
    v_r3: &mut Vec<f32>,
) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let dt = traj.dt;
    let gain = traj.drive_gain;
    let alpha_val = traj.alpha;
    let eps = traj.epsilon;
    let pst_dim = pst_linear_dim(n);

    let mut grad_drives = vec![0.0; n_steps * 3 * n];

    for step in (0..n_steps).rev() {
        let off = step * n;               // ← offset for trajectory slices [n_steps × n]
        let grad_off = step * 3 * n;       // ← offset for grad_drives [n_steps × 3n]

        // State BEFORE step t = pre-state
        let th1_t = &traj.th1_pre[off..off + n];
        let th2_t = &traj.th2_pre[off..off + n];
        let th3_t = &traj.th3_pre[off..off + n];
        let r1_t  = &traj.r1_pre[off..off + n];
        let r2_t  = &traj.r2_pre[off..off + n];
        let r3_t  = &traj.r3_pre[off..off + n];

        // Drive at this step
        let d1_t = &traj.drive1[off..off + n];
        let d2_t = &traj.drive2[off..off + n];
        let d3_t = &traj.drive3[off..off + n];

            // ── Local gradient w.r.t. drive ──
            // dθ_next/d(drive) = dt · gain / (r + ε)
            // dr_next/d(drive) = dt · gain
            //
            // Gate: TriOscillator clamps r to [0, 10] after each step (non-differentiable).
            // When r is near the clamp boundary (r < 0.01 or r > 9.9), the true gradients
            // dr/dr, dθ/dr, and dr_dd through the clamp are 0. We apply a per-oscillator
            // threshold gate to zero out these spurious gradient paths.
            let mut r1_gate = vec![1.0; n];
            let mut r2_gate = vec![1.0; n];
            let mut r3_gate = vec![1.0; n];
            for i in 0..n {
                if r1_t[i] < 0.01 || r1_t[i] > 9.9 { r1_gate[i] = 0.0; }
                if r2_t[i] < 0.01 || r2_t[i] > 9.9 { r2_gate[i] = 0.0; }
                if r3_t[i] < 0.01 || r3_t[i] > 9.9 { r3_gate[i] = 0.0; }
            }

            let dr_dd = dt * gain;
            for i in 0..n {
                let dth_dd1 = dt * gain / (r1_t[i] + eps);
                let dth_dd2 = dt * gain / (r2_t[i] + eps);
                let dth_dd3 = dt * gain / (r3_t[i] + eps);

                // dr_dd gated: when clamp engages, dr_next/d(drive) = 0
                grad_drives[grad_off + i]         = v_th1[i] * dth_dd1 + v_r1[i] * dr_dd * r1_gate[i];
                grad_drives[grad_off + n + i]     = v_th2[i] * dth_dd2 + v_r2[i] * dr_dd * r2_gate[i];
                grad_drives[grad_off + 2 * n + i] = v_th3[i] * dth_dd3 + v_r3[i] * dr_dd * r3_gate[i];
            }

            // ── Backprop to previous state ──
            if step > 0 {


            // Sin/cos at pre-state (for coupling derivative)
            let mut sin1 = vec![0.0; n];
            let mut cos1 = vec![0.0; n];
            let mut sin2 = vec![0.0; n];
            let mut cos2 = vec![0.0; n];
            let mut sin3 = vec![0.0; n];
            let mut cos3 = vec![0.0; n];
            for i in 0..n {
                sin1[i] = th1_t[i].sin();
                cos1[i] = th1_t[i].cos();
                sin2[i] = th2_t[i].sin();
                cos2[i] = th2_t[i].cos();
                sin3[i] = th3_t[i].sin();
                cos3[i] = th3_t[i].cos();
            }

            // Phase Gate at pre-state
            let r1_op = order_param(&sin1, &cos1, n);
            let r2_op = order_param(&sin2, &cos2, n);
            let r3_op = order_param(&sin3, &cos3, n);
            let g1 = phase_gate_sigmoid(r1_op, DEFAULT_GAMMA[0]);
            let g2 = phase_gate_sigmoid(r2_op, DEFAULT_GAMMA[1]);
            let g3 = phase_gate_sigmoid(r3_op, DEFAULT_GAMMA[2]);

            // ── Coupling self-derivative: d(coupling[i])/dθ[i] ──
            // d(cpl1[i])/dθ1[i] = -Σⱼ w12[i,j]·r2[j]·cos(θ2[j]-θ1[i])·g2
            //                     -Σⱼ w13[i,j]·r3[j]·cos(θ3[j]-θ1[i])·g3
            // Uses uniform coupling: w[i,j] = w_total / n for all i,j
            let dcpl1 = coupling_self_deriv(n, traj.w12, r2_t, &sin2, &cos2, &cos1, &sin1,
                                            g2, traj.w13, r3_t, &sin3, &cos3, g3);
            let dcpl2 = coupling_self_deriv2(n, traj.w12, r1_t, &sin1, &cos2, &sin2, &cos1,
                                             g1, traj.w23, r3_t, &sin3, &cos3, g3);
            let dcpl3 = coupling_self_deriv3(n, traj.w13, r1_t, &sin1, &cos3, &sin3, &cos1,
                                             g1, traj.w23, r2_t, &sin2, &cos2, g2);

            // dθ_next/dθ = 1 + dt · d(coupling)/dθ  (diagonal approx)
            // dr_next/dr = 1 + dt · (α - 3·r²)
            // dθ_next/dr = -dt · drive · gain / (r + ε)²

            let mut v_th1_prev = vec![0.0; n];
            let mut v_th2_prev = vec![0.0; n];
            let mut v_th3_prev = vec![0.0; n];
            let mut v_r1_prev = vec![0.0; n];
            let mut v_r2_prev = vec![0.0; n];
            let mut v_r3_prev = vec![0.0; n];

            for i in 0..n {
                // dθ/dθ diagonal (from coupling + no explicit cross terms)
                let dth1_i = 1.0 + dt * dcpl1[i];
                let dth2_i = 1.0 + dt * dcpl2[i];
                let dth3_i = 1.0 + dt * dcpl3[i];

                // Gate: TriOscillator clamps r to [0.001, 10.0] after each step (non-differentiable).
                // Use precomputed r_gate from above (based on r_next_raw before clamp).
                let r1g = r1_gate[i];
                let r2g = r2_gate[i];
                let r3g = r3_gate[i];

                let dr1_i = (1.0 + dt * (alpha_val - 3.0 * r1_t[i] * r1_t[i])) * r1g;
                let dr2_i = (1.0 + dt * (alpha_val - 3.0 * r2_t[i] * r2_t[i])) * r2g;
                let dr3_i = (1.0 + dt * (alpha_val - 3.0 * r3_t[i] * r3_t[i])) * r3g;

                // dθ/dr = -dt · drive · gain / (r + ε)²
                // Gated: when clamp engages (r_next_raw outside [0,10]), the true dθ/dr is 0.
                let dth1_dr1 = -dt * d1_t[i] * gain / (r1_t[i] + eps).powi(2) * r1g;
                let dth2_dr2 = -dt * d2_t[i] * gain / (r2_t[i] + eps).powi(2) * r2g;
                let dth3_dr3 = -dt * d3_t[i] * gain / (r3_t[i] + eps).powi(2) * r3g;

                v_th1_prev[i] = v_th1[i] * dth1_i;
                v_th2_prev[i] = v_th2[i] * dth2_i;
                v_th3_prev[i] = v_th3[i] * dth3_i;

                v_r1_prev[i] = v_r1[i] * dr1_i + v_th1[i] * dth1_dr1;
                v_r2_prev[i] = v_r2[i] * dr2_i + v_th2[i] * dth2_dr2;
                v_r3_prev[i] = v_r3[i] * dr3_i + v_th3[i] * dth3_dr3;
            }

            // ── Velocity gradient contribution (step N-1 only) ──
            // velocity features at PST pos 6n..9n
            // d(velocity_i)/d(prev_θ_i) = -1
            // → v_th1_prev[i] += dL/d(vel1_i) · (-1) = -grad_pst[6n + i]
            if step == n_steps - 1 {
                let vel_start = 6 * n;
                if vel_start + 3 * n <= pst_dim {
                    for i in 0..n {
                        v_th1_prev[i] -= grad_pst[vel_start + i];
                        v_th2_prev[i] -= grad_pst[vel_start + n + i];
                        v_th3_prev[i] -= grad_pst[vel_start + 2 * n + i];
                    }
                }
            }

            *v_th1 = v_th1_prev;
            *v_th2 = v_th2_prev;
            *v_th3 = v_th3_prev;
            *v_r1 = v_r1_prev;
            *v_r2 = v_r2_prev;
            *v_r3 = v_r3_prev;
        }
    }

    grad_drives
}

// ═══════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════

/// Order parameter R = |mean(exp(iθ))|
fn order_param(sin_vec: &[f32], cos_vec: &[f32], n: usize) -> f32 {
    let sum_s: f64 = sin_vec.iter().map(|&s| s as f64).sum();
    let sum_c: f64 = cos_vec.iter().map(|&c| c as f64).sum();
    ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32
}

/// Phase Gate sigmoid: 1 / (1 + exp(-G·(r - γ))) with G = 4.0
fn phase_gate_sigmoid(r: f32, gamma: f32) -> f32 {
    1.0 / (1.0 + (-DEFAULT_PHASE_GATE_GAIN * (r - gamma)).exp())
}

/// d(cpl1)/dθ1: self-derivative of θ1's coupling w.r.t. θ1
/// Uniform coupling: w[i,j] = w_total / n for all i,j.
fn coupling_self_deriv(
    n: usize,
    w12: f32, r2: &[f32],
    sin2: &[f32], cos2: &[f32], cos1: &[f32], sin1: &[f32],
    g2: f32,
    w13: f32, r3: &[f32], sin3: &[f32], cos3: &[f32], g3: f32,
) -> Vec<f32> {
    let w12_n = (w12 / n as f32) as f64;
    let w13_n = (w13 / n as f32) as f64;
    let mut dcpl = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0f64;
        let c1 = cos1[i] as f64;
        let s1 = sin1[i] as f64;
        // w12 coupling
        for j in 0..n {
            // cos(θ2[j] - θ1[i]) = cos2[j]·cos1[i] + sin2[j]·sin1[i]
            let cos_diff = cos2[j] as f64 * c1 + sin2[j] as f64 * s1;
            sum += w12_n * r2[j] as f64 * cos_diff * g2 as f64;
        }
        // w13 coupling
        for j in 0..n {
            let cos_diff = cos3[j] as f64 * c1 + sin3[j] as f64 * s1;
            sum += w13_n * r3[j] as f64 * cos_diff * g3 as f64;
        }
        dcpl[i] = -sum as f32;
    }
    dcpl
}

/// d(cpl2)/dθ2 — uniform coupling.
fn coupling_self_deriv2(
    n: usize,
    w12: f32, r1: &[f32],
    sin1: &[f32], cos2: &[f32], sin2: &[f32], cos1: &[f32],
    g1: f32,
    w23: f32, r3: &[f32],
    sin3: &[f32], cos3: &[f32],
    g3: f32,
) -> Vec<f32> {
    let w12_n = (w12 / n as f32) as f64;
    let w23_n = (w23 / n as f32) as f64;
    let mut dcpl = vec![0.0; n];
    let s1_: Vec<f64> = sin1.iter().map(|&x| x as f64).collect();
    let c1_: Vec<f64> = cos1.iter().map(|&x| x as f64).collect();
    let r1_: Vec<f64> = r1.iter().map(|&x| x as f64).collect();
    for i in 0..n {
        let c2 = cos2[i] as f64;
        let s2 = sin2[i] as f64;
        // w12^T coupling (θ1 → θ2, uniform → w12_n for all j)
        let mut sum = 0.0f64;
        for j in 0..n {
            let cos_diff = c1_[j] * c2 + s1_[j] * s2;
            sum += w12_n * r1_[j] * cos_diff * g1 as f64;
        }
        // w23 coupling
        for j in 0..n {
            let cos_diff = cos3[j] as f64 * c2 + sin3[j] as f64 * s2;
            sum += w23_n * r3[j] as f64 * cos_diff * g3 as f64;
        }
        dcpl[i] = -sum as f32;
    }
    dcpl
}

/// d(cpl3)/dθ3 — uniform coupling.
fn coupling_self_deriv3(
    n: usize,
    w13: f32, r1: &[f32],
    sin1: &[f32], cos3: &[f32], sin3: &[f32], cos1: &[f32],
    g1: f32,
    w23: f32, r2: &[f32],
    sin2: &[f32], cos2: &[f32],
    g2: f32,
) -> Vec<f32> {
    let w13_n = (w13 / n as f32) as f64;
    let w23_n = (w23 / n as f32) as f64;
    let mut dcpl = vec![0.0; n];
    let s1_: Vec<f64> = sin1.iter().map(|&x| x as f64).collect();
    let c1_: Vec<f64> = cos1.iter().map(|&x| x as f64).collect();
    let r1_: Vec<f64> = r1.iter().map(|&x| x as f64).collect();
    for i in 0..n {
        let c3 = cos3[i] as f64;
        let s3 = sin3[i] as f64;
        // w13^T coupling
        let mut sum = 0.0f64;
        for j in 0..n {
            let cos_diff = c1_[j] * c3 + s1_[j] * s3;
            sum += w13_n * r1_[j] * cos_diff * g1 as f64;
        }
        // w23^T coupling
        for j in 0..n {
            let cos_diff = cos2[j] as f64 * c3 + sin2[j] as f64 * s3;
            sum += w23_n * r2[j] as f64 * cos_diff * g2 as f64;
        }
        dcpl[i] = -sum as f32;
    }
    dcpl
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t3spike_forward_backward_consistency() {
        let n_osc = 16;
        let n_steps = 10;
        let dt = 0.05;
        let drive_gain = 10.0;
        let alpha = 1.0;
        let freq_min = 0.5;
        let freq_max = 3.0;

        // Random drives
        let mut drives = vec![0.0; n_steps * 3 * n_osc];
        for i in 0..drives.len() {
            drives[i] = (i as f32 * 0.1).sin() * 2.0; // deterministic
        }

        // Forward
        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, 0.3, 0.15, 0.08, &drives,
        );

        // Check PST dimension
        let expected_dim = pst_linear_dim(n_osc);
        assert_eq!(pst.len(), expected_dim);

        // Backward with unit gradient
        let mut grad_pst = vec![0.0; expected_dim];
        grad_pst[0] = 1.0; // gradient w.r.t. first PST feature
        let grad_drives = t3spike_backward(&grad_pst, &traj);
        assert_eq!(grad_drives.len(), n_steps * 3 * n_osc);

        // Non-zero gradient
        let nonzero = grad_drives.iter().filter(|&&g| g.abs() > 1e-10).count();
        assert!(nonzero > 0, "backward should produce non-zero gradients");

        // All features
        for pst_idx in 0..expected_dim.min(10) {
            let mut gp = vec![0.0; expected_dim];
            gp[pst_idx] = 1.0;
            let gd = t3spike_backward(&gp, &traj);
            let nz = gd.iter().filter(|&&g| g.abs() > 1e-10).count();
            assert!(nz > 0, "pst feature {} produces zero gradient", pst_idx);
        }
    }

    #[test]
    fn test_pst_jacobian_feature_192_fd() {
        let n_osc = 16;
        let n_steps = 10;
        let dt = 0.05;
        let drive_gain = 10.0;
        let alpha = 1.0;
        let freq_min = 0.5;
        let freq_max = 3.0;

        let mut drives = vec![0.0; n_steps * 3 * n_osc];
        for i in 0..drives.len() {
            drives[i] = (i as f32 * 0.1).sin() * 2.0;
        }

        let (pst, traj) = t3spike_forward(
            n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, 0.3, 0.15, 0.08, &drives,
        );

        let pst_dim = pst_linear_dim(n_osc);
        let feat_idx = 192;

        // Analytical
        let mut grad_pst = vec![0.0; pst_dim];
        grad_pst[feat_idx] = 1.0;
        let ana_grad = t3spike_backward(&grad_pst, &traj);

        // FD for drive[0,0]
        let eps_fd = 1e-4f32;
        let mut dp = drives.clone(); dp[0] += eps_fd;
        let mut dm = drives.clone(); dm[0] -= eps_fd;
        let (pp, _) = t3spike_forward(n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, 0.3, 0.15, 0.08, &dp);
        let (pm, _) = t3spike_forward(n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, 0.3, 0.15, 0.08, &dm);
        let fd = (pp[feat_idx] - pm[feat_idx]) / (2.0 * eps_fd);

        println!("\n=== Feature {} (sin²1[0]) FD check ===", feat_idx);
        println!("PST[{}] = {:.6}", feat_idx, pst[feat_idx]);
        println!("FD:      {:.6}", fd);
        println!("Analytical: {:.6}", ana_grad[0]);
        println!("Ratio:   {:.2}x", ana_grad[0] / fd);
        println!("Ana grad_drives[9*3n] (last step): {:.6}", ana_grad[9 * 3 * n_osc]);
    }
}
