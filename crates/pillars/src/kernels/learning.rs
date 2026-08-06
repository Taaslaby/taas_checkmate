//! # Learning Microkernels — e-prop, STDP, BCM & SORN
//!
//! Batch kernels for SNN learning:
//! - [`surrogate_gradient`] — fast sigmoid surrogate (e-prop)
//! - [`eprop_update`] — standalone e-prop weight update `Δw = η·L_j·e_ji`
//! - [`stdp_update`] — batch STDP weight update on flat arrays (SIMD + tail)
//! - [`bcm_step`] — Bienenstock-Cooper-Munro (1982) rate plasticity
//!   with sliding threshold θ_M, plus [`homeostatic_scale`] (Turrigiano 1998)
//! - SORN (Self-Organizing Recurrent Network): intrinsic plasticity,
//!   synaptic normalization and the full [`SornNetwork`] loop.

use wide::f32x8;

use crate::core::util::{load8, store8};

// ═══════════════════════════════════════════════════════════
//  e-prop
// ═══════════════════════════════════════════════════════════

/// Surrogate gradient: derivative of the spike threshold function.
///
/// Fast sigmoid surrogate: `σ'(v) = 1 / (1 + |v − v_th|/τ)²`.
#[inline(always)]
pub fn surrogate_gradient(v: f32, v_thresh: f32, tau_surrogate: f32) -> f32 {
    let diff = (v - v_thresh).abs() / tau_surrogate;
    1.0 / ((1.0 + diff) * (1.0 + diff))
}

/// E-prop standalone weight update: `Δw = η · L_j · e_ji`.
///
/// All three arrays have length `n`; the update is elementwise.
pub fn eprop_update(weights: &mut [f32], eligibility: &[f32], learning_signals: &[f32], lr: f32) {
    let n = weights
        .len()
        .min(eligibility.len())
        .min(learning_signals.len());
    for i in 0..n {
        weights[i] += lr * learning_signals[i] * eligibility[i];
    }
}

// ═══════════════════════════════════════════════════════════
//  STDP batch update
// ═══════════════════════════════════════════════════════════

/// STDP batch update on flat weight arrays: `w = clamp(w + lr_ltp·tr_post − lr_ltd·tr_pre)`.
///
/// Used by SORN and simple rate-based plasticity where traces are already
/// computed (not per-lane spike events).
///
/// SIMD lanes and scalar tails are bit-identical (pure FMA/clamp arithmetic).
pub fn stdp_update(
    weights: &mut [f32],
    traces_pre: &[f32],
    traces_post: &[f32],
    lr_ltd: f32,
    lr_ltp: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = weights.len().min(traces_pre.len()).min(traces_post.len());
    let n8 = n & !7;
    let ltp_s = f32x8::splat(lr_ltp);
    let ltd_s = f32x8::splat(lr_ltd);
    let wmin_s = f32x8::splat(w_min);
    let wmax_s = f32x8::splat(w_max);
    for off in (0..n8).step_by(8) {
        unsafe {
            let mut w = load8(weights.as_ptr(), off);
            let tp = load8(traces_pre.as_ptr(), off);
            let tpo = load8(traces_post.as_ptr(), off);
            w = w + ltp_s * tpo - ltd_s * tp;
            w = w.max(wmin_s).min(wmax_s);
            store8(weights.as_mut_ptr(), off, w);
        }
    }
    for i in n8..n {
        weights[i] =
            (weights[i] + lr_ltp * traces_post[i] - lr_ltd * traces_pre[i]).clamp(w_min, w_max);
    }
}

// ═══════════════════════════════════════════════════════════
//  BCM — Bienenstock, Cooper & Munro (1982)
//  Rate-based plasticity with sliding threshold θ_M
// ═══════════════════════════════════════════════════════════

/// BCM plasticity step for a synapse population.
///
/// Rate-based Hebbian learning with a sliding modification threshold:
///
/// ```text
/// x'  = x + (pre − x)·inv_tau_x·dt      (pre-synaptic trace)
/// y'  = y + (post − y)·inv_tau_y·dt     (post-synaptic trace)
/// θ'  = θ + (y² − θ)·inv_tau_θ·dt       (sliding threshold, clamped ≥ 0)
/// Δw  = η·y·(y−θ)·x·dt − decay·w·dt     (LTP when y > θ, LTD when y < θ)
/// ```
///
/// All six slices share length `n = min(...)`; τ values are floored at 1 ms
/// (matches the legacy `crates/kernels` behaviour). SIMD lanes and the scalar
/// tail use identical multiply/divide structure — results agree per element.
///
/// Reference: Bienenstock, Cooper & Munro (1982) J. Neurosci.,
/// "Theory for the development of neuron selectivity".
#[allow(clippy::too_many_arguments)]
pub fn bcm_step(
    x: &mut [f32],       // pre-synaptic trace
    y: &mut [f32],       // post-synaptic trace
    theta_m: &mut [f32], // sliding modification threshold
    weights: &mut [f32],
    pre_spikes: &[f32],  // pre firing rate / spike indicator
    post_spikes: &[f32], // post firing rate / spike indicator
    dt: f32,
    tau_x: f32,
    tau_y: f32,
    tau_theta: f32,
    eta: f32,
    decay: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = weights
        .len()
        .min(x.len())
        .min(y.len())
        .min(theta_m.len())
        .min(pre_spikes.len())
        .min(post_spikes.len());
    let n8 = n & !7;

    let dt_s = f32x8::splat(dt);
    let eta_s = f32x8::splat(eta);
    let decay_s = f32x8::splat(decay);
    let wmin_s = f32x8::splat(w_min);
    let wmax_s = f32x8::splat(w_max);
    let zero = f32x8::splat(0.0);
    let inv_tau_x_s = f32x8::splat(1.0 / tau_x.max(1.0));
    let inv_tau_y_s = f32x8::splat(1.0 / tau_y.max(1.0));
    let inv_tau_th_s = f32x8::splat(1.0 / tau_theta.max(1.0));

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut xx = load8(x.as_ptr(), off);
            let mut yy = load8(y.as_ptr(), off);
            let mut th = load8(theta_m.as_ptr(), off);
            let mut ww = load8(weights.as_ptr(), off);
            let pre = load8(pre_spikes.as_ptr(), off);
            let post = load8(post_spikes.as_ptr(), off);

            // 1. Exponential-moving-average traces
            xx += (pre - xx) * inv_tau_x_s * dt_s;
            yy += (post - yy) * inv_tau_y_s * dt_s;

            // 2. Sliding threshold θ_M = low-pass of y², clamped ≥ 0
            th += (yy * yy - th) * inv_tau_th_s * dt_s;
            th = th.max(zero);

            // 3. BCM weight update: Δw = η·y·(y−θ)·x·dt − decay·w·dt
            let bcm_drive = yy * (yy - th) * xx;
            ww += eta_s * bcm_drive * dt_s - decay_s * ww * dt_s;
            ww = ww.max(wmin_s).min(wmax_s);

            store8(x.as_mut_ptr(), off, xx);
            store8(y.as_mut_ptr(), off, yy);
            store8(theta_m.as_mut_ptr(), off, th);
            store8(weights.as_mut_ptr(), off, ww);
        }
    }

    // Scalar tail — same operations, same order
    let inv_tau_x = 1.0 / tau_x.max(1.0);
    let inv_tau_y = 1.0 / tau_y.max(1.0);
    let inv_tau_th = 1.0 / tau_theta.max(1.0);
    for i in n8..n {
        x[i] += (pre_spikes[i] - x[i]) * inv_tau_x * dt;
        y[i] += (post_spikes[i] - y[i]) * inv_tau_y * dt;
        theta_m[i] += (y[i] * y[i] - theta_m[i]) * inv_tau_th * dt;
        theta_m[i] = theta_m[i].max(0.0);
        let bcm_drive = y[i] * (y[i] - theta_m[i]) * x[i];
        weights[i] += eta * bcm_drive * dt - decay * weights[i] * dt;
        weights[i] = weights[i].clamp(w_min, w_max);
    }
}

/// Homeostatic synaptic scaling (Turrigiano 1998).
///
/// Multiplicative scaling toward a target firing rate:
///
/// ```text
/// Δw = η_h·(r_target − y)·w·dt
/// ```
///
/// `y` is the post-synaptic activity trace (same value for all synapses of
/// one neuron); weights are clamped ≥ 0. SIMD + scalar tail, bit-consistent.
pub fn homeostatic_scale(weights: &mut [f32], y: &[f32], dt: f32, eta_h: f32, target_rate: f32) {
    let n = weights.len().min(y.len());
    let n8 = n & !7;
    let dt_s = f32x8::splat(dt);
    let eta_s = f32x8::splat(eta_h);
    let target_s = f32x8::splat(target_rate);
    let zero = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        unsafe {
            let ww = load8(weights.as_ptr(), off);
            let yy = load8(y.as_ptr(), off);
            let scale = eta_s * (target_s - yy) * dt_s;
            let ww = (ww + scale * ww).max(zero);
            store8(weights.as_mut_ptr(), off, ww);
        }
    }
    for i in n8..n {
        let scale = eta_h * (target_rate - y[i]) * dt;
        weights[i] = (weights[i] + scale * weights[i]).max(0.0);
    }
}

// ═══════════════════════════════════════════════════════════
//  SORN — Self-Organizing Recurrent Network
// ═══════════════════════════════════════════════════════════

/// Intrinsic plasticity: adapts neuron thresholds to maintain a target
/// firing rate. `θ += lr_ip·(rate − target)`, decayed and clamped to [0, 50].
pub fn intrinsic_plasticity(
    theta: &mut [f32],
    spikes: &[i32],
    lr_ip: f32,
    target_rate: f32,
    decay: f32,
) {
    for (t, &s) in theta.iter_mut().zip(spikes.iter()) {
        let rate = if s != 0 { 1.0 } else { 0.0 };
        *t += lr_ip * (rate - target_rate);
        *t *= decay;
        *t = t.clamp(0.0, 50.0);
    }
}

/// Synaptic normalization: rescales each post-neuron's incoming weights to
/// `target_sum`. `weights` is row-major `[n_pre × n_post]`.
pub fn synaptic_normalization(
    weights: &mut [f32],
    n_pre: usize,
    n_post: usize,
    target_sum: f32,
    eps: f32,
) {
    assert!(weights.len() >= n_pre * n_post);
    // Column sums (per post-neuron)
    let mut col_scales = vec![1.0f32; n_post];
    for (j, scale) in col_scales.iter_mut().enumerate() {
        let mut sum: f32 = 0.0;
        for i in 0..n_pre {
            sum += weights[i * n_post + j];
        }
        if sum > eps {
            *scale = target_sum / sum;
        }
    }
    for row in weights.chunks_exact_mut(n_post) {
        for j in 0..n_post {
            row[j] *= col_scales[j];
        }
    }
}

/// SORN network — Self-Organizing Recurrent Neural Network.
///
/// Each learning step applies:
/// 1. **STDP** — spike-timing dependent plasticity
/// 2. **Intrinsic plasticity** — adaptive thresholds
/// 3. **Synaptic normalization** — weight renormalization
///
/// Reference: Lazar, Pipa & Triesch (2009) "SORN: a self-organizing
/// recurrent neural network".
pub struct SornNetwork {
    pub weights: Vec<f32>,
    pub theta: Vec<f32>,
    pub rates: Vec<f32>,
    pub n: usize,
    pub lr_stdp_ltd: f32,
    pub lr_stdp_ltp: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub w_target_sum: f32,
    pub lr_ip: f32,
    pub target_rate: f32,
    pub ip_decay: f32,
    pub sn_eps: f32,
}

impl SornNetwork {
    pub fn new(n: usize, w_init: f32, w_target_sum: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            weights: vec![w_init; n_pad * n_pad],
            theta: vec![0.0; n_pad],
            rates: vec![0.0; n_pad],
            n: n_pad,
            lr_stdp_ltd: 0.01,
            lr_stdp_ltp: 0.005,
            w_min: 0.0,
            w_max: 2.0,
            w_target_sum,
            lr_ip: 0.001,
            target_rate: 0.01,
            ip_decay: 0.999,
            sn_eps: 1e-8,
        }
    }

    /// One SORN learning step: STDP → intrinsic plasticity → normalization.
    pub fn step(&mut self, spikes: &[i32], traces_pre: &[f32], traces_post: &[f32]) {
        stdp_update(
            &mut self.weights,
            traces_pre,
            traces_post,
            self.lr_stdp_ltd,
            self.lr_stdp_ltp,
            self.w_min,
            self.w_max,
        );
        intrinsic_plasticity(
            &mut self.theta,
            spikes,
            self.lr_ip,
            self.target_rate,
            self.ip_decay,
        );
        synaptic_normalization(
            &mut self.weights,
            self.n,
            self.n,
            self.w_target_sum,
            self.sn_eps,
        );
    }

    pub fn reset(&mut self) {
        self.weights.fill(0.1);
        self.theta.fill(0.0);
        self.rates.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surrogate_gradient_peaks_at_threshold() {
        let g0 = surrogate_gradient(-54.0, -54.0, 1.0);
        assert!((g0 - 1.0).abs() < 1e-6, "at threshold → 1.0, got {g0}");
        let g_far = surrogate_gradient(-70.0, -54.0, 1.0);
        assert!(g_far < g0, "far from threshold → smaller gradient");
    }

    #[test]
    fn eprop_update_basic() {
        let n = 16;
        let mut weights = vec![0.5f32; n];
        let eligibility = vec![1.0f32; n];
        let learning_signals = vec![0.1f32; n];
        eprop_update(&mut weights, &eligibility, &learning_signals, 0.1);
        for i in 0..n {
            assert!(
                (weights[i] - 0.51).abs() < 1e-5,
                "w should be 0.51, got {}",
                weights[i]
            );
        }
    }

    #[test]
    fn intrinsic_plasticity_pushes_toward_target() {
        let n = 8;
        let mut theta = vec![10.0f32; n];
        let spikes = vec![0i32; n]; // below target_rate=0.01 → theta decreases
        intrinsic_plasticity(&mut theta, &spikes, 0.1, 0.01, 0.99);
        assert!(theta[0] < 10.0, "below-target firing should lower theta");
    }

    #[test]
    fn synaptic_normalization_preserves_ratios() {
        let n_pre = 2;
        let n_post = 2;
        let mut w = vec![1.0f32, 2.0, 3.0, 4.0];
        // Column sums: col0=4, col1=6 → scales: 0.25, 1/6
        synaptic_normalization(&mut w, n_pre, n_post, 1.0, 1e-8);
        let col0: f32 = w[0] + w[2];
        let col1: f32 = w[1] + w[3];
        assert!(
            (col0 - 1.0).abs() < 1e-5,
            "col0 sum should be 1, got {col0}"
        );
        assert!(
            (col1 - 1.0).abs() < 1e-5,
            "col1 sum should be 1, got {col1}"
        );
    }

    #[test]
    fn stdp_update_basic() {
        let n = 16;
        let mut weights = vec![0.5f32; n];
        let traces_pre = vec![0.0f32; n];
        let traces_post = vec![1.0f32; n];
        stdp_update(&mut weights, &traces_pre, &traces_post, 0.1, 0.1, 0.0, 1.0);
        for i in 0..n {
            assert!((weights[i] - 0.6).abs() < 0.01, "stdp[{i}]: {}", weights[i]);
        }
    }

    #[test]
    fn stdp_update_simd_tail_consistent() {
        // n = 10 → 8 SIMD lanes + 2 scalar tail, both must produce identical bits.
        let n = 10;
        let mut w_simd = vec![0.3f32; n];
        let traces_pre: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let traces_post: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.02).collect();
        stdp_update(&mut w_simd, &traces_pre, &traces_post, 0.05, 0.1, 0.0, 1.0);
        // Reference computed scalar (identical arithmetic order for f32).
        for i in 0..n {
            let expected = (0.3 + 0.1 * traces_post[i] - 0.05 * traces_pre[i]).clamp(0.0, 1.0);
            assert!(
                (w_simd[i] - expected).abs() < 1e-6,
                "stdp lane/tail mismatch at {i}: got {}, expected {}",
                w_simd[i],
                expected
            );
        }
    }

    #[test]
    fn sorn_step_runs_without_panic() {
        let mut sorn = SornNetwork::new(8, 0.1, 1.0);
        let spikes = vec![0i32; 8];
        let traces_pre = vec![0.0f32; 64];
        let traces_post = vec![0.0f32; 64];
        sorn.step(&spikes, &traces_pre, &traces_post);
        assert!(
            sorn.weights.iter().all(|&w| w.is_finite()),
            "weights must stay finite"
        );
    }

    // ─── BCM ────────────────────────────────────────────────────

    fn run_bcm(n: usize, post_val: f32, theta_init: f32, steps: usize) -> (Vec<f32>, Vec<f32>) {
        let dt = 1.0;
        let tau_x = 20.0;
        let tau_y = 20.0;
        let tau_theta = 100.0;
        let eta = 0.01;
        let decay = 0.001;
        let w_min = 0.0;
        let w_max = 10.0;

        let mut x = vec![0.5f32; n];
        let mut y = vec![post_val; n];
        let mut theta = vec![theta_init; n];
        let mut weights = vec![1.0f32; n];
        let pre = vec![0.5f32; n];
        let post = vec![post_val; n];

        for _ in 0..steps {
            bcm_step(
                &mut x,
                &mut y,
                &mut theta,
                &mut weights,
                &pre,
                &post,
                dt,
                tau_x,
                tau_y,
                tau_theta,
                eta,
                decay,
                w_min,
                w_max,
            );
        }
        (weights, theta)
    }

    #[test]
    fn bcm_bidirectional_plasticity() {
        // LTP: post (10) > θ (5) → weight increases
        let (w_ltp, _) = run_bcm(8, 10.0, 5.0, 1);
        assert!(
            w_ltp[0] > 1.0,
            "BCM LTP: weight should increase, got {:.6}",
            w_ltp[0]
        );

        // LTD: post (1) < θ (5) → weight decreases
        let (w_ltd, _) = run_bcm(8, 1.0, 5.0, 1);
        assert!(
            w_ltd[0] < 1.0,
            "BCM LTD: weight should decrease, got {:.6}",
            w_ltd[0]
        );

        assert!(
            w_ltp[0] > w_ltd[0],
            "LTP ({:.6}) > LTD ({:.6})",
            w_ltp[0],
            w_ltd[0]
        );
    }

    #[test]
    fn bcm_threshold_follows_activity() {
        // θ_M tracks y² ≈ 21 (post=5, τ_θ=100, 50 steps): θ should rise but < 30
        let (_, theta) = run_bcm(8, 5.0, 0.0, 50);
        assert!(theta[0] > 2.0, "BCM: θ_M should rise, got {:.4}", theta[0]);
        assert!(
            theta[0] < 30.0,
            "BCM: θ_M should be <30, got {:.4}",
            theta[0]
        );
    }

    #[test]
    fn bcm_non_aligned_n_simd_tail_consistent() {
        // n=13 → 8 SIMD lanes + 5 scalar tail; identical params ⇒ same result
        let (weights, theta) = run_bcm(13, 8.0, 5.0, 100);
        assert!(
            (weights[0] - weights[12]).abs() < 1e-5,
            "n=13 lane/tail mismatch: nrn0={:.6}, nrn12={:.6}",
            weights[0],
            weights[12]
        );
        assert!(
            (theta[0] - theta[12]).abs() < 1e-5,
            "n=13 θ mismatch: nrn0={:.6}, nrn12={:.6}",
            theta[0],
            theta[12]
        );
        assert!(
            weights.iter().all(|&w| w.is_finite()),
            "weights must be finite"
        );
    }

    #[test]
    fn homeostatic_scaling_direction() {
        let dt = 1.0;
        let eta_h = 0.01;
        let target = 5.0;

        // High activity (10 > 5) → weights shrink
        let mut w_hi = vec![1.0f32; 8];
        let y_hi = vec![10.0f32; 8];
        homeostatic_scale(&mut w_hi, &y_hi, dt, eta_h, target);
        assert!(
            w_hi[0] < 1.0,
            "high activity should shrink weights, got {}",
            w_hi[0]
        );

        // Low activity (2 < 5) → weights grow
        let mut w_lo = vec![1.0f32; 8];
        let y_lo = vec![2.0f32; 8];
        homeostatic_scale(&mut w_lo, &y_lo, dt, eta_h, target);
        assert!(
            w_lo[0] > 1.0,
            "low activity should grow weights, got {}",
            w_lo[0]
        );

        // At target → no change
        let mut w_at = vec![1.0f32; 8];
        let y_at = vec![5.0f32; 8];
        homeostatic_scale(&mut w_at, &y_at, dt, eta_h, target);
        assert!(
            (w_at[0] - 1.0).abs() < 1e-6,
            "at target: {}, want 1.0",
            w_at[0]
        );
    }
}
