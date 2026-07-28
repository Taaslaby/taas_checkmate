#![allow(non_snake_case)]
use super::{load8, load8_mut, store8};
use wide::f32x8;

// ═══════════════════════════════════════════════════════
//  BCM Rule — Bienenstock, Cooper & Munro (1982)
//  Rate-based plasticity with sliding threshold θ_M
// ═══════════════════════════════════════════════════════
//
//  Standard BCM formulation:
//    dw/dt = η · y · (y - θ_M) · x − η_decay · w
//    θ_M = τ_θ · E[y²]       (low-pass filtered y²)
//    τ_θ · dθ_M/dt = -θ_M + y²
//
//  Our implementation uses exponential moving averages:
//    x[t] = (1-1/τ_x)·x[t-1] + (1/τ_x)·pre_spike[t]
//    y[t] = (1-1/τ_y)·y[t-1] + (1/τ_y)·post_spike[t]
//    θ_M[t] = (1-1/τ_θ)·θ_M[t-1] + (1/τ_θ)·y[t]²
//    Δw = η · y · (y - θ_M) · x · dt  −  decay · w · dt
//
//  BCM slope: 
//    y < θ_M → LTD (weight decreases)
//    y > θ_M → LTP (weight increases)
//
//  Reference: Bienenstock, Cooper, Munro (1982) J. Neurosci.
//            "Theory for the development of neuron selectivity"

/// Step BCM plasticity for `n` synapses.
///
/// Updates: pre-trace, post-trace, threshold θ_M, and weights.
///
/// # Parameters
/// - `x`: presynaptic trace [n]
/// - `y`: postsynaptic trace [n]
/// - `theta_m`: modification threshold [n]
/// - `weights`: synaptic weights [n]
/// - `pre_spikes`: presynaptic spike indicators [n] (0/1 or rate)
/// - `post_spikes`: postsynaptic spike indicators [n] (0/1 or rate)
/// - `dt`: time step (ms)
/// - `tau_x`: pre-trace time constant (ms)
/// - `tau_y`: post-trace time constant (ms)
/// - `tau_theta`: threshold time constant (ms)
/// - `eta`: BCM learning rate
/// - `decay`: weight decay
/// - `w_min`, `w_max`: weight bounds
///
/// # Safety
/// All pointers valid for `n` elements.
pub unsafe fn bcm_step(
    x: *mut f32,          // pre-synaptic trace
    y: *mut f32,          // post-synaptic trace
    theta_m: *mut f32,    // sliding threshold
    weights: *mut f32,
    pre_spikes: *const f32,   // pre firing rate / spike indicator
    post_spikes: *const f32,  // post firing rate / spike indicator
    dt: f32,
    tau_x: f32,
    tau_y: f32,
    tau_theta: f32,
    eta: f32,
    decay: f32,
    w_min: f32,
    w_max: f32,
    n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let eta_s = f32x8::splat(eta);
    let decay_s = f32x8::splat(decay);
    let wmin = f32x8::splat(w_min);
    let wmax = f32x8::splat(w_max);
    let zero = f32x8::splat(0.0);

    // Exponential decay factors for traces
    let tau_x_s = f32x8::splat(tau_x.max(1.0));
    let tau_y_s = f32x8::splat(tau_y.max(1.0));
    let tau_th_s = f32x8::splat(tau_theta.max(1.0));
    let n8 = n & !7;
    for off in (0..n8).step_by(8) {
        let mut xx = load8_mut(x, off);
        let mut yy = load8_mut(y, off);
        let mut th = load8_mut(theta_m, off);
        let mut ww = load8_mut(weights, off);
        let pre = load8(pre_spikes, off);
        let post = load8(post_spikes, off);

        // 1. Update traces (exponential moving average)
        // x += (pre - x) / τ_x · dt
        // y += (post - y) / τ_y · dt
        let dx = (pre - xx) / tau_x_s * dt_s;
        let dy = (post - yy) / tau_y_s * dt_s;
        xx += dx;
        yy += dy;

        // 2. Update sliding threshold θ_M
        // θ_M += (y² - θ_M) / τ_θ · dt
        let dth = (yy * yy - th) / tau_th_s * dt_s;
        th += dth;
        th = th.max(zero); // θ_M ≥ 0

        // 3. BCM weight update
        // Δw = η · y · (y - θ_M) · x · dt  −  decay · w · dt
        let bcm_drive = yy * (yy - th) * xx; // y·(y-θ_M)·x
        let dw = eta_s * bcm_drive * dt_s - decay_s * ww * dt_s;
        ww += dw;
        ww = ww.min(wmax).max(wmin);

        store8(x, off, xx);
        store8(y, off, yy);
        store8(theta_m, off, th);
        store8(weights, off, ww);
    }

    // Scalar tail
    if n8 < n {
        let x_sl = std::slice::from_raw_parts_mut(x, n);
        let y_sl = std::slice::from_raw_parts_mut(y, n);
        let th_sl = std::slice::from_raw_parts_mut(theta_m, n);
        let w_sl = std::slice::from_raw_parts_mut(weights, n);
        let pre_sl = std::slice::from_raw_parts(pre_spikes, n);
        let post_sl = std::slice::from_raw_parts(post_spikes, n);
        for i in n8..n {
            // Traces
            x_sl[i] += (pre_sl[i] - x_sl[i]) / tau_x.max(1.0) * dt;
            y_sl[i] += (post_sl[i] - y_sl[i]) / tau_y.max(1.0) * dt;
            // Threshold
            th_sl[i] += (y_sl[i] * y_sl[i] - th_sl[i]) / tau_theta.max(1.0) * dt;
            th_sl[i] = th_sl[i].max(0.0);
            // BCM weight update
            let bcm_drive = y_sl[i] * (y_sl[i] - th_sl[i]) * x_sl[i];
            w_sl[i] += eta * bcm_drive * dt - decay * w_sl[i] * dt;
            w_sl[i] = w_sl[i].clamp(w_min, w_max);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Homeostatic Synaptic Scaling
//  Turrigiano (1998) — Nature
//  Global normalization of synaptic weights
// ═══════════════════════════════════════════════════════
//
//  Multiplicative scaling:
//    w ← w · (r_target / r_mean)
//
//  Or in incremental form (preferred for SIMD):
//    Δw = η_h · (r_target − r_mean) · w · dt
//
//  Where:
//    r_mean = mean postsynaptic firing rate across the neuron
//    r_target = target firing rate
//
//  This is cell-wide: all synapses on a neuron are scaled together.

/// Step homeostatic synaptic scaling for `n` synapses belonging to one neuron.
///
/// Multiplicative scaling: Δw = η · (r_target − r_mean) · w · dt
///
/// r_mean is computed from the `y` trace (running average of post-spike activity).
///
/// # Parameters
/// - `weights`: synaptic weights [n]
/// - `y`: postsynaptic trace [n] (same value for all synapses of one neuron)
/// - `n`: number of synapses
/// - `dt`: time step (ms)
/// - `eta_h`: homeostatic learning rate
/// - `target_rate`: target firing rate (Hz)
///
/// # Safety
/// All pointers valid for `n` elements.
pub(crate) unsafe fn homeostatic_scale(
    weights: *mut f32,
    y: *const f32,     // postsynaptic trace (per weight, typically same value)
    n: usize,
    dt: f32,
    eta_h: f32,
    target_rate: f32,
) {
    let dt_s = f32x8::splat(dt);
    let eta_s = f32x8::splat(eta_h);
    let target_s = f32x8::splat(target_rate);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut ww = load8_mut(weights, off);
        let yy = load8(y, off);    // post-synaptic activity (firing rate proxy)

        // Δw = η · (r_target − y) · w · dt
        let scale = eta_s * (target_s - yy) * dt_s;
        ww = ww + scale * ww;
        // Clamp to prevent negative weights
        ww = ww.max(zero);
        store8(weights, off, ww);
    }

    if n8 < n {
        let w_sl = std::slice::from_raw_parts_mut(weights, n);
        let y_sl = std::slice::from_raw_parts(y, n);
        for i in n8..n {
            let scale = eta_h * (target_rate - y_sl[i]) * dt;
            w_sl[i] = w_sl[i] + scale * w_sl[i];
            w_sl[i] = w_sl[i].max(0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// BCM — Weight increases when post > θ_M (LTP), decreases when post < θ_M (LTD).
    /// Strong post (y=10) with threshold θ_M=5 → y > θ_M → LTP
    /// Weak post (y=1) with threshold θ_M=5 → y < θ_M → LTD
    #[test]
    fn bcm_bidirectional_plasticity() {
        let n = 8;
        let dt = 1.0;
        let tau_x = 20.0;
        let tau_y = 20.0;
        let tau_theta = 100.0;
        let eta = 0.01;
        let decay = 0.001;
        let w_min = 0.0;
        let w_max = 10.0;

        // Test LTP: strong post activity
        let mut x = vec![0.5f32; n];
        let mut y = vec![10.0f32; n];  // strong post
        let mut theta = vec![5.0f32; n]; // threshold below y → LTP
        let mut weights = vec![1.0f32; n];
        let pre = vec![0.5f32; n];
        let post = vec![10.0f32; n];

        unsafe {
            bcm_step(
                x.as_mut_ptr(), y.as_mut_ptr(), theta.as_mut_ptr(),
                weights.as_mut_ptr(), pre.as_ptr(), post.as_ptr(),
                dt, tau_x, tau_y, tau_theta, eta, decay, w_min, w_max, n,
            );
        }

        // LTP: y > θ_M → weight increases
        assert!(weights[0] > 1.0, "BCM LTP: weight should increase, got {:.6}", weights[0]);

        // Test LTD: weak post activity
        let mut x_ltd = vec![0.5f32; n];
        let mut y_ltd = vec![1.0f32; n];  // weak post
        let mut theta_ltd = vec![5.0f32; n]; // threshold above y → LTD
        let mut weights_ltd = vec![1.0f32; n];
        let pre_ltd = vec![0.5f32; n];
        let post_ltd = vec![1.0f32; n];

        unsafe {
            bcm_step(
                x_ltd.as_mut_ptr(), y_ltd.as_mut_ptr(), theta_ltd.as_mut_ptr(),
                weights_ltd.as_mut_ptr(), pre_ltd.as_ptr(), post_ltd.as_ptr(),
                dt, tau_x, tau_y, tau_theta, eta, decay, w_min, w_max, n,
            );
        }

        // LTD: y < θ_M → weight decreases
        assert!(weights_ltd[0] < 1.0, "BCM LTD: weight should decrease, got {:.6}", weights_ltd[0]);

        // LTP weight should be > LTD weight
        assert!(weights[0] > weights_ltd[0],
            "BCM: LTP ({:.6}) > LTD ({:.6})", weights[0], weights_ltd[0]);
    }

    /// BCM — Sliding threshold θ_M follows y²
    #[test]
    fn bcm_threshold_follows_activity() {
        let n = 8;
        let dt = 1.0;
        let tau_x = 20.0;
        let tau_y = 20.0;
        let tau_theta = 5.0; // fast threshold dynamics
        let eta = 0.01;
        let decay = 0.001;
        let w_min = 0.0;
        let w_max = 10.0;

        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut theta = vec![0.0f32; n];  // start at 0
        let mut weights = vec![1.0f32; n];
        let pre = vec![0.0f32; n];  // no pre activity to isolate threshold
        let post = vec![5.0f32; n]; // moderate post activity

        // Run multiple steps to let threshold converge
        for _ in 0..50 {
            unsafe {
                bcm_step(
                    x.as_mut_ptr(), y.as_mut_ptr(), theta.as_mut_ptr(),
                    weights.as_mut_ptr(), pre.as_ptr(), post.as_ptr(),
                    dt, tau_x, tau_y, tau_theta, eta, decay, w_min, w_max, n,
                );
            }
        }

        // θ_M should have risen significantly toward y²
        // y trace converges to 5 with τ_y=20ms: after 50 steps at dt=1: y ≈ 5·(1-exp(-50/20)) ≈ 4.59
        // θ_M should be tracking y² ≈ 21
        assert!(theta[0] > 12.0, "BCM: θ_M should be >12, got {:.4}", theta[0]);
        assert!(theta[0] < 30.0, "BCM: θ_M should be <30, got {:.4}", theta[0]);
    }

    /// BCM — Non-aligned n (SIMD + scalar tail consistency)
    #[test]
    fn bcm_non_aligned_n() {
        let n = 13;
        let dt = 1.0;
        let tau_x = 20.0;
        let tau_y = 20.0;
        let tau_theta = 50.0;
        let eta = 0.01;
        let decay = 0.001;
        let w_min = 0.0;
        let w_max = 10.0;

        let mut x = vec![0.5f32; n];
        let mut y = vec![8.0f32; n];
        let mut theta = vec![5.0f32; n];
        let mut weights = vec![1.0f32; n];
        let pre = vec![0.5f32; n];
        let post = vec![8.0f32; n];

        for _ in 0..100 {
            unsafe {
                bcm_step(
                    x.as_mut_ptr(), y.as_mut_ptr(), theta.as_mut_ptr(),
                    weights.as_mut_ptr(), pre.as_ptr(), post.as_ptr(),
                    dt, tau_x, tau_y, tau_theta, eta, decay, w_min, w_max, n,
                );
            }
        }

        // All neurons with same params should have similar weights
        let diff_w = (weights[0] - weights[12]).abs();
        let diff_th = (theta[0] - theta[12]).abs();
        assert!(diff_w < 1.0, "BCM n=13: weight mismatch: nrn0={:.4}, nrn12={:.4}", weights[0], weights[12]);
        assert!(diff_th < 1.0, "BCM n=13: theta mismatch: nrn0={:.4}, nrn12={:.4}", theta[0], theta[12]);
        // No NaN
        for i in 0..n {
            assert!(!weights[i].is_nan(), "BCM n=13: weights[{i}] is NaN");
            assert!(!theta[i].is_nan(), "BCM n=13: theta[{i}] is NaN");
        }
    }

    /// Homeostatic scaling: high activity reduces weights.
    #[test]
    fn homeostatic_scaling_high_activity() {
        let n = 8;
        let dt = 1.0;
        let eta_h = 0.01;
        let target_rate = 5.0; // Hz

        let mut weights = vec![1.0f32; n];
        let y = vec![10.0f32; n]; // high post activity

        unsafe {
            homeostatic_scale(
                weights.as_mut_ptr(), y.as_ptr(), n, dt, eta_h, target_rate,
            );
        }

        // y=10 > target=5 → Δw = η·(rtar - y)·w·dt = 0.01·(5-10)·1·1 = -0.05
        // w = 1.0 - 0.05 = 0.95
        for i in 0..n {
            assert!(weights[i] < 1.0, "Homeostatic: high activity should reduce weights, got {:.6}", weights[i]);
        }
    }

    /// Homeostatic scaling: low activity increases weights.
    #[test]
    fn homeostatic_scaling_low_activity() {
        let n = 8;
        let dt = 1.0;
        let eta_h = 0.01;
        let target_rate = 5.0;

        let mut weights = vec![1.0f32; n];
        let y = vec![2.0f32; n]; // low post activity

        unsafe {
            homeostatic_scale(
                weights.as_mut_ptr(), y.as_ptr(), n, dt, eta_h, target_rate,
            );
        }

        // y=2 < target=5 → Δw = η·(rtar - y)·w·dt = 0.01·(5-2)·1·1 = 0.03
        // w = 1.0 + 0.03 = 1.03
        for i in 0..n {
            assert!(weights[i] > 1.0, "Homeostatic: low activity should increase weights, got {:.6}", weights[i]);
        }
    }

    /// Homeostatic scaling: at target rate, no change.
    #[test]
    fn homeostatic_scaling_at_target() {
        let n = 8;
        let dt = 1.0;
        let eta_h = 0.01;
        let target_rate = 5.0;

        let mut weights = vec![1.0f32; n];
        let y = vec![5.0f32; n]; // exactly at target

        unsafe {
            homeostatic_scale(
                weights.as_mut_ptr(), y.as_ptr(), n, dt, eta_h, target_rate,
            );
        }

        // y = target → Δw = 0
        for i in 0..n {
            assert!((weights[i] - 1.0).abs() < 1e-6,
                "Homeostatic: at target rate weight should not change, got {:.6}", weights[i]);
        }
    }
}
