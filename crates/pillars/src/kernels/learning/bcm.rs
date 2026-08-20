use crate::core::util::{load8, store8};

use wide::f32x8;

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
