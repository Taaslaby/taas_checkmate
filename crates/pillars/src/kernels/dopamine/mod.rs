//! # Dopamine-Modulated STDP — 3-factor plasticity for RL
//!
//! Combines:
//!   1. STDP eligibility (pre-post spike timing)
//!   2. Dopamine gate (reward signal)
//!   3. Eligibility trace decay
//!
//! Equations per synapse:
//! ```text
//! tr_pre  += 1 on pre-spike,  decays otherwise
//! tr_post += 1 on post-spike, decays otherwise
//! e       += A_ltp·tr_pre·post_spike − A_ltd·tr_post·pre_spike
//! e       *= exp(−dt/τ_e)                          (eligibility decay)
//! Δw       = η·D·e
//! ```
//!
//! Based on: Izhikevich (2007) "Solving the distal reward problem through
//! linkage of STDP and dopamine signaling" — Cerebral Cortex.

use wide::f32x8;
use wide::CmpGt;

use crate::core::util::{load8, load8_i32, store8};
use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd};

/// Update dopamine-modulated STDP traces and eligibility for `n` synapses.
///
/// # Args
/// * `trace_pre` — pre-synaptic spike trace `[n]` (in/out)
/// * `trace_post` — post-synaptic spike trace `[n]` (in/out)
/// * `eligibility` — eligibility trace e `[n]` (in/out)
/// * `pre_spikes` — pre-synaptic spikes `[n]` (0/1)
/// * `post_spikes` — post-synaptic spikes `[n]` (0/1)
/// * `dt` — time step (ms)
/// * `tau_pre` — pre-trace time constant (ms)
/// * `tau_post` — post-trace time constant (ms)
/// * `tau_e` — eligibility trace time constant (ms)
/// * `a_ltp` — LTP amplitude (positive STDP)
/// * `a_ltd` — LTD amplitude (negative STDP)
#[allow(clippy::too_many_arguments)]
pub fn dopamine_traces_step(
    trace_pre: &mut [f32],
    trace_post: &mut [f32],
    eligibility: &mut [f32],
    pre_spikes: &[i32],
    post_spikes: &[i32],
    dt: f32,
    tau_pre: f32,
    tau_post: f32,
    tau_e: f32,
    a_ltp: f32,
    a_ltd: f32,
) {
    let n = trace_pre.len();
    assert_eq!(trace_post.len(), n, "dopamine: trace_post length mismatch");
    assert_eq!(
        eligibility.len(),
        n,
        "dopamine: eligibility length mismatch"
    );
    assert_eq!(pre_spikes.len(), n, "dopamine: pre_spikes length mismatch");
    assert_eq!(
        post_spikes.len(),
        n,
        "dopamine: post_spikes length mismatch"
    );

    // BUGFIX: clamp to EPSILON (was 1.0) — prevents div-by-zero without
    // silently changing dynamics. Both SIMD and scalar paths use EPSILON so
    // they stay consistent.
    let epsi = f32::EPSILON;
    let dt_s = f32x8::splat(dt);
    let tau_pre_s = f32x8::splat(tau_pre.max(epsi));
    let tau_post_s = f32x8::splat(tau_post.max(epsi));
    let tau_e_s = f32x8::splat(tau_e.max(epsi));
    let a_ltp_s = f32x8::splat(a_ltp);
    let a_ltd_s = f32x8::splat(a_ltd);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut tp = load8(trace_pre.as_ptr(), off);
            let mut tpo = load8(trace_post.as_ptr(), off);
            let mut ee = load8(eligibility.as_ptr(), off);
            let pre = load8_i32(pre_spikes.as_ptr(), off);
            let post = load8_i32(post_spikes.as_ptr(), off);

            // Decay traces (exponential Euler)
            let decay_pre = fast_exp_poly_simd(-dt_s / tau_pre_s);
            let decay_post = fast_exp_poly_simd(-dt_s / tau_post_s);
            tp *= decay_pre;
            tpo *= decay_post;

            // Spike-triggered trace increase
            let pre_mask = pre.cmp_gt(zero);
            let post_mask = post.cmp_gt(zero);
            tp += pre_mask.blend(one, zero);
            tpo += post_mask.blend(one, zero);

            // STDP eligibility: e += A_ltp·tr_pre·post − A_ltd·tr_post·pre
            let ltp_term = a_ltp_s * tp * post_mask.blend(one, zero);
            let ltd_term = a_ltd_s * tpo * pre_mask.blend(one, zero);
            ee = ee + ltp_term - ltd_term;

            // Eligibility decay (exponential Euler)
            let decay_e = fast_exp_poly_simd(-dt_s / tau_e_s);
            ee *= decay_e;

            store8(trace_pre.as_mut_ptr(), off, tp);
            store8(trace_post.as_mut_ptr(), off, tpo);
            store8(eligibility.as_mut_ptr(), off, ee);
        }
    }

    // Scalar tail — same EPSILON clamp as SIMD path for consistency.
    for i in n8..n {
        let decay_pre = fast_exp(-dt / tau_pre.max(epsi));
        let decay_post = fast_exp(-dt / tau_post.max(epsi));
        trace_pre[i] *= decay_pre;
        trace_post[i] *= decay_post;
        if pre_spikes[i] != 0 {
            trace_pre[i] += 1.0;
        }
        if post_spikes[i] != 0 {
            trace_post[i] += 1.0;
        }
        let ltp = if post_spikes[i] != 0 {
            a_ltp * trace_pre[i]
        } else {
            0.0
        };
        let ltd = if pre_spikes[i] != 0 {
            a_ltd * trace_post[i]
        } else {
            0.0
        };
        eligibility[i] += ltp - ltd;
        let decay_e = fast_exp(-dt / tau_e.max(epsi));
        eligibility[i] *= decay_e;
    }
}

/// Apply dopamine-modulated weight update: `Δw = η·D·e`.
///
/// The clamp is part of the UPDATE, not a global pass: only synapses with
/// non-zero eligibility (`Δ ≠ 0`) are clamped to `[w_min, w_max]`. Synapses
/// with `e == 0` keep their exact value even if it lies outside the bounds —
/// innate/over-sized weights must survive zero-dopamine and frozen-path
/// updates (a global clamp would silently truncate them every call).
///
/// * `weights` — weight matrix `[n]` (in/out)
/// * `eligibility` — eligibility traces `[n]`
/// * `dopamine` — scalar dopamine signal (D)
/// * `lr` — learning rate η
pub fn dopamine_weight_update(
    weights: &mut [f32],
    eligibility: &[f32],
    dopamine: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
) {
    let n = weights.len();
    assert_eq!(
        eligibility.len(),
        n,
        "dopamine_weight: eligibility length mismatch"
    );
    let lr_dopa = f32x8::splat(lr * dopamine);
    let wmin_s = f32x8::splat(w_min);
    let wmax_s = f32x8::splat(w_max);
    let zero_s = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        unsafe {
            let w = load8(weights.as_ptr(), off);
            let ee = load8(eligibility.as_ptr(), off);
            let delta = lr_dopa * ee;
            let updated = (w + delta).max(wmin_s).min(wmax_s);
            // e != 0 → synapse participates: clamped update. e == 0 →
            // preserve value exactly (no re-clamp of innate weights).
            let participates = ee.abs().cmp_gt(zero_s);
            let out = participates.blend(updated, w);
            store8(weights.as_mut_ptr(), off, out);
        }
    }

    for i in n8..n {
        let delta = lr * dopamine * eligibility[i];
        if delta == 0.0 {
            continue;
        }
        weights[i] = (weights[i] + delta).clamp(w_min, w_max);
    }
}

#[cfg(test)]
mod tests;
