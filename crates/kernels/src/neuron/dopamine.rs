#![allow(non_snake_case)]
use super::{fast_exp_simd, load8, load8_i32, load8_mut, store8};
use wide::{f32x8, CmpGt};

// ═══════════════════════════════════════════════════════
//  Dopamine-Modulated STDP — 3-factor plasticity for RL
// ═══════════════════════════════════════════════════════
//
// Combines:
//   1. STDP eligibility (pre-post spike timing)
//   2. Dopamine gate (reward signal)
//   3. Eligibility trace decay
//
// Equations per synapse:
//   tr_pre += 1 on pre-spike, decays otherwise
//   tr_post += 1 on post-spike, decays otherwise
//   e += A_ltp · tr_pre · post_spike − A_ltd · tr_post · pre_spike
//   e *= (1 − dt/τ_e)  (eligibility decay)
//   Δw = η · D · e
//
// Based on: Izhikevich (2007) "Solving the distal reward problem
// through linkage of STDP and dopamine signaling" — Cerebral Cortex

/// Update dopamine-modulated STDP traces and eligibility for `n` synapses.
/// Uses `f32x8` SIMD.
///
/// # Args
/// * `trace_pre` — pre-synaptic spike trace [n] (in/out)
/// * `trace_post` — post-synaptic spike trace [n] (in/out)
/// * `eligibility` — eligibility trace e [n] (in/out)
/// * `pre_spikes` — pre-synaptic spikes [n] (0/1)
/// * `post_spikes` — post-synaptic spikes [n] (0/1)
/// * `dt` — time step (ms)
/// * `tau_pre` — pre-trace time constant (ms)
/// * `tau_post` — post-trace time constant (ms)
/// * `tau_e` — eligibility trace time constant (ms)
/// * `a_ltp` — LTP amplitude (positive STDP)
/// * `a_ltd` — LTD amplitude (negative STDP)
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn dopamine_traces_step(
    trace_pre: *mut f32,
    trace_post: *mut f32,
    eligibility: *mut f32,
    pre_spikes: *const i32,
    post_spikes: *const i32,
    dt: f32,
    tau_pre: f32,
    tau_post: f32,
    tau_e: f32,
    a_ltp: f32,
    a_ltd: f32,
    n: usize,
) {
    let dt_s = f32x8::splat(dt);
    // BUGFIX: clamp to EPSILON (was 1.0) — prevents div-by-zero without silently changing dynamics
    let epsi = f32::EPSILON;
    let tau_pre_s = f32x8::splat(tau_pre.max(epsi));
    let tau_post_s = f32x8::splat(tau_post.max(epsi));
    let tau_e_s = f32x8::splat(tau_e.max(epsi));
    let a_ltp_s = f32x8::splat(a_ltp);
    let a_ltd_s = f32x8::splat(a_ltd);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut tp = load8_mut(trace_pre, off);
        let mut tpo = load8_mut(trace_post, off);
        let mut ee = load8_mut(eligibility, off);
        let pre = load8_i32(pre_spikes, off);
        let post = load8_i32(post_spikes, off);

        // Decay traces (exponential Euler)
        let decay_pre = fast_exp_simd(-dt_s / tau_pre_s);
        let decay_post = fast_exp_simd(-dt_s / tau_post_s);
        tp *= decay_pre;
        tpo *= decay_post;

        // Spike-triggered trace increase
        let pre_mask = pre.cmp_gt(zero);
        let post_mask = post.cmp_gt(zero);
        tp += pre_mask.blend(one, zero);
        tpo += post_mask.blend(one, zero);

        // STDP eligibility: e += A_ltp · tr_pre · post_spike − A_ltd · tr_post · pre_spike
        let ltp_term = a_ltp_s * tp * post_mask.blend(one, zero);
        let ltd_term = a_ltd_s * tpo * pre_mask.blend(one, zero);
        ee = ee + ltp_term - ltd_term;

        // Eligibility decay (exponential Euler)
        let decay_e = fast_exp_simd(-dt_s / tau_e_s);
        ee *= decay_e;

        store8(trace_pre, off, tp);
        store8(trace_post, off, tpo);
        store8(eligibility, off, ee);
    }

    // Scalar tail (n & 7)
    // NOTE: must use same clamping as SIMD path (f32::EPSILON) to avoid SIMD/scalar mismatch
    if n8 < n {
        let epsi = f32::EPSILON;
        let tp = std::slice::from_raw_parts_mut(trace_pre, n);
        let tpo = std::slice::from_raw_parts_mut(trace_post, n);
        let ee = std::slice::from_raw_parts_mut(eligibility, n);
        let pre = std::slice::from_raw_parts(pre_spikes, n);
        let post = std::slice::from_raw_parts(post_spikes, n);
        for i in n8..n {
            let decay_pre = (-dt / tau_pre.max(epsi)).exp();
            let decay_post = (-dt / tau_post.max(epsi)).exp();
            tp[i] *= decay_pre;
            tpo[i] *= decay_post;
            if pre[i] != 0 {
                tp[i] += 1.0;
            }
            if post[i] != 0 {
                tpo[i] += 1.0;
            }
            let ltp = if post[i] != 0 { a_ltp * tp[i] } else { 0.0 };
            let ltd = if pre[i] != 0 { a_ltd * tpo[i] } else { 0.0 };
            ee[i] += ltp - ltd;
            let decay_e = (-dt / tau_e.max(epsi)).exp();
            ee[i] *= decay_e;
        }
    }
}

/// Apply dopamine-modulated weight update: Δw = η · D · e.
/// Uses `f32x8` SIMD.
///
/// # Args
/// * `weights` — weight matrix [n] (in/out)
/// * `eligibility` — eligibility traces [n]
/// * `dopamine` — scalar dopamine signal (D)
/// * `lr` — learning rate η
/// * `w_min` — minimum weight (clamp)
/// * `w_max` — maximum weight (clamp)
/// * `n` — number of synapses
///
/// # Safety
/// `weights[n]`, `eligibility[n]` valid, non-overlapping.
pub(crate) unsafe fn dopamine_weight_update(
    weights: *mut f32,
    eligibility: *const f32,
    dopamine: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
    n: usize,
) {
    let lr_dopa = f32x8::splat(lr * dopamine);
    let wmin_s = f32x8::splat(w_min);
    let wmax_s = f32x8::splat(w_max);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let ee = load8(eligibility, off);
        w += lr_dopa * ee;
        w = w.min(wmax_s).max(wmin_s);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let ee = std::slice::from_raw_parts(eligibility, n);
        for i in n8..n {
            w[i] += lr * dopamine * ee[i];
            w[i] = w[i].clamp(w_min, w_max);
        }
    }
}
