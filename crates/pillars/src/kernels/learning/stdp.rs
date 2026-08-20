use crate::core::util::{load8, store8};

use wide::f32x8;

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

/// STDP with homeostatic weight-sum regulation (волна 8, транш 1b).
///
/// Same update as [`stdp_update`] plus a global pull of the total weight
/// sum toward `target_sum`:
///
/// ```text
/// homeo = (target_sum − Σw) / n
/// w'    = clamp(w + lr_ltp·tr_post − lr_ltd·tr_pre + homeo, 0, w_max)
/// ```
///
/// Canonical port of the legacy `kernels::neuron::stp::stdp_homeostatic`
/// (бит-в-бит: та же формула суммы, тот же порядок арифметики SIMD/tail).
pub fn stdp_homeostatic(
    weights: &mut [f32],
    traces_pre: &[f32],
    traces_post: &[f32],
    lr_ltd: f32,
    lr_ltp: f32,
    target_sum: f32,
    w_max: f32,
) {
    let n = weights.len().min(traces_pre.len()).min(traces_post.len());
    if n == 0 {
        return;
    }

    // Global weight sum (SIMD reduce + scalar tail, тот же порядок, что в легаси).
    let n8 = n & !7;
    let mut sum = 0.0f32;
    for off in (0..n8).step_by(8) {
        let wv = unsafe { load8(weights.as_ptr(), off) };
        let a = wv.to_array();
        sum += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    }
    for &val in &weights[n8..n] {
        sum += val;
    }

    let homeo = (target_sum - sum) / n as f32;
    let ltd_s = f32x8::splat(lr_ltd);
    let ltp_s = f32x8::splat(lr_ltp);
    let homeo_s = f32x8::splat(homeo);
    let wmax_s = f32x8::splat(w_max);
    let zero_s = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        unsafe {
            let mut w = load8(weights.as_ptr(), off);
            let pre = load8(traces_pre.as_ptr(), off);
            let post = load8(traces_post.as_ptr(), off);
            w = w + ltp_s * post - ltd_s * pre + homeo_s;
            w = w.min(wmax_s).max(zero_s);
            store8(weights.as_mut_ptr(), off, w);
        }
    }
    for i in n8..n {
        weights[i] += lr_ltp * traces_post[i] - lr_ltd * traces_pre[i] + homeo;
        weights[i] = weights[i].max(0.0).min(w_max);
    }
}
