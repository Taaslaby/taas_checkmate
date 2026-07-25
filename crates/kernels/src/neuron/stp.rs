#![allow(non_snake_case)]
use super::{load8, load8_i32, load8_mut, store8};
use wide::{f32x8, CmpGt};

// ═══════════════════════════════════════════════════════════
// TaasLab yakutia \\ ТаасЛаб якутия \\ дойдум инники кэскилэ
// ═════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════
//  STP (Short-Term Plasticity) — Tsodyks-Markram SIMD kernel
// ═══════════════════════════════════════════════════════

/// Step `n` STP synapses (Tsodyks-Markram model). Uses `f32x8` SIMD.
///
/// Equations:
///   du/dt = (u0-u)/τ_f + u0·(1-u)·s      (facilitation)
///   dx/dt = (1-x)/τ_d − u·x·s             (depression)
///   I_out += A·u·x·s                        (synaptic current)
///
/// Where `s = 1` if pre_spike, else `s = 0`.
/// When no spike, only the decay terms apply.
///
/// # Safety
/// All pointer args valid for `n` elements, non-overlapping.
pub(crate) unsafe fn stp_step(
    x: *mut f32,
    u: *mut f32,
    pre_spikes: *const i32,
    tau_d: *const f32,
    tau_f: *const f32,
    u0: *const f32,
    a: *const f32,
    I_out: *mut f32,
    dt: f32,
    n: usize,
) {
    let dt_s = f32x8::splat(dt);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let eps = f32x8::splat(1e-10);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut xx = load8_mut(x, off);
        let mut uu = load8_mut(u, off);
        let sp = load8_i32(pre_spikes, off);
        let td = load8(tau_d, off).max(eps);
        let tf = load8(tau_f, off).max(eps);
        let u0v = load8(u0, off);
        let av = load8(a, off);
        let mut io = load8_mut(I_out, off);

        // s = 1 where pre_spike, else 0
        let s_one = sp.cmp_gt(zero).blend(one, zero);

        // Save pre-update values for both facilitation and depression
        // Tsodyks & Markram (1997): all spike-dependent terms use pre-spike u/x
        let uu0 = uu;
        let xx0 = xx;

        // Common decay (always applied): u += dt·(u0-u)/τ_f
        uu += dt_s * (u0v - uu0) / tf;
        // Common decay (always applied): x += dt·(1-x)/τ_d
        xx += dt_s * (one - xx0) / td;

        // Facilitation on spike: + u0·(1-u_old)  (pre-spike u)
        uu += s_one * u0v * (one - uu0);
        // Depression on spike: − u_old · x_old  (both pre-spike)
        xx += s_one * (-uu0 * xx0); // u_old here — parallel Euler
                                    // Synaptic current: + A·u_new·x_new where spike
        io += s_one * av * uu * xx;

        store8(x, off, xx);
        store8(u, off, uu);
        store8(I_out, off, io);
    }

    // Scalar tail (n & 7)
    // BUGFIX: save pre-decay u/x for spike-dependent terms (matching SIMD)
    if n8 < n {
        let x_sl = std::slice::from_raw_parts_mut(x, n);
        let u_sl = std::slice::from_raw_parts_mut(u, n);
        let pre = std::slice::from_raw_parts(pre_spikes, n);
        let td = std::slice::from_raw_parts(tau_d, n);
        let tf = std::slice::from_raw_parts(tau_f, n);
        let u0v = std::slice::from_raw_parts(u0, n);
        let av = std::slice::from_raw_parts(a, n);
        let io = std::slice::from_raw_parts_mut(I_out, n);
        for i in n8..n {
            let s = if pre[i] != 0 { 1.0 } else { 0.0 };
            let tf_i = tf[i].max(1e-10);
            let td_i = td[i].max(1e-10);
            // Save pre-spike values for BOTH facilitation and depression (Tsodyks-Markram 1997)
            let u_pre = u_sl[i];
            let x_pre = x_sl[i];
            // Decay (always applied)
            u_sl[i] += dt * (u0v[i] - u_pre) / tf_i;
            x_sl[i] += dt * (1.0 - x_pre) / td_i;
            // Spike-dependent terms (PRE-spike values for both u and x — parallel Euler)
            u_sl[i] += u0v[i] * (1.0 - u_pre) * s;
            x_sl[i] -= u_pre * x_pre * s; // u_pre here — parallel Euler
            io[i] += av[i] * u_sl[i] * x_sl[i] * s;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  STDP (Spike-Timing-Dependent Plasticity)
// ═══════════════════════════════════════════════════════

/// STDP weight update: w += A⁺·pre − A⁻·post, clamped to [w_min, w_max].
/// Uses `f32x8` SIMD.
///
/// # Safety
/// `weights[n]`, `traces_pre[n]`, `traces_post[n]` valid, non-overlapping.
pub(crate) unsafe fn stdp_update(
    weights: *mut f32,
    traces_pre: *const f32,
    traces_post: *const f32,
    lr_ltd: f32,
    lr_ltp: f32,
    w_min: f32,
    w_max: f32,
    n: usize,
) {
    let ltd = f32x8::splat(lr_ltd);
    let ltp = f32x8::splat(lr_ltp);
    let wmin = f32x8::splat(w_min);
    let wmax = f32x8::splat(w_max);
    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let pre = load8(traces_pre, off);
        let post = load8(traces_post, off);
        w = w + ltp * post - ltd * pre;
        w = w.min(wmax).max(wmin);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let pre = std::slice::from_raw_parts(traces_pre, n);
        let post = std::slice::from_raw_parts(traces_post, n);
        for i in n8..n {
            w[i] += lr_ltp * post[i] - lr_ltd * pre[i];
            w[i] = w[i].clamp(w_min, w_max);
        }
    }
}

/// STDP with homeostatic weight-sum regulation. Uses `f32x8` SIMD.
///
/// # Safety
/// `weights[n]`, `traces_pre[n]`, `traces_post[n]` valid, non-overlapping.
pub(crate) unsafe fn stdp_homeostatic(
    weights: *mut f32,
    traces_pre: *const f32,
    traces_post: *const f32,
    lr_ltd: f32,
    lr_ltp: f32,
    target_sum: f32,
    w_max: f32,
    n: usize,
) {
    // Compute sum via SIMD reduction
    let n8 = n & !7;
    let mut sum = 0.0f32;
    {
        let w_slc = std::slice::from_raw_parts(weights, n);
        for off in (0..n8).step_by(8) {
            let wv = load8(weights, off);
            let a: [f32; 8] = wv.to_array();
            sum += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
        }
        for &val in w_slc[n8..].iter() {
            sum += val;
        }
    }

    let ltd = f32x8::splat(lr_ltd);
    let ltp = f32x8::splat(lr_ltp);
    let homeo = f32x8::splat((target_sum - sum) / n as f32);
    let wmax = f32x8::splat(w_max);
    let zero = f32x8::splat(0.0);

    for off in (0..n8).step_by(8) {
        let mut w = load8_mut(weights, off);
        let pre = load8(traces_pre, off);
        let post = load8(traces_post, off);
        w = w + ltp * post - ltd * pre + homeo;
        w = w.min(wmax).max(zero);
        store8(weights, off, w);
    }

    if n8 < n {
        let w = std::slice::from_raw_parts_mut(weights, n);
        let pre = std::slice::from_raw_parts(traces_pre, n);
        let post = std::slice::from_raw_parts(traces_post, n);
        for i in n8..n {
            w[i] += lr_ltp * post[i] - lr_ltd * pre[i] + (target_sum - sum) / n as f32;
            w[i] = w[i].max(0.0).min(w_max);
        }
    }
}
