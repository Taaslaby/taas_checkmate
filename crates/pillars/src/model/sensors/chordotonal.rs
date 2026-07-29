#![allow(non_snake_case)]

//! Chordotonal Sensory Organ — SIMD-Accelerated Mechanoreceptor
//!
//! Models Drosophila chordotonal (scolopidial) mechanoreceptors:
//!   - Load-sensitive dendrite: receptor potential ∝ contact force
//!   - Stretch-sensitive: joint angle deviation
//!   - Velocity-sensitive: phasic component ∝ dF/dt
//!   - Slow adaptation: negative feedback (sensory adaptation)
//!   - Firing rate: sigmoid transfer function
//!
//! Equations:
//!   τ_sens · dV_sens/dt = -V_sens + w_load·load + w_stretch·stretch + w_baseline
//!   τ_adapt · dA/dt = -A + b_adapt · rate
//!   rate = max_rate · σ((V_sens - A - threshold) / slope)
//!   σ(x) = 1 / (1 + exp(-x))
//!
//! Params layout (column-major, size 9 × n):
//!   [0..n]      dt (ms)
//!   [n..2n]     tau_sens (ms)
//!   [2n..3n]    tau_adapt (ms)
//!   [3n..4n]    b_adapt
//!   [4n..5n]    load_gain
//!   [5n..6n]    stretch_gain
//!   [6n..7n]    baseline
//!   [7n..8n]    threshold
//!   [8n..9n]    max_rate (Hz)

use wide::f32x8;
use crate::model::util::{load8, store8};

/// Step chordotonal sensory neurons — converts mechanical load to firing rate.
///
/// # Args
/// * `v_sens` — receptor potential (state), updated in-place
/// * `adapt` — adaptation variable (state), updated in-place
/// * `rates` — output firing rates (Hz), written each step
/// * `loads` — mechanical load/contact force (normalized 0..1)
/// * `stretches` — stretch/joint angle deviation (normalized -1..1)
/// * `d_loads` — derivative of load (velocity sensitivity, optional, can be null)
/// * `params` — per-channel parameters (9 × n, column-major)
/// * `n` — number of channels
///
/// # Safety
/// All non-null pointers must be valid for `n` elements.
pub unsafe fn chordotonal_step(
    v_sens: *mut f32,
    adapt: *mut f32,
    rates: *mut f32,
    loads: *const f32,
    stretches: *const f32,
    d_loads: *const f32,
    params: *const f32,
    n: usize,
) {
    let p = std::slice::from_raw_parts(params, n * 9);
    let dt_splat = f32x8::splat(p[0]);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let clamp_max = f32x8::splat(50.0);
    let clamp_min = f32x8::splat(-50.0);
    let eps = f32x8::splat(1e-6);

    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vs = load8(v_sens, off);
        let mut ad = load8(adapt, off);
        let load_v = load8(loads, off);
        let stretch_v = load8(stretches, off);
        let dload_v = if !d_loads.is_null() { load8(d_loads, off) } else { zero };

        let tau_sens_v = load8(params.wrapping_add(n), off).max(eps);
        let tau_adapt_v = load8(params.wrapping_add(2 * n), off).max(eps);
        let b_adapt_v = load8(params.wrapping_add(3 * n), off);
        let load_gain_v = load8(params.wrapping_add(4 * n), off);
        let stretch_gain_v = load8(params.wrapping_add(5 * n), off);
        let baseline_v = load8(params.wrapping_add(6 * n), off);
        let threshold_v = load8(params.wrapping_add(7 * n), off);
        let max_rate_v = load8(params.wrapping_add(8 * n), off);

        // 1. Receptor potential
        let i_mech = load_gain_v * load_v + stretch_gain_v * stretch_v + dload_v + baseline_v;
        vs += dt_splat * (-vs + i_mech) / tau_sens_v;

        // 2. Raw rate for adaptation (pre-adaptation)
        let raw_diff = (vs - threshold_v).max(clamp_min).min(clamp_max);
        let raw_sig = one / (one + (-raw_diff).exp());
        let raw_rate = max_rate_v * raw_sig;

        // 3. Adaptation
        ad += dt_splat * (-ad + b_adapt_v * raw_rate) / tau_adapt_v;
        ad = ad.max(zero);

        // 4. Final rate with adaptation
        let adapted_diff = (vs - ad - threshold_v).max(clamp_min).min(clamp_max);
        let adapted_sig = one / (one + (-adapted_diff).exp());
        let final_rate = max_rate_v * adapted_sig;

        store8(v_sens, off, vs);
        store8(adapt, off, ad);
        store8(rates, off, final_rate);
    }

    // Scalar tail
    if n8 < n {
        for i in n8..n {
            let tau_sens = p[n + i].max(1e-6);
            let tau_adapt = p[2 * n + i].max(1e-6);
            let b_adapt = p[3 * n + i];
            let load_gain = p[4 * n + i];
            let stretch_gain = p[5 * n + i];
            let baseline = p[6 * n + i];
            let threshold = p[7 * n + i];
            let max_rate = p[8 * n + i];
            let dt = p[0];

            let load = *loads.add(i);
            let stretch = *stretches.add(i);
            let dload = if !d_loads.is_null() { *d_loads.add(i) } else { 0.0 };

            let vs_i = v_sens.add(i);
            let ad_i = adapt.add(i);

            *vs_i += dt * (-*vs_i + load_gain * load + stretch_gain * stretch + dload + baseline) / tau_sens;
            let raw_diff = (*vs_i - threshold).clamp(-50.0, 50.0);
            let raw_sig = 1.0 / (1.0 + (-raw_diff).exp());
            *ad_i += dt * (-*ad_i + b_adapt * max_rate * raw_sig) / tau_adapt;
            *ad_i = (*ad_i).max(0.0);

            let adapted_diff = (*vs_i - *ad_i - threshold).clamp(-50.0, 50.0);
            let adapted_sig = 1.0 / (1.0 + (-adapted_diff).exp());
            *rates.add(i) = max_rate * adapted_sig;
        }
    }
}
