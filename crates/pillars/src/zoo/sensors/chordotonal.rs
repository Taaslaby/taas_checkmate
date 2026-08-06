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

use crate::core::lane::SimdLane;
use crate::core::util::{load8, store8};
use wide::f32x8;

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
    // FIX #5: Early return for empty arrays — prevents UB on params slice
    if n == 0 {
        return;
    }
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
        let dload_v = if !d_loads.is_null() {
            load8(d_loads, off)
        } else {
            zero
        };

        let tau_sens_v = load8(params.wrapping_add(n), off).max(eps);
        let tau_adapt_v = load8(params.wrapping_add(2 * n), off).max(eps);
        let b_adapt_v = load8(params.wrapping_add(3 * n), off);
        let load_gain_v = load8(params.wrapping_add(4 * n), off);
        let stretch_gain_v = load8(params.wrapping_add(5 * n), off);
        let baseline_v = load8(params.wrapping_add(6 * n), off);
        let threshold_v = load8(params.wrapping_add(7 * n), off);
        let max_rate_v = load8(params.wrapping_add(8 * n), off);

        // 1. Receptor potential (inv_tau to avoid vector division)
        let inv_tau_sens = tau_sens_v.recip();
        let inv_tau_adapt = tau_adapt_v.recip();
        let i_mech = load_gain_v * load_v + stretch_gain_v * stretch_v + dload_v + baseline_v;
        vs += dt_splat * (-vs + i_mech) * inv_tau_sens;

        // 2. Raw rate for adaptation (pre-adaptation)
        let raw_diff = (vs - threshold_v).max(clamp_min).min(clamp_max);
        let raw_sig = one / (one + <f32x8 as SimdLane>::exp(-raw_diff));
        let raw_rate = max_rate_v * raw_sig;

        // 3. Adaptation (inv_tau to avoid vector division)
        ad += dt_splat * (-ad + b_adapt_v * raw_rate) * inv_tau_adapt;
        ad = ad.max(zero);

        // 4. Final rate with adaptation
        let adapted_diff = (vs - ad - threshold_v).max(clamp_min).min(clamp_max);
        let adapted_sig = one / (one + <f32x8 as SimdLane>::exp(-adapted_diff));
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
            let dload = if !d_loads.is_null() {
                *d_loads.add(i)
            } else {
                0.0
            };

            let vs_i = v_sens.add(i);
            let ad_i = adapt.add(i);

            let inv_tau_s = 1.0 / tau_sens;
            let inv_tau_a = 1.0 / tau_adapt;
            *vs_i += dt
                * (-*vs_i + load_gain * load + stretch_gain * stretch + dload + baseline)
                * inv_tau_s;
            let raw_diff = (*vs_i - threshold).clamp(-50.0, 50.0);
            let raw_sig = 1.0 / (1.0 + <f32 as SimdLane>::exp(-raw_diff));
            *ad_i += dt * (-*ad_i + b_adapt * max_rate * raw_sig) * inv_tau_a;
            *ad_i = (*ad_i).max(0.0);

            let adapted_diff = (*vs_i - *ad_i - threshold).clamp(-50.0, 50.0);
            let adapted_sig = 1.0 / (1.0 + <f32 as SimdLane>::exp(-adapted_diff));
            *rates.add(i) = max_rate * adapted_sig;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sensor_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[i] = 1.0; // dt
            p[n + i] = 10.0; // tau_sens
            p[2 * n + i] = 100.0; // tau_adapt
            p[3 * n + i] = 0.01; // b_adapt
            p[4 * n + i] = 50.0; // load_gain
            p[5 * n + i] = 20.0; // stretch_gain
            p[6 * n + i] = 0.0; // baseline
            p[7 * n + i] = 0.5; // threshold
            p[8 * n + i] = 200.0; // max_rate (Hz)
        }
        p
    }

    #[test]
    fn test_chordotonal_basic_load() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![1.0f32; n]; // max load
        let stretches = vec![0.0f32; n];

        unsafe {
            chordotonal_step(
                v_sens.as_mut_ptr(),
                adapt.as_mut_ptr(),
                rates.as_mut_ptr(),
                loads.as_ptr(),
                stretches.as_ptr(),
                std::ptr::null(),
                params.as_ptr(),
                n,
            );
        }

        // With max load and no adaptation yet, v_sens should rise
        assert!(v_sens[0] > 0.0, "v_sens should depolarize: {}", v_sens[0]);
        // Rate should be positive
        assert!(rates[0] > 0.0, "rate should be positive: {}", rates[0]);
        // Rate should be <= max_rate
        assert!(rates[0] <= 200.0, "rate capped by max_rate: {}", rates[0]);
    }

    #[test]
    fn test_chordotonal_zero_input() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];
        let stretches = vec![0.0f32; n];

        unsafe {
            chordotonal_step(
                v_sens.as_mut_ptr(),
                adapt.as_mut_ptr(),
                rates.as_mut_ptr(),
                loads.as_ptr(),
                stretches.as_ptr(),
                std::ptr::null(),
                params.as_ptr(),
                n,
            );
        }

        // With zero input and baseline=0, v_sens should stay near 0
        assert!(
            v_sens[0].abs() < 1e-4,
            "v_sens should stay near 0 with no load: {}",
            v_sens[0]
        );
        // Rate is sigmoid(threshold) * max_rate ≈ 200/(1+exp(0.5)) ≈ 75 Hz
        // Baseline=0, threshold=0.5 → quiescent firing at ~75 Hz with max 200
        assert!(
            rates[0] > 0.0 && rates[0] <= 200.0,
            "rate should be in valid range: {}",
            rates[0]
        );
    }

    #[test]
    fn test_chordotonal_empty_input() {
        unsafe {
            chordotonal_step(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
            );
        }
        // Must not crash with n=0
    }

    #[test]
    fn test_chordotonal_non_aligned() {
        let n = 10; // not a multiple of 8
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![1.0f32; n];
        let stretches = vec![0.0f32; n];

        unsafe {
            chordotonal_step(
                v_sens.as_mut_ptr(),
                adapt.as_mut_ptr(),
                rates.as_mut_ptr(),
                loads.as_ptr(),
                stretches.as_ptr(),
                std::ptr::null(),
                params.as_ptr(),
                n,
            );
        }

        // All neurons should behave the same
        for (i, &v_i) in v_sens.iter().enumerate() {
            assert!(v_i > 0.0, "v_sens[{i}] should depolarize");
            assert!(v_i.is_finite());
        }
        // Aligned and tail parts should match
        assert!(
            (v_sens[7] - v_sens[9]).abs() < 0.01,
            "SIMD v_sens[7]={:.4} != tail v_sens[9]={:.4}",
            v_sens[7],
            v_sens[9]
        );
    }

    #[test]
    fn test_chordotonal_stress_silence() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];
        let stretches = vec![0.0f32; n];

        for _ in 0..1000 {
            unsafe {
                chordotonal_step(
                    v_sens.as_mut_ptr(),
                    adapt.as_mut_ptr(),
                    rates.as_mut_ptr(),
                    loads.as_ptr(),
                    stretches.as_ptr(),
                    std::ptr::null(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &v_i) in v_sens.iter().enumerate() {
                assert!(v_i.is_finite(), "v_sens[{i}] NaN at silence");
            }
        }
        // After many steps with zero input, v_sens should settle near 0
        assert!(
            v_sens[0].abs() < 0.01,
            "v_sens should settle: {}",
            v_sens[0]
        );
    }

    #[test]
    fn test_chordotonal_stress_explosion() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![100.0f32; n]; // extreme load
        let stretches = vec![10.0f32; n];

        for _ in 0..1000 {
            unsafe {
                chordotonal_step(
                    v_sens.as_mut_ptr(),
                    adapt.as_mut_ptr(),
                    rates.as_mut_ptr(),
                    loads.as_ptr(),
                    stretches.as_ptr(),
                    std::ptr::null(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert!(v_sens[i].is_finite(), "v_sens[{i}] NaN at explosion");
                assert!(!v_sens[i].is_nan(), "v_sens[{i}] NaN");
                assert!(rates[i].is_finite());
            }
        }
        // Rate should be non-negative and finite
        assert!(rates[0] >= 0.0 && rates[0] <= 200.0);
    }

    #[test]
    fn test_chordotonal_velocity_sensitivity() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];
        let stretches = vec![0.0f32; n];
        let d_loads = vec![5.0f32; n]; // positive velocity

        unsafe {
            chordotonal_step(
                v_sens.as_mut_ptr(),
                adapt.as_mut_ptr(),
                rates.as_mut_ptr(),
                loads.as_ptr(),
                stretches.as_ptr(),
                d_loads.as_ptr(),
                params.as_ptr(),
                n,
            );
        }

        // Velocity sensitivity should produce a positive v_sens
        assert!(v_sens[0] > 0.0, "d_load should depolarize: {}", v_sens[0]);
        assert!(
            rates[0] > 0.0,
            "rate should be positive with d_load: {}",
            rates[0]
        );
    }

    #[test]
    fn test_chordotonal_stretch_sensitivity() {
        let n = 8;
        let params = make_sensor_params(n);
        let mut v_sens = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];
        let stretches = vec![1.0f32; n];

        unsafe {
            chordotonal_step(
                v_sens.as_mut_ptr(),
                adapt.as_mut_ptr(),
                rates.as_mut_ptr(),
                loads.as_ptr(),
                stretches.as_ptr(),
                std::ptr::null(),
                params.as_ptr(),
                n,
            );
        }

        // Stretch should depolarize
        assert!(v_sens[0] > 0.0, "stretch should depolarize: {}", v_sens[0]);
        // Inhibitory stretch (negative) should hyperpolarize
        let mut v_sens2 = vec![0.0f32; n];
        let mut adapt2 = vec![0.0f32; n];
        let mut rates2 = vec![0.0f32; n];
        let neg_stretches = vec![-1.0f32; n];
        unsafe {
            chordotonal_step(
                v_sens2.as_mut_ptr(),
                adapt2.as_mut_ptr(),
                rates2.as_mut_ptr(),
                loads.as_ptr(),
                neg_stretches.as_ptr(),
                std::ptr::null(),
                params.as_ptr(),
                n,
            );
        }
        assert!(
            v_sens2[0] < 0.0,
            "negative stretch should hyperpolarize: {}",
            v_sens2[0]
        );
    }
}
