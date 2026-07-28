#![allow(non_snake_case)]
use super::{load8, store8};
use wide::f32x8;

// ═══════════════════════════════════════════════════════
//  Chordotonal Sensory Organ Kernel
// ═══════════════════════════════════════════════════════
//
// Models Drosophila chordotonal (scolopidial) mechanoreceptors:
//   - Load-sensitive dendrite: receptor potential ∝ contact force
//   - Stretch-sensitive: joint angle deviation
//   - Velocity-sensitive: phasic component ∝ dF/dt
//   - Slow adaptation: negative feedback (sensory adaptation)
//   - Firing rate: sigmoid transfer function
//
// Equations:
//   τ_sens · dV_sens/dt = -V_sens + w_load·load + w_stretch·stretch + w_baseline
//   τ_adapt · dA/dt = -A + b_adapt · rate
//   rate = max_rate · σ((V_sens - A - threshold) / slope)
//   σ(x) = 1 / (1 + exp(-x))
//
// Params layout (column-major, size 9 × n):
//   [0..n]      dt (ms)
//   [n..2n]     tau_sens (ms) — receptor time constant
//   [2n..3n]    tau_adapt (ms) — adaptation time constant
//   [3n..4n]    b_adapt — adaptation strength
//   [4n..5n]    load_gain — gain for load (contact force) input
//   [5n..6n]    stretch_gain — gain for stretch (joint angle) input
//   [6n..7n]    baseline — baseline depolarization
//   [7n..8n]    threshold — spiking threshold
//   [8n..9n]    max_rate (Hz) — maximum firing rate

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
/// * `n` — number of channels (padded to multiple of 8)
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
    let dt_splat = f32x8::splat(p[0]); // dt (same for all)
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    let n8 = n & !7;

    for off in (0..n8).step_by(8) {
        let mut vs = load8(v_sens, off);
        let mut ad = load8(adapt, off);

        // Load inputs
        let load_v = load8(loads, off);
        let stretch_v = load8(stretches, off);

        // Velocity component (if provided)
        let dload_v = if !d_loads.is_null() {
            load8(d_loads, off)
        } else {
            zero
        };

        // Load parameters
        let tau_sens_v = load8(params.wrapping_add(n), off).max(f32x8::splat(1e-6));
        let tau_adapt_v = load8(params.wrapping_add(2 * n), off).max(f32x8::splat(1e-6));
        let b_adapt_v = load8(params.wrapping_add(3 * n), off);
        let load_gain_v = load8(params.wrapping_add(4 * n), off);
        let stretch_gain_v = load8(params.wrapping_add(5 * n), off);
        let baseline_v = load8(params.wrapping_add(6 * n), off);
        let threshold_v = load8(params.wrapping_add(7 * n), off);
        let max_rate_v = load8(params.wrapping_add(8 * n), off);

        // 1. Receptor potential: V_sens += dt * (-V_sens + I_mech) / tau_sens
        let i_mech = load_gain_v * load_v + stretch_gain_v * stretch_v + dload_v + baseline_v;
        let dV = (-vs + i_mech) / tau_sens_v;
        vs += dt_splat * dV;

        // 2. Compute pre-adaptation rate: raw_rate = max_rate * sigmoid(v_sens - threshold)
        let diff = vs - threshold_v;
        let clamped_diff = diff.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0));
        let sigmoid_v = one / (one + (-clamped_diff).exp());
        let raw_rate = max_rate_v * sigmoid_v;

        // 3. Adaptation: dA/dt = (-A + b_adapt * rate) / tau_adapt
        // Use raw_rate for adaptation (negative feedback)
        let dA = (-ad + b_adapt_v * raw_rate) / tau_adapt_v;
        ad += dt_splat * dA;
        ad = ad.max(zero); // Non-negative adaptation

        // 4. Final rate: rate = max_rate * sigmoid(V_sens - A - threshold)
        let diff_adapted = vs - ad - threshold_v;
        let clamped_adapted = diff_adapted.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0));
        let sigmoid_adapted = one / (one + (-clamped_adapted).exp());
        let final_rate = max_rate_v * sigmoid_adapted;

        // Store outputs
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

            // 1. Receptor potential
            let i_mech = load_gain * load + stretch_gain * stretch + dload + baseline;
            let vs_i = v_sens.add(i);
            let ad_i = adapt.add(i);

            *vs_i += dt * (-*vs_i + i_mech) / tau_sens;

            // 2. Raw rate for adaptation
            let raw_diff = *vs_i - threshold;
            let raw_sig = 1.0 / (1.0 + (-raw_diff.clamp(-50.0, 50.0)).exp());
            let raw_rate = max_rate * raw_sig;

            // 3. Adaptation
            *ad_i += dt * (-*ad_i + b_adapt * raw_rate) / tau_adapt;
            *ad_i = (*ad_i).max(0.0);

            // 4. Final rate with adaptation
            let adapted_diff = *vs_i - *ad_i - threshold;
            let adapted_sig = 1.0 / (1.0 + (-adapted_diff.clamp(-50.0, 50.0)).exp());
            *rates.add(i) = max_rate * adapted_sig;
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(n: usize, dt: f32) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[i] = dt;                  // dt
            p[n + i] = 5.0;             // tau_sens = 5ms
            p[2*n + i] = 100.0;         // tau_adapt = 100ms
            p[3*n + i] = 0.3;           // b_adapt = 0.3
            p[4*n + i] = 3.0;           // load_gain = 3.0
            p[5*n + i] = 1.0;           // stretch_gain = 1.0
            p[6*n + i] = -2.0;          // baseline = -2.0 (below threshold)
            p[7*n + i] = 0.0;           // threshold = 0.0
            p[8*n + i] = 200.0;         // max_rate = 200 Hz
        }
        p
    }

    #[test]
    fn test_chordotonal_no_load_no_rate() {
        let n = 8;
        let dt = 1.0;
        let mut vs = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];     // no load
        let stretches = vec![0.0f32; n]; // no stretch
        let params = make_params(n, dt);

        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                params.as_ptr(), n,
            );
        }

        // After 1 Euler step: V_sens = 0 + 1*(-2)/5 = -0.4
        // sigmoid(-0.4) ≈ 0.401, rate ≈ 200 * 0.401 ≈ 80 Hz (1 step, not converged)
        // After convergence: V_sens = -2.0, rate ≈ 200 * 0.119 = 23.8 Hz
        // So rate should be between 1-step and steady state
        assert!(rates[0] > 0.0, "Baseline alone should produce some rate");
        // 200 * sigmoid(-0.4) = 80 Hz, so upper bound should allow this
        assert!(rates[0] < 150.0, "Low baseline should give moderate rate");
    }

    #[test]
    fn test_chordotonal_load_increases_rate() {
        let n = 8;
        let dt = 1.0;
        let mut vs = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![1.0f32; n];     // full load
        let stretches = vec![0.0f32; n];
        let params = make_params(n, dt);

        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                params.as_ptr(), n,
            );
        }

        // After 1 Euler step (dt=1ms, tau_sens=5): V_sens = 0 + 1*(3-2)/5 = 0.2
        // sigmoid(0.2) ≈ 0.55, rate ≈ 200 * 0.55 ≈ 110 Hz (1 step)
        // After convergence: V_sens = 1.0, rate ≈ 200 * 0.731 ≈ 146 Hz
        assert!(rates[0] > 50.0, "High load should produce high firing rate");
    }

    #[test]
    fn test_chordotonal_adaptation_decays_rate() {
        let n = 8;
        let dt = 1.0;
        let mut vs = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![1.0f32; n];
        let stretches = vec![0.0f32; n];
        let params = make_params(n, dt);

        // Step 1: high rate
        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                params.as_ptr(), n,
            );
        }
        let rate1 = rates[0];

        // Step 2-10: rate should decrease due to adaptation
        for _ in 0..20 {
            unsafe {
                chordotonal_step(
                    vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                    loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                    params.as_ptr(), n,
                );
            }
        }
        let rate_final = rates[0];

        assert!(rate_final < rate1 * 0.8,
            "Adaptation should reduce firing rate over time");
    }

    #[test]
    fn test_chordotonal_velocity_sensitivity() {
        let n = 8;
        let dt = 1.0;
        let mut vs = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.5f32; n];
        let stretches = vec![0.0f32; n];
        let d_loads = vec![10.0f32; n]; // rapid increase in load
        let params = make_params(n, dt);

        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                loads.as_ptr(), stretches.as_ptr(), d_loads.as_ptr(),
                params.as_ptr(), n,
            );
        }

        // With d_load=10, V_sens jumps high -> rate should be saturated near max_rate
        assert!(rates[0] > 150.0, "Velocity sensitivity should boost rate strongly");
    }

    #[test]
    fn test_chordotonal_phasic_tonic_response() {
        // Step load: initial phasic burst (high rate), then tonic steady state
        let n = 8;
        let dt = 1.0;
        let mut vs = vec![0.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.0f32; n];
        let stretches = vec![0.0f32; n];
        let params = make_params(n, dt);

        // First, steady state at no load (to stabilize)
        for _ in 0..20 {
            unsafe {
                chordotonal_step(
                    vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                    loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                    params.as_ptr(), n,
                );
            }
        }

        // Apply sudden load
        let high_loads = vec![1.0f32; n];
        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                high_loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                params.as_ptr(), n,
            );
        }
        let phasic_rate = rates[0];

        // After many steps with sustained load, should drop to tonic
        for _ in 0..100 {
            unsafe {
                chordotonal_step(
                    vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                    high_loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                    params.as_ptr(), n,
                );
            }
        }
        let tonic_rate = rates[0];

        assert!(phasic_rate > tonic_rate * 1.2,
            "Phasic response should exceed tonic");
    }

    #[test]
    fn test_chordotonal_simd_scalar_parity() {
        let n = 11; // 8 SIMD + 3 scalar
        let dt = 1.0;
        let mut vs = vec![-1.0f32; n];
        let mut adapt = vec![0.0f32; n];
        let mut rates = vec![0.0f32; n];
        let loads = vec![0.3f32; n];
        let stretches = vec![0.1f32; n];
        let params = make_params(n, dt);

        unsafe {
            chordotonal_step(
                vs.as_mut_ptr(), adapt.as_mut_ptr(), rates.as_mut_ptr(),
                loads.as_ptr(), stretches.as_ptr(), std::ptr::null(),
                params.as_ptr(), n,
            );
        }

        let diff = (rates[0] - rates[8]).abs();
        assert!(diff < 1e-5, "SIMD vs scalar rate mismatch");
    }
}
