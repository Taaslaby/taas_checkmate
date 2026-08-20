//! Adaptive Exponential Integrate-and-Fire (AdEx) Neuron Model
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Equations
//! ```text
//! C·dV/dt = g_L·(V - E_L) + g_L·Δ_T·exp((V - V_th)/Δ_T) - w + I_inj
//! τ_w·dw/dt = a·(V - E_L) - w
//! ```
//! On spike (V ≥ V_peak): V → V_reset, w → w + b
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE exponential computation
//!   (prevents NaN from exp((v_reset - v_thresh)/Δ_T) when v_reset ≈ v_thresh)
//! - **Peak detection**: spike triggers at V_peak (not V_thresh) — the exponential
//!   term creates the spike upstroke naturally, peak detection catches it
//! - **Pre-inverted inv_delta_t**: no division in hot path

#![allow(clippy::module_inception)] // deliberate: `pub mod adex` mirrors the file module

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

/// Adaptive Exponential Integrate-and-Fire Neuron.
///
/// # Fields
/// - `#[state]` vars: membrane potential `v`, adaptation variable `w`,
///   refractory counter `refr`
/// - `#[param]` vars: `dt`, `inv_tau_m`, `inv_tau_w`, `v_rest`, `v_thresh`,
///   `v_reset`, `delta_t`, `inv_delta_t`, `a`, `b`, `tau_ref`, `v_peak`
#[dual_neuron]
pub mod adex {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct AdExModel {
        #[state]
        pub v: f32,
        #[state]
        pub w: f32,
        #[state]
        pub refr: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub inv_tau_m: f32,
        #[param]
        pub inv_tau_w: f32,
        #[param]
        pub v_rest: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub v_reset: f32,
        #[param]
        pub delta_t: f32,
        #[param]
        pub inv_delta_t: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub tau_ref: f32,
        #[param]
        pub v_peak: f32,
    }

    impl ComputeNeuron for AdExModel {
        type State<S: SimdLane> = AdExState<S>;
        type Params<S: SimdLane> = AdExParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut AdExState<S>,
            params: &AdExParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let is_refr = state.refr.cmp_gt(zero);

            // ─── Refractory clamp at INPUT ─────────────────────────────
            // Clamping v to v_reset BEFORE the exponential computation
            // prevents NaN from exp((v_reset - v_thresh) / Δ_T) edge cases.
            let v = is_refr.cond().then(params.v_reset).else_(state.v);

            // ─── Exponential spike term ─────────────────────────────────
            // Pure SIMD: no division, (v - v_th) * inv_delta_t.
            // Clamp arg to [-10, 10] to prevent exp overflow.
            let exp_arg = (v - params.v_thresh) * params.inv_delta_t;
            let exp_val = exp_arg.max(S::splat(-10.0)).min(S::splat(10.0)).exp();
            let exp_term = params.delta_t * exp_val;

            // ─── Membrane potential (Forward Euler) ─────────────────────
            // dv = (v_rest - v + Δ_T·exp((v-V_th)/Δ_T) - w + I) · inv_τ_m · dt
            let dv = (params.v_rest - v + exp_term - state.w + ctx.i_syn)
                * params.inv_tau_m
                * ctx.dt_simd();
            let v_new = v + dv;

            // ─── Adaptation variable ────────────────────────────────────
            // dw = (a·(v - v_rest) - w) · inv_τ_w · dt
            // NOTE: uses pre-refractory v for stability, not v_new
            let dw = (params.a * (v - params.v_rest) - state.w) * params.inv_tau_w * ctx.dt_simd();
            let w_new = state.w + dw;

            // ─── Spike detection at peak threshold ──────────────────────
            // V_peak > V_thresh — catches the spike after the exponential
            // upstroke, not at the subthreshold exponential foot.
            let spike = v_new.cmp_ge(params.v_peak);
            ctx.emit_spike(spike);

            // ─── Reset on spike ─────────────────────────────────────────
            state.v = spike.cond().then(params.v_reset).else_(v_new);
            state.w = spike.cond().then(w_new + params.b).else_(w_new);

            // ─── Refractory timer ───────────────────────────────────────
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::adex::AdExModel` etc.
pub use adex::*;

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * AdExModel::N_PARAMS];
        for i in 0..n {
            p[AdExModelParamIdx::dt as usize * n + i] = 0.05;
            p[AdExModelParamIdx::inv_tau_m as usize * n + i] = 1.0 / 5.0;
            p[AdExModelParamIdx::inv_tau_w as usize * n + i] = 1.0 / 100.0;
            p[AdExModelParamIdx::v_rest as usize * n + i] = -65.0;
            p[AdExModelParamIdx::v_thresh as usize * n + i] = -50.0;
            p[AdExModelParamIdx::v_reset as usize * n + i] = -65.0;
            p[AdExModelParamIdx::delta_t as usize * n + i] = 2.0;
            p[AdExModelParamIdx::inv_delta_t as usize * n + i] = 0.5;
            p[AdExModelParamIdx::a as usize * n + i] = 0.5;
            p[AdExModelParamIdx::b as usize * n + i] = 7.0;
            p[AdExModelParamIdx::tau_ref as usize * n + i] = 2.0;
            p[AdExModelParamIdx::v_peak as usize * n + i] = 0.0;
        }
        p
    }

    #[test]
    fn test_adex_spikes_with_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut w = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![15.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<AdExModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx should spike with I=15");
    }

    #[test]
    fn test_adex_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut w = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for _ in 0..500 {
            unsafe {
                run_neuron::<AdExModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        assert_eq!(
            spikes.iter().sum::<i32>(),
            0,
            "AdEx should not spike with I=0"
        );
    }

    #[test]
    fn test_adex_non_aligned() {
        let n = 11;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut w = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![20.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<AdExModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx n=11 should spike");
    }

    #[test]
    fn test_adex_population() {
        let mut pop = NeuralPopulation::<AdExModel>::new(8);
        pop.fill_param(0, 0.05);
        pop.fill_param(1, 1.0 / 5.0);
        pop.fill_param(2, 1.0 / 100.0);
        pop.fill_param(3, -65.0);
        pop.fill_param(4, -50.0);
        pop.fill_param(5, -65.0);
        pop.fill_param(6, 2.0);
        pop.fill_param(7, 0.5);
        pop.fill_param(8, 0.5);
        pop.fill_param(9, 7.0);
        pop.fill_param(10, 2.0);
        pop.fill_param(11, 0.0);
        pop.fill_state(0, -65.0);

        let i_in = vec![15.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;

        for _ in 0..1000 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "AdEx pop should spike with I=15, got {total}");
    }

    #[test]
    fn test_adex_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut w = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<AdExModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test]
    fn test_adex_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut w = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<AdExModel>(
                    v.as_mut_ptr(),
                    w.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite");
            }
        }
    }
}
