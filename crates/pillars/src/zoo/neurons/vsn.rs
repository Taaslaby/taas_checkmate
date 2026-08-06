//! VSN — Variable Spiking Neuron
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Surrogate gradient**: stored as real state, not lost — enables training
//! - **Leakage mixing**: v = β·v_new + (1−β)·v_rest

#![allow(clippy::module_inception)] // deliberate: `pub mod vsn` mirrors the file module

use crate::core::context::SimdContext; // used by the surrogate_output helper below
use crate::core::lane::SimdLane; // used by the surrogate_output helper below
use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum VSNParamIdx {
    DT = 0,             // ms
    Inv_Tau_M = 1,      // 1/τ_m (ms⁻¹)
    V_Rest = 2,         // mV
    V_Thresh = 3,       // mV
    V_Reset = 4,        // mV
    Tau_Ref = 5,        // ms
    Beta = 6,           // leakage mixing factor
    Surrogate_Gain = 7, // gain for surrogate gradient
}

impl From<VSNParamIdx> for usize {
    #[inline(always)]
    fn from(idx: VSNParamIdx) -> Self {
        idx as usize
    }
}

/// Variable Spiking Neuron with a stored surrogate gradient.
///
/// # States (3)
/// - `v`: membrane potential (mV)
/// - `refr`: refractory counter (ms)
/// - `surrogate`: surrogate gradient (1/(1+|v−θ|)²)
///
/// # Params (8)
/// - `dt`, `inv_tau_m`, `v_rest`, `v_thresh`, `v_reset`,
///   `tau_ref`, `beta`, `surrogate_gain`
#[dual_neuron]
pub mod vsn {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct VSNModel {
        #[state]
        pub v: f32,
        #[state]
        pub refr: f32,
        #[state]
        pub surrogate: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub inv_tau_m: f32,
        #[param]
        pub v_rest: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub v_reset: f32,
        #[param]
        pub tau_ref: f32,
        #[param]
        pub beta: f32,
        /// Gain for surrogate gradient — declared for the trainer, unused in step.
        #[param]
        pub surrogate_gain: f32,
    }

    impl ComputeNeuron for VSNModel {
        type State<S: SimdLane> = VSNState<S>;
        type Params<S: SimdLane> = VSNParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut VSNState<S>,
            params: &VSNParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let one = S::one();

            // ─── Refractory clamp at INPUT ─────────────────────────────
            let is_refr = state.refr.cmp_gt(zero);
            let v = is_refr.cond().then(params.v_reset).else_(state.v);

            // ─── LIF membrane ──────────────────────────────────────────
            let dv = (params.v_rest - v + ctx.i_syn) * params.inv_tau_m * ctx.dt_simd();
            let v_new = v + dv;

            // ─── Leakage mixing: v = β·v_new + (1−β)·v_rest ────────────
            let v_mix = params.beta * v_new + (one - params.beta) * params.v_rest;

            // ─── Surrogate gradient — stored in state (not lost!) ──────
            let diff = (v_mix - params.v_thresh).abs();
            let surrogate = one / ((one + diff) * (one + diff));
            state.surrogate = surrogate;

            // ─── Spike detection & reset ───────────────────────────────
            let spike = v_mix.cmp_ge(params.v_thresh);
            ctx.emit_spike(spike);
            state.v = spike.cond().then(params.v_reset).else_(v_mix);

            // ─── Refractory counter ────────────────────────────────────
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::vsn::VSNModel` etc.
pub use vsn::*;

impl VSNModel {
    /// Read the surrogate gradient written by the last step.
    #[inline(always)]
    pub fn surrogate_output<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        ctx.s(2)
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 8];
        for i in 0..n {
            p[VSNParamIdx::DT as usize * n + i] = 0.5;
            p[VSNParamIdx::Inv_Tau_M as usize * n + i] = 1.0 / 10.0;
            p[VSNParamIdx::V_Rest as usize * n + i] = -70.0;
            p[VSNParamIdx::V_Thresh as usize * n + i] = -54.0;
            p[VSNParamIdx::V_Reset as usize * n + i] = -70.0;
            p[VSNParamIdx::Tau_Ref as usize * n + i] = 2.0;
            p[VSNParamIdx::Beta as usize * n + i] = 0.9;
            p[VSNParamIdx::Surrogate_Gain as usize * n + i] = 10.0;
        }
        p
    }

    #[test]
    fn test_vsn_spikes_with_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut surrogate = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![100.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..500 {
            unsafe {
                run_neuron::<VSNModel>(
                    v.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    surrogate.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_syn.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "VSN should spike with I=100, got {total}");
        assert!(
            surrogate[0] != 0.0,
            "surrogate gradient should be non-zero, got {:.4}",
            surrogate[0]
        );
    }

    #[test]
    fn test_vsn_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut surrogate = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for _ in 0..100 {
            unsafe {
                run_neuron::<VSNModel>(
                    v.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    surrogate.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_syn.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        assert_eq!(
            spikes.iter().sum::<i32>(),
            0,
            "VSN should not spike with I=0"
        );
    }

    #[test]
    fn test_vsn_beta_variation() {
        let n = 8;
        let mut p_high = make_params(n);
        let mut p_low = make_params(n);
        for i in 0..n {
            p_high[VSNParamIdx::Beta as usize * n + i] = 0.99;
            p_low[VSNParamIdx::Beta as usize * n + i] = 0.01;
        }
        let mut v_high = vec![-70.0; n];
        let mut v_low = vec![-70.0; n];
        let mut refr_h = vec![0.0; n];
        let mut refr_l = vec![0.0; n];
        let mut surr_h = vec![0.0; n];
        let mut surr_l = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![100.0; n];
        let mut spikes_h = vec![0i32; n];
        let mut spikes_l = vec![0i32; n];

        for _ in 0..200 {
            unsafe {
                run_neuron::<VSNModel>(
                    v_high.as_mut_ptr(),
                    refr_h.as_mut_ptr(),
                    surr_h.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_syn.as_ptr(),
                    spikes_h.as_mut_ptr(),
                    p_high.as_ptr(),
                    n,
                );
            }
            unsafe {
                run_neuron::<VSNModel>(
                    v_low.as_mut_ptr(),
                    refr_l.as_mut_ptr(),
                    surr_l.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_syn.as_ptr(),
                    spikes_l.as_mut_ptr(),
                    p_low.as_ptr(),
                    n,
                );
            }
        }
        let total_h: i32 = spikes_h.iter().sum();
        let total_l: i32 = spikes_l.iter().sum();
        assert!(total_h > 0, "High-beta should spike, got total_h={total_h}");
        assert!(
            total_h > total_l,
            "High-beta ({total_h}) should spike more than low-beta ({total_l})"
        );
    }

    #[test]
    fn test_vsn_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut surr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<VSNModel>(
                    v.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    surr.as_mut_ptr(),
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
                assert!(!v_i.is_nan(), "v[{i}] NaN {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test]
    fn test_vsn_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut surr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<VSNModel>(
                    v.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    surr.as_mut_ptr(),
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
                assert!(!v_i.is_nan(), "v[{i}] NaN {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite {step}");
            }
        }
    }
}
