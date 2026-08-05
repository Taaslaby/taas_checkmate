//! Rate — simple firing-rate neuron (exponential Euler toward clamped input).
//!
//! The rate model the legacy connectome `sim_rate` used:
//! `r ← r·exp(−dt/τ) + clamp(I, 0, 1)·(1 − exp(−dt/τ))`.
//!
//! `dt` is in **seconds** (the connectome rate config passes `dt_ms / 1000`).
//! The model emits a "spike" (mask 1) whenever the rate crosses `0.5`, so it
//! composes with the pillars spike-routing engine (`Brain`/`Network`).

use pillars_macros::dual_neuron;

#[dual_neuron]
pub mod rate {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// Firing-rate neuron: `r ← r·exp(−dt/τ) + clamp(I, 0, 1)·(1 − exp(−dt/τ))`.
    ///
    /// `dt` is param column 0 (the engine reads it via `ctx.dt_simd()`); the
    /// connectome rate config passes seconds (`dt_ms / 1000`).
    #[derive(NeuronModel)]
    pub struct RateModel {
        #[state]
        pub r: f32,
        #[param]
        pub dt: f32, // seconds
        #[param]
        pub inv_tau: f32,
    }

    impl ComputeNeuron for RateModel {
        type State<S: SimdLane> = RateState<S>;
        type Params<S: SimdLane> = RateParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let decay = (-ctx.dt_simd() * params.inv_tau).exp();
            let input = ctx.i_syn.clamp01();
            let r_new = decay.mul_add(state.r, input * (S::one() - decay));
            state.r = r_new;
            let spike = r_new.cmp_gt(S::splat(0.5));
            ctx.emit_spike(spike);
        }
    }
}

pub use rate::*;
