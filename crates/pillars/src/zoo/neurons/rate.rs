//! -------------------------------------------------------------------------
//! Filename:    rate.rs
//! Class:       Neural zoo
//! Purpose:     Rate — simple firing-rate neuron (exponential Euler toward clamped input)
//! Description:
//!
//! The rate model the legacy connectome `sim_rate` used:
//! `r ← r·exp(−dt/τ) + clamp(I, 0, 1)·(1 − exp(−dt/τ))`.
//!
//! `dt` is in **seconds** (the connectome rate config passes `dt_ms / 1000`).
//! The model emits a "spike" (mask 1) whenever the rate crosses `0.5`, so it
//! composes with the pillars spike-routing engine (`Brain`/`Network`).
//!
//! coder: Anton Atlasov
//! -------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeuralPopulation;

    #[test]
    fn test_rate_integrates_toward_input() {
        let mut pop = NeuralPopulation::<RateModel>::new(8);
        pop.fill_param(RateModelParamIdx::dt.into(), 0.001); // seconds
        pop.fill_param(RateModelParamIdx::inv_tau.into(), 1.0);
        pop.fill_state(RateModelStateIdx::r.into(), 0.0);

        let i_in = vec![1.0f32; 8]; // saturating input rate
        let mut spikes = vec![0i32; 8];

        // One step: r = 0·exp(−dt·inv_tau) + 1·(1 − exp(−dt·inv_tau)) ≈ dt.
        pop.step(&i_in, &mut spikes);
        let r1 = pop.state_slice(0)[0];
        assert!(r1 > 0.0009 && r1 < 0.0011, "r after 1 step = {r1}");

        // Saturate toward 1 (exponential Euler fixed point at input), spiking
        // once r crosses 0.5.
        let mut total = 0i32;
        for _ in 0..5000 {
            pop.step(&i_in, &mut spikes);
            total += spikes[0];
        }
        assert!(
            pop.state_slice(0)[0] > 0.99,
            "r should saturate near 1, got {}",
            pop.state_slice(0)[0]
        );
        assert!(total > 0, "rate neuron must spike after crossing r=0.5");
    }
}
