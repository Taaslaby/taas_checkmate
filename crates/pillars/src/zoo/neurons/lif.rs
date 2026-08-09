//! Leaky Integrate-and-Fire (LIF) Neuron Model
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: no divisions, pure FMA arithmetic
//!
//! Params: [0]dt, [1]inv_tau_m, [2]v_rest, [3]v_thresh, [4]v_reset, [5]tau_ref
//! State: [0]v, [1]refractory_counter

use pillars_macros::dual_neuron;

#[dual_neuron]
pub mod lif {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// LIF neuron: membrane potential with leak + spike + refractory.
    #[derive(NeuronModel)]
    pub struct LIFModel {
        #[state]
        pub v: f32,
        #[state]
        pub refr: f32,

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
    }

    impl ComputeNeuron for LIFModel {
        type State<S: SimdLane> = LIFState<S>;
        type Params<S: SimdLane> = LIFParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut LIFState<S>,
            params: &LIFParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let is_refr = state.refr.cmp_gt(zero);

            // ─── Legacy parity: absolute refractory FREEZES v (no integration, no spike) ───
            // Legacy kernels (kernels::neuron::lif) skip dynamics entirely while refr > 0:
            //   vv = refr_mask.blend(vv, vv + dv);   sm = (vv >= v_th) & !refr
            // Clamping v to v_reset at INPUT (old design) let v drift during refractory and
            // allowed spike breakthrough at high currents — the exact bug fixed here.

            // Forward Euler on the raw v (not clamped)
            let dv = (params.v_rest - state.v + ctx.i_syn) * params.inv_tau_m * ctx.dt();
            let v_new = state.v + dv;

            // Spike detection — gated on NOT refractory (legacy: sm & !refr)
            let spike_raw = v_new.cmp_ge(params.v_thresh);
            let spike = is_refr.cond().then(zero).else_(spike_raw);
            ctx.emit_spike(spike);

            // On spike reset to v_reset; during refractory v stays FROZEN
            let v_final = spike.cond().then(params.v_reset).else_(v_new);
            state.v = is_refr.cond().then(state.v).else_(v_final);

            // Refractory timer
            let refr_dec = state.refr - ctx.dt();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

pub use lif::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * LIFModel::N_PARAMS];
        for i in 0..n {
            p[i] = 0.5;
            p[n + i] = 1.0 / 10.0;
            p[2 * n + i] = -70.0;
            p[3 * n + i] = -54.0;
            p[4 * n + i] = -70.0;
            p[5 * n + i] = 2.0;
        }
        p
    }

    #[test]
    fn test_lif_basic_spiking() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-70.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..100 {
            unsafe {
                run_neuron::<LIFModel>(
                    v.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
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
        assert!(total > 0, "LIF should spike with I=30, got {total}");
    }

    #[test]
    fn test_lif_population_spiking() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 1.0 / 10.0);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_in = vec![30.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;

        for _ in 0..100 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(
            total > 0,
            "LIF population should spike with I=30, got {total}"
        );
    }

    #[test]
    fn test_lif_population_non_aligned() {
        let mut pop = NeuralPopulation::<LIFModel>::new(10);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 1.0 / 10.0);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_in = vec![30.0; 10];
        let mut spikes = vec![0i32; 10];

        for _ in 0..20 {
            pop.step(&i_in, &mut spikes);
        }

        let v8 = pop.state_slice(0)[8];
        let v0 = pop.state_slice(0)[0];
        assert!(
            (v8 - v0).abs() < 0.001,
            "Tail v[8]={:.3} != SIMD v[0]={:.3}",
            v8,
            v0
        );
    }
}
