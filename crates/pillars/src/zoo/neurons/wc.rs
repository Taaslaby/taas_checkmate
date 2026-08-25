//! -------------------------------------------------------------------------
//! Filename:    wc.rs
//! Class:       Neural zoo
//! Purpose:     Wilson-Cowan (WC) — excitatory-inhibitory rate model
//! Description:
//!
//! Population-level rate dynamics: E (excitatory) and I (inhibitory) populations.
//! No spikes — continuous rate output via sigmoid transfer function.
//!
//! # Equations
//! ```text
//! τ_E·dE/dt = -E + σ(w_EE·E - w_EI·I + I_ext)
//! τ_I·dI/dt = -I + σ(w_IE·E - w_II·I)
//! σ(x) = 1/(1+exp(-x/β))   (β is a width parameter, not a gain: the code
//! uses `sigmoid(x, 0, β)`, which divides the input by β)
//! ```
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! coder: Anton Atlasov
//! -------------------------------------------------------------------------

#![allow(clippy::module_inception)] // deliberate: `pub mod wc` mirrors the file module

use pillars_macros::dual_neuron;

/// Wilson-Cowan (1972) — excitatory-inhibitory rate model.
/// No spiking output (HAS_SPIKE = false).
#[dual_neuron]
pub mod wc {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct WCModel {
        #[state]
        pub e: f32, // excitatory population rate
        #[state]
        pub i: f32, // inhibitory population rate

        #[param]
        pub dt: f32,
        #[param]
        pub inv_tau_e: f32,
        #[param]
        pub inv_tau_i: f32,
        #[param]
        pub w_ee: f32,
        #[param]
        pub w_ei: f32,
        #[param]
        pub w_ie: f32,
        #[param]
        pub w_ii: f32,
        #[param]
        pub beta: f32,
    }

    impl ComputeNeuron for WCModel {
        type State<S: SimdLane> = WCState<S>;
        type Params<S: SimdLane> = WCParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let input_e = params.w_ee * state.e - params.w_ei * state.i + ctx.i_syn;
            let input_i = params.w_ie * state.e - params.w_ii * state.i;

            let beta_s = params.beta.to_array()[0];
            let f_e = S::sigmoid(input_e, 0.0, beta_s);
            let f_i = S::sigmoid(input_i, 0.0, beta_s);

            state.e = state.e + (-state.e + f_e) * params.inv_tau_e * ctx.dt_simd();
            state.i = state.i + (-state.i + f_i) * params.inv_tau_i * ctx.dt_simd();
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::wc::WCModel` etc.
pub use wc::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * WCModel::N_PARAMS];
        for i in 0..n {
            p[i] = 0.1;
            p[n + i] = 0.1;
            p[2 * n + i] = 0.1;
            p[3 * n + i] = 1.2;
            p[4 * n + i] = 0.4;
            p[5 * n + i] = 0.6;
            p[6 * n + i] = 0.8;
            p[7 * n + i] = 4.0;
        }
        p
    }

    #[test]
    fn test_wc_converges_to_equilibrium() {
        let n = 8;
        let params = make_params(n);
        let mut e = vec![0.0; n];
        let mut i = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_e = vec![0.5; n];
        let mut spikes = vec![0i32; n];
        for _ in 0..500 {
            unsafe {
                run_neuron::<WCModel>(
                    e.as_mut_ptr(),
                    i.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_e.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        for idx in 0..n {
            assert!(
                e[idx] > 0.0 && e[idx] < 1.0,
                "E[{idx}]={} should be in (0,1)",
                e[idx]
            );
            assert!(
                i[idx] > 0.0 && i[idx] < 1.0,
                "I[{idx}]={} should be in (0,1)",
                i[idx]
            );
        }
    }

    #[test]
    fn test_wc_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut e = vec![0.0; n];
        let mut i = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_e = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<WCModel>(
                    e.as_mut_ptr(),
                    i.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_e.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &e_i) in e.iter().enumerate() {
                assert!(!e_i.is_nan(), "E[{i}] NaN {step}");
            }
        }
    }
}
