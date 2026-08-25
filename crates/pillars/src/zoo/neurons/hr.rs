//! -------------------------------------------------------------------------
//! Filename:    hr.rs
//! Class:       Neural zoo
//! Purpose:     Hindmarsh-Rose (HR) — 3-variable bursting neuron
//! Description:
//!
//! 100% polynomial — zero exp() calls. Pure SIMD multiplications.
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! coder: Anton Atlasov
//! -------------------------------------------------------------------------

#![allow(clippy::module_inception)] // deliberate: `pub mod hr` mirrors the file module

use pillars_macros::dual_neuron;

/// Hindmarsh-Rose (1984) — 3-variable bursting neuron.
/// States: x (membrane), y (recovery), z (slow adaptation).
/// Pure polynomial dynamics — no exp() calls.
#[dual_neuron]
pub mod hr {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct HRModel {
        #[state]
        pub x: f32,
        #[state]
        pub y: f32,
        #[state]
        pub z: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub c: f32,
        #[param]
        pub d: f32,
        #[param]
        pub r: f32,
        #[param]
        pub s: f32,
        #[param]
        pub x_r: f32,
        #[param]
        pub v_thresh: f32,
    }

    impl ComputeNeuron for HRModel {
        type State<S: SimdLane> = HRState<S>;
        type Params<S: SimdLane> = HRParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let x2 = state.x * state.x;

            // Horner's method: dx/dt = y + x²·(b - a·x) - z + I
            let poly_x = x2 * (params.b - params.a * state.x);
            let dx = state.y + poly_x - state.z + ctx.i_syn;
            let dy = params.c - params.d * x2 - state.y;
            let dz = params.r * (params.s * (state.x - params.x_r) - state.z);

            state.x = state.x + dx * ctx.dt_simd();
            state.y = state.y + dy * ctx.dt_simd();
            state.z = state.z + dz * ctx.dt_simd();

            let spike = state.x.cmp_ge(params.v_thresh);
            ctx.emit_spike(spike);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::hr::HRModel` etc.
pub use hr::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * HRModel::N_PARAMS];
        for i in 0..n {
            p[i] = 0.01;
            p[n + i] = 1.0;
            p[2 * n + i] = 3.0;
            p[3 * n + i] = 1.0;
            p[4 * n + i] = 5.0;
            p[5 * n + i] = 0.006;
            p[6 * n + i] = 4.0;
            p[7 * n + i] = -1.6;
            p[8 * n + i] = 1.0;
        }
        p
    }

    #[test]
    fn test_hr_bursts_with_current() {
        let n = 8;
        let params = make_params(n);
        let mut x = vec![-1.6; n];
        let mut y = vec![-12.0; n];
        let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![3.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..10000 {
            unsafe {
                run_neuron::<HRModel>(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    z.as_mut_ptr(),
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
        assert!(total > 0, "HR should burst with I=3");
    }

    #[test]
    fn test_hr_resting_no_spikes() {
        let n = 8;
        let params = make_params(n);
        let mut x = vec![-1.6; n];
        let mut y = vec![-12.0; n];
        let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for _ in 0..1000 {
            unsafe {
                run_neuron::<HRModel>(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    z.as_mut_ptr(),
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
            "HR should not spike with I=0"
        );
    }

    #[test]
    fn test_hr_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut x = vec![-1.6; n];
        let mut y = vec![-12.0; n];
        let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<HRModel>(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    z.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &x_i) in x.iter().enumerate() {
                assert!(!x_i.is_nan(), "x[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test]
    fn test_hr_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut x = vec![-1.6; n];
        let mut y = vec![-12.0; n];
        let mut z = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<HRModel>(
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    z.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for (i, &x_i) in x.iter().enumerate() {
                assert!(!x_i.is_nan(), "x[{i}] NaN step {step}");
                assert!(x_i.is_finite(), "x[{i}] non-finite");
            }
        }
    }
}
