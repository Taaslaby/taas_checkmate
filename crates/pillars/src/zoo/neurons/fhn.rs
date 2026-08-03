//! FitzHugh-Nagumo (FHN) Neuron Model
//!
//! 2D simplification of Hodgkin-Huxley: `v` (membrane potential) and `w` (recovery).
//! Edge-triggered spike detection.
//!
//! # Equations
//! ```text
//! dv/dt = v - v³/3 - w + I
//! dw/dt = (v + a - b·w) / τ_w
//! ```
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};

/// FitzHugh-Nagumo (1961) — simplified 2D neuron model.
#[derive(NeuronModel)]
pub struct FHNModel {
    #[state] pub v: f32,
    #[state] pub w: f32,
    #[state] pub was_below: f32,

    #[param] pub dt: f32,
    #[param] pub inv_tau_w: f32,
    #[param] pub a: f32,
    #[param] pub b: f32,
    #[param] pub v_thresh: f32,
}

impl ComputeNeuron for FHNModel {
    type State<S: SimdLane> = FHNState<S>;
    type Params<S: SimdLane> = FHNParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut Self::State<S>,
        params: &Self::Params<S>,
        ctx: &mut StepContext<S>,
    ) {
        // Horner's method: v - v³/3 = v·(1 - v²/3)
        // Saves 1 vmulps: 3 multiplications → 2 multiplications
        let one_third = S::splat(1.0 / 3.0);
        let v_sq = state.v.square();  // 1 multiply
        let v_poly = state.v * (S::one() - v_sq * one_third);  // 2 multiplies total

        let dv = (v_poly - state.w + ctx.i_syn) * ctx.dt_simd();
        let dw = (state.v + params.a - params.b * state.w) * params.inv_tau_w * ctx.dt_simd();

        let v_new = state.v + dv;
        let w_new = state.w + dw;

        // Edge-triggered spike
        let is_above = v_new.cmp_ge(params.v_thresh);
        let was_below = state.was_below.cmp_gt(S::zero());
        let spike = is_above.and(was_below);
        ctx.emit_spike(spike);

        // Update was_below flag for next step
        // was_below = 1 if v < thresh, 0 if v >= thresh
        let flag = is_above.cond().then(S::zero()).else_(S::one());

        state.v = v_new;
        state.w = w_new;
        state.was_below = flag;
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * FHNModel::N_PARAMS];
        for i in 0..n {
            p[i] = 0.01;
            p[n + i] = 1.0 / 10.0;
            p[2 * n + i] = 0.7;
            p[3 * n + i] = 0.8;
            p[4 * n + i] = 0.5;
        }
        p
    }

    #[test]
    fn test_fhn_spikes_with_current() {
        let n = 8; let params = make_params(n);
        let mut v = vec![0.0; n]; let mut w = vec![0.0; n]; let mut flag = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.5; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe { run_neuron::<FHNModel>(
                v.as_mut_ptr(), w.as_mut_ptr(), flag.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "FHN should spike with I=0.5");
    }

    #[test]
    fn test_fhn_silent_without_current() {
        let n = 8; let params = make_params(n);
        let mut v = vec![0.0; n]; let mut w = vec![0.0; n]; let mut flag = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for _ in 0..1000 {
            unsafe { run_neuron::<FHNModel>(
                v.as_mut_ptr(), w.as_mut_ptr(), flag.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "FHN should not spike with I=0");
    }

    #[test]
    fn test_fhn_stress_silence() {
        let n = 8; let params = make_params(n);
        let mut v = vec![0.0; n]; let mut w = vec![0.0; n]; let mut flag = vec![1.0; n];
        let mut dummy = vec![0.0; n]; let I = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<FHNModel>(
                v.as_mut_ptr(), w.as_mut_ptr(), flag.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for (i, &v_i) in v.iter().enumerate() { assert!(!v_i.is_nan(), "v[{i}] NaN {step}"); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }
}
