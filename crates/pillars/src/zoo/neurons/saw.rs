//! -------------------------------------------------------------------------
//! Filename:    saw.rs
//! Class:       Neural zoo
//! Purpose:     SAW (Spike-Adaptive-Width) — Izhikevich-like neuron.
//! Description:
//!
//! Izhikevich-style quadratic neuron with polar-wave trace dynamics.
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! coder: Anton Atlasov
//! -------------------------------

#![allow(clippy::module_inception)] // deliberate: `pub mod saw` mirrors the file module

use pillars_macros::dual_neuron;

/// State: v (membrane), u (recovery), tr (trace), refr (refractory counter)
#[dual_neuron]
pub mod saw {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct SAWModel {
        #[state]
        pub v: f32,
        #[state]
        pub u: f32,
        #[state]
        pub tr: f32,
        #[state]
        pub refr: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub trace_decay: f32,
        #[param]
        pub noise_amp: f32,
        #[param]
        pub tau_ref: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub v_reset: f32,
    }

    impl ComputeNeuron for SAWModel {
        type State<S: SimdLane> = SAWState<S>;
        type Params<S: SimdLane> = SAWParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let one = S::one();
            let is_refr = state.refr.cmp_gt(zero).cond();

            // ─── Legacy parity: absolute refractory FREEZES v (no integration, no spike) ───
            // Легаси batch-кернел (удалён в волну 7) пропускал динамику в рефрактерности:
            //   vv += refr_mask.blend(zero, dt·dv);  sm = (vv >= v_th) & !refr
            // Clamping v to v_reset at INPUT (old design) let v drift during refractory.
            let v = state.v;

            // ─── Single Euler step — EXACT legacy parity ───
            // Legacy kernels use ONE forward-Euler step (no substepping):
            //   dv = 0.04·v² + 5·v + 140 − u + i;  vv += dt·dv
            // (The old pillars double-substep drifted spike timing by ~10% vs legacy.)
            let c04 = S::splat(0.04);
            let c5 = S::splat(5.0);
            let c140 = S::splat(140.0);

            let v_sq = v * v;
            let dv = c04.mul_add(v_sq, c5 * v) + c140 - state.u + ctx.i_syn;
            let v_new = v + dv * ctx.dt_simd();

            // Recovery variable (only outside refractory)
            let du = params.a * (params.b * v - state.u);
            let u_new = is_refr.then(state.u).else_(state.u + du * ctx.dt_simd());

            // Spike detection (no spiking in refractory)
            let spike_raw = v_new.cmp_ge(params.v_thresh);
            let spike = is_refr.then(zero).else_(spike_raw);
            ctx.emit_spike(spike);

            // State updates — v frozen during refractory (legacy parity)
            let v_final = spike.cond().then(params.v_reset).else_(v_new);
            state.v = is_refr.then(state.v).else_(v_final);
            state.u = u_new + spike.cond().then(S::splat(8.0)).else_(zero);

            // polar-wave trace (frozen during refractory — legacy parity)
            // Legacy: on spike tr = tr·td + (1 − tr); else tr·(1 − td)
            let tr_inc = state.tr * params.trace_decay + (one - state.tr);
            let tr_dec = state.tr * (one - params.trace_decay);
            let tr_new = spike.cond().then(tr_inc).else_(tr_dec).max(zero).min(one);
            state.tr = is_refr.then(state.tr).else_(tr_new);

            // Refractory timer
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::saw::SAWModel` etc.
pub use saw::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let mut p = vec![0.0f32; n * SAWModel::N_PARAMS];
        for i in 0..n {
            p[i] = 1.0;
            p[n + i] = 0.02;
            p[2 * n + i] = 0.2;
            p[3 * n + i] = 0.01;
            p[4 * n + i] = 0.0;
            p[5 * n + i] = tau_ref;
            p[6 * n + i] = 30.0;
            p[7 * n + i] = -65.0;
        }
        p
    }

    #[test]
    fn test_saw_spikes_with_current() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![15.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..500 {
            unsafe {
                run_neuron::<SAWModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    tr.as_mut_ptr(),
                    refr.as_mut_ptr(),
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
        assert!(total > 0);
    }

    #[test]
    fn test_saw_silent_without_current() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for _ in 0..500 {
            unsafe {
                run_neuron::<SAWModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    tr.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test]
    fn test_saw_non_aligned_n() {
        let n = 11;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..1000 {
            unsafe {
                run_neuron::<SAWModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    tr.as_mut_ptr(),
                    refr.as_mut_ptr(),
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
        assert!(total > 0);
        for i in 0..n {
            assert!(!v[i].is_nan());
            assert!(!u[i].is_nan());
        }
    }

    #[test]
    fn test_saw_stress_silence() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<SAWModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    tr.as_mut_ptr(),
                    refr.as_mut_ptr(),
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
    fn test_saw_stress_explosion() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut tr = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<SAWModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    tr.as_mut_ptr(),
                    refr.as_mut_ptr(),
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
