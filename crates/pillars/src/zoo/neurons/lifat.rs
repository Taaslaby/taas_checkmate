//! -------------------------------------------------------------------------
//! Filename:    lifat.rs
//! Class:       Neural zoo
//! Purpose:     LIF-AT (LIF with Adaptive Threshold) — GLIF2-equivalent.
//! Description:
//!
//! Extends LIF with a dynamic threshold offset `θ(t)` (spike-frequency
//! adaptation, Allen Institute GLIF2 family):
//! ```text
//! τ_m·dv/dt = -(v - v_rest) + I            (Forward Euler, frozen in refractory)
//! θ(t+dt)   = θ(t)·exp(-dt/τ_θ)            (exponential decay — ALWAYS, even in refractory)
//! if v ≥ v_thresh + θ: spike; v ← v_reset; θ ← θ + β; refr ← τ_ref
//! ```
//!
//! Canonical port of the legacy `kernels::lif_at_step` (same semantics:
//! θ decays during refractory, spike only when non-refractory). The legacy
//! kernel used the `fast_exp` polynomial for θ decay (~1e-5 rel err); this
//! model uses exact `exp` — guarded by `crates/kernels/tests/lifat_parity.rs`
//! (θ within 1e-3 rel, spike counts equal).
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! coder: Anton Atlasov
//! -------------------------------

#![allow(clippy::module_inception)] // deliberate: `pub mod lifat` mirrors the file module

use pillars_macros::dual_neuron;

/// LIF-AT (GLIF2) — LIF + adaptive threshold θ.
#[dual_neuron]
pub mod lifat {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct LIFAtModel {
        #[state]
        pub v: f32,
        #[state]
        pub theta: f32,
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
        pub inv_tau_theta: f32,
        #[param]
        pub beta: f32,
        #[param]
        pub tau_ref: f32,
    }

    impl ComputeNeuron for LIFAtModel {
        type State<S: SimdLane> = LIFAtState<S>;
        type Params<S: SimdLane> = LIFAtParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let is_refr = state.refr.cmp_gt(zero);

            // v — Forward Euler, frozen during refractory (legacy parity).
            // inv_tau_m == 0 (τ_m ≤ 0) → no integration, same as legacy guard.
            let dv = (params.v_rest - state.v + ctx.i_syn) * params.inv_tau_m * ctx.dt_simd();
            let v_new = is_refr.cond().then(state.v).else_(state.v + dv);

            // θ — exponential decay ALWAYS (even during refractory).
            let theta_decay = (-ctx.dt_simd() * params.inv_tau_theta).exp();
            let theta_new = state.theta * theta_decay;

            // Spike: v ≥ v_thresh + θ, only non-refractory.
            let spike = is_refr
                .cond()
                .then(zero)
                .else_(v_new.cmp_ge(params.v_thresh + theta_new));
            ctx.emit_spike(spike);

            state.v = spike.cond().then(params.v_reset).else_(v_new);
            // θ инкремент — через blend, НЕ арифметикой над маской: маска спайка
            // это NaN(true)/0(false), и `spike * beta` протёк бы NaN в θ.
            state.theta = spike.cond().then(theta_new + params.beta).else_(theta_new);

            // Refractory counter: decrement always, set to τ_ref on spike.
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::lifat::LIFAtModel` etc.
pub use lifat::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;

    const DT: f32 = 0.5;
    const TAU_M: f32 = 10.0;
    const V_REST: f32 = -70.0;
    const V_THRESH: f32 = -54.0;
    const V_RESET: f32 = -70.0;
    const TAU_THETA: f32 = 20.0;
    const BETA: f32 = 5.0;
    const TAU_REF: f32 = 3.0;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * LIFAtModel::N_PARAMS];
        for i in 0..n {
            p[i] = DT;
            p[n + i] = 1.0 / TAU_M;
            p[2 * n + i] = V_REST;
            p[3 * n + i] = V_THRESH;
            p[4 * n + i] = V_RESET;
            p[5 * n + i] = 1.0 / TAU_THETA;
            p[6 * n + i] = BETA;
            p[7 * n + i] = TAU_REF;
        }
        p
    }

    fn run(n: usize, I: f32, steps: usize) -> (Vec<f32>, Vec<f32>, i32) {
        let params = make_params(n);
        let mut v = vec![V_REST; n];
        let mut theta = vec![BETA * 2.0; n]; // старт с поднятым порогом
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![I; n];
        let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..steps {
            unsafe {
                run_neuron::<LIFAtModel>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            total += spikes.iter().sum::<i32>();
        }
        (v, theta, total)
    }

    /// θ затухает экспоненциально без спайков: θ·exp(-dt/τ_θ).
    #[test]
    fn test_lifat_theta_decays_without_spike() {
        let (_, theta, total) = run(8, 0.0, 1);
        let expected = (BETA * 2.0) * (-DT / TAU_THETA).exp();
        for &t in &theta {
            assert!(
                (t - expected).abs() < 1e-4,
                "θ после 1 шага: {t}, ожидалось {expected}"
            );
        }
        assert_eq!(total, 0, "θ-затухание не должно давать спайков при I=0");
    }

    /// Сильный ток → спайки, θ инкрементируется на β после спайка.
    #[test]
    fn test_lifat_spikes_and_increments_theta() {
        let (v, theta, total) = run(8, 40.0, 500);
        assert!(total > 0, "LIF-AT должен спайкать при I=40");
        assert!(
            theta.iter().any(|&t| t > 0.3),
            "θ должна была инкрементироваться после спайков: {theta:?}"
        );
        for &vv in &v {
            assert!(vv.is_finite() && !vv.is_nan(), "v non-finite");
        }
    }

    /// Абсолютная рефрактерность: спайк возможен только если нейрон НЕ входил
    /// в шаг в рефрактерности (нет back-to-back).
    #[test]
    fn test_lifat_no_back_to_back_spikes() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![V_REST; n];
        let mut theta = vec![BETA * 2.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1000.0; n]; // гигантский ток (клампится раннером до ±100)
        let mut spikes = vec![0i32; n];

        let mut fired = false;
        for _ in 0..2000 {
            let refr_before = refr.clone();
            unsafe {
                run_neuron::<LIFAtModel>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                if spikes[i] != 0 {
                    assert!(
                        refr_before[i] <= 0.0,
                        "back-to-back спайк внутри рефрактерного окна: refr_before={}",
                        refr_before[i]
                    );
                    fired = true;
                }
            }
        }
        assert!(fired, "LIF-AT должен спайкать хотя бы раз при I=1000");
    }

    /// Стресс: никаких NaN при огромном токе.
    #[test]
    fn test_lifat_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![V_REST; n];
        let mut theta = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_in = vec![1000.0; n];
        let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe {
                run_neuron::<LIFAtModel>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert!(
                    v[i].is_finite() && theta[i].is_finite(),
                    "non-finite at step {step}: v={} θ={}",
                    v[i],
                    theta[i]
                );
            }
        }
    }
}
