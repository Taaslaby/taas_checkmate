//! Izhikevich Simple Model of Spiking Neurons
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Refractory at input**: v clamped to reset BEFORE substepping
//! - **Substepping**: two 0.5·dt steps for Forward Euler stability
//! - **Horner**: v·(0.04·v + 5.0) + 140 — polynomial optimization
//! - **Per-type threshold**: params.v_thresh from calibration v2
//!
//! The `IzhikevichModelParamIdx` enum is generated for manual param setting via `fill_param()`.
//!
//! # Do NOT merge with the other two Izhikevich families
//! `izhikevich_2007` (conductance-based, different ODE) and
//! `izhikevich_dual` (single-step Euler, guarded bit-parity) share the
//! `0.04·v² + 5·v + 140` polynomial with this model, but each rounds it
//! differently (substep ×2 here vs one step there). Unifying the step bodies
//! would break the legacy/NeuroML bit-exactness the test suites pin down.
//!
//! # Dual-source (CPU SIMD + CUDA)
//! The `#[dual_neuron]` wrapper emits `IZHIKEVICH_CUDA_SOURCE: &str` — the
//! same step body transpiled to CUDA C++ (NVRTC-ready). `pillars-cuda`
//! imports that string; no CUDA runtime is needed to build `pillars`.

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

#[dual_neuron]
pub mod izhikevich {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    /// Izhikevich Simple Model of Spiking Neurons.
    ///
    /// # Fields
    /// - `#[state]` vars: membrane potential `v`, recovery variable `u`, refractory counter `refr`
    /// - `#[param]` vars: `dt`, `a`, `b`, `c` (reset), `d` (spike bump), `tau_ref`, `v_thresh`
    #[derive(NeuronModel)]
    pub struct IzhikevichModel {
        #[state]
        pub v: f32,
        #[state]
        pub u: f32,
        #[state]
        pub refr: f32,

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
        pub tau_ref: f32,
        #[param]
        pub v_thresh: f32,
    }

    impl ComputeNeuron for IzhikevichModel {
        type State<S: SimdLane> = IzhikevichState<S>;
        type Params<S: SimdLane> = IzhikevichParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut IzhikevichState<S>,
            params: &IzhikevichParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let is_refr = state.refr.cmp_gt(zero);

            // Refractory clamp at INPUT: v = v_reset (c) during refractory
            let v = is_refr.cond().then(params.c).else_(state.v);

            // Substepping: two 0.5·dt steps for Forward Euler stability
            let half_dt = ctx.dt_simd() * S::splat(0.5);

            let c04 = S::splat(0.04);
            let c5 = S::splat(5.0);
            let c140 = S::splat(140.0);

            // Sub-step 1: Horner v·(0.04·v + 5.0)
            let v_poly1 = v * (c04 * v + c5);
            let dv1 = (v_poly1 + c140 - state.u + ctx.i_syn) * half_dt;
            let v_mid = v + dv1;

            // Sub-step 2
            let v_poly2 = v_mid * (c04 * v_mid + c5);
            let dv2 = (v_poly2 + c140 - state.u + ctx.i_syn) * half_dt;

            // Абсолютная рефрактерность: суммарный dv ЗАМОРОЖЕН (v остаётся на
            // уровне c). Иначе при высоком токе v_new пробивает порог прямо внутри
            // рефрактерного периода (back-to-back спайк). Легаси fused-ядро делает
            // то же самое: vv += refr_mask.blend(zero, dt_s * dv).
            let dv_total = is_refr.cond().then(zero).else_(dv1 + dv2);
            let v_new = v + dv_total;

            // u update only when NOT in refractory.
            // Legacy parity: du computed from the PRE-integration v (not v_new) —
            // matching the old SIMD kernel `du = a·(b·v − u)`. Using v_new inflated
            // u at spike (v_new is above threshold), which throttled firing rate by
            // ~20% and broke FS pattern tests (45 vs 57 spikes at I=10).
            let du = params.a * (params.b * v - state.u) * ctx.dt_simd();
            let u_new = is_refr.cond().then(state.u).else_(state.u + du);

            // Spike detection: v_reset < v_thresh prevents false spikes in refractory
            let spike = v_new.cmp_ge(params.v_thresh);
            ctx.emit_spike(spike);

            // Reset on spike
            state.v = spike.cond().then(params.c).else_(v_new); // v = c
            state.u = spike.cond().then(u_new + params.d).else_(u_new); // u += d

            // Refractory timer
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::izhikevich::IzhikevichModel` etc.
pub use izhikevich::*;

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0; n * IzhikevichModel::N_PARAMS];
        for i in 0..n {
            p[IzhikevichModelParamIdx::dt as usize * n + i] = 0.5;
            p[IzhikevichModelParamIdx::a as usize * n + i] = 0.02;
            p[IzhikevichModelParamIdx::b as usize * n + i] = 0.2;
            p[IzhikevichModelParamIdx::c as usize * n + i] = -65.0;
            p[IzhikevichModelParamIdx::d as usize * n + i] = 8.0;
            p[IzhikevichModelParamIdx::tau_ref as usize * n + i] = 2.0;
            p[IzhikevichModelParamIdx::v_thresh as usize * n + i] = 30.0;
        }
        p
    }

    #[test]
    fn test_izhikevich_spikes() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![10.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe {
                run_neuron::<IzhikevichModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
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
        assert!(total > 0, "Izhikevich should spike with I=10, got {total}");
    }

    #[test]
    fn test_izhikevich_silent_without_current() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for _ in 0..100 {
            unsafe {
                run_neuron::<IzhikevichModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
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
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes with I=0");
    }

    #[test]
    fn test_izhikevich_population() {
        let mut pop = NeuralPopulation::<IzhikevichModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.02);
        pop.fill_param(2, 0.2);
        pop.fill_param(3, -65.0);
        pop.fill_param(4, 8.0);
        pop.fill_param(5, 2.0);
        pop.fill_param(6, 30.0);
        pop.fill_state(0, -65.0);
        pop.fill_state(1, 0.0);

        let i_in = vec![10.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;

        for _ in 0..500 {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(
            total > 0,
            "Izh population should spike with I=10, got {total}"
        );
    }

    #[test]
    fn test_izhikevich_stress_silence() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe {
                run_neuron::<IzhikevichModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
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
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!(!u[i].is_nan(), "u[{i}] NaN at step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes at I=0");
    }

    #[test]
    fn test_izhikevich_stress_explosion() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe {
                run_neuron::<IzhikevichModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
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
                assert!(!v_i.is_nan(), "v[{i}] NaN at step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite at step {step}");
            }
        }
    }

    /// Абсолютная рефрактерность: после спайка dv ЗАМОРОЖЕН (v держится на c),
    /// поэтому нейрон не спайкает back-to-back внутри рефрактерного окна.
    /// Инвариант: спайк возможен только если на ВХОДЕ шага нейрон НЕ был в
    /// рефрактерности (refr_before <= 0). Регрессия — «пробитие рефрактерности».
    #[test]
    fn test_izhikevich_refractory_freeze_no_back_to_back() {
        let n = 8;
        let params = make_params(n);
        let mut v = vec![-65.0; n];
        let mut u = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n]; // гигантский ток (клампится раннером до ±100)
        let mut spikes = vec![0i32; n];

        let mut fired_once = false;
        for _ in 0..2000 {
            let refr_before = refr.clone(); // состояние рефрактерности ДО шага
            unsafe {
                run_neuron::<IzhikevichModel>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
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
            for i in 0..n {
                if spikes[i] != 0 {
                    // Спайк запрещён, если нейрон входил в шаг в рефрактерности
                    assert!(
                        refr_before[i] <= 0.0,
                        "back-to-back spike inside refractory window at neuron {i}: refr_before={}",
                        refr_before[i]
                    );
                    fired_once = true;
                }
            }
        }
        assert!(fired_once, "neuron should spike at least once with I=1000");
    }
}
