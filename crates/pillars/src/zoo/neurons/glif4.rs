//! GLIF4 — Generalized LIF with threshold adaptation + after-spike current
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: g += dt·(−g·inv_tau_ahp); θ += dt·(−θ·inv_tau_theta)
//! - **Unbounded**: g_ahp and θ accumulate without .min() clamps

#![allow(clippy::module_inception)] // deliberate: `pub mod glif4` mirrors the file module

use pillars_macros::dual_neuron;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum GLIF4ParamIdx {
    DT = 0,            // ms
    G_Leak = 1,        // mS/cm²
    E_Leak = 2,        // mV
    V_Thresh_0 = 3,    // mV — baseline threshold
    V_Reset = 4,       // mV
    Inv_Tau_AHP = 5,   // 1/τ_ahp (ms⁻¹)
    Delta_G = 6,       // g_ahp increment on spike
    E_AHP = 7,         // mV
    Inv_Tau_Theta = 8, // 1/τ_θ (ms⁻¹)
    Delta_Theta = 9,   // θ increment on spike (mV)
    Tau_Ref = 10,      // ms
}

impl From<GLIF4ParamIdx> for usize {
    #[inline(always)]
    fn from(idx: GLIF4ParamIdx) -> Self {
        idx as usize
    }
}

/// Generalized LIF with adaptive threshold θ + after-spike current.
///
/// # States (4)
/// - `v`: membrane potential (mV)
/// - `th`: adaptive threshold offset θ (mV)
/// - `g`: after-spike conductance g_ahp
/// - `refr`: refractory counter (ms)
///
/// # Params (11)
/// - `dt`, `g_leak`, `e_leak`, `v_thresh_0`, `v_reset`,
///   `inv_tau_ahp`, `delta_g`, `e_ahp`, `inv_tau_theta`,
///   `delta_theta`, `tau_ref`
#[dual_neuron]
pub mod glif4 {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    pub struct GLIF4Model {
        #[state]
        pub v: f32,
        #[state]
        pub th: f32,
        #[state]
        pub g: f32,
        #[state]
        pub refr: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub g_leak: f32,
        #[param]
        pub e_leak: f32,
        #[param]
        pub v_thresh_0: f32,
        #[param]
        pub v_reset: f32,
        #[param]
        pub inv_tau_ahp: f32,
        #[param]
        pub delta_g: f32,
        #[param]
        pub e_ahp: f32,
        #[param]
        pub inv_tau_theta: f32,
        #[param]
        pub delta_theta: f32,
        #[param]
        pub tau_ref: f32,
    }

    impl ComputeNeuron for GLIF4Model {
        type State<S: SimdLane> = GLIF4State<S>;
        type Params<S: SimdLane> = GLIF4Params<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut GLIF4State<S>,
            params: &GLIF4Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();

            // ─── Refractory clamp at INPUT ─────────────────────────────
            let is_refr = state.refr.cmp_gt(zero);
            let v = is_refr.cond().then(params.v_reset).else_(state.v);

            // Effective threshold = v_thresh_0 + θ
            let v_th_eff = params.v_thresh_0 + state.th;

            // ─── GLIF4 dynamics ────────────────────────────────────────
            // i_total = I + g_leak·E_leak + g_ahp·E_ahp − (g_leak+g_ahp)·v
            let i_total = (ctx.i_syn + params.g_leak * params.e_leak + state.g * params.e_ahp)
                - (params.g_leak + state.g) * v;
            let v_new = v + i_total * ctx.dt_simd();

            // ─── Forward Euler for θ and g (both unbounded) ────────────
            let theta_decay = state.th + ctx.dt_simd() * (-state.th * params.inv_tau_theta);
            let g_decay = state.g + ctx.dt_simd() * (-state.g * params.inv_tau_ahp);

            // ─── Spike detection against adaptive threshold ────────────
            let spike = v_new.cmp_ge(v_th_eff);
            ctx.emit_spike(spike);

            // ─── Reset on spike ────────────────────────────────────────
            state.v = spike.cond().then(params.v_reset).else_(v_new);
            state.th = spike
                .cond()
                .then(theta_decay + params.delta_theta)
                .else_(theta_decay);
            state.g = spike.cond().then(g_decay + params.delta_g).else_(g_decay);

            // ─── Refractory counter ────────────────────────────────────
            let refr_dec = state.refr - ctx.dt_simd();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

// Re-export so existing paths keep working:
// `pillars::zoo::neurons::glif4::GLIF4Model` etc.
pub use glif4::*;

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let inv_tau_ahp = 1.0 / 30.0;
        let inv_tau_theta = 1.0 / 30.0;
        let mut p = vec![0.0f32; n * 11];
        for i in 0..n {
            p[GLIF4ParamIdx::DT as usize * n + i] = 1.0;
            p[GLIF4ParamIdx::G_Leak as usize * n + i] = 0.1;
            p[GLIF4ParamIdx::E_Leak as usize * n + i] = -65.0;
            p[GLIF4ParamIdx::V_Thresh_0 as usize * n + i] = -50.0;
            p[GLIF4ParamIdx::V_Reset as usize * n + i] = -65.0;
            p[GLIF4ParamIdx::Inv_Tau_AHP as usize * n + i] = inv_tau_ahp;
            p[GLIF4ParamIdx::Delta_G as usize * n + i] = 0.01;
            p[GLIF4ParamIdx::E_AHP as usize * n + i] = -80.0;
            p[GLIF4ParamIdx::Inv_Tau_Theta as usize * n + i] = inv_tau_theta;
            p[GLIF4ParamIdx::Delta_Theta as usize * n + i] = 3.0;
            p[GLIF4ParamIdx::Tau_Ref as usize * n + i] = tau_ref;
        }
        p
    }

    #[test]
    fn test_glif4_basic_spiking() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe {
                run_neuron::<GLIF4Model>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    g.as_mut_ptr(),
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
        assert!(total > 0, "GLIF4 should spike with I=30");
        assert!(theta[0] > 0.0, "theta should accumulate");
    }

    #[test]
    fn test_glif4_theta_decay() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut theta = vec![10.0; n];
        let mut g = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron::<GLIF4Model>(
                v.as_mut_ptr(),
                theta.as_mut_ptr(),
                g.as_mut_ptr(),
                refr.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                I.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        let expected = 10.0 + 1.0 * (-10.0 / 30.0);
        assert!(
            (theta[0] - expected).abs() < 0.02,
            "theta FE decay: got {:.4}, expected {:.4}",
            theta[0],
            expected
        );
    }

    #[test]
    fn test_glif4_non_aligned_n() {
        let n = 13;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n];
        let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe {
                run_neuron::<GLIF4Model>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    g.as_mut_ptr(),
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
        assert!(total > 0, "GLIF4 n=13 should spike, got {total}");
        for i in 0..n {
            assert!(!v[i].is_nan(), "v[{i}] is NaN");
            assert!(!theta[i].is_nan(), "theta[{i}] is NaN");
            assert!(!g[i].is_nan(), "g[{i}] is NaN");
        }
    }

    #[test]
    fn test_glif4_stress_silence() {
        let n = 8;
        let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n];
        let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe {
                run_neuron::<GLIF4Model>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    g.as_mut_ptr(),
                    refr.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!theta[i].is_nan(), "θ[{i}] NaN step {step}");
                assert!(!g[i].is_nan(), "g[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "GLIF4 silence: no spikes");
    }

    #[test]
    fn test_glif4_stress_explosion() {
        let n = 8;
        let mut params = make_params(n, 2.0);
        for i in 0..n {
            params[GLIF4ParamIdx::Delta_G as usize * n + i] = 0.1;
            params[GLIF4ParamIdx::Delta_Theta as usize * n + i] = 1.0;
        }
        let mut v = vec![-65.0; n];
        let mut theta = vec![0.0; n];
        let mut g = vec![0.0; n];
        let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n];
        let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe {
                run_neuron::<GLIF4Model>(
                    v.as_mut_ptr(),
                    theta.as_mut_ptr(),
                    g.as_mut_ptr(),
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
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite step {step}");
            }
        }
        assert!(
            spikes.iter().sum::<i32>() > 0,
            "GLIF4 explosion: should spike"
        );
    }
}
