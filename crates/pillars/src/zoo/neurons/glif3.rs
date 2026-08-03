//! GLIF3 — Generalized LIF with after-spike current (AHP)
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Refractory at input**: v clamped to v_reset BEFORE any computation
//! - **Forward Euler**: g += dt·(−g·inv_tau_ahp) — unbounded accumulation
//! - **No divisions**: all τ pre-inverted at init

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum GLIF3ParamIdx {
    DT = 0,           // ms
    G_Leak = 1,       // mS/cm²
    E_Leak = 2,       // mV
    V_Thresh = 3,     // mV
    V_Reset = 4,      // mV
    Inv_Tau_AHP = 5,  // 1/τ_ahp (ms⁻¹)
    Delta_G = 6,      // g_ahp increment
    E_AHP = 7,        // mV
    Tau_Ref = 8,      // ms
}

impl From<GLIF3ParamIdx> for usize {
    #[inline(always)] fn from(idx: GLIF3ParamIdx) -> Self { idx as usize }
}

/// Generalized LIF with after-spike hyperpolarizing current.
///
/// # States (3)
/// - `v`: membrane potential (mV)
/// - `g`: after-spike conductance g_ahp
/// - `refr`: refractory counter (ms)
///
/// # Params (9)
/// - `dt`, `g_leak`, `e_leak`, `v_thresh`, `v_reset`,
///   `inv_tau_ahp`, `delta_g`, `e_ahp`, `tau_ref`
#[derive(NeuronModel)]
pub struct GLIF3Model {
    #[state] pub v: f32,
    #[state] pub g: f32,
    #[state] pub refr: f32,

    #[param] pub dt: f32,
    #[param] pub g_leak: f32,
    #[param] pub e_leak: f32,
    #[param] pub v_thresh: f32,
    #[param] pub v_reset: f32,
    #[param] pub inv_tau_ahp: f32,
    #[param] pub delta_g: f32,
    #[param] pub e_ahp: f32,
    #[param] pub tau_ref: f32,
}

impl ComputeNeuron for GLIF3Model {
    type State<S: SimdLane> = GLIF3State<S>;
    type Params<S: SimdLane> = GLIF3Params<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut GLIF3State<S>,
        params: &GLIF3Params<S>,
        ctx: &mut StepContext<S>,
    ) {
        let zero = S::zero();

        // ─── Refractory clamp at INPUT ─────────────────────────────
        let is_refr = state.refr.cmp_gt(zero);
        let v = is_refr.cond().then(params.v_reset).else_(state.v);

        // ─── GLIF3 dynamics ────────────────────────────────────────
        // i_total = I + g_leak·E_leak + g_ahp·E_ahp − (g_leak+g_ahp)·v
        let i_total = (ctx.i_syn + params.g_leak * params.e_leak + state.g * params.e_ahp)
            - (params.g_leak + state.g) * v;
        let v_new = v + i_total * ctx.dt_simd();

        // ─── Forward Euler for g_ahp — unbounded (no .min clamp) ───
        let g_new = state.g + ctx.dt_simd() * (-state.g * params.inv_tau_ahp);

        // ─── Spike detection ───────────────────────────────────────
        let spike = v_new.cmp_ge(params.v_thresh);
        ctx.emit_spike(spike);

        // ─── Reset on spike ────────────────────────────────────────
        state.v = spike.cond().then(params.v_reset).else_(v_new);
        state.g = spike.cond().then(g_new + params.delta_g).else_(g_new);

        // ─── Refractory counter ────────────────────────────────────
        let refr_dec = state.refr - ctx.dt_simd();
        let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
        state.refr = refr_set.max(zero);
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize, tau_ref: f32) -> Vec<f32> {
        let inv_tau_ahp = 1.0 / 50.0;
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[GLIF3ParamIdx::DT as usize * n + i] = 1.0;
            p[GLIF3ParamIdx::G_Leak as usize * n + i] = 0.1;
            p[GLIF3ParamIdx::E_Leak as usize * n + i] = -65.0;
            p[GLIF3ParamIdx::V_Thresh as usize * n + i] = -50.0;
            p[GLIF3ParamIdx::V_Reset as usize * n + i] = -65.0;
            p[GLIF3ParamIdx::Inv_Tau_AHP as usize * n + i] = inv_tau_ahp;
            p[GLIF3ParamIdx::Delta_G as usize * n + i] = 0.01;
            p[GLIF3ParamIdx::E_AHP as usize * n + i] = -80.0;
            p[GLIF3ParamIdx::Tau_Ref as usize * n + i] = tau_ref;
        }
        p
    }

    #[test]
    fn test_glif3_basic_spiking() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        let mut total = 0i32;
        for _ in 0..500 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "GLIF3 should spike with I=30");
        assert!(g[0] > 0.0, "g_ahp should accumulate");
    }

    #[test]
    fn test_glif3_refractory_clamp() {
        let n = 8; let params = make_params(n, 5.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![30.0; n]; let mut spikes = vec![0i32; n];

        unsafe { run_neuron::<GLIF3Model>(
            v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!((v[0] - (-65.0)).abs() < 0.1, "After spike v should be v_reset, got {:.2}", v[0]);
        assert!(spikes[0] == 1, "Should have spiked");

        unsafe { run_neuron::<GLIF3Model>(
            v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
            dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        assert!((v[0] - (-65.0)).abs() < 0.1,
            "During refractory v clamped to v_reset, got {:.2}", v[0]);
    }

    #[test]
    fn test_glif3_stress_silence() {
        let n = 8; let params = make_params(n, 2.0);
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![0.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!g[i].is_nan(), "g[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0);
    }

    #[test]
    fn test_glif3_stress_explosion() {
        let n = 8; let mut params = make_params(n, 2.0);
        for i in 0..n { params[GLIF3ParamIdx::Delta_G as usize * n + i] = 0.1; }
        let mut v = vec![-65.0; n]; let mut g = vec![0.0; n]; let mut refr = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let I = vec![1000.0; n]; let mut spikes = vec![0i32; n];

        for step in 0..10000 {
            unsafe { run_neuron::<GLIF3Model>(
                v.as_mut_ptr(), g.as_mut_ptr(), refr.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for (i, &v_i) in v.iter().enumerate() {
                assert!(!v_i.is_nan(), "v[{i}] NaN step {step}");
                assert!(v_i.is_finite(), "v[{i}] non-finite");
            }
        }
        assert!(spikes.iter().sum::<i32>() > 0, "GLIF3 explosion: should spike");
    }
}
