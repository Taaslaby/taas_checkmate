//! -------------------------------------------------------------------------
//! Filename:    astrocyte.rs
//! Class:       Neural body
//! Purpose:     Astrocyte — Hidden-State Glial Modulation (Shen et al. 2024)
//! Description:
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//!   Principles
//! - No spikes: glia, not a neuron → `#[no_spike]`
//! - Ca²⁺ wave: dCa/dt = (0.1 − Ca)·inv_tau_ca + α_ca·I
//! - Nernst correction: E_K_eff = E_K + 26.7·ln(K) (K ≥ 0.5)
//!
//! Author/Concept: Anton Atlasov
//! -------------------------------------------------------------------------

use crate::core::compute::{ComputeNeuron, StepContext};
use crate::core::context::SimdContext;
use crate::core::lane::SimdLane;
use pillars_macros::NeuronModel;

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════

/// Astrocyte: hidden-state glial modulation.
///
/// # States (4)
/// - `ca`: intracellular Ca²⁺ (0..1)
/// - `glu`: glutamate (0..1)
/// - `k`: extracellular K⁺
/// - `gain`: volume transmission gain = 1 + μ·Ca
///
/// # Params (9)
/// - `dt`, `inv_tau_ca`, `inv_tau_glu`, `inv_tau_k`,
///   `alpha_ca`, `alpha_glu`, `alpha_k`, `mu`, `g_gap`
#[derive(NeuronModel)]
#[no_spike]
pub struct AstrocyteModel {
    #[state]
    pub ca: f32,
    #[state]
    pub glu: f32,
    #[state]
    pub k: f32,
    #[state]
    pub gain: f32,

    #[param]
    pub dt: f32,
    #[param]
    pub inv_tau_ca: f32,
    #[param]
    pub inv_tau_glu: f32,
    #[param]
    pub inv_tau_k: f32,
    #[param]
    pub alpha_ca: f32,
    #[param]
    pub alpha_glu: f32,
    #[param]
    pub alpha_k: f32,
    #[param]
    pub mu: f32,
    /// Gap-junction strength — declared for the synaptic engine, unused in step.
    #[param]
    pub g_gap: f32,
}

impl ComputeNeuron for AstrocyteModel {
    type State<S: SimdLane> = AstrocyteState<S>;
    type Params<S: SimdLane> = AstrocyteParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut AstrocyteState<S>,
        params: &AstrocyteParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        let one = S::one();
        let zero = S::zero();
        let dt = ctx.dt_simd();

        // Input clamped to [0,1] — synaptic drive
        let inp = ctx.i_syn.clamp01();

        // Ca²⁺: relaxes to 0.1, driven by input
        let ca_inf = S::splat(0.1);
        let new_ca = (state.ca
            + dt * ((ca_inf - state.ca) * params.inv_tau_ca + params.alpha_ca * inp))
            .clamp01();

        // Glutamate: pure decay + input-driven influx
        let new_glu = (state.glu
            + dt * ((-state.glu) * params.inv_tau_glu + params.alpha_glu * inp))
            .clamp01();

        // K⁺: pure decay + input-driven influx (unbounded above 0)
        let new_k =
            (state.k + dt * ((-state.k) * params.inv_tau_k + params.alpha_k * inp)).max(zero);

        // Volume transmission gain: M = 1 + μ·Ca
        let new_gain = one + params.mu * new_ca;

        state.ca = new_ca;
        state.glu = new_glu;
        state.k = new_k;
        state.gain = new_gain;
    }
}

impl AstrocyteModel {
    /// Volume transmission gain M = 1 + μ·Ca.
    #[inline(always)]
    pub fn gain<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        ctx.s(3)
    }

    /// Effective K⁺ reversal potential with Nernst correction:
    /// `E_K_eff = E_K + 26.7·ln(K)` — rising K⁺ depolarises the membrane.
    #[inline(always)]
    pub fn ek_eff<S: SimdLane>(ctx: &SimdContext<S>, e_k: f32) -> S {
        let k = ctx.s(2).max(S::splat(0.5));
        let nernst = S::splat(26.7) * k.ln();
        S::splat(e_k) + nernst
    }

    /// EAAT (glutamate reuptake) factor: 1 − Glu.
    #[inline(always)]
    pub fn eaat_factor<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        S::one() - ctx.s(1)
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::MAX_STATES;
    use crate::core::population::run_neuron;
    use std::marker::PhantomData;
    use std::ptr::NonNull;

    fn make_params(n: usize) -> Vec<f32> {
        let inv_tau_ca = 1.0 / 1000.0;
        let inv_tau_glu = 1.0 / 10.0;
        let inv_tau_k = 1.0 / 100.0;
        let mut p = vec![0.0f32; n * 9];
        for i in 0..n {
            p[AstrocyteModelParamIdx::dt as usize * n + i] = 1.0;
            p[AstrocyteModelParamIdx::inv_tau_ca as usize * n + i] = inv_tau_ca;
            p[AstrocyteModelParamIdx::inv_tau_glu as usize * n + i] = inv_tau_glu;
            p[AstrocyteModelParamIdx::inv_tau_k as usize * n + i] = inv_tau_k;
            p[AstrocyteModelParamIdx::alpha_ca as usize * n + i] = 0.1;
            p[AstrocyteModelParamIdx::alpha_glu as usize * n + i] = 0.5;
            p[AstrocyteModelParamIdx::alpha_k as usize * n + i] = 0.3;
            p[AstrocyteModelParamIdx::mu as usize * n + i] = 0.5;
            p[AstrocyteModelParamIdx::g_gap as usize * n + i] = 0.1;
        }
        p
    }

    #[test]
    fn test_astrocyte_ca_wave() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.0; n];
        let mut glu = vec![0.0; n];
        let mut k = vec![0.0; n];
        let mut gain = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![1.0; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron::<AstrocyteModel>(
                ca.as_mut_ptr(),
                glu.as_mut_ptr(),
                k.as_mut_ptr(),
                gain.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                i_syn.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(ca[0] > 0.0, "Ca should rise with input: {:.4}", ca[0]);
        assert!(gain[0] >= 1.0, "Gain should be ≥ 1.0: {:.4}", gain[0]);
    }

    #[test]
    fn test_astrocyte_silent_decay() {
        let n = 8;
        let params = make_params(n);
        let mut ca = vec![0.5; n];
        let mut glu = vec![0.5; n];
        let mut k = vec![0.5; n];
        let mut gain = vec![1.25; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        unsafe {
            run_neuron::<AstrocyteModel>(
                ca.as_mut_ptr(),
                glu.as_mut_ptr(),
                k.as_mut_ptr(),
                gain.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                i_syn.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        assert!(ca[0] < 0.5, "Ca should decay without input: {:.4}", ca[0]);
        assert!(
            glu[0] < 0.5,
            "Glu should decay without input: {:.4}",
            glu[0]
        );
    }

    #[test]
    fn test_astrocyte_ek_eff() {
        type S = wide::f32x8;
        let mut ctx = SimdContext::<S> {
            state: [S::splat(0.0); MAX_STATES],
            params_ptr: NonNull::dangling(),
            param_count: 9,
            n: 8,
            off: 0,
            i_syn: S::splat(0.0),
            i_pre: S::splat(0.0),
            i_post: S::splat(0.0),
            dt_val: 1.0,
            spike_mask: S::splat(0.0),
            _marker: PhantomData,
        };
        // s(2)=K+ ext, s(3)=Ca²⁺ — остальные каналы нулевые.
        ctx.state[2] = S::splat(2.0);
        ctx.state[3] = S::splat(1.0);

        let ek = AstrocyteModel::ek_eff(&ctx, -80.0);
        let val = ek.to_array()[0];
        assert!(
            (val - (-61.5)).abs() < 1.0,
            "E_K_eff with K=2.0 should be ≈ -61.5 (depolarised), got {:.2}",
            val
        );
    }
}
