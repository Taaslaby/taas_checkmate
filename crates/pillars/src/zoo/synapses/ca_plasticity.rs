//! 🔬 CaPlasticity — Calcium-Based Synaptic Plasticity (Plastic Arbor, 2025)
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Clean Bus**: pre/post spikes arrive via `ctx.i_pre` / `ctx.i_post`
//! - **Soft bounds** (+ε): weight can never freeze at the 0/1 boundary
//! - **No spikes**: a plasticity rule, not a neuron → `#[no_spike]`

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};
use crate::core::context::SimdContext;
use crate::ops::spike::spike_to_float;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum CaPlastParamIdx {
    DT = 0,          // ms
    Inv_Tau_Ca = 1,  // 1/τ_ca (ms⁻¹)
    Alpha = 2,       // Ca²⁺ influx per spike
    Eta = 3,         // learning rate
    Target = 4,      // target Ca²⁺ product
    W_Init = 5,      // initial weight
}

impl From<CaPlastParamIdx> for usize {
    #[inline(always)] fn from(idx: CaPlastParamIdx) -> Self { idx as usize }
}

/// Calcium-based synaptic plasticity rule.
///
/// # States (3)
/// - `ca_pre`: pre-synaptic Ca²⁺ concentration
/// - `ca_post`: post-synaptic Ca²⁺ concentration
/// - `w`: synaptic weight (soft-bounded in 0..1)
///
/// # Params (6)
/// - `dt`, `inv_tau_ca`, `alpha`, `eta`, `target`, `w_init`
#[derive(NeuronModel)]
#[no_spike]
pub struct CaPlasticityModel {
    #[state] pub ca_pre: f32,
    #[state] pub ca_post: f32,
    #[state] pub w: f32,

    #[param] pub dt: f32,
    #[param] pub inv_tau_ca: f32,
    #[param] pub alpha: f32,
    #[param] pub eta: f32,
    #[param] pub target: f32,
    /// Initial weight — applied via state init, kept for SoA layout compat.
    #[param] pub w_init: f32,
}

impl ComputeNeuron for CaPlasticityModel {
    type State<S: SimdLane> = CaPlasticityState<S>;
    type Params<S: SimdLane> = CaPlasticityParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut CaPlasticityState<S>,
        params: &CaPlasticityParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        let zero = S::zero();
        let one = S::one();

        // ─── Ca²⁺ decay (exact exponential) ─────────────────────────
        let decay = (-ctx.dt_simd() * params.inv_tau_ca).exp();

        // ─── Clean Bus: pre/post spikes via dedicated buses ─────────
        let pre_spike = spike_to_float(ctx.i_pre);
        let post_spike = spike_to_float(ctx.i_post);

        // ─── Pre-synaptic Ca²⁺: decay + spike-triggered influx ─────
        let new_ca_pre = (state.ca_pre * decay + params.alpha * pre_spike).max(zero);

        // ─── Post-synaptic Ca²⁺: decay + spike-triggered influx ────
        let new_ca_post = (state.ca_post * decay + params.alpha * post_spike).max(zero);

        // ─── Weight update with soft bounds (never freezes) ─────────
        let ca_product = new_ca_pre * new_ca_post;
        let dw_drive = ca_product - params.target;

        let eps = S::splat(0.01);
        let soft_clamp = ((one - state.w) + eps) * ((state.w - zero) + eps);

        let dw = params.eta * dw_drive * soft_clamp * ctx.dt_simd();
        let new_w = (state.w + dw).max(zero).min(one);

        state.ca_pre = new_ca_pre;
        state.ca_post = new_ca_post;
        state.w = new_w;
    }
}

impl CaPlasticityModel {
    /// Current synaptic weight.
    #[inline(always)]
    pub fn weight<S: SimdLane>(ctx: &SimdContext<S>) -> S { ctx.s(2) }
    /// Ca²⁺ product (drive for plasticity).
    #[inline(always)]
    pub fn ca_product<S: SimdLane>(ctx: &SimdContext<S>) -> S { ctx.s(0) * ctx.s(1) }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::population::run_neuron_ext;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 6];
        for i in 0..n {
            p[CaPlastParamIdx::DT as usize * n + i] = 1.0;
            p[CaPlastParamIdx::Inv_Tau_Ca as usize * n + i] = 1.0 / 20.0;
            p[CaPlastParamIdx::Alpha as usize * n + i] = 0.1;
            p[CaPlastParamIdx::Eta as usize * n + i] = 0.01;
            p[CaPlastParamIdx::Target as usize * n + i] = 0.01;
            p[CaPlastParamIdx::W_Init as usize * n + i] = 0.5;
        }
        p
    }

    fn run_ca(
        ca_pre: &mut [f32], ca_post: &mut [f32], w: &mut [f32],
        i_pre: &[f32], i_post: &[f32], params: &[f32],
    ) {
        let n = ca_pre.len();
        let mut spikes = vec![0i32; n];
        let mut dummy = vec![0.0f32; n];
        let i_in = vec![0.0f32; n];
        unsafe {
            run_neuron_ext::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_in.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
                i_pre.as_ptr(), i_post.as_ptr(),
            );
        }
    }

    #[test]
    fn test_ca_pre_post_potentiation() {
        let n = 8; let params = make_params(n);
        let mut ca_pre = vec![0.0; n]; let mut ca_post = vec![0.0; n]; let mut w = vec![0.5; n];
        let i_pre = vec![1.0; n]; let i_post = vec![1.0; n];
        for _ in 0..20 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &params); }
        assert!(ca_pre[0] > 0.01, "Ca_pre should accumulate: {:.4}", ca_pre[0]);
        assert!(ca_post[0] > 0.01, "Ca_post should accumulate: {:.4}", ca_post[0]);
        assert!(w[0] > 0.5, "Weight should potentiate with pre+post pairing, got {:.4}", w[0]);
    }

    #[test]
    fn test_ca_high_freq_unbounded_accumulation() {
        let n = 8; let params = make_params(n);
        let mut ca_pre = vec![0.0; n]; let mut ca_post = vec![0.0; n]; let mut w = vec![0.5; n];
        let i_pre = vec![1.0; n]; let i_post = vec![0.0; n];
        for _ in 0..50 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &params); }
        assert!(ca_pre[0] > 1.0, "Ca_pre should accumulate > 1.0 at high frequency, got {:.4}", ca_pre[0]);
    }

    #[test]
    fn test_ca_weight_bounds() {
        let n = 8; let mut p = make_params(n);
        for i in 0..n { p[CaPlastParamIdx::Eta as usize * n + i] = 1.0; }
        let mut ca_pre = vec![0.0; n]; let mut ca_post = vec![0.0; n]; let mut w = vec![0.5; n];
        let i_pre = vec![10.0; n]; let i_post = vec![10.0; n];
        for _ in 0..100 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &p); }
        assert!(w[0] >= 0.0 && w[0] <= 1.0, "Weight must stay in [0,1], got {:.4}", w[0]);
    }

    #[test]
    fn test_ca_decay_without_spikes() {
        let n = 8; let params = make_params(n);
        let mut ca_pre = vec![0.5; n]; let mut ca_post = vec![0.5; n]; let mut w = vec![0.5; n];
        let i_pre = vec![0.0; n]; let i_post = vec![0.0; n];
        for _ in 0..10 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &params); }
        assert!(ca_pre[0] < 0.5, "Ca_pre should decay: {:.4}", ca_pre[0]);
        assert!(ca_post[0] < 0.5, "Ca_post should decay: {:.4}", ca_post[0]);
    }

    #[test]
    fn test_ca_pre_only_accumulation() {
        let n = 8; let params = make_params(n);
        let mut ca_pre = vec![0.0; n]; let mut ca_post = vec![0.0; n]; let mut w = vec![0.5; n];
        let i_pre = vec![1.0; n]; let i_post = vec![0.0; n];
        for _ in 0..10 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &params); }
        assert!(ca_pre[0] > 0.01, "Ca_pre should accumulate on pre-only: {:.4}", ca_pre[0]);
        assert!(ca_post[0] < 0.001, "Ca_post should stay near 0 on pre-only: {:.4}", ca_post[0]);
    }

    #[test]
    fn test_ca_post_only_accumulation() {
        let n = 8; let params = make_params(n);
        let mut ca_pre = vec![0.0; n]; let mut ca_post = vec![0.0; n]; let mut w = vec![0.5; n];
        let i_pre = vec![0.0; n]; let i_post = vec![1.0; n];
        for _ in 0..10 { run_ca(&mut ca_pre, &mut ca_post, &mut w, &i_pre, &i_post, &params); }
        assert!(ca_pre[0] < 0.001, "Ca_pre should stay near 0 on post-only: {:.4}", ca_pre[0]);
        assert!(ca_post[0] > 0.01, "Ca_post should accumulate on post-only: {:.4}", ca_post[0]);
    }
}
