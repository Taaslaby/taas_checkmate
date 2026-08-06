//! Exp2Syn — Dual-Exponential Synapse (from NEURON exp2syn.mod)
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **No spikes**: a synapse, not a neuron → `#[no_spike]`
//! - **Exact exponential decay** (stable for fast rise τ₁ = 0.1 ms)
//! - **Ohm's law**: I = g·(E_rev − V) — sign convention for injection

use crate::core::compute::{ComputeNeuron, StepContext};
use crate::core::context::SimdContext;
use crate::core::lane::SimdLane;
use pillars_macros::NeuronModel;

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum Exp2SynParamIdx {
    DT = 0,       // ms
    Inv_Tau1 = 1, // 1/τ₁ (ms⁻¹) — rise rate
    Inv_Tau2 = 2, // 1/τ₂ (ms⁻¹) — decay rate
    E_Rev = 3,    // mV — reversal potential
    Factor = 4,   // normalization factor
    Unused = 5,   // padding
}

impl From<Exp2SynParamIdx> for usize {
    #[inline(always)]
    fn from(idx: Exp2SynParamIdx) -> Self {
        idx as usize
    }
}

/// Dual-exponential synapse: two exponentials, conductance g = (B − A)⁺.
///
/// # States (2)
/// - `a`: rise exponential
/// - `b`: decay exponential
///
/// # Params (6)
/// - `dt`, `inv_tau1`, `inv_tau2`, `e_rev`, `factor`, `unused`
#[derive(NeuronModel)]
#[no_spike]
pub struct Exp2SynModel {
    #[state]
    pub a: f32,
    #[state]
    pub b: f32,

    #[param]
    pub dt: f32,
    #[param]
    pub inv_tau1: f32,
    #[param]
    pub inv_tau2: f32,
    #[param]
    pub e_rev: f32,
    #[param]
    pub factor: f32,
    /// Padding slot — kept for SoA layout backward-compat.
    #[param]
    pub unused: f32,
}

impl ComputeNeuron for Exp2SynModel {
    type State<S: SimdLane> = Exp2SynState<S>;
    type Params<S: SimdLane> = Exp2SynParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut Exp2SynState<S>,
        params: &Exp2SynParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        // Exact exponential decay — no Forward Euler, no division
        let a = state.a * (-ctx.dt_simd() * params.inv_tau1).exp();
        let b = state.b * (-ctx.dt_simd() * params.inv_tau2).exp();

        // Pre-synaptic spike injects normalized current into both exponentials
        let a_inj = a + ctx.i_syn * params.factor;
        let b_inj = b + ctx.i_syn * params.factor;

        state.a = a_inj.max(S::zero());
        state.b = b_inj.max(S::zero());
    }
}

impl Exp2SynModel {
    /// Synaptic conductance: g = (B − A)⁺
    #[inline(always)]
    pub fn conductance<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        (ctx.s(1) - ctx.s(0)).max(S::zero())
    }

    /// Synaptic current (Ohm's law, injected-current sign): I = g·(E_rev − V)
    #[inline(always)]
    pub fn current<S: SimdLane>(ctx: &SimdContext<S>, v: S) -> S {
        let g = Self::conductance(ctx);
        g * (ctx.p_enum(Exp2SynParamIdx::E_Rev) - v)
    }
}

/// NEURON exp2syn normalization: peak conductance = 1.0.
pub fn exp2syn_factor(tau1: f32, tau2: f32) -> f32 {
    assert!(
        tau2 > tau1,
        "Exp2Syn: tau2 ({}) must be > tau1 ({})",
        tau2,
        tau1
    );
    let tp = (tau1 * tau2) / (tau2 - tau1) * (tau2 / tau1).ln();
    1.0 / ((-(-tp / tau1).exp()) + (-tp / tau2).exp())
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: I — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let tau1 = 0.1;
        let tau2 = 10.0;
        let inv_tau1 = 1.0 / tau1;
        let inv_tau2 = 1.0 / tau2;
        let factor = exp2syn_factor(tau1, tau2);
        let mut p = vec![0.0f32; n * 6];
        for i in 0..n {
            p[Exp2SynParamIdx::DT as usize * n + i] = 1.0;
            p[Exp2SynParamIdx::Inv_Tau1 as usize * n + i] = inv_tau1;
            p[Exp2SynParamIdx::Inv_Tau2 as usize * n + i] = inv_tau2;
            p[Exp2SynParamIdx::E_Rev as usize * n + i] = 0.0;
            p[Exp2SynParamIdx::Factor as usize * n + i] = factor;
            p[Exp2SynParamIdx::Unused as usize * n + i] = 0.0;
        }
        p
    }

    #[test]
    fn test_exp2syn_factor() {
        let f = exp2syn_factor(0.1, 10.0);
        assert!(f > 0.0 && f < 10.0, "factor should be reasonable: {:.4}", f);
    }

    #[test]
    fn test_exp2syn_dual_exp_response() {
        let n = 8;
        let params = make_params(n);
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let mut dummy = vec![0.0f32; n];
        let mut dummy6 = vec![0.0f32; n];
        let mut I = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut g_trace = Vec::new();

        I[0] = 1.0;
        unsafe {
            run_neuron::<Exp2SynModel>(
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                I.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        I[0] = 0.0;
        let g0 = (b[0] - a[0]).max(0.0);
        assert!(
            (g0 - 0.0).abs() < 1e-4,
            "g should be 0 right after injection (A=B), got {:.6}",
            g0
        );
        g_trace.push(g0);

        for _ in 1..100 {
            unsafe {
                run_neuron::<Exp2SynModel>(
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            g_trace.push((b[0] - a[0]).max(0.0));
        }
        assert!(
            g_trace[99] < 0.1,
            "g at 100ms should be near 0, got {:.6}",
            g_trace[99]
        );
        let peak_time = g_trace
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(peak_time < 10, "peak should be early, got step {peak_time}");
    }

    #[test]
    fn test_exp2syn_current_sign() {
        let n = 1;
        let params = make_params(n);
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let mut dummy = vec![0.0f32; n];
        let mut dummy6 = vec![0.0f32; n];
        let mut I = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        I[0] = 1.0;
        for _ in 0..4 {
            unsafe {
                run_neuron::<Exp2SynModel>(
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            I[0] = 0.0;
        }
        let g_val = (b[0] - a[0]).max(0.0);
        let i_syn = g_val * (0.0 - (-65.0));
        assert!(
            i_syn > 0.0,
            "Excitatory current should be positive, got {:.6}",
            i_syn
        );
        let i_syn_inh = g_val * (-80.0 - (-65.0));
        assert!(
            i_syn_inh < 0.0,
            "Inhibitory current should be negative, got {:.6}",
            i_syn_inh
        );
    }

    #[test]
    fn test_exp2syn_conductance_shape() {
        let n = 1;
        let params = make_params(n);
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let mut dummy = vec![0.0f32; n];
        let mut dummy6 = vec![0.0f32; n];
        let mut I = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        I[0] = 1.0;
        unsafe {
            run_neuron::<Exp2SynModel>(
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                dummy6.as_mut_ptr(),
                I.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        I[0] = 0.0;

        let mut g_vals = Vec::new();
        for _ in 0..50 {
            unsafe {
                run_neuron::<Exp2SynModel>(
                    a.as_mut_ptr(),
                    b.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    dummy6.as_mut_ptr(),
                    I.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            g_vals.push((b[0] - a[0]).max(0.0));
        }
        let peak = g_vals.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            (peak - 1.0).abs() < 0.5,
            "peak conductance should be ~1.0, got {:.4}",
            peak
        );
        assert!(
            g_vals[49] < peak * 0.5,
            "end g ({:.4}) should be < half peak ({:.4})",
            g_vals[49],
            peak
        );
    }
}
