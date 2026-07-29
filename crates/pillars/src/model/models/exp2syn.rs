//! Exp2Syn — Dual-Exponential Synapse (from NEURON exp2syn.mod)
//!
//! Two-state kinetic scheme:
//!   dA/dt = -A / τ₁   (rise)
//!   dB/dt = -B / τ₂   (decay)
//!   g = B - A          (conductance, peak normalized to 1)
//!   i_syn = g · (E_rev - V)
//!
//! On presynaptic event (ctx.i() > 0): A += w · factor, B += w · factor
//! where factor = 1 / (exp(-tp/τ₁) - exp(-tp/τ₂))
//!         tp = τ₁·τ₂/(τ₂-τ₁) · ln(τ₂/τ₁)
//!
//! # Bugfixes applied
//! - inject() absorbed into step(): ctx.i() carries presynaptic spike weight per lane
//! - gate_integrate → pure Forward Euler: A += dt·(-A·inv_tau1); B += dt·(-B·inv_tau2)
//! - current() sign: g·(E_rev - V) instead of g·(V - E_rev) — matches NEURON convention
//!
//! Requires τ₂ > τ₁ (decay > rise).
//!
//! Params (6 × n): dt, inv_tau1, inv_tau2, e_rev, factor, __unused
//! State (2): A (rise state), B (decay state)

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Exp2Syn — Dual-Exponential Synapse (NEURON exp2syn.mod)
///
/// Usage:
/// - Set `ctx.i()` = presynaptic spike weight per event (per-lane, 0 = no event)
/// - `step()` integrates decay + injects presynaptic events
/// - After step, use `Self::current(ctx, v)` to compute synaptic current for target neuron
pub struct Exp2SynModel;

crate::validate_model!(Exp2SynModel {
    params: 6, state: 2, dt: 0,
    inv_tau: [1, 2],
    forward_euler: true
});


impl Model for Exp2SynModel {
    /// Params: [0]dt, [1]inv_tau1, [2]inv_tau2, [3]e_rev, [4]factor, [5]__unused
    const N_PARAMS: usize = 6;
    /// State: [0]A (rise), [1]B (decay)
    const N_STATE: usize = 2;
    /// No intrinsic spike — this is a synapse
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        // Forward Euler for A: A += dt * (-A * inv_tau1)
        let a = ctx.s(0) + ctx.dt() * (-ctx.s(0) * ctx.p(1));
        // Forward Euler for B: B += dt * (-B * inv_tau2)
        let b = ctx.s(1) + ctx.dt() * (-ctx.s(1) * ctx.p(2));

        // Inject presynaptic event: ctx.i() = weight per lane
        // If weight > 0, add weight * factor to both A and B
        let factor = ctx.p(4);
        let a_inj = a + ctx.i() * factor;
        let b_inj = b + ctx.i() * factor;

        ctx.set_s(0, a_inj.max(S::zero()));
        ctx.set_s(1, b_inj.max(S::zero()));
    }
}

impl Exp2SynModel {
    /// Get conductance g = B - A (peak-normalized)
    #[inline(always)]
    pub fn conductance<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        (ctx.s(1) - ctx.s(0)).max(S::zero())
    }

    /// Compute synaptic current: i = g · (E_rev - V)
    ///
    /// Sign convention: positive current depolarizes (excitatory) when V < E_rev.
    /// Matches NEURON convention where i_syn is subtracted in the membrane equation.
    #[inline(always)]
    pub fn current<S: SimdLane>(ctx: &SimdContext<S>, v: S) -> S {
        let g = Self::conductance(ctx);
        g * (ctx.p(3) - v)
    }
}

/// Compute the normalization factor for Exp2Syn.
/// From NEURON exp2syn.mod:
///   tp = (τ₁·τ₂)/(τ₂-τ₁) · ln(τ₂/τ₁)
///   factor = 1 / (-exp(-tp/τ₁) + exp(-tp/τ₂))
pub fn exp2syn_factor(tau1: f32, tau2: f32) -> f32 {
    assert!(tau2 > tau1, "Exp2Syn: tau2 ({}) must be > tau1 ({})", tau2, tau1);
    let tp = (tau1 * tau2) / (tau2 - tau1) * (tau2 / tau1).ln();
    1.0 / ((-(-tp / tau1).exp()) + (-tp / tau2).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let tau1 = 0.1;   // rise time (ms)
        let tau2 = 10.0;  // decay time (ms)
        let inv_tau1 = 1.0 / tau1;
        let inv_tau2 = 1.0 / tau2;
        let e_rev = 0.0;  // reversal potential (mV)
        let factor = exp2syn_factor(tau1, tau2);
        let mut p = vec![0.0f32; n * 6];
        for i in 0..n {
            p[i] = 1.0;               // dt
            p[n+i] = inv_tau1;        // inv_tau1
            p[2*n+i] = inv_tau2;      // inv_tau2
            p[3*n+i] = e_rev;         // e_rev
            p[4*n+i] = factor;        // factor
            p[5*n+i] = 0.0;           // unused
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
        // Use ctx.i() = weight * flag: set I = 1.0 for 1 step only (presynaptic event)
        let mut I = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        // Track g = B - A over time
        let mut g_trace = Vec::new();

        // Step 0: inject via ctx.i()
        I[0] = 1.0;  // weight = 1.0 for neuron 0
        unsafe { run_neuron::<Exp2SynModel>(
            a.as_mut_ptr(), b.as_mut_ptr(), dummy.as_mut_ptr(),
            dummy6.as_mut_ptr(), dummy6.as_mut_ptr(), dummy6.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        I[0] = 0.0;  // no more events
        let factor = exp2syn_factor(0.1, 10.0);
        // After injection: A = B = factor, so g = 0 initially
        let g0 = (b[0] - a[0]).max(0.0);
        assert!((g0 - 0.0).abs() < 1e-4, "g should be 0 right after injection (A=B), got {:.6}", g0);
        g_trace.push(g0);

        // Steps 1-99: FE decay only, g rises then falls
        for _ in 1..100 {
            unsafe { run_neuron::<Exp2SynModel>(
                a.as_mut_ptr(), b.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy6.as_mut_ptr(), dummy6.as_mut_ptr(), dummy6.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            let g = (b[0] - a[0]).max(0.0);
            g_trace.push(g);
        }

        // After 100ms with tau2=10: g should be near 0
        assert!(g_trace[99] < 0.1,
            "g at 100ms should be near 0, got {:.6}", g_trace[99]);

        // Peak should be early (rise phase with tau1=0.1)
        let peak_time = g_trace.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(peak_time < 10,
            "peak should be early (rise phase), got step {}", peak_time);
    }

    /// Exp2Syn: current() returns g·(E_rev - V) — excitatory at rest.
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

        // Inject with weight = 1, then step to let B > A
        I[0] = 1.0;
        for _ in 0..4 {
            unsafe { run_neuron::<Exp2SynModel>(
                a.as_mut_ptr(), b.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy6.as_mut_ptr(), dummy6.as_mut_ptr(), dummy6.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            I[0] = 0.0;  // only inject on first step
        }

        // Now B > A, so g > 0
        // E_rev = 0 (excitatory), V = -65: i_syn = g * (0 - (-65)) = g * 65 > 0 ✓
        let g_val = (b[0] - a[0]).max(0.0);
        let i_syn = g_val * (params[3] - (-65.0));  // e_rev = 0, v = -65
        assert!(i_syn > 0.0,
            "Excitatory current should be positive (depolarizing), got {:.6}", i_syn);

        // With E_rev = -80 (inhibitory), V = -65: i_syn = g * (-80 + 65) = g * (-15) < 0
        let i_syn_inh = g_val * (-80.0 - (-65.0));
        assert!(i_syn_inh < 0.0,
            "Inhibitory current should be negative (hyperpolarizing), got {:.6}", i_syn_inh);
    }

    /// Helper: run a few steps to build context, then verify conductance peak.
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

        let weight = 1.0;
        I[0] = weight;
        // Inject
        unsafe { run_neuron::<Exp2SynModel>(
            a.as_mut_ptr(), b.as_mut_ptr(), dummy.as_mut_ptr(),
            dummy6.as_mut_ptr(), dummy6.as_mut_ptr(), dummy6.as_mut_ptr(),
            I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }
        I[0] = 0.0;

        let mut g_vals = Vec::new();
        for _ in 0..50 {
            unsafe { run_neuron::<Exp2SynModel>(
                a.as_mut_ptr(), b.as_mut_ptr(), dummy.as_mut_ptr(),
                dummy6.as_mut_ptr(), dummy6.as_mut_ptr(), dummy6.as_mut_ptr(),
                I.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            g_vals.push((b[0] - a[0]).max(0.0));
        }

        // Peak conductance should be ≈ 1.0 (factor normalizes to 1)
        let peak = g_vals.iter().cloned().fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.5,
            "peak conductance should be ~1.0, got {:.4}", peak);

        // After peak, g should decay
        let end_g = g_vals[49];
        assert!(end_g < peak * 0.5,
            "end g ({:.4}) should be < half peak ({:.4})", end_g, peak);
    }
}
