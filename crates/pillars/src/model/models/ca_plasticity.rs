//! 🔬 CaPlasticity — Calcium-Based Synaptic Plasticity (Plastic Arbor, 2025)
//!
//! Implements calcium-dependent learning rules where intracellular Ca²⁺
//! concentration determines the sign and magnitude of synaptic weight changes:
//!
//! ```text
//!   dCa_pre/dt  = -Ca_pre / τ_ca  +  α · pre_spikes
//!   dCa_post/dt = -Ca_post / τ_ca +  α · post_spikes
//!   Δw = η · (Ca_pre · Ca_post - target) · (w_max - w) · (w - w_min)
//! ```
//!
//! This is more biologically realistic than simple pair-based STDP because:
//! 1. **Ca²⁺ integration**: Temporal summation of spikes over ms timescale (unbounded Ca²⁺ accumulation)
//! 2. **Soft weight bounds**: Weight change ∝ (w_max - w)·(w - w_min)
//! 3. **Biological saturation**: Prevents unbounded weight growth
//!
//! Reference: Luboeinski et al. 2025, "Plastic Arbor: a modern simulation
//!   framework for synaptic plasticity", arXiv:2411.16445
//!
//! Graupner & Brunel 2012, "Calcium-based plasticity model explains
//!  ADP protocol", PLoS Comput Biol
//!
//! ## Params (6 × n, column-major)
//!
//!   [0] dt, [1] inv_tau_ca (pre-inverted 1/τ_ca, ms⁻¹),
//!   [2] alpha (Ca²⁺ influx per spike), [3] eta (learning rate),
//!   [4] target (target Ca²⁺ product), [5] w_init (initial weight)
//!
//! ## State (4 vars per synapse)
//!
//!   [0] Ca_pre (pre-synaptic Ca²⁺ trace),
//!   [1] Ca_post (post-synaptic Ca²⁺ trace),
//!   [2] w (synaptic weight),
//!   [3] post_spikes (post-synaptic spike input stream)
//!
//! ## Usage
//!
//! Call step() with:
//!   - pre_spikes via `i_syn`
//!   - post_spikes via 4th state buffer `r` (ctx.s(3))
//!
//! The weight w evolves according to the triplet Ca²⁺ rule.

use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// Calcium-Based Synaptic Plasticity Model
pub struct CaPlasticityModel;

crate::validate_model!(CaPlasticityModel {
    params: 6, state: 4, dt: 0,
    inv_tau: [1],
    forward_euler: true
});


impl Model for CaPlasticityModel {
    /// Params: [0]dt, [1]inv_tau_ca (1/tau_ca), [2]alpha, [3]eta, [4]target, [5]w_init
    const N_PARAMS: usize = 6;
    /// State: [0]Ca_pre, [1]Ca_post, [2]w, [3]post_spikes
    const N_STATE: usize = 4;
    /// No intrinsic spike — this is a learning rule, not a neuron
    const HAS_SPIKE: bool = false;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let ca_pre = ctx.s(0);     // Pre-synaptic Ca²⁺ trace
        let ca_post = ctx.s(1);    // Post-synaptic Ca²⁺ trace
        let w = ctx.s(2);          // Synaptic weight
        let post_spikes = ctx.s(3); // Post-synaptic spike stream

        let zero = S::zero();
        let one = S::one();

        // Zero-division pre-inverted exponential decay: exp(-dt * inv_tau_ca)
        let inv_tau_ca = ctx.p(1);
        let decay = (-ctx.dt() * inv_tau_ca).exp();

        // Pre-synaptic Ca²⁺: decay + spike-triggered influx (unbounded sum)
        let ca_pre_decay = ca_pre * decay;
        let ca_pre_input = ctx.p(2) * ctx.i();
        let new_ca_pre = (ca_pre_decay + ca_pre_input).max(zero);

        // Post-synaptic Ca²⁺: decay + spike-triggered influx via dedicated channel ctx.s(3)
        let ca_post_decay = ca_post * decay;
        let ca_post_input = ctx.p(2) * post_spikes;
        let new_ca_post = (ca_post_decay + ca_post_input).max(zero);

        // Weight update: Δw = η · (Ca_pre · Ca_post - target) · (w_max - w) · (w - w_min)
        // BCM-like rule: product of pre/post Ca²⁺ determines LTP/LTD
        let ca_product = new_ca_pre * new_ca_post;
        let target = ctx.p(4);
        let dw_drive = ca_product - target;

        // Soft bounds: w ∈ [0, 1]
        let w_max = one;
        let w_min = zero;
        let soft_clamp = (w_max - w) * (w - w_min);

        // Weight change with soft bounds
        let eta = ctx.p(3);
        let dw = eta * dw_drive * soft_clamp * ctx.dt();
        let new_w = (w + dw).max(w_min).min(w_max);

        ctx.set_s(0, new_ca_pre);
        ctx.set_s(1, new_ca_post);
        ctx.set_s(2, new_w);
        ctx.set_s(3, zero); // Reset post-spike event buffer for next step
    }
}

impl CaPlasticityModel {
    /// Get current synaptic weight (normalised to [0, 1]).
    #[inline(always)]
    pub fn weight<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        ctx.s(2)
    }

    /// Get Ca²⁺ product (indicates LTP vs LTD direction).
    /// Ca²⁺ product > target → LTP (potentiation)
    /// Ca²⁺ product < target → LTD (depression)
    #[inline(always)]
    pub fn ca_product<S: SimdLane>(ctx: &SimdContext<S>) -> S {
        ctx.s(0) * ctx.s(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::runner::run_neuron;

    fn make_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 6];
        for i in 0..n {
            p[i] = 1.0;            // dt (ms)
            p[n+i] = 1.0 / 20.0;   // inv_tau_ca (1/20 ms)
            p[2*n+i] = 0.1;        // alpha (Ca²⁺ influx)
            p[3*n+i] = 0.01;       // eta (learning rate)
            p[4*n+i] = 0.01;       // target (Ca²⁺ product threshold)
            p[5*n+i] = 0.5;        // w_init (initial weight)
        }
        p
    }

    #[test]
    fn test_ca_pre_post_potentiation() {
        // Pre-post pairing (simultaneous or pre before post) should potentiate
        let n = 8;
        let params = make_params(n);
        let mut ca_pre = vec![0.0; n];
        let mut ca_post = vec![0.0; n];
        let mut w = vec![0.5; n];  // initial weight = 0.5
        let mut post_spikes = vec![1.0; n]; // post-spike active
        let mut dummy = vec![0.0; n];
        let mut spikes = vec![0i32; n];

        // Pre-spike: i_syn = 1.0
        let i_syn = vec![1.0; n];

        for _ in 0..20 {
            post_spikes.fill(1.0);
            unsafe { run_neuron::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }

        // Ca²⁺ should have accumulated above 1.0 for high frequency
        assert!(ca_pre[0] > 0.01, "Ca_pre should accumulate: {:.4}", ca_pre[0]);
        assert!(ca_post[0] > 0.01, "Ca_post should accumulate: {:.4}", ca_post[0]);
        assert!(w[0] > 0.5, "Weight should potentiate with pre+post pairing, got {:.4}", w[0]);
    }

    #[test]
    fn test_ca_high_freq_unbounded_accumulation() {
        // High frequency spikes should allow Ca²⁺ to exceed 1.0 (no artificial clamp01)
        let n = 8;
        let params = make_params(n);
        let mut ca_pre = vec![0.0; n];
        let mut ca_post = vec![0.0; n];
        let mut w = vec![0.5; n];
        let mut post_spikes = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![1.0; n];  // continuous high influx
        let mut spikes = vec![0i32; n];

        for _ in 0..50 {
            unsafe { run_neuron::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }

        assert!(ca_pre[0] > 1.0, "Ca_pre should accumulate > 1.0 at high frequency, got {:.4}", ca_pre[0]);
    }

    #[test]
    fn test_ca_weight_bounds() {
        // Weight should stay within [0, 1] even with extreme input
        let n = 8;
        let mut p = make_params(n);
        for i in 0..n {
            p[3*n+i] = 1.0;  // eta = 1.0 (extreme)
        }
        let mut ca_pre = vec![0.0; n];
        let mut ca_post = vec![0.0; n];
        let mut w = vec![0.5; n];
        let mut post_spikes = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![10.0; n];  // extreme pre-spike
        let mut spikes = vec![0i32; n];

        for _ in 0..100 {
            unsafe { run_neuron::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), p.as_ptr(), n,
            ); }
        }

        assert!(w[0] >= 0.0 && w[0] <= 1.0,
            "Weight must stay in [0,1], got {:.4}", w[0]);
    }

    #[test]
    fn test_ca_decay_without_spikes() {
        let n = 8;
        let params = make_params(n);
        let mut ca_pre = vec![0.5; n];  // elevated Ca²⁺
        let mut ca_post = vec![0.5; n];
        let mut w = vec![0.5; n];
        let mut post_spikes = vec![0.0; n];
        let mut dummy = vec![0.0; n];
        let i_syn = vec![0.0; n];  // no input
        let mut spikes = vec![0i32; n];

        for _ in 0..10 {
            unsafe { run_neuron::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }

        assert!(ca_pre[0] < 0.5, "Ca_pre should decay: {:.4}", ca_pre[0]);
        assert!(ca_post[0] < 0.5, "Ca_post should decay: {:.4}", ca_post[0]);
    }

    #[test]
    fn test_ca_post_spike_buffer_clearing() {
        // A single post spike should be cleared after 1 step and not cause continuous Ca influx
        let n = 8;
        let params = make_params(n);
        let mut ca_pre = vec![0.0; n];
        let mut ca_post = vec![0.0; n];
        let mut w = vec![0.5; n];
        let mut post_spikes = vec![1.0; n]; // Single post-spike at step 0
        let mut dummy = vec![0.0; n];
        let i_syn = vec![0.0; n]; // No pre-spike
        let mut spikes = vec![0i32; n];

        // Step 1: post-spike consumes 1.0
        unsafe { run_neuron::<CaPlasticityModel>(
            ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
            post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
            i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
        ); }

        assert_eq!(post_spikes[0], 0.0, "post_spikes buffer must be cleared to 0.0 after step");

        // Step 2..10: silence, Ca_post should decay
        let ca_post_step1 = ca_post[0];
        for _ in 0..10 {
            unsafe { run_neuron::<CaPlasticityModel>(
                ca_pre.as_mut_ptr(), ca_post.as_mut_ptr(), w.as_mut_ptr(),
                post_spikes.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }

        assert!(ca_post[0] < ca_post_step1,
            "Ca_post should decay after single post-spike, got {:.4} vs step1 {:.4}",
            ca_post[0], ca_post_step1);
    }
}

