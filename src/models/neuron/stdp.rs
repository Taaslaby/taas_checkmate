//! STDP (Spike-Timing Dependent Plasticity) + dopamine-modulated STDP.
//!
//! Safe Rust wrappers over SIMD kernels (`kernels::stdp_update`,
//! `kernels::stdp_homeostatic`, `kernels::dopamine_traces_step`,
//! `kernels::dopamine_weight_update`).

use crate::kernels;

// ─── STDP ────────────────────────────────────────────────

pub fn stdp_update(weights: &mut [f32], traces_pre: &[f32], traces_post: &[f32],
    lr_ltd: f32, lr_ltp: f32, w_min: f32, w_max: f32) {
    kernels::stdp_update(weights, traces_pre, traces_post, lr_ltd, lr_ltp, w_min, w_max);
}

pub fn stdp_homeostatic(weights: &mut [f32], traces_pre: &[f32], traces_post: &[f32],
    lr_ltd: f32, lr_ltp: f32, target_sum: f32, w_max: f32) {
    kernels::stdp_homeostatic(weights, traces_pre, traces_post, lr_ltd, lr_ltp, target_sum, w_max);
}

// ═══════════════════════════════════════════════════════════
//  Dopamine-Modulated STDP — 3-factor plasticity for RL
// ═══════════════════════════════════════════════════════════

/// Dopamine-modulated STDP synapse with eligibility traces.
///
/// Based on: Izhikevich (2007) "Solving the distal reward problem
/// through linkage of STDP and dopamine signaling" — Cerebral Cortex
///
/// # Example
/// ```
/// use taas_checkmate::neuron::DopamineSTDP;
///
/// let n = 64;
/// let mut stdp = DopamineSTDP::new(n, 1.0);
///
/// let pre_spikes = vec![0i32; n];
/// let post_spikes = vec![0i32; n];
/// stdp.step(&pre_spikes, &post_spikes);
///
/// let mut weights = vec![0.5f32; n];
/// stdp.apply_dopamine(&mut weights, 1.0, 0.01);
/// stdp.clear_eligibility();
/// ```
pub struct DopamineSTDP {
    pub trace_pre: Vec<f32>,
    pub trace_post: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_pre: f32,
    pub tau_post: f32,
    pub tau_e: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub w_min: f32,
    pub w_max: f32,
}

impl DopamineSTDP {
    /// Create new DopamineSTDP with `n` synapses (padded to multiple of 8).
    ///
    /// Default params (from Izhikevich 2007):
    /// - τ_pre = 20 ms, τ_post = 20 ms, τ_e = 1000 ms
    /// - A_ltp = 0.1, A_ltd = 0.12, w_min = 0.0, w_max = 1.0
    pub fn new(n: usize, dt: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            trace_pre: vec![0.0; n_pad],
            trace_post: vec![0.0; n_pad],
            eligibility: vec![0.0; n_pad],
            n: n_pad,
            dt,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 1000.0,
            a_ltp: 0.1,
            a_ltd: 0.12,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// One step: update pre/post traces and eligibility trace.
    pub fn step(&mut self, pre_spikes: &[i32], post_spikes: &[i32]) {
        let n = self.n.min(pre_spikes.len()).min(post_spikes.len());
        let n8 = n & !7;
        if n8 == 0 { return; }

        kernels::dopamine_traces_step(
            &mut self.trace_pre[..n8],
            &mut self.trace_post[..n8],
            &mut self.eligibility[..n8],
            &pre_spikes[..n8],
            &post_spikes[..n8],
            self.dt,
            self.tau_pre,
            self.tau_post,
            self.tau_e,
            self.a_ltp,
            self.a_ltd,
        );

        // Scalar tail (n & 7)
        if n8 < n {
            for i in n8..n {
                let decay_pre = 1.0 - self.dt / self.tau_pre.max(1.0);
                let decay_post = 1.0 - self.dt / self.tau_post.max(1.0);
                self.trace_pre[i] *= decay_pre;
                self.trace_post[i] *= decay_post;
                if pre_spikes[i] != 0 { self.trace_pre[i] += 1.0; }
                if post_spikes[i] != 0 { self.trace_post[i] += 1.0; }
                let ltp = if post_spikes[i] != 0 { self.a_ltp * self.trace_pre[i] } else { 0.0 };
                let ltd = if pre_spikes[i] != 0 { self.a_ltd * self.trace_post[i] } else { 0.0 };
                self.eligibility[i] += ltp - ltd;
                let decay_e = 1.0 - self.dt / self.tau_e.max(1.0);
                self.eligibility[i] *= decay_e;
            }
        }
    }

    /// Apply dopamine-modulated weight update: Δw = η · D · e.
    pub fn apply_dopamine(&self, weights: &mut [f32], dopamine: f32, lr: f32) {
        let n = self.n.min(weights.len());
        let n8 = n & !7;
        if n8 == 0 { return; }

        kernels::dopamine_weight_update(
            &mut weights[..n8],
            &self.eligibility[..n8],
            dopamine,
            lr,
            self.w_min,
            self.w_max,
        );

        if n8 < n {
            for i in n8..n {
                weights[i] += lr * dopamine * self.eligibility[i];
                weights[i] = weights[i].clamp(self.w_min, self.w_max);
            }
        }
    }

    pub fn clear_eligibility(&mut self) {
        self.eligibility.fill(0.0);
    }

    pub fn reset(&mut self) {
        self.trace_pre.fill(0.0);
        self.trace_post.fill(0.0);
        self.eligibility.fill(0.0);
    }
}
