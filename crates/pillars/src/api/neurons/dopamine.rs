//! Dopamine-modulated STDP — 3-factor plasticity for RL.

// ═══════════════════════════════════════════════════════════
//  Dopamine-Modulated STDP — 3-factor plasticity for RL
// ═══════════════════════════════════════════════════════════

/// Dopamine-modulated STDP synapse with eligibility traces.
///
/// Based on: Izhikevich (2007) "Solving the distal reward problem
/// through linkage of STDP and dopamine signaling" — Cerebral Cortex.
///
/// Канон — `crate::kernels::dopamine` (SIMD + scalar tail внутри канона).
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
        if n == 0 {
            return;
        }
        crate::kernels::dopamine::dopamine_traces_step(
            &mut self.trace_pre[..n],
            &mut self.trace_post[..n],
            &mut self.eligibility[..n],
            &pre_spikes[..n],
            &post_spikes[..n],
            self.dt,
            self.tau_pre,
            self.tau_post,
            self.tau_e,
            self.a_ltp,
            self.a_ltd,
        );
    }

    /// Apply dopamine-modulated weight update: Δw = η · D · e.
    pub fn apply_dopamine(&self, weights: &mut [f32], dopamine: f32, lr: f32) {
        let n = self.n.min(weights.len());
        if n == 0 {
            return;
        }
        crate::kernels::dopamine::dopamine_weight_update(
            &mut weights[..n],
            &self.eligibility[..n],
            dopamine,
            lr,
            self.w_min,
            self.w_max,
        );
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
