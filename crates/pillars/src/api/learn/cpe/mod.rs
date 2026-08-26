//! -------------------------------------------------------------------------
//! Filename:    src/cpe/mod.rs
//! Class:       cpe
//! Purpose:     ChaosPatternEngine (CPE)
//! Description: Contrastive dopamine-STDP + T2-SPIGRAD on Kuramoto
//!              connectomes. Compares pattern (meaningful input) vs chaos
//!              (random noise); dopamine delta gates plasticity:
//!              delta = reward + beta*[(R_pat - alpha*|H_pat-H0|) - (R_chaos - alpha*|H_chaos-H0|)].
//!              delta>0 → STDP strengthens; delta<0 → weakens.
//!              Levels: 1) Kuramoto phase dynamics (kernels::phase),
//!              2) dopamine-STDP (contrastive), 3) T2-SPIGRAD (optional).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

pub mod coherence;
pub mod config;
pub mod dopamine;
pub mod hybrid;
pub mod stdp;

pub use coherence::{detect_spikes, phase_coherence, spike_entropy};
pub use config::CpeConfig;
pub use dopamine::{dopamine_delta, dopamine_timing_gate};
pub use hybrid::{hybrid_update_w_in, hybrid_update_w_out};
pub use stdp::{stdp_update, EligibilityTrace};

/// Kuramoto step on sparse CSR connectome.
///
/// Wraps `crate::kernels::phase::kuramoto_prepare` + `kuramoto_update`.
/// Coupling uses the angle-difference identity
/// `sin(θ_j − θ_i) = sin_j·cos_i − cos_j·sin_i` on the vectors produced by
/// `kuramoto_prepare` — same math as the canonical
/// [`kuramoto_step_sparse`](crate::kernels::phase::kuramoto_step_sparse),
/// no per-edge `sin()` call.
pub fn kuramoto_step(
    phases: &mut [f32],
    omegas: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    coupling_from_spikes: Option<&[f32]>,
    dt: f32,
) {
    let n = phases.len();
    let mut cos_out = vec![0.0f32; n];
    let mut sin_out = vec![0.0f32; n];

    // Wrap phases into [0, τ) and compute the shared sin/cos vectors.
    crate::kernels::phase::kuramoto_prepare(phases, &mut cos_out, &mut sin_out);

    // coupling[i] = Σ_j w_ij · sin(θ_j − θ_i), via the identity above.
    let mut coupling = vec![0.0f32; n];
    for i in 0..n {
        let cos_i = cos_out[i];
        let sin_i = sin_out[i];
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j < n {
                coupling[i] += values[idx] * (sin_out[j] * cos_i - cos_out[j] * sin_i);
            }
        }
    }

    // Add explicit coupling if provided
    if let Some(ext) = coupling_from_spikes {
        for i in 0..n.min(ext.len()) {
            coupling[i] += ext[i];
        }
    }

    // Update phases
    crate::kernels::phase::kuramoto_update(phases, omegas, &coupling, dt);
}

/// # ChaosPatternEngine
///
/// High-level orchestrator for chaos/pattern contrastive learning.
pub struct CpeEngine {
    pub config: CpeConfig,
}

impl CpeEngine {
    pub fn new(config: CpeConfig) -> Self {
        Self { config }
    }

    /// Run one Kuramoto step on a sparse connectome.
    pub fn step(
        &self,
        phases: &mut [f32],
        omegas: &[f32],
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
    ) {
        kuramoto_step(
            phases,
            omegas,
            row_ptr,
            col_indices,
            values,
            None,
            self.config.dt,
        );
    }

    /// Compute phase coherence from current phases.
    pub fn coherence(&self, phases: &[f32]) -> f32 {
        phase_coherence(phases)
    }

    /// Detect spikes (rising edge through threshold).
    pub fn spikes(&self, phases: &[f32], prev: &[f32]) -> Vec<i32> {
        detect_spikes(phases, prev, self.config.spike_threshold)
    }

    /// Compute dopamine δ.
    pub fn delta(
        &self,
        reward: f32,
        pat_r: f32,
        pat_h_err: f32,
        chaos_r: f32,
        chaos_h_err: f32,
    ) -> f32 {
        dopamine_delta(&self.config, reward, pat_r, pat_h_err, chaos_r, chaos_h_err)
    }

    /// Apply dopamine-STDP update to connectome weights.
    pub fn stdp_update(&self, w: &mut [f32], delta: f32, elig: &[f32], gate: f32) {
        stdp_update(w, delta, elig, gate, &self.config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpe_engine_creation() {
        let engine = CpeEngine::new(CpeConfig::default());
        assert!((engine.config.lr_stdp - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_kuramoto_step_simple() {
        let n = 8;
        let mut phases = vec![0.0f32; n];
        let omegas = vec![3.0f32; n];
        let row_ptr = vec![0i32, 1, 2, 3, 4, 5, 6, 7, 8];
        let col_indices: Vec<i32> = (1..9).map(|x| x % 8).collect();
        let values = vec![1.0f32; 8];

        kuramoto_step(
            &mut phases,
            &omegas,
            &row_ptr,
            &col_indices,
            &values,
            None,
            0.01,
        );
        assert!(phases.iter().any(|&p| p > 0.0), "phases should advance");
    }

    #[test]
    fn test_full_delta_cycle() {
        let engine = CpeEngine::new(CpeConfig::default());
        let n = 16;
        let mut phases = vec![0.0f32; n];
        let omegas = vec![3.0f32; n];
        let rp = vec![
            0i32, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
        ];
        let ci: Vec<i32> = (0..32).map(|i| (i + 1) % 16).collect();
        let w = vec![1.0f32; 32];

        // Pattern phase: run 10 Kuramoto steps
        let mut pat_coh = 0.0f32;
        let mut spike_counts = vec![0.0f32; n];
        let mut prev = phases.clone();
        for _ in 0..10 {
            kuramoto_step(&mut phases, &omegas, &rp, &ci, &w, None, 0.01);
            pat_coh += engine.coherence(&phases);
            let sp = engine.spikes(&phases, &prev);
            for (i, &s) in sp.iter().enumerate() {
                spike_counts[i] += s as f32;
            }
            prev = phases.clone();
        }
        let pat_r = pat_coh / 10.0;
        let (_, pat_h_err) = spike_entropy(&spike_counts, 10, 3.0);

        // Delta with chaos reference
        let delta = engine.delta(1.0, pat_r, pat_h_err, 0.3, 1.0);
        assert!(delta.is_finite(), "delta should be finite");
    }
}
