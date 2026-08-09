//! STDP — Spike-Timing-Dependent Plasticity with eligibility trace.
//!
//! Δw = η · δ · e · gate
//! where:
//! - e: eligibility trace (τ_trace ≈ 85ms)
//! - δ: dopamine signal (chaos/pattern comparison)
//! - gate: timing gate (critical vs non-critical steps)

use super::config::CpeConfig;

/// Eligibility trace: per-synapse accumulation of recent co-activity.
///
/// On pre-neuron spike: `elig[j] += ELIG_ADD` for all outgoing synapses.
/// Every step: `elig *= ELIG_DECAY` (exponential decay).
pub struct EligibilityTrace {
    pub trace: Vec<f32>,
    n_syn: usize,
}

impl EligibilityTrace {
    pub fn new(n_syn: usize) -> Self {
        Self {
            trace: vec![0.0f32; n_syn],
            n_syn,
        }
    }

    /// Update eligibility from spike pattern using CSR adjacency.
    ///
    /// For each neuron that spiked, add `config.elig_add` to all its
    /// outgoing synapses. Then decay the entire trace.
    pub fn update(&mut self, spikes: &[i32], rp: &[i32], config: &CpeConfig) {
        let n = rp.len() - 1;
        for pre in 0..n {
            if pre >= spikes.len() || spikes[pre] == 0 {
                continue;
            }
            let start = rp[pre] as usize;
            let end = rp[pre + 1] as usize;
            for idx in start..end {
                if idx < self.n_syn {
                    self.trace[idx] += config.elig_add;
                }
            }
        }
        // Decay
        for t in &mut self.trace {
            *t *= config.elig_decay;
        }
    }

    /// Reset trace to zero.
    pub fn reset(&mut self) {
        for t in &mut self.trace {
            *t = 0.0;
        }
    }
}

/// Apply dopamine-STDP update to connectome weights.
///
/// Δw = lr_stdp · δ · e · gate
/// w = clip(w, w_min, w_max) — sign-preserving
pub fn stdp_update(w: &mut [f32], delta: f32, elig: &[f32], gate: f32, config: &CpeConfig) {
    let lr = config.lr_stdp;
    let n = w.len().min(elig.len());
    for i in 0..n {
        let is_excitatory = w[i] >= 0.0;
        w[i] += lr * delta * elig[i] * gate;
        // Sign-preserving clip based on initial synapse sign
        if is_excitatory {
            w[i] = w[i].clamp(config.w_min, config.w_max);
        } else {
            w[i] = w[i].clamp(-config.w_max, -config.w_min);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eligibility_trace_basic() {
        let mut et = EligibilityTrace::new(5);
        let spikes = vec![1i32, 1, 1, 1, 1];
        let rp = vec![0i32, 1, 2, 3, 4, 5]; // 5 neurons, 1 synapse each
        let config = CpeConfig::default();

        et.update(&spikes, &rp, &config);
        // Spiking neurons add ELIG_ADD=0.15 then decay 0.90 -> 0.135
        for &t in &et.trace {
            assert!((t - 0.135).abs() < 1e-6, "trace should be 0.135, got {t}");
        }
    }

    #[test]
    fn test_stdp_update_positive_delta() {
        let mut w = vec![1.0f32; 5];
        let elig = vec![1.0f32; 5];
        let config = CpeConfig::default();
        stdp_update(&mut w, 1.0, &elig, 1.0, &config);
        for &wi in &w {
            assert!(wi > 1.0, "positive delta should strengthen weights");
        }
    }

    #[test]
    fn test_stdp_update_negative_delta() {
        let mut w = vec![2.0f32; 5];
        let elig = vec![1.0f32; 5];
        let config = CpeConfig::default();
        stdp_update(&mut w, -1.0, &elig, 1.0, &config);
        for &wi in &w {
            assert!(wi < 2.0, "negative delta should weaken weights");
        }
    }

    #[test]
    fn test_stdp_sign_preserving_clip() {
        let mut w = vec![-0.5f32, 0.5];
        let elig = vec![100.0f32; 2];
        let config = CpeConfig::default();
        stdp_update(&mut w, 10.0, &elig, 1.0, &config);
        // Positive weight should be clipped to w_max (3.0)
        assert!(
            (w[1] - 3.0).abs() < 1e-4,
            "positive should clip to 3.0, got {}",
            w[1]
        );
        // Negative weight should be clipped to -w_min (-0.01) or stay negative
        assert!(w[0] < 0.0, "negative should stay negative, got {}", w[0]);
    }
}
