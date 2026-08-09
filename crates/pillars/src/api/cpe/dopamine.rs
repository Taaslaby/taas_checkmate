//! Dopamine gate — δ = reward + β · [(R_pat − α·|H_pat−H₀|) − (R_chaos − α·|H_chaos−H₀|)]

use super::config::CpeConfig;

/// Compute dopamine signal δ.
///
/// # Formula
/// ```text
/// δ = reward + β · [(R_pat − α·|H_pat−H₀|) − (R_chaos − α·|H_chaos−H₀|)]
/// ```
///
/// Where:
/// - `reward`: task reward (e.g. Pong hits − misses)
/// - `R_pat`: phase coherence under pattern (Pong)
/// - `H_pat`: spike entropy under pattern
/// - `R_chaos`: phase coherence under chaos (noise)
/// - `H_chaos`: spike entropy under chaos
/// - `H₀`: target entropy
/// - `α`: entropy penalty weight
/// - `β`: chaos/pattern gap weight
///
/// δ > 0 → network does better than chaos → STDP strengthens.
/// δ < 0 → network does worse than chaos → STDP weakens.
pub fn dopamine_delta(
    config: &CpeConfig,
    reward: f32,
    pattern_r: f32,
    pattern_h_err: f32,
    chaos_r: f32,
    chaos_h_err: f32,
) -> f32 {
    let pat_util = pattern_r - config.alpha * pattern_h_err;
    let chaos_util = chaos_r - config.alpha * chaos_h_err;
    reward + config.beta * (pat_util - chaos_util)
}

/// Apply timing gate based on fraction of steps in critical zone.
///
/// Gate scales from 0 (no critical activity) to 1 (≥50% critical).
pub fn dopamine_timing_gate(critical_steps: usize, total_steps: usize) -> f32 {
    let total = total_steps.max(1) as f32;
    let critical = critical_steps as f32;
    (critical / total * 2.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_positive_when_pattern_beats_chaos() {
        let config = CpeConfig::default();
        // Pattern: high coherence, low entropy error
        // Chaos: low coherence, high entropy error
        let delta = dopamine_delta(&config, 1.0, 0.9, 0.1, 0.5, 0.8);
        assert!(delta > 0.0, "δ should be positive when pattern beats chaos");
    }

    #[test]
    fn test_delta_negative_when_chaos_beats_pattern() {
        let config = CpeConfig::default();
        let delta = dopamine_delta(&config, 0.0, 0.5, 1.0, 0.9, 0.1);
        assert!(delta < 0.0, "δ should be negative when chaos beats pattern");
    }

    #[test]
    fn test_timing_gate_full() {
        let gate = dopamine_timing_gate(50, 50);
        assert!((gate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_timing_gate_half() {
        let gate = dopamine_timing_gate(25, 100);
        assert!((gate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_timing_gate_zero() {
        let gate = dopamine_timing_gate(0, 100);
        assert!((gate - 0.0).abs() < 1e-6);
    }
}
