//! Phase coherence & spike entropy — the heart of Chaos/Pattern comparison.

/// Compute phase coherence R ∈ [0, 1] from Kuramoto oscillator phases.
///
/// R = |Σ exp(i·θⱼ)| / N
/// where θⱼ is the phase of oscillator j.
///
/// High R → oscillators are synchronised (signal).
/// Low R → oscillators are desynchronised (noise/chaos).
pub fn phase_coherence(phases: &[f32]) -> f32 {
    let n = phases.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    for &theta in phases {
        sum_cos += theta.cos();
        sum_sin += theta.sin();
    }
    (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n
}

/// Compute spike entropy H and its deviation from target.
///
/// H = — Σ pᵢ · log₂(pᵢ)
/// where pᵢ = spike_countᵢ / total_steps
///
/// Returns (H, |H — H_target|).
pub fn spike_entropy(counts: &[f32], total_steps: usize, h_target: f32) -> (f32, f32) {
    let total = total_steps.max(1) as f32;
    let eps = 1e-10f32;
    let mut h = 0.0f32;
    for &c in counts {
        let p = (c / total).clamp(eps, 1.0);
        h -= p * p.log2();
    }
    let err = (h - h_target).abs();
    (h, err)
}

/// Detect spikes: phase crosses threshold from below to above.
pub fn detect_spikes(phases: &[f32], phases_prev: &[f32], threshold: f32) -> Vec<i32> {
    let n = phases.len().min(phases_prev.len());
    let mut spikes = vec![0i32; n];
    for i in 0..n {
        if phases[i] > threshold && phases_prev[i] <= threshold {
            spikes[i] = 1;
        }
    }
    spikes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_coherence_all_aligned() {
        let phases = vec![0.0f32, 0.0, 0.0, 0.0];
        let r = phase_coherence(&phases);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "aligned phases should have R=1, got {r}"
        );
    }

    #[test]
    fn test_phase_coherence_uniform() {
        let phases = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let r = phase_coherence(&phases);
        assert!(r < 0.5, "uniform phases should have low R, got {r}");
    }

    #[test]
    fn test_spike_entropy_uniform() {
        let counts = vec![10.0f32; 8];
        let (h, err) = spike_entropy(&counts, 100, 3.0);
        // 8 uniform bins: p=0.1 each, H = -8 * 0.1 * log2(0.1) = 2.658
        assert!(
            (h - 2.658_f32).abs() < 0.01,
            "uniform 8-bin entropy should be ~2.658, got {h}"
        );
        let expected_err = (2.658_f32 - 3.0_f32).abs();
        assert!((err - expected_err).abs() < 0.01);
    }

    #[test]
    fn test_detect_spikes() {
        let phases = vec![0.0f32, 3.0, 1.0, 4.0];
        let prev = vec![0.0f32, 1.0, 0.5, 2.0];
        let spikes = detect_spikes(&phases, &prev, 2.5);
        assert_eq!(
            spikes,
            vec![0, 1, 0, 1],
            "should detect rising edge through 2.5"
        );
    }
}
