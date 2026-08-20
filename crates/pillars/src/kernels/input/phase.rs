// ═══════════════════════════════════════════════════════════
//  Phase encoding
// ═══════════════════════════════════════════════════════════

/// Phase encoding: advance `phases` by `signal·gain` and emit a spike when
/// the precomputed `cos(phase) > threshold`.
///
/// Returns the number of spikes.
pub fn phase_encode(
    spikes_out: &mut [i32],
    phases: &mut [f32],
    cos_phases: &[f32],
    signal: &[f32],
    threshold: f32,
    gain: f32,
) -> usize {
    let n = spikes_out
        .len()
        .min(phases.len())
        .min(cos_phases.len())
        .min(signal.len());
    let mut spike_count = 0;
    for i in 0..n {
        phases[i] += signal[i] * gain;
        if cos_phases[i] > threshold {
            spikes_out[i] = 1;
            spike_count += 1;
        } else {
            spikes_out[i] = 0;
        }
    }
    spike_count
}
