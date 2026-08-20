// ═══════════════════════════════════════════════════════════
//  Population rate decoder
// ═══════════════════════════════════════════════════════════

/// Population rate decoder: average spike count per neuron over a sliding
/// window. `spike_buffer` layout is `[time × neuron]` — `buffer[t·n + i]`.
pub fn population_rate_decode(
    rates_out: &mut [f32],
    spike_buffer: &[i32],
    n_neurons: usize,
    window: usize,
) {
    let n_out = n_neurons.min(rates_out.len());
    let n_steps = window.min(spike_buffer.len() / n_neurons.max(1));
    let divisor = n_steps.max(1) as f32;
    for i in 0..n_out {
        let mut count = 0;
        for t in 0..n_steps {
            if spike_buffer[t * n_neurons + i] != 0 {
                count += 1;
            }
        }
        rates_out[i] = count as f32 / divisor;
    }
}
