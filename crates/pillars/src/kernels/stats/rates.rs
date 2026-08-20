// ═══════════════════════════════════════════════════════════
//  Firing rates
// ═══════════════════════════════════════════════════════════

/// Firing rates per neuron (Hz) from a spike buffer `[time × neuron]`.
pub fn firing_rates(spike_buffer: &[i32], n_neurons: usize, n_steps: usize, dt: f32) -> Vec<f32> {
    let mut rates = vec![0.0f32; n_neurons];
    let available_steps = spike_buffer.len().checked_div(n_neurons).unwrap_or(0);
    let observed_steps = n_steps.min(available_steps);
    for t in 0..observed_steps {
        for i in 0..n_neurons {
            if spike_buffer[t * n_neurons + i] != 0 {
                rates[i] += 1.0;
            }
        }
    }
    let duration = (observed_steps as f32) * dt;
    if duration > 0.0 {
        for r in rates.iter_mut() {
            *r /= duration;
        }
    }
    rates
}

/// Mean firing rate (Hz) across all neurons.
pub fn mean_firing_rate(spike_buffer: &[i32], n_neurons: usize, n_steps: usize, dt: f32) -> f32 {
    let rates = firing_rates(spike_buffer, n_neurons, n_steps, dt);
    if rates.is_empty() {
        0.0
    } else {
        rates.iter().sum::<f32>() / rates.len() as f32
    }
}
