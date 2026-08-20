// ═══════════════════════════════════════════════════════════
//  ISI
// ═══════════════════════════════════════════════════════════

/// Interspike intervals (ms) for a single neuron's spike train.
/// Returns `[0.0]` if fewer than two spikes.
pub fn compute_isi(spikes: &[i32], dt: f32) -> Vec<f32> {
    let mut isi = Vec::new();
    let mut last_spike: Option<usize> = None;
    for (t, &s) in spikes.iter().enumerate() {
        if s != 0 {
            if let Some(last) = last_spike {
                let interval = (t - last) as f32 * dt * 1000.0;
                if interval > 0.0 {
                    isi.push(interval);
                }
            }
            last_spike = Some(t);
        }
    }
    if isi.is_empty() {
        isi.push(0.0);
    }
    isi
}

/// Coefficient of variation (CV) of ISIs for each neuron.
///
/// `CV = σ_isi / μ_isi`; CV ≈ 1 → Poisson, CV < 1 → regular, CV > 1 → bursting.
pub fn compute_cvs(spike_buffer: &[i32], n_neurons: usize, n_steps: usize, dt: f32) -> Vec<f32> {
    let mut cvs = vec![0.0f32; n_neurons];
    if n_neurons == 0 || n_steps == 0 {
        return cvs;
    }
    let observed_steps = n_steps.min(spike_buffer.len() / n_neurons);
    if observed_steps == 0 {
        return cvs;
    }

    for i in 0..n_neurons {
        let mut isi_values = Vec::new();
        let mut last_spike: Option<usize> = None;
        for t in 0..observed_steps {
            if spike_buffer[t * n_neurons + i] != 0 {
                if let Some(last) = last_spike {
                    let interval = (t - last) as f32 * dt * 1000.0;
                    if interval > 0.0 {
                        isi_values.push(interval);
                    }
                }
                last_spike = Some(t);
            }
        }
        if isi_values.len() >= 2 {
            let mean_isi: f32 = isi_values.iter().sum::<f32>() / isi_values.len() as f32;
            if mean_isi > 0.0 {
                let variance: f32 = isi_values
                    .iter()
                    .map(|&x| (x - mean_isi).powi(2))
                    .sum::<f32>()
                    / (isi_values.len() - 1) as f32;
                cvs[i] = variance.sqrt() / mean_isi;
            }
        }
    }
    cvs
}
