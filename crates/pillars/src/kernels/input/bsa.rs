use core::f32::consts::PI;

// ═══════════════════════════════════════════════════════════
//  BSA (Bens Spiker Algorithm)
// ═══════════════════════════════════════════════════════════

/// BSA: encode an analog signal into a spike train.
///
/// Returns the number of spikes emitted.
///
/// At each timestep, if `|signal[t] − reconstructed[t]| > threshold`, emit a
/// spike and add `kernel ⊗ sign(err)` to the reconstruction.
pub fn bsa_encode(
    spikes_out: &mut [i32],
    signal: &[f32],
    kernel: &[f32],
    reconstructed: &mut [f32],
    threshold: f32,
) -> usize {
    let n = spikes_out.len().min(signal.len()).min(reconstructed.len());
    let k_len = kernel.len();
    let mut spike_count = 0;

    for t in 0..n {
        let err = signal[t] - reconstructed[t];
        if err.abs() > threshold {
            spikes_out[t] = 1;
            spike_count += 1;
            for k in 0..k_len {
                let idx = t + k;
                if idx < n {
                    reconstructed[idx] += kernel[k] * err.signum();
                }
            }
        } else {
            spikes_out[t] = 0;
        }
    }
    spike_count
}

/// Hamming window kernel for BSA encoding (normalized to sum 1).
pub fn bsa_kernel_hamming(len: usize) -> Vec<f32> {
    if len == 0 {
        return vec![];
    }
    if len == 1 {
        return vec![1.0];
    }
    let mut k = vec![0.0; len];
    let inv = 1.0 / (len - 1) as f32;
    for i in 0..len {
        k[i] = 0.54 - 0.46 * (2.0 * PI * i as f32 * inv).cos();
    }
    let sum: f32 = k.iter().sum();
    if sum > 1e-10 {
        for kk in k.iter_mut() {
            *kk /= sum;
        }
    }
    k
}
