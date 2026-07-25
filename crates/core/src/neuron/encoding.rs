//! Spike encoding algorithms — BSA, Phase, TTFS.
//!
//! Convert analog signals to spike trains for SNN input encoding.

use std::f32::consts::PI;

// ─── BSA ──────────────────────────────────────────────────

/// BSA (Bens Spiker Algorithm): encode analog signal into spike train.
///
/// Returns: количество спайков
pub fn bsa_encode(
    spikes_out: &mut [i32],
    signal: &[f32],
    kernel: &[f32],
    reconstructed: &mut [f32],
    threshold: f32,
) -> usize {
    let n = spikes_out.len().min(signal.len());
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

/// Hamming window kernel for BSA encoding.
pub fn bsa_kernel_hamming(len: usize) -> Vec<f32> {
    if len == 0 {
        return vec![];
    }
    if len == 1 {
        return vec![1.0];
    } // avoid 0/0 NaN
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

// ─── Phase Encoding ───────────────────────────────────────

/// Phase Encoding: кодирует значение сигнала в фазу T3 осциллятора.
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

// ─── TTFS ─────────────────────────────────────────────────

/// TTFS (Time-To-First-Spike): кодирует значение амплитудой/задержкой.
///
/// value ∈ [0, 1] → spike через t = (1 - value) × max_delay шагов
pub struct TtfsEncoder {
    pub max_delay: usize,
    pub current_step: usize,
    pub fire_times: Vec<usize>,
    pub fired: Vec<bool>,
}

impl TtfsEncoder {
    pub fn new(n: usize, max_delay: usize) -> Self {
        Self {
            max_delay,
            current_step: 0,
            fire_times: vec![0; n],
            fired: vec![false; n],
        }
    }

    /// Encode values into TTFS spikes over multiple steps.
    /// Возвращает true если есть активные спайки на этом шаге.
    pub fn encode_step(&mut self, values: &[f32], spikes_out: &mut [i32]) -> bool {
        let n = spikes_out
            .len()
            .min(values.len())
            .min(self.fire_times.len());
        let mut any_spike = false;

        if self.current_step == 0 {
            for i in 0..n {
                let val = values[i].clamp(0.0, 1.0);
                self.fire_times[i] = ((1.0 - val) * self.max_delay as f32) as usize;
                self.fired[i] = false;
            }
        }

        for i in 0..n {
            if !self.fired[i] && self.current_step >= self.fire_times[i] {
                spikes_out[i] = 1;
                self.fired[i] = true;
                any_spike = true;
            } else {
                spikes_out[i] = 0;
            }
        }

        self.current_step += 1;
        if self.current_step > self.max_delay {
            self.current_step = 0;
        }
        any_spike
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
        self.fired.fill(false);
        self.fire_times.fill(0);
    }
}
