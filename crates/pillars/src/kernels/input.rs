//! # Input Encoding Microkernels — Poisson, BSA, TTFS
//!
//! Converts analog signals into spike trains for SNN input encoding:
//! - [`PoissonGenerator`] — rate-coded Poisson spike trains
//! - [`bsa_encode`] — Bens Spiker Algorithm
//! - [`phase_encode`] — encode value into T3 oscillator phase
//! - [`TtfsEncoder`] — Time-To-First-Spike encoding

use core::f32::consts::PI;

/// 24-bit RNG scale — for Poisson spike generation (LCG state >> 8).
use crate::kernels::RNG_SCALE_24;

// ═══════════════════════════════════════════════════════════
//  Poisson generator
// ═══════════════════════════════════════════════════════════

/// Poisson spike generator with per-neuron LCG RNG streams.
///
/// Each neuron has its own rate (Hz). At each timestep:
/// `P(spike) = rate[i]·dt`. Rate-coded input for SNNs.
pub struct PoissonGenerator {
    pub rates: Vec<f32>,
    pub rng_states: Vec<u32>,
    n: usize,
    dt: f32,
}

impl PoissonGenerator {
    /// Create a generator for `n` neurons with step `dt` (ms).
    /// `n` is padded to a multiple of 8 for SIMD-friendly layouts.
    pub fn new(n: usize, dt: f32, seed: u32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        let mut rng_states = Vec::with_capacity(n_pad);
        let mut s = seed;
        for _ in 0..n_pad {
            rng_states.push(s);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        }
        Self {
            rates: vec![10.0; n_pad],
            rng_states,
            n: n_pad,
            dt,
        }
    }

    pub fn set_rate(&mut self, idx: usize, rate_hz: f32) {
        if idx < self.n {
            self.rates[idx] = rate_hz;
        }
    }

    pub fn set_rates(&mut self, rates: &[f32]) {
        let n = self.n.min(rates.len());
        self.rates[..n].copy_from_slice(&rates[..n]);
    }

    /// Generate one step of Poisson spikes into `spikes_out` (0/1).
    pub fn step(&mut self, spikes_out: &mut [i32]) {
        let n = self.n.min(spikes_out.len());
        for i in 0..n {
            let p = (self.rates[i] * self.dt).clamp(0.0, 1.0);
            if p > 0.0 {
                self.rng_states[i] = self.rng_states[i].wrapping_mul(1664525).wrapping_add(1013904223);
                let r = (self.rng_states[i] >> 8) as f32 * RNG_SCALE_24;
                spikes_out[i] = if r < p { 1 } else { 0 };
            } else {
                spikes_out[i] = 0;
            }
        }
        for s in spikes_out[n..].iter_mut() {
            *s = 0;
        }
    }

    pub fn dt(&self) -> f32 {
        self.dt
    }
    pub fn rates(&self) -> &[f32] {
        &self.rates[..self.n]
    }
    pub fn n_neurons(&self) -> usize {
        self.n
    }
}

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
    let n = spikes_out.len().min(phases.len()).min(cos_phases.len()).min(signal.len());
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

// ═══════════════════════════════════════════════════════════
//  TTFS (Time-To-First-Spike)
// ═══════════════════════════════════════════════════════════

/// TTFS: encodes a value in [0, 1] as a spike delayed by `(1 − value)·max_delay`
/// steps — the larger the value, the earlier the spike.
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
    /// Returns `true` if there are active spikes on this step.
    pub fn encode_step(&mut self, values: &[f32], spikes_out: &mut [i32]) -> bool {
        let n = spikes_out.len().min(values.len()).min(self.fire_times.len());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_zero_rate_no_spikes() {
        let mut gen = PoissonGenerator::new(16, 1.0, 42);
        gen.set_rates(&[0.0f32; 16]);
        let mut spikes = vec![0i32; 16];
        for _ in 0..1000 {
            gen.step(&mut spikes);
        }
        assert!(spikes.iter().all(|&s| s == 0), "0 Hz rate must not spike");
    }

    #[test]
    fn poisson_high_rate_spikes_sometimes() {
        let mut gen = PoissonGenerator::new(16, 1.0, 42);
        gen.set_rates(&[100.0f32; 16]); // p = 100·1.0 clamped to 1.0
        let mut spikes = vec![0i32; 16];
        let mut total = 0;
        for _ in 0..100 {
            gen.step(&mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "high rate should spike");
    }

    #[test]
    fn poisson_rates_kept_in_padded_bounds() {
        let gen = PoissonGenerator::new(10, 0.5, 1);
        assert_eq!(gen.n_neurons(), 16, "n padded to multiple of 8");
    }

    #[test]
    fn bsa_encodes_step_signal() {
        let signal = vec![1.0f32; 8];
        let kernel = bsa_kernel_hamming(4);
        let mut spikes = vec![0i32; 8];
        let mut recon = vec![0.0f32; 8];
        let count = bsa_encode(&mut spikes, &signal, &kernel, &mut recon, 0.5);
        assert!(count > 0, "step signal should produce spikes");
        // Reconstruction should track the signal (last sample near 1)
        assert!((recon[7] - signal[7]).abs() < 0.6, "recon should approach signal");
    }

    #[test]
    fn bsa_kernel_normalized() {
        let k = bsa_kernel_hamming(8);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "kernel should be normalized, sum={sum}");
    }

    #[test]
    fn ttfs_higher_value_spikes_earlier() {
        let mut enc = TtfsEncoder::new(2, 10);
        let values = vec![1.0f32, 0.0];
        let mut spikes = vec![0i32; 2];
        let mut fired_early = false;
        for _ in 0..12 {
            let any = enc.encode_step(&values, &mut spikes);
            if any && spikes[0] == 1 && spikes[1] == 0 {
                fired_early = true;
            }
        }
        assert!(fired_early, "value=1 should spike before value=0");
    }

    #[test]
    fn phase_encode_basic() {
        let mut spikes = vec![0i32; 4];
        let mut phases = vec![0.0f32; 4];
        let cos_phases = vec![0.5f32; 4];
        let signal = vec![0.1f32; 4];
        let count = phase_encode(&mut spikes, &mut phases, &cos_phases, &signal, 0.3, 1.0);
        assert_eq!(count, 4, "cos=0.5 > 0.3 → all spike");
        assert!((phases[0] - 0.1).abs() < 1e-6, "phase should advance by signal·gain");
    }
}
