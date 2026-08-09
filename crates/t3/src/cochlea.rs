//! # Cochlea — auditory filter bank with envelope followers.
//!
//! Из оригинального taas_chaser cochlea.rs.
//!
//! ```text
//! Audio → [Biquad Bandpass × N] → [Envelope Follower × N] → envelopes
//!                                    ↓ (threshold)
//!                                 spike detection
//! ```

use super::TWO_PI;

/// Validate Biquad design parameters.
///
/// # Panics
/// If `fs <= 0`, `q <= 0`, or `fc` outside `(0, fs/2)` — any of these
/// produces an unstable or degenerate IIR filter (or a non-finite
/// coefficient via `w0.sin()` / `w0.cos()`).
fn validate_filter_params(fc: f32, q: f32, fs: f32) {
    assert!(
        fs > 0.0 && fs.is_finite(),
        "Biquad: sample_rate must be finite and > 0, got {fs}"
    );
    assert!(
        q > 0.0 && q.is_finite(),
        "Biquad: Q must be finite and > 0, got {q}"
    );
    assert!(
        fc.is_finite() && fc > 0.0 && fc < fs / 2.0,
        "Biquad: center frequency {fc} must be in (0, fs/2 = {})",
        fs / 2.0
    );
}

// ─── Biquad IIR Filter ───────────────────────────────────

/// Biquad IIR filter (Transposed Direct Form II).
///
/// Audio EQ Cookbook formulas for bandpass, lowpass, highpass.
pub struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
    pub x1: f32,
    pub x2: f32,
}

impl Biquad {
    /// Create passthrough filter.
    pub fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
        }
    }

    /// Init as bandpass: fc, q, sample_rate
    ///
    /// # Panics
    /// If `fs <= 0`, `q <= 0`, or `fc` outside `(0, fs/2)`.
    pub fn bandpass(fc: f32, q: f32, fs: f32) -> Self {
        validate_filter_params(fc, q, fs);
        let w0 = TWO_PI * fc / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: -2.0 * w0.cos() / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
        }
    }

    /// Init as lowpass: fc, q, sample_rate
    ///
    /// # Panics
    /// If `fs <= 0`, `q <= 0`, or `fc` outside `(0, fs/2)`.
    pub fn lowpass(fc: f32, q: f32, fs: f32) -> Self {
        validate_filter_params(fc, q, fs);
        let w0 = TWO_PI * fc / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let cos_w0 = w0.cos();
        Self {
            b0: (1.0 - cos_w0) / (2.0 * a0),
            b1: (1.0 - cos_w0) / a0,
            b2: (1.0 - cos_w0) / (2.0 * a0),
            a1: -2.0 * cos_w0 / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
        }
    }

    /// Tick one sample: Direct Form II transposed
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.x1;
        self.x1 = self.b1 * x - self.a1 * y + self.x2;
        self.x2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
    }
}

impl Default for Biquad {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Cochlea ─────────────────────────────────────────────

/// Auditory filter bank: Biquad bandpass filters + envelope followers.
///
/// - `n_bands` фильтров, логарифмически распределённых от `low_hz` до `high_hz`
/// - Каждый канал: Biquad bandpass → envelope follower (leaky integrator)
/// - `smoothing`: константа envelope (0.995 ≈ 4.5ms @ 44.1KHz)
pub struct Cochlea {
    pub filters: Vec<Biquad>,
    pub envelopes: Vec<f32>,
    pub center_freqs: Vec<f32>,
    pub n_bands: usize,
    pub sample_rate: f32,
    pub smoothing: f32,
    pub q: f32,
}

impl Cochlea {
    pub fn new(n_bands: usize, low_hz: f32, high_hz: f32, sample_rate: f32) -> Self {
        // Parameter validation: an unstable filter bank silently destroys
        // the signal (IIR blow-up → NaN) instead of failing fast.
        assert!(
            sample_rate > 0.0 && sample_rate.is_finite(),
            "Cochlea: sample_rate must be finite and > 0, got {sample_rate}"
        );
        assert!(
            low_hz > 0.0 && low_hz.is_finite(),
            "Cochlea: low_hz must be finite and > 0, got {low_hz}"
        );
        assert!(
            high_hz > low_hz && high_hz.is_finite(),
            "Cochlea: high_hz ({high_hz}) must be > low_hz ({low_hz})"
        );
        assert!(
            high_hz < sample_rate / 2.0,
            "Cochlea: high_hz ({high_hz}) must be below Nyquist (fs/2 = {})",
            sample_rate / 2.0
        );

        let q = 0.5;
        let smoothing = 0.995;
        let log_low = low_hz.ln();
        let log_high = high_hz.ln();
        let n = n_bands.max(1);

        let mut filters = Vec::with_capacity(n);
        let mut center_freqs = Vec::with_capacity(n);
        for i in 0..n {
            // n == 1 → single filter at low_hz (avoids 0/0 → NaN in t).
            let t = if n > 1 {
                i as f32 / (n - 1) as f32
            } else {
                0.0
            };
            // Clamp bounds are ordered (min ≤ max) so `clamp` cannot panic.
            let upper = (sample_rate / 2.0 - 1.0).max(20.0);
            let fc = (log_low + t * (log_high - log_low))
                .exp()
                .clamp(20.0, upper);
            filters.push(Biquad::bandpass(fc, q, sample_rate));
            center_freqs.push(fc);
        }

        Self {
            filters,
            envelopes: vec![0.0; n],
            center_freqs,
            n_bands: n,
            sample_rate,
            smoothing,
            q,
        }
    }

    /// Process one audio sample through all filters.
    /// Returns envelope vector slice.
    pub fn tick(&mut self, sample: f32) -> &[f32] {
        for i in 0..self.n_bands {
            let y = self.filters[i].tick(sample);
            self.envelopes[i] =
                self.smoothing * self.envelopes[i] + (1.0 - self.smoothing) * y.abs();
        }
        &self.envelopes
    }

    /// Reset all filters and envelopes.
    pub fn reset(&mut self) {
        for f in self.filters.iter_mut() {
            f.reset();
        }
        self.envelopes.fill(0.0);
    }

    /// Detect spikes: returns indices where envelope > threshold.
    pub fn detect_spikes(&self, threshold: f32) -> Vec<usize> {
        self.envelopes
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get current envelopes as slice.
    pub fn get_envelopes(&self) -> &[f32] {
        &self.envelopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_bandpass() {
        let mut bp = Biquad::bandpass(440.0, 0.5, 16000.0);
        let y = bp.tick(1.0);
        assert!(y.is_finite());
    }

    #[test]
    fn test_cochlea_tick() {
        // high_hz strictly below Nyquist (fs/2 = 8000): a band at exactly
        // Nyquist would be a degenerate filter and is rejected by validation.
        let mut c = Cochlea::new(8, 100.0, 7500.0, 16000.0);
        assert_eq!(c.n_bands, 8);
        assert!(c.center_freqs[0] > 90.0 && c.center_freqs[0] < 110.0);
        // Process a sine wave
        for i in 0..1000 {
            let sample = (std::f32::consts::PI * 2.0 * 440.0 * i as f32 / 16000.0).sin();
            c.tick(sample);
        }
        // At least one envelope should respond to 440Hz
        assert!(
            c.envelopes.iter().any(|&e| e > 0.01),
            "envelopes should respond to 440Hz: {:?}",
            c.envelopes
        );
    }

    #[test]
    fn test_cochlea_spike_detection() {
        let mut c = Cochlea::new(4, 100.0, 4000.0, 16000.0);
        for i in 0..500 {
            let sample = (std::f32::consts::PI * 2.0 * 1000.0 * i as f32 / 16000.0).sin();
            c.tick(sample);
        }
        let spikes = c.detect_spikes(0.1);
        assert!(!spikes.is_empty(), "should detect spikes near 1KHz");
    }

    #[test]
    fn test_biquad_reset() {
        let mut bp = Biquad::bandpass(440.0, 0.5, 16000.0);
        bp.tick(1.0);
        bp.reset();
        assert_eq!(bp.x1, 0.0);
        assert_eq!(bp.x2, 0.0);
    }
}
