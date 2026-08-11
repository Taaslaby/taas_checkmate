//! # Spike Statistics Microkernels
//!
//! Firing rates, interspike intervals (ISI), coefficient of variation (CV),
//! population rate decoding and the [`SpikeRaster`] event collector.

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

// ═══════════════════════════════════════════════════════════
//  SpikeRaster — spike event collector
// ═══════════════════════════════════════════════════════════

/// SpikeRaster collects spike events in sparse `(step, neuron)` format.
///
/// ```rust
/// use pillars::kernels::stats::SpikeRaster;
/// let n_neurons = 64;
/// let mut raster = SpikeRaster::new();
/// let spikes = [0i32; 64];
/// raster.record(0, &spikes, n_neurons);
/// let (dense, _) = raster.to_dense(n_neurons);
/// ```
pub struct SpikeRaster {
    events: Vec<(usize, usize)>,
    n_steps: usize,
}

impl SpikeRaster {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            n_steps: 0,
        }
    }

    /// Record one step of spikes.
    pub fn record(&mut self, step: usize, spikes: &[i32], n_neurons: usize) {
        let n = spikes.len().min(n_neurons);
        for i in 0..n {
            if spikes[i] != 0 {
                self.events.push((step, i));
            }
        }
        self.n_steps = self.n_steps.max(step + 1);
    }

    /// Convert to a dense matrix `[n_steps × n_neurons]` (1 = spike).
    pub fn to_dense(&self, n_neurons: usize) -> (Vec<i32>, usize) {
        let n_steps = self.n_steps;
        let dense_len = n_steps
            .checked_mul(n_neurons)
            .expect("SpikeRaster::to_dense: matrix size overflows usize");
        let mut dense = vec![0i32; dense_len];
        for &(t, i) in &self.events {
            if t < n_steps && i < n_neurons {
                dense[t * n_neurons + i] = 1;
            }
        }
        (dense, n_steps)
    }

    /// Export spike raster as CSV: `time_ms,neuron_id` — one row per spike.
    /// Compatible with matplotlib, Excel and plotting tools.
    pub fn to_csv(&self, dt: f32) -> String {
        use std::fmt::Write;
        let mut csv = String::with_capacity(self.events.len() * 24);
        let _ = writeln!(csv, "time_ms,neuron_id");
        for &(step, neuron) in &self.events {
            let t_ms = step as f32 * dt;
            let _ = writeln!(csv, "{:.2},{neuron}", t_ms);
        }
        csv
    }

    pub fn events(&self) -> &[(usize, usize)] {
        &self.events
    }
    pub fn total_spikes(&self) -> usize {
        self.events.len()
    }
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    pub fn reset(&mut self) {
        self.events.clear();
        self.n_steps = 0;
    }
}

impl Default for SpikeRaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: create CSV from a [`SpikeRaster`].
pub fn spike_raster_to_csv(raster: &SpikeRaster, dt: f32) -> String {
    raster.to_csv(dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firing_rates_basic() {
        // 2 neurons, 10 steps, neuron 0 spikes every step, neuron 1 never.
        let mut buf = vec![0i32; 20];
        for t in 0..10 {
            buf[t * 2] = 1;
        }
        let rates = firing_rates(&buf, 2, 10, 1.0);
        assert!(
            (rates[0] - 1.0).abs() < 1e-5,
            "neuron 0: 1 spike/ms → 1 Hz/ms"
        );
        assert_eq!(rates[1], 0.0, "neuron 1 silent");
    }

    #[test]
    fn compute_isi_and_cvs_basic() {
        // Regular firing every 10 steps, dt=0.001s (1ms) → ISI = 10·0.001·1000 = 10ms, CV ≈ 0.
        let mut buf = vec![0i32; 100];
        for t in (0..100).step_by(10) {
            buf[t] = 1;
        }
        let isi = compute_isi(&buf[..100], 0.001);
        assert!(
            (isi[0] - 10.0).abs() < 1e-3,
            "regular ISI should be 10ms, got {}",
            isi[0]
        );
        let cvs = compute_cvs(&buf, 1, 100, 0.001);
        assert!(cvs[0] < 0.05, "regular firing → CV ≈ 0, got {}", cvs[0]);
    }

    #[test]
    fn statistics_use_available_frames_when_request_is_longer() {
        // Only two frames are available although callers request ten. The
        // denominator must be two, and CV must not index past the buffer.
        let buf = [1i32, 0, 1, 0];
        let rates = firing_rates(&buf, 2, 10, 1.0);
        assert_eq!(rates, vec![1.0, 0.0]);
        let cvs = compute_cvs(&buf, 2, 10, 1.0);
        assert_eq!(cvs.len(), 2);
    }

    #[test]
    fn population_rate_decode_empty_population_is_safe() {
        let mut rates = [1.0f32];
        population_rate_decode(&mut rates, &[], 0, 10);
        assert_eq!(rates, [1.0]);
    }

    #[test]
    fn population_rate_decode_basic() {
        let mut buf = vec![0i32; 10];
        for t in 0..5 {
            buf[t] = 1;
        }
        let mut rates = vec![0.0f32; 1];
        population_rate_decode(&mut rates, &buf, 1, 10);
        assert!((rates[0] - 0.5).abs() < 1e-5, "5/10 window → 0.5");
    }

    #[test]
    fn spike_raster_roundtrip() {
        let mut raster = SpikeRaster::new();
        let spikes = [1i32, 0, 0, 1];
        raster.record(0, &spikes, 4);
        raster.record(1, &spikes, 4);
        let (dense, n_steps) = raster.to_dense(4);
        assert_eq!(n_steps, 2);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[1], 0);
        assert_eq!(dense[4], 1);
        let csv = raster.to_csv(0.5);
        assert!(csv.starts_with("time_ms,neuron_id\n"), "csv header");
        assert!(csv.contains("0.00,0"), "csv first event");
    }
}
