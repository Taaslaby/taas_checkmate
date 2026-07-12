//! Spike statistics — firing rates, ISI, CV, SpikeRaster, population rate decode.

use rayon::prelude::*;

// ─── Population Rate Decoder ─────────────────────────────

/// Populational rate decoder: считает спайки в скользящем окне.
pub fn population_rate_decode(
    rates_out: &mut [f32],
    spike_buffer: &[i32],
    n_neurons: usize,
    window: usize,
) {
    let n_out = n_neurons.min(rates_out.len());
    let n_steps = window.min(spike_buffer.len() / n_neurons.max(1));
    let divisor = n_steps.max(1) as f32;
    rates_out[..n_out].par_iter_mut().enumerate().for_each(|(i, rate)| {
        let mut count = 0;
        for t in 0..n_steps {
            if spike_buffer[t * n_neurons + i] != 0 { count += 1; }
        }
        *rate = count as f32 / divisor;
    });
}

// ─── Firing Rates ────────────────────────────────────────

/// Compute firing rates for each neuron from a spike buffer.
/// Buffer layout: `[time × neuron]` — `spike_buffer[t * n_neurons + i]`.
pub fn firing_rates(
    spike_buffer: &[i32],
    n_neurons: usize,
    n_steps: usize,
    dt: f32,
) -> Vec<f32> {
    let mut rates = vec![0.0f32; n_neurons];
    for t in 0..n_steps.min(spike_buffer.len() / n_neurons.max(1)) {
        for i in 0..n_neurons {
            if spike_buffer[t * n_neurons + i] != 0 { rates[i] += 1.0; }
        }
    }
    let duration = (n_steps as f32) * dt;
    if duration > 0.0 { for r in rates.iter_mut() { *r /= duration; } }
    rates
}

/// Compute mean firing rate across all neurons.
pub fn mean_firing_rate(spike_buffer: &[i32], n_neurons: usize, n_steps: usize, dt: f32) -> f32 {
    let rates = firing_rates(spike_buffer, n_neurons, n_steps, dt);
    if rates.is_empty() { 0.0 } else { rates.iter().sum::<f32>() / rates.len() as f32 }
}

// ─── ISI ─────────────────────────────────────────────────

/// Compute Interspike Interval (ISI) for a single neuron's spike train.
pub fn compute_isi(spikes: &[i32], dt: f32) -> Vec<f32> {
    let mut isi = Vec::new();
    let mut last_spike: Option<usize> = None;
    for (t, &s) in spikes.iter().enumerate() {
        if s != 0 {
            if let Some(last) = last_spike {
                let interval = (t - last) as f32 * dt * 1000.0;
                if interval > 0.0 { isi.push(interval); }
            }
            last_spike = Some(t);
        }
    }
    if isi.is_empty() { isi.push(0.0); }
    isi
}

// ─── CV ──────────────────────────────────────────────────

/// Compute Coefficient of Variation (CV) of ISIs for each neuron.
///
/// CV = σ_isi / μ_isi
/// CV ≈ 1 → Poisson, CV < 1 → regular, CV > 1 → bursting
pub fn compute_cvs(spike_buffer: &[i32], n_neurons: usize, n_steps: usize, dt: f32) -> Vec<f32> {
    let mut cvs = vec![0.0f32; n_neurons];
    if n_neurons == 0 || n_steps == 0 { return cvs; }

    for i in 0..n_neurons {
        let mut isi_values = Vec::new();
        let mut last_spike: Option<usize> = None;
        for t in 0..n_steps {
            if spike_buffer[t * n_neurons + i] != 0 {
                if let Some(last) = last_spike {
                    let interval = (t - last) as f32 * dt * 1000.0;
                    if interval > 0.0 { isi_values.push(interval); }
                }
                last_spike = Some(t);
            }
        }
        if isi_values.len() >= 2 {
            let mean_isi: f32 = isi_values.iter().sum::<f32>() / isi_values.len() as f32;
            if mean_isi > 0.0 {
                let variance: f32 = isi_values.iter()
                    .map(|&x| (x - mean_isi).powi(2)).sum::<f32>() / (isi_values.len() - 1) as f32;
                cvs[i] = variance.sqrt() / mean_isi;
            }
        }
    }
    cvs
}

// ═══════════════════════════════════════════════════════════
//  SpikeRaster — spike event collector
// ═══════════════════════════════════════════════════════════

/// SpikeRaster — собирает спайк-события в разрежённом формате.
///
/// # Example
/// ```
/// use taas_core::neuron::SpikeRaster;
///
/// let n_neurons = 64;
/// let mut raster = SpikeRaster::new();
/// let spikes = [0i32; 64];
/// raster.record(0, &spikes, n_neurons);
///
/// let (dense, _) = raster.to_dense(n_neurons);
/// ```
pub struct SpikeRaster {
    events: Vec<(usize, usize)>,
    n_steps: usize,
}

impl SpikeRaster {
    pub fn new() -> Self {
        Self { events: Vec::new(), n_steps: 0 }
    }

    /// Record one step of spikes.
    pub fn record(&mut self, step: usize, spikes: &[i32], n_neurons: usize) {
        let n = spikes.len().min(n_neurons);
        for i in 0..n {
            if spikes[i] != 0 { self.events.push((step, i)); }
        }
        self.n_steps = self.n_steps.max(step + 1);
    }

    /// Convert to dense matrix: `[n_steps × n_neurons]` — 1 = spike.
    pub fn to_dense(&self, n_neurons: usize) -> (Vec<i32>, usize) {
        let n_steps = self.n_steps;
        let mut dense = vec![0i32; n_steps * n_neurons];
        for &(t, i) in &self.events {
            if t < n_steps && i < n_neurons { dense[t * n_neurons + i] = 1; }
        }
        (dense, n_steps)
    }

    /// Export spike raster as CSV string.
    /// Columns: `time_ms, neuron_id` — one row per spike.
    /// Compatible with matplotlib, Excel, and plotting tools.
    pub fn to_csv(&self, dt: f32) -> String {
        use std::fmt::Write;
        let mut csv = String::with_capacity(self.events.len() * 24);
        let _ = writeln!(csv, "time_ms,neuron_id");
        for &(step, neuron) in &self.events {
            let t_ms = step as f32 * dt;
            let _ = writeln!(csv, "{:.2},{}", t_ms, neuron);
        }
        csv
    }

    pub fn events(&self) -> &[(usize, usize)] { &self.events }
    pub fn total_spikes(&self) -> usize { self.events.len() }
    pub fn n_steps(&self) -> usize { self.n_steps }

    pub fn reset(&mut self) {
        self.events.clear();
        self.n_steps = 0;
    }
}

impl Default for SpikeRaster {
    fn default() -> Self { Self::new() }
}

/// Convenience function: create CSV from a SpikeRaster.
/// Equivalent to `raster.to_csv(dt)`.
pub fn spike_raster_to_csv(raster: &SpikeRaster, dt: f32) -> String {
    raster.to_csv(dt)
}
