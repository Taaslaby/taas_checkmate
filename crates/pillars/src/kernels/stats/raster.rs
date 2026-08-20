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
