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
    /// Create a generator for `n` neurons with step `dt` (seconds — the spike
    /// probability is `p = rate[Hz]·dt`, NOT `rate·dt/1000`).
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
                self.rng_states[i] = self.rng_states[i]
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223);
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
