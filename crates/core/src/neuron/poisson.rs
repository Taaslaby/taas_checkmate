//! Poisson spike generator — generates Poisson spike trains at given rates.
//!
//! Uses per-neuron LCG RNG streams for independent spike generation.
//!
//! если что это не отравление это просто генератор случайных чисел ахахах

use rayon::prelude::*;

/// Poisson spike generator.
///
/// Each neuron has its own rate (Hz). At each timestep:
///   P(spike) = rate[i] * dt
///
/// For rate coding input to SNN.
pub struct PoissonGenerator {
    pub rates: Vec<f32>,
    pub rng_states: Vec<u32>,
    n: usize,
    dt: f32,
}

impl PoissonGenerator {
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

    /// Generate one step of Poisson spikes.
    pub fn step(&mut self, spikes_out: &mut [i32]) {
        let n = self.n.min(spikes_out.len());
        let dt = self.dt;
        self.rng_states[..n]
            .par_chunks_exact_mut(8)
            .zip(self.rates[..n].par_chunks_exact(8))
            .zip(spikes_out[..n].par_chunks_exact_mut(8))
            .for_each(|((states8, rates8), sp8)| {
                for i in 0..8 {
                    let p = (rates8[i] * dt).clamp(0.0, 1.0);
                    if p > 0.0 {
                        states8[i] = states8[i].wrapping_mul(1664525).wrapping_add(1013904223);
                        let r = (states8[i] >> 8) as f32 * super::RNG_SCALE_24;
                        sp8[i] = if r < p { 1 } else { 0 };
                    } else {
                        sp8[i] = 0;
                    }
                }
            });
        for i in n..self.n.min(spikes_out.len()) {
            spikes_out[i] = 0;
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
