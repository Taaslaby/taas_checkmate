use super::*;

use crate::api::population::{CableCa, CableVoltage};
use crate::api::Population;
use crate::core::population::I_SYN_CLAMP_RANGE;

/// A population of `n` identical cable neurons, each discretized from the same
/// morphology. Injected current (nA) is delivered to the soma (root
/// compartment); the observable state is the soma voltage (column 0); a spike
/// is emitted when the soma voltage rises through 0 mV.
///
/// This is the bridge from the cable engine (Фаза 5d) into
/// [`crate::api::Network`]: a NeuroML `<cell>` with morphology/biophysics lowers
/// to this boxed population, so a morphologically-detailed cell actually runs
/// in the network loop.
pub(crate) struct CablePopulation {
    n: usize,
    /// Root (soma) compartment index — identical across all neurons (same tree).
    soma_idx: usize,
    /// One cable solver per neuron.
    neurons: Vec<ActiveCableSolver>,
    /// Soma voltage mirror (contiguous, for `state_slice(0)`).
    soma_v: Vec<f32>,
    /// **Per-compartment** voltage mirror, neuron-major flat
    /// (`v[neuron*n_comps + comp]`) — the cable view for the GUI/analysis
    /// ([`Population::cable_voltage`](crate::api::population::Population)).
    comp_v: Vec<f32>,
    /// **Per-compartment** intracellular Ca²⁺ mirror, neuron-major flat
    /// (`ca[neuron*n_comps + comp]`, mM) — the Ca view for the GUI/analysis
    /// ([`Population::cable_ca`](crate::api::population::Population)). Always
    /// present (compartments without a pool stay 0), cheap to mirror.
    comp_ca: Vec<f32>,
    /// Per-neuron spike-detection flag (was-below-threshold last step).
    was_below: Vec<bool>,
    /// Reused soma-only injection buffer (length = compartments per neuron).
    soma_inj: Vec<f32>,
}

impl CablePopulation {
    /// Build `n` identical cable neurons from a discretized tree, at rest
    /// (`V = e_leak`), with no active channels. Attach channels per compartment
    /// afterwards via [`set_channels`](Self::set_channels).
    pub fn new(tree: &CompartmentTree, n: usize, e_leak: f32, dt: f32, celsius: f32) -> Self {
        let soma_idx = tree.root;
        let n_comps = tree.len();
        let neurons = (0..n)
            .map(|_| ActiveCableSolver::new(tree.clone(), e_leak, dt, celsius))
            .collect();
        Self {
            n,
            soma_idx,
            neurons,
            soma_v: vec![e_leak; n],
            comp_v: vec![e_leak; n * n_comps],
            comp_ca: vec![0.0; n * n_comps],
            was_below: vec![true; n],
            soma_inj: vec![0.0; n_comps],
        }
    }

    /// Number of neurons. Test-only — production sizes the population from
    /// the NeuroML `<population size>` (the `Population` trait `len`).
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Attach active channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_channels(comp, channels.clone());
        }
    }

    /// Attach active Markov channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_markov_channels(comp, channels.clone());
        }
    }

    /// Attach stochastic (discrete-population) channels to compartment `comp`
    /// of every neuron, re-seeding each neuron's RNG with the neuron index so
    /// the `n` copies of the cell get **independent** (but reproducible) noise
    /// streams — otherwise every neuron would follow the identical noisy
    /// trajectory.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_stochastic_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticMarkovActiveChannel>,
    ) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut chans = channels.clone();
            for ch in &mut chans {
                ch.reseed_for_neuron(i);
            }
            neuron.set_stochastic_channels(comp, chans);
        }
    }

    /// Attach stochastic Nernst (Ca²⁺) channel populations to compartment
    /// `comp` of every neuron, re-seeding each neuron's RNG with the neuron
    /// index (independent, reproducible noise streams).
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_stochastic_nernst_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticNernstActiveChannel>,
    ) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let mut chans = channels.clone();
            for ch in &mut chans {
                ch.reseed_for_neuron(i);
            }
            neuron.set_stochastic_nernst_channels(comp, chans);
        }
    }

    /// Attach Nernst (Ca²⁺) channels to compartment `comp` of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_nernst_channels(&mut self, comp: usize, channels: Vec<NernstActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_nernst_channels(comp, channels.clone());
        }
    }

    /// Attach GHK (Ca²⁺, `channelDensityGHK2`) channels to compartment `comp`
    /// of every neuron.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_ghk_channels(&mut self, comp: usize, channels: Vec<GhkActiveChannel>) {
        for neuron in &mut self.neurons {
            neuron.set_ghk_channels(comp, channels.clone());
        }
    }

    /// Override the passive leak of compartment `comp` of every neuron with an
    /// explicit conductance `g_us` (µS) at reversal `e_rev` (mV) — per-group
    /// leak densities. The tree's uniform `g_leak_us` must be zeroed for the
    /// compartment (compile.rs) so the two don't double-count.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_leak(&mut self, comp: usize, g_us: f32, e_rev: f32) {
        for neuron in &mut self.neurons {
            neuron.set_leak(comp, g_us, e_rev);
        }
    }

    /// Attach a Ca²⁺ pool to compartment `comp` of every neuron, initialised to
    /// `initial_mm`.
    ///
    /// # Panics
    /// If `comp >= compartments`.
    pub fn set_ca_pool(&mut self, comp: usize, pool: CaPool, initial_mm: f32) {
        for neuron in &mut self.neurons {
            neuron.set_ca_pool(comp, pool, initial_mm);
        }
        // Seed the Ca mirror with the initial concentration (the per-step
        // mirror in `step` only reflects concentrations *after* a step).
        let n_comps = self.soma_inj.len();
        for i in 0..self.n {
            self.comp_ca[i * n_comps + comp] = initial_mm;
        }
    }

    /// Soma (root) [Ca²⁺] (mM) of neuron `neuron`.
    /// Test-only convenience (the GUI reads Ca via [`Population::cable_ca`]).
    #[cfg(test)]
    pub fn ca_soma(&self, neuron: usize) -> f32 {
        self.neurons[neuron].ca(self.soma_idx)
    }
}

/// SplitMix64-style finaliser — mixes a per-neuron salt into a deterministic
/// 64-bit value so each neuron's noise stream is independent yet reproducible.
pub(crate) fn mix_seed(base: u64, salt: u64) -> u64 {
    let mut z = base ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Population for CablePopulation {
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        assert_eq!(i_in.len(), self.n, "i_in length mismatch");
        assert_eq!(spikes_out.len(), self.n, "spikes_out length mismatch");
        if self.n == 0 {
            return;
        }
        let soma = self.soma_idx;
        let n_comps = self.soma_inj.len();
        for i in 0..self.n {
            self.soma_inj[soma] = i_in[i].clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
            self.neurons[i].step(&self.soma_inj);
            let v = self.neurons[i].v(soma);
            self.soma_v[i] = v;
            // Mirror the whole compartment vector (neuron-major flat) for the
            // cable view — cheap relative to the Hines solve.
            let base = i * n_comps;
            for (j, cv) in self.comp_v[base..base + n_comps].iter_mut().enumerate() {
                *cv = self.neurons[i].v(j);
            }
            for (j, cc) in self.comp_ca[base..base + n_comps].iter_mut().enumerate() {
                *cc = self.neurons[i].ca(j);
            }
            let was = self.was_below[i];
            let is_above = v >= 0.0;
            spikes_out[i] = i32::from(is_above && was);
            self.was_below[i] = !is_above;
        }
        self.soma_inj[soma] = 0.0;
    }

    fn len(&self) -> usize {
        self.n
    }

    fn state_slice(&self, col: usize) -> &[f32] {
        if col == 0 {
            &self.soma_v
        } else {
            &[]
        }
    }

    fn param_slice(&self, _col: usize) -> &[f32] {
        &[]
    }

    fn param_slice_mut(&mut self, _col: usize) -> &mut [f32] {
        &mut []
    }

    fn fill_param(&mut self, _col: usize, _value: f32) {
        // No per-neuron parameter columns: the cable constants live on the
        // (shared) compartment tree, not as per-neuron columns.
    }

    fn fill_state(&mut self, col: usize, value: f32) {
        assert_eq!(
            col, 0,
            "CablePopulation: only state column 0 (soma V) is exposed"
        );
        self.soma_v.fill(value);
        for neuron in &mut self.neurons {
            neuron.set_v(self.soma_idx, value);
        }
        // Keep the cable mirror consistent with the soma override.
        for i in 0..self.n {
            self.comp_v[i * self.soma_inj.len() + self.soma_idx] = value;
        }
    }

    fn cable_voltage(&self) -> Option<CableVoltage<'_>> {
        let n_comps = self.soma_inj.len();
        Some(CableVoltage {
            n_neurons: self.n,
            n_comps,
            v: &self.comp_v,
        })
    }

    fn cable_ca(&self) -> Option<CableCa<'_>> {
        let n_comps = self.soma_inj.len();
        Some(CableCa {
            n_neurons: self.n,
            n_comps,
            ca: &self.comp_ca,
        })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
