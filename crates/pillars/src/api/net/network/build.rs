use super::*;
use super::super::delay::DelayBuffer;
use super::super::graph::{SparseGraph, MAX_DELAY_SLOTS};
use super::super::population::{CableCa, CableVoltage, Population, PopulationId};
use crate::core::buf::AlignedVec;
use crate::zoo::Model;

impl Network {
    /// Build the sparse graph from accumulated edges.
    ///
    /// Must be called after all `add_population` / `connect` calls.
    /// After `build()`, new populations cannot be added.
    ///
    /// # Panics
    /// If called twice: `build()` **drains** `Self::edges`, so a second call
    /// would silently rebuild the graph with no connections. Rebuilding is
    /// rejected explicitly rather than corrupting the graph.
    pub fn build(&mut self) {
        assert!(
            self.graph.is_none(),
            "Network::build() called twice — the graph is already built; connections would be lost"
        );
        let n_syn = self.edges.len();
        let mut from = Vec::with_capacity(n_syn);
        let mut to = Vec::with_capacity(n_syn);
        let mut weights = Vec::with_capacity(n_syn);
        let mut delays = Vec::with_capacity(n_syn);
        let mut syn_types = Vec::with_capacity(n_syn);
        let mut plast_types = Vec::with_capacity(n_syn);

        for (f, t, w, d, st, pt) in self.edges.drain(..) {
            from.push(f);
            to.push(t);
            weights.push(w);
            delays.push(d);
            syn_types.push(st);
            plast_types.push(pt);
        }

        for &st in &syn_types {
            assert!(
                (st as usize) <= self.kinetics.len(),
                "Network::build: syn_type {st} exceeds registered kinetics ({})",
                self.kinetics.len()
            );
        }
        for &pt in &plast_types {
            assert!(
                (pt as usize) <= self.plast_profiles.len(),
                "Network::build: plast_type {pt} exceeds registered plasticity profiles ({})",
                self.plast_profiles.len()
            );
        }
        for src in &self.poisson_sources {
            for &(_, _, _, st, pt) in &src.outgoing {
                assert!(
                    (st as usize) <= self.kinetics.len(),
                    "Network::build: Poisson-source syn_type {st} exceeds registered kinetics ({})",
                    self.kinetics.len()
                );
                assert!(
                    (pt as usize) <= self.plast_profiles.len(),
                    "Network::build: Poisson-source plast_type {pt} exceeds registered plasticity ({})",
                    self.plast_profiles.len()
                );
            }
        }

        // Per-edge short-term plasticity state (R, U, last-spike step).
        self.plast_types = plast_types;
        let profiles = &self.plast_profiles;
        self.plast_r = vec![1.0; n_syn];
        self.plast_u = self
            .plast_types
            .iter()
            .map(|&pt| {
                if pt == 0 {
                    0.0
                } else {
                    profiles[(pt - 1) as usize].u0
                }
            })
            .collect();
        self.plast_last = vec![0u32; n_syn];

        self.graph = Some(SparseGraph::from_triplets_with_types(
            self.total_neurons,
            &from,
            &to,
            &weights,
            &delays,
            &syn_types,
        ));

        // Allocate delay ring buffer + scratch
        self.delay = DelayBuffer::new(self.total_neurons);
        // Allocate kinetic runtime state + per-profile delay rings.
        for k in &mut self.kinetics {
            k.allocate(self.total_neurons);
        }
        self.kinetic_delay = (0..self.kinetics.len())
            .map(|_| DelayBuffer::new(self.total_neurons))
            .collect();
        self.kin_scratch = AlignedVec::new(self.total_neurons, 0.0);
        self.v_scratch = AlignedVec::new(self.total_neurons, 0.0);
        self.i_in = AlignedVec::new(self.total_neurons, 0.0);
        self.spikes_out = vec![0i32; self.total_neurons];
        self.active_scratch = Vec::with_capacity(self.total_neurons / 20 + 8);
    }

    /// Per-edge short-term plasticity profile ids (`0` = none, `1..` = a
    /// profile from [`Network::register_plasticity`]). Sized `n_edges` after
    /// [`Network::build`] — lets callers verify plasticity routing.
    pub fn plast_types(&self) -> &[u8] {
        &self.plast_types
    }

    /// True if the network has been built and is ready to step.
    #[inline(always)]
    pub fn is_built(&self) -> bool {
        self.graph.is_some()
    }

    /// Number of populations.
    #[inline(always)]
    pub fn num_populations(&self) -> usize {
        self.pops.len()
    }

    /// Total number of neurons across all populations.
    #[inline(always)]
    pub fn total_neurons(&self) -> usize {
        self.total_neurons
    }

    /// Current simulation step counter.
    #[inline(always)]
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Read spikes for a population (updated after `step()`).
    ///
    /// Returns an empty slice if the population index is invalid.
    pub fn spikes(&self, pop: PopulationId) -> &[i32] {
        if pop >= self.pops.len() {
            return &[];
        }
        let start = self.pop_offsets[pop];
        let end = self.pop_offsets[pop + 1];
        &self.spikes_out[start..end]
    }

    /// Read a state column for a population.
    pub fn state(&self, pop: PopulationId, col: usize) -> &[f32] {
        if pop >= self.pops.len() {
            return &[];
        }
        self.pops[pop].state_slice(col)
    }

    /// Per-compartment membrane potentials of a **cable** population (a
    /// morphologically-detailed cell), if `pop` is one. Returns `None` for
    /// point-neuron populations and out-of-range ids.
    pub fn cable_voltage(&self, pop: PopulationId) -> Option<CableVoltage<'_>> {
        self.pops.get(pop).and_then(|p| p.cable_voltage())
    }

    /// Per-compartment intracellular Ca²⁺ concentrations of a **cable**
    /// population that tracks a Ca²⁺ `species`, if `pop` is one. Returns `None`
    /// for point-neuron populations, Ca-less cables and out-of-range ids.
    pub fn cable_ca(&self, pop: PopulationId) -> Option<CableCa<'_>> {
        self.pops.get(pop).and_then(|p| p.cable_ca())
    }

    /// Read a parameter column for a population.
    pub fn param_slice(&self, pop: PopulationId, col: usize) -> &[f32] {
        if pop >= self.pops.len() {
            return &[];
        }
        self.pops[pop].param_slice(col)
    }

    /// Read-only access to the built sparse graph (CSR), if `build()` was
    /// called. Used by GPU backends to mirror the projection graph.
    pub fn graph(&self) -> Option<&SparseGraph> {
        self.graph.as_ref()
    }

    /// Mutable access to a population by ID.
    pub fn pop_mut(&mut self, pop: PopulationId) -> Option<&mut dyn Population> {
        self.pops.get_mut(pop).map(|b| b.as_mut())
    }

    /// Typed mutable access to a population (safe downcast).
    ///
    /// Returns `None` if `pop` is invalid or the population is not of type `M`.
    pub fn pop_typed_mut<M: Model + 'static + Send + Sync>(
        &mut self,
        pop: PopulationId,
    ) -> Option<&mut crate::api::NeuralPopulation<M>> {
        self.pops
            .get_mut(pop)?
            .as_any_mut()
            .downcast_mut::<crate::api::NeuralPopulation<M>>()
    }

    /// Configure a population's model parameters / state via its typed model.
    ///
    /// The ergonomic replacement for `pop_typed_mut` — the closure gets the
    /// typed [`crate::api::NeuralPopulation<M>`] directly, so you can `fill_param`/
    /// `fill_state` by the model's column indices.
    ///
    /// # Panics
    /// - If `pop` is out of range
    /// - If the population is **not** of model type `M` (create it with
    ///   [`Network::add_population`] using the same `M`)
    pub fn configure<M: Model + 'static + Send + Sync>(
        &mut self,
        pop: PopulationId,
        f: impl FnOnce(&mut crate::api::NeuralPopulation<M>),
    ) {
        let p = self
            .pop_typed_mut::<M>(pop)
            .expect("Network::configure: population is not of the requested model type M");
        f(p);
    }

    /// Inject external current into specific neurons of a population.
    ///
    /// The current is added to the delay-buffer slot `step_counter % MAX_DELAY`
    /// — the **same** slot the next [`Network::step`] reads, so it drives the
    /// population in that step (same-step arrival, not one step later),
    /// summing with any synaptically-delivered current. This is the safe
    /// way to drive populations from the outside — the node editor's
    /// "Stimulus" node and integration tests use it.
    ///
    /// # Panics
    /// - If `build()` has not been called
    /// - If `pop` is out of range
    /// - If `indices`/`values` lengths differ
    /// - If any neuron index exceeds the population size
    pub fn inject_current(&mut self, pop: PopulationId, indices: &[u32], values: &[f32]) {
        assert!(
            self.graph.is_some(),
            "inject_current() called before build()"
        );
        assert_eq!(
            indices.len(),
            values.len(),
            "inject_current: indices/values length mismatch"
        );
        assert!(
            pop < self.pops.len(),
            "inject_current: population {pop} out of range (have {})",
            self.pops.len()
        );

        let start = self.pop_offsets[pop];
        let count = self.pop_offsets[pop + 1] - start;
        let slot = self.step_counter % MAX_DELAY_SLOTS;
        for i in 0..indices.len() {
            let idx = indices[i] as usize;
            assert!(
                idx < count,
                "inject_current: neuron index {idx} out of bounds (population has {count})"
            );
            assert!(
                values[i].is_finite(),
                "inject_current: value {i} must be finite, got {}",
                values[i]
            );
            self.delay.add(slot, start + idx, values[i]);
        }
    }
}
