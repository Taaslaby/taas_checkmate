//! Brain — pillars-native connectome orchestrator.
//!
//! Builds a [`pillars::api::Network`] from a [`ConnectomeData`] and drives the
//! simulation loop: [`NeuralPopulation`]s, CSR spike routing with axonal delays
//! (ring buffer), external current injection.
//!
//! This replaces the legacy `taas-connectome` "assembled from corners"
//! simulators — the engine here is 100% pillars.
//!
//! [`NeuralPopulation`]: pillars::NeuralPopulation

use std::collections::HashMap;

use pillars::api::{Edge, Network, PopulationId};
use pillars::zoo::Model;

use crate::data::ConnectomeData;

/// Drives a connectome on the pillars engine.
///
/// `M` is the neuron model for every connectome neuron. Signal flows along the
/// CSR edges; each spike adds its edge weight to the target's input current
/// with the edge's axonal delay.
///
/// Two construction modes:
/// - [`Brain::from_connectome`] — one population for the whole connectome.
/// - [`Brain::from_groups`] — one population per named group, with
///   cross-group spike routing (the "sensory → motor" style orchestration).
pub struct Brain<M: Model + 'static> {
    net: Network,
    /// Single-population mode: the whole connectome.
    pop: Option<PopulationId>,
    /// Group mode: group name → population id.
    pops: HashMap<String, PopulationId>,
    _marker: std::marker::PhantomData<M>,
}

impl<M: Model + 'static> Brain<M> {
    /// Build a one-population brain: every connectome neuron is an `M` neuron,
    /// wired exactly as the CSR graph says (weights + axonal delays).
    ///
    /// Call [`Brain::configure`] afterwards to set the model parameters and
    /// initial state, then [`Brain::step`] to run.
    pub fn from_connectome(conn: &ConnectomeData) -> Self {
        let n = conn.n_neurons();
        let mut net = Network::new();
        let pop = net.add_population::<M>(n);

        // Flatten the CSR graph into edges (weight + axonal delay each).
        let g = &conn.graph;
        let mut edges = Vec::with_capacity(conn.n_synapses());
        for i in 0..n {
            let row = g.row(i);
            for (k, &j) in row.col.iter().enumerate() {
                edges.push(Edge::new(i as u32, j, row.weights[k], row.delays[k]));
            }
        }
        net.connect_edges(pop, pop, &edges);
        net.build();

        Self {
            net,
            pop: Some(pop),
            pops: HashMap::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Build a group-mode brain: one population per named group, connected
    /// across groups per the CSR graph.
    ///
    /// # Panics
    /// - If `conn.groups` is empty
    /// - If any neuron is in multiple groups or in no group
    pub fn from_groups(conn: &ConnectomeData) -> Self {
        assert!(!conn.groups.is_empty(), "from_groups requires at least one group");
        let n = conn.n_neurons();
        let mut net = Network::new();

        // 1. One population per group (ordered by group index), plus a
        //    global → (group, local) map to translate CSR edges.
        let mut group_pop_ids = Vec::with_capacity(conn.groups.len());
        let mut pops = HashMap::new();
        let mut global_to_local = vec![(usize::MAX, usize::MAX); n];
        for (gi, group) in conn.groups.iter().enumerate() {
            let pop = net.add_population::<M>(group.indices.len());
            group_pop_ids.push(pop);
            pops.insert(group.name.clone(), pop);
            for (li, &neuron) in group.indices.iter().enumerate() {
                assert!(neuron < n, "group neuron index {neuron} out of bounds");
                assert!(
                    global_to_local[neuron].0 == usize::MAX,
                    "neuron {neuron} in multiple groups"
                );
                global_to_local[neuron] = (gi, li);
            }
        }
        for (neuron, &(gi, _)) in global_to_local.iter().enumerate() {
            assert!(gi != usize::MAX, "neuron {neuron} is not in any group");
        }

        // 2. Cross-group routing: translate each CSR edge to (pop, local) pairs.
        let g = &conn.graph;
        for i in 0..n {
            let (pi, li) = global_to_local[i];
            let row = g.row(i);
            for (k, &j) in row.col.iter().enumerate() {
                let (pj, lj) = global_to_local[j as usize];
                net.connect(
                    group_pop_ids[pi],
                    &[li as u32],
                    group_pop_ids[pj],
                    &[lj as u32],
                    &[row.weights[k]],
                    &[row.delays[k]],
                );
            }
        }
        net.build();

        Self {
            net,
            pop: None,
            pops,
            _marker: std::marker::PhantomData,
        }
    }

    /// Configure the single population (set parameters / initial state).
    ///
    /// # Panics
    /// In group mode — use [`Brain::configure_group`] instead.
    pub fn configure(&mut self, f: impl FnOnce(&mut pillars::NeuralPopulation<M>)) {
        let pop = self.pop.expect("group mode: use configure_group");
        self.net.configure::<M>(pop, f);
    }

    /// Configure one group's population.
    ///
    /// # Panics
    /// If `name` is not a group.
    pub fn configure_group(&mut self, name: &str, f: impl FnOnce(&mut pillars::NeuralPopulation<M>)) {
        let pop = *self.pops.get(name).expect("unknown group");
        self.net.configure::<M>(pop, f);
    }

    /// Advance one simulation step (all populations + spike routing).
    pub fn step(&mut self) {
        self.net.step();
    }

    /// Inject external current into specific neurons of the single population.
    /// Sums with any synaptically-delivered current.
    ///
    /// # Panics
    /// In group mode — use [`Brain::group_inject`] instead.
    pub fn inject(&mut self, indices: &[u32], values: &[f32]) {
        let pop = self.pop.expect("group mode: use group_inject");
        self.net.inject_current(pop, indices, values);
    }

    /// Inject external current into neurons of one group.
    ///
    /// # Panics
    /// If `name` is not a group.
    pub fn group_inject(&mut self, name: &str, indices: &[u32], values: &[f32]) {
        let pop = *self.pops.get(name).expect("unknown group");
        self.net.inject_current(pop, indices, values);
    }

    /// Current spike mask (0/1 per neuron) of the single population.
    ///
    /// # Panics
    /// In group mode — use [`Brain::group_spikes`] instead.
    pub fn spikes(&self) -> &[i32] {
        let pop = self.pop.expect("group mode: use group_spikes");
        self.net.spikes(pop)
    }

    /// Current spike mask of one group.
    ///
    /// # Panics
    /// If `name` is not a group.
    pub fn group_spikes(&self, name: &str) -> &[i32] {
        let pop = *self.pops.get(name).expect("unknown group");
        self.net.spikes(pop)
    }

    /// One state column (e.g. column 0 = membrane potential) of the single population.
    ///
    /// # Panics
    /// In group mode — use [`Brain::group_state`] instead.
    pub fn state(&self, col: usize) -> &[f32] {
        let pop = self.pop.expect("group mode: use group_state");
        self.net.state(pop, col)
    }

    /// One state column of one group.
    ///
    /// # Panics
    /// If `name` is not a group.
    pub fn group_state(&self, name: &str, col: usize) -> &[f32] {
        let pop = *self.pops.get(name).expect("unknown group");
        self.net.state(pop, col)
    }

    /// Number of steps run so far.
    pub fn step_counter(&self) -> usize {
        self.net.step_counter()
    }
}
