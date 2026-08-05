//! Connectome data — pillars-native CSR graph + per-synapse receptor types.
//!
//! The legacy `taas-connectome::ConnectomeData` was a CSV/adjacency struct
//! carrying its own spike-queue machinery. Here the graph IS a
//! [`pillars::api::SparseGraph`] (the single source of truth for adjacency);
//! this crate only adds the connectome-specific per-synapse receptor types
//! and named neuron groups that `SparseGraph` does not model.

use pillars::api::SparseGraph;

/// A named group of neurons (e.g. `"sensory"`, `"motor"`, `"dn"`).
#[derive(Debug, Clone, Default)]
pub struct NeuronGroup {
    pub name: String,
    pub indices: Vec<usize>,
}

/// Pillars-native connectome: a CSR [`SparseGraph`] plus per-synapse receptor
/// types and named neuron groups.
#[derive(Debug, Clone)]
pub struct ConnectomeData {
    pub graph: SparseGraph,
    /// Per-synapse neurotransmitter / receptor type, parallel to `graph.col_idx`
    /// (e.g. 0 = AMPA, 1 = NMDA, 2 = GABA_A, …; see `pillars::kernels::synapse`).
    pub receptor_types: Vec<u8>,
    /// Named neuron groups.
    pub groups: Vec<NeuronGroup>,
}

impl ConnectomeData {
    /// Build from CSR arrays in the legacy connectome format:
    /// `row_ptr`/`col_idx`/`weights` per synapse, optional per-synapse axonal
    /// `delays` (steps) and `receptor_types`.
    ///
    /// # Panics
    /// - If `col_idx`/`weights`/`delays`/`receptor_types` lengths mismatch
    /// - If the CSR shape is invalid (see [`SparseGraph::from_csr`])
    pub fn from_csr(
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        weights: Vec<f32>,
        delays: Vec<u8>,
        receptor_types: Vec<u8>,
    ) -> Self {
        let graph = SparseGraph::from_csr(row_ptr, col_idx, &weights, &delays);
        assert_eq!(
            receptor_types.len(),
            graph.num_synapses(),
            "receptor_types length must equal the synapse count"
        );
        Self {
            graph,
            receptor_types,
            groups: Vec::new(),
        }
    }

    #[inline]
    pub fn n_neurons(&self) -> usize {
        self.graph.num_neurons()
    }

    #[inline]
    pub fn n_synapses(&self) -> usize {
        self.graph.num_synapses()
    }
}
