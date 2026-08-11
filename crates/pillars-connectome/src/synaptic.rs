//! SynapticBrain — conductance-based connectome with receptor kinetics.
//!
//! [`Brain`](crate::Brain) routes **instant current** (each spike adds its
//! edge weight to the target's input). This module is the conductance-based
//! variant: every synapse carries a [`ReceptorType`] (AMPA/NMDA/GABA/…), a
//! pre-spike *jumps* the post-synaptic conductance, and the resulting current
//! is `Σ g·(E_rev − V)` via `pillars::kernels::synapse::SynapticKinetics`
//! (NMDA has a voltage-dependent Mg²⁺ block).

use pillars::api::SparseGraph;
use pillars::kernels::synapse::{DecayFactors, ReceptorType, SynapticKinetics};
use pillars::zoo::Model;
use pillars::NeuralPopulation;

use crate::brain::sparse_graph;
use crate::data::{build_receptor_types_from_csr, ConnectomeData};

/// Drives a connectome with conductance-based (AMPA/NMDA/GABA) synapses.
///
/// Same neuron engine as [`Brain`](crate::Brain) (`NeuralPopulation<M>`), but
/// the step loop owns the receptor kinetics instead of delegating to the
/// instant-current `api::Network`.
pub struct SynapticBrain<M: Model + 'static + Send + Sync> {
    pop: NeuralPopulation<M>,
    kinetics: SynapticKinetics,
    factors: DecayFactors,
    graph: SparseGraph,
    /// Per-synapse receptor type (u8, parallel to `graph.col_idx`).
    receptor_types: Vec<u8>,
    n: usize,
}

impl<M: Model + 'static + Send + Sync> SynapticBrain<M> {
    /// Build from a connectome. `dt` is the simulation time step (ms), used
    /// for the conductance decay factors.
    ///
    /// # Requires
    /// `M`'s **state column 0** to be the membrane potential `v` (true for
    /// the zoo point models — LIF, Izhikevich, HH, …).
    pub fn from_connectome(conn: &ConnectomeData, dt: f32) -> Self {
        // Fallback: if no per-synapse receptor types were built yet, derive
        // them from the per-neuron NT signs (nt > 0 → AMPA, nt ≤ 0 → GABA_A).
        let receptor_types = if conn.receptor_types.is_empty() {
            build_receptor_types_from_csr(&conn.row_ptr, conn.n_synapses, &conn.nt_types)
        } else {
            conn.receptor_types.clone()
        };
        Self::from_sparse_graph(sparse_graph(conn), receptor_types, dt)
    }

    /// Build from a raw pillars [`SparseGraph`] + explicit per-synapse receptor
    /// types (used where per-synapse delays matter — e.g. the python
    /// `SynapticConnectomeBrain` facade).
    pub fn from_sparse_graph(graph: SparseGraph, receptor_types: Vec<u8>, dt: f32) -> Self {
        let n = graph.num_neurons();
        Self {
            pop: NeuralPopulation::<M>::new(n),
            kinetics: SynapticKinetics::new(n),
            factors: DecayFactors::new(dt),
            graph,
            receptor_types,
            n,
        }
    }

    /// Configure the neuron population (set parameters / initial state).
    pub fn configure(&mut self, f: impl FnOnce(&mut NeuralPopulation<M>)) {
        f(&mut self.pop);
    }

    /// Restore the conductance state (e.g. carry-over from a previous run).
    pub fn set_kinetics(&mut self, k: &SynapticKinetics) {
        self.kinetics = k.clone();
    }

    /// The current conductance state (e.g. to persist across runs).
    pub fn kinetics(&self) -> &SynapticKinetics {
        &self.kinetics
    }

    /// Advance one step with `external_i` (per-neuron input current).
    ///
    /// Returns the spike mask (0/1 per neuron) for this step.
    ///
    /// Per-step lifecycle (matches the legacy conductance sims):
    /// 1. decay all conductances by `exp(−dt/τ_r)`
    /// 2. `I_syn[i] = Σ_r g_r[i]·(E_rev_r − V[i])` (NMDA Mg²⁺-blocked)
    /// 3. step neurons with `i_in = external_i + I_syn`
    /// 4. for each spike, jump the post-synaptic conductances of its
    ///    outgoing synapses (feeds the **next** step's `I_syn`)
    pub fn step(&mut self, external_i: &[f32]) -> Vec<i32> {
        // 1. Decay.
        self.kinetics.decay_all(&self.factors);

        // 2. I_syn from conductances and the pre-step membrane potential.
        let v = self.pop.state_slice(0);
        let mut i_syn = vec![0.0; self.n];
        self.kinetics.compute_currents(v, &mut i_syn);

        // 3. i_in = external + I_syn (the population clamps to ±100 anyway).
        let mut i_in = Vec::with_capacity(self.n);
        for (i, &s) in i_syn.iter().enumerate() {
            i_in.push(s + external_i.get(i).copied().unwrap_or(0.0));
        }

        // 4. Neuron step.
        let mut spikes = vec![0i32; self.n];
        self.pop.step(&i_in, &mut spikes);

        // 5. Jump post-synaptic conductances on spikes.
        for (pre, &sp) in spikes.iter().enumerate() {
            if sp == 0 {
                continue;
            }
            let start = self.graph.row_ptr()[pre] as usize;
            let end = self.graph.row_ptr()[pre + 1] as usize;
            for si in start..end {
                let post = self.graph.col_idx()[si] as usize;
                let w = self.graph.weights()[si];
                let rec = ReceptorType::from_u8(self.receptor_types[si]);
                self.kinetics.jump(post, rec, w);
            }
        }
        spikes
    }

    /// One state column of the population (column 0 = membrane potential).
    pub fn state(&self, col: usize) -> &[f32] {
        self.pop.state_slice(col)
    }

    /// Number of neurons.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the brain has no neurons.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
