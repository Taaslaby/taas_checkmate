//! # taas-connectome — Universal connectome brain simulator.
//!
//! Simulates LIF, Izhikevich, Rate, and ψ-wave dynamics on any CSR connectome.
//!
//! ## Architecture
//!
//! ```text
//! ConnectomeData (CSR matrix + NT types + neuron groups)
//!       │
//! BrainSimulator (LIF / Izhikevich / Rate / Psi)
//!       │
//! SpikeBuffer ───→ SpikeRecorder (DN/MN rates)
//! ```
//!
//! ## Usage (Python via PyO3)
//!
//! ```python
//! import taas_checkmate as tcm
//!
//! # Build connectome from CSR arrays
//! conn = tcm.ConnectomeData(row_ptr, col_indices, values,
//!                           nt_types, [("dn", dn_ids), ("mn", mn_ids)])
//!
//! # Create simulator
//! sim = tcm.BrainSimulator(conn.n_neurons, "lif")
//!
//! # Run simulation
//! spike_buf = sim.step_batch(conn, n_steps=50, dt_ms=0.5, external_I=...)
//!
//! # Extract rates
//! recorder = tcm.SpikeRecorder(conn.n_neurons)
//! rates = recorder.record_group(spike_buf, n_steps=50, dt_ms=0.5,
//!                                indices=conn.group("mn"), max_rate=100.0)
//! ```

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::manual_memcpy)]

pub mod cortical;
pub mod data;
pub mod propagate;
pub mod rcm;
pub mod record;
pub mod runner;
pub mod sim;
pub mod types;

pub use cortical::{
    LaminarConnectomeBuilder, LaminarRule, LayerAssignment, LayerId, LayerStats, ProjectionType,
};
pub use data::{
    BrainIO, ConnectomeData, InputChannel, InputEncoding, NeuronGroup, OutputChannel, OutputReadout,
};
pub use rcm::{
    apply_permutation_to_csr, apply_permutation_to_groups, apply_permutation_to_state, bandwidth,
    build_adjacency, fiedler_vector, mean_bandwidth, reverse_cuthill_mckee, spectral_ordering,
};
pub use record::{SpikeRecorder, SpikeRecordings};
pub use runner::{BrainConfig, BrainRunner, LearningRule, NeuronConfig, PoissonGroup};
pub use sim::{
    csr_propagate, csr_propagate_conductance_from_queue, csr_propagate_from_queue,
    hh_population_pacemaking, hh_population_standard, pad_to, push_spikes_to_queue, sim_gco,
    sim_gco_record, sim_hh_pillars, sim_hh_synaptic_pillars, sim_izhikevich,
    sim_izhikevich_dual, sim_izhikevich_synaptic, sim_lif, sim_lif_synaptic, sim_rate,
    BrainState, GcoParams, IzhikevichDualParams, IzhikevichParams, LifParams, NeuronModel,
    RateParams, SpikeQueue,
};
pub use taas_kernels::neuron::synapse::{DecayFactors, ReceptorType, SynapticKinetics};

#[cfg(feature = "cuda")]
pub use sim::gpu::{sim_izhikevich_gpu, sim_lif_gpu};
