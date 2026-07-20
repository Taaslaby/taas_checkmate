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
#![allow(clippy::too_many_arguments)]

pub mod data;
pub mod rcm;
pub mod record;
pub mod sim;

pub use data::{
    ConnectomeData, NeuronGroup,
    BrainIO, InputChannel, InputEncoding, OutputChannel, OutputReadout,
};
pub use record::{SpikeRecorder, SpikeRecordings};
pub use rcm::{reverse_cuthill_mckee, build_adjacency, bandwidth, mean_bandwidth,
    apply_permutation_to_csr, apply_permutation_to_state, apply_permutation_to_groups,
    spectral_ordering, fiedler_vector};
pub use sim::{
    BrainState, NeuronModel, LifParams, IzhikevichParams, RateParams, PsiParams,
    sim_lif, sim_lif_synaptic, sim_hh, sim_hh_synaptic,
    sim_izhikevich, sim_izhikevich_synaptic, sim_rate, sim_psi, sim_psi_record,
    csr_propagate, csr_propagate_from_queue, csr_propagate_conductance_from_queue,
    push_spikes_to_queue, SpikeQueue,
};
pub use taas_kernels::neuron::synapse::{SynapticKinetics, DecayFactors, ReceptorType};

#[cfg(feature = "cuda")]
pub use sim::gpu::{sim_lif_gpu, sim_izhikevich_gpu};
