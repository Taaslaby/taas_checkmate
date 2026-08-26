//! -------------------------------------------------------------------------
//! Filename:    src/api/network/mod.rs
//! Class:       network
//! Purpose:     Network — Spike Routing Orchestrator
//! Description: Connects NeuralPopulations via SparseGraph and drives the
//!              simulation loop. Uses a ring buffer of currents (Approach 2)
//!              for axonal delays — on spike, weights are written to a
//!              future ring buffer slot; each step reads and clears it.
//!
//! Architecture:
//! ```text
//!     Neuron step (all pops)
//!           ↓
//!     spikes_out  →  compact_spikes  →  active_indices
//!           ↓                              ↓
//!     [current slot]   ←  push: for each active neuron,
//!                              for each outgoing synapse,
//!                              delay[(step + delay.max(1)) % 32] += weight
//!                              (delay 0 = next step)
//!           ↓
//!     Clear current slot, advance step_counter
//! ```
//! Module boundaries: population (type erasure) | graph (CSR) | delay
//! (ring buffer) | push (event-driven scatter seq+par) | edge/sources
//! (synapse specs) | core/build/step (orchestration, this module).
//! Push vs Pull: Pull scans all neurons every step; Push scatters only
//! for active neurons (1-5%), 20-100x cheaper at typical sparsity.
//!
//! Usage:
//! ```rust
//! use pillars::api::Network;
//! use pillars::zoo::neurons::LIFModel;
//! let mut net = Network::new();
//! let sensory = net.add_population::<LIFModel>(100);
//! let motor = net.add_population::<LIFModel>(50);
//! for pop in [sensory, motor] {
//!     let lif = net.pop_typed_mut::<LIFModel>(pop).unwrap();
//!     lif.fill_param(0, 0.5);   // dt (ms)
//!     lif.fill_param(1, 0.1);   // 1/tau
//!     lif.fill_param(2, -70.0); // v_rest
//!     lif.fill_param(3, -54.0); // v_thresh
//!     lif.fill_param(4, -70.0); // v_reset
//! }
//! net.connect(sensory, &[0,1,2,3,4,5,6,7,8,9], motor, &[0,0,1,1,2,2,3,3,4,4], &[0.8; 10], &[0; 10]);
//! net.build();
//! for _ in 0..1000 { net.step(); }
//! ```
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

mod build;
mod core;
mod edge;
mod sources;
mod step;
#[cfg(test)]
mod tests;

pub use core::Network;
pub use edge::Edge;
pub use sources::SigmoidGradedSynapse;
pub(crate) use sources::{poisson_spike, PoissonSourceRuntime, SigmoidGradedRuntime};
pub(crate) use step::KineticRuntime;

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}
