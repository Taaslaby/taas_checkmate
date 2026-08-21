//! # Network — Spike Routing Orchestrator
//!
//! Connects [`NeuralPopulation`](crate::api::NeuralPopulation)s via a [`SparseGraph`] and drives
//! the simulation loop. Uses a **ring buffer of currents** (Approach 2)
//! for axonal delays — on spike, weights are written to a future ring
//! buffer slot; each step reads the current slot and clears it.
//!
//! ## Architecture
//!
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
//!
//! ## Module boundaries (anti-monolith)
//!
//! | Concern | Module |
//! |---------|--------|
//! | Type erasure (`Population`, `PopulationId`) | [`population`](crate::api::population) |
//! | CSR adjacency | [`graph`](crate::api::graph) |
//! | Delay ring buffer | [`delay`](crate::api::delay) |
//! | Event-driven scatter (seq + par) | [`push`](crate::api::push) |
//! | Edge / synapse spec types | [`edge`](self), [`sources`](self) |
//! | **Orchestration (this module)** | [`core`](self), [`build`](self), [`step`](self) |
//!
//! ## Push vs Pull
//!
//! | Approach | Operation |
//! |:---------|:----------|
//! | ❌ Pull  | For every neuron, scan incoming synapses → `i_in[i] += w` |
//! | ✅ Push  | For every **active** neuron (1–5%), scatter to future slots |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pillars::api::{Network, SparseGraph};
//! use pillars::zoo::neurons::LIFModel;
//!
//! let mut net = Network::new();
//!
//! // Add populations
//! let sensory = net.add_population::<LIFModel>(100).unwrap();
//! let motor = net.add_population::<LIFModel>(50).unwrap();
//!
//! // Connect sensory[0..10] → motor[0..5] with weight 0.8, no delay
//! net.connect(
//!     sensory, &[0,1,2,3,4,5,6,7,8,9],
//!     motor,   &[0,0,1,1,2,2,3,3,4,4],
//!     &[0.8; 10], &[0; 10],
//! );
//!
//! // Build graph and run
//! net.build();
//! for _ in 0..1000 {
//!     net.step();
//! }
//!
//! println!("Motor spikes: {:?}", &net.spikes(motor)[..10]);
//! ```

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
