//! -------------------------------------------------------------------------
//! Filename:    src/conductance/mod.rs
//! Class:       conductance
//! Purpose:     ConductancePopulation — data-driven HH population
//! Description: Population for conductance-based cells whose ion channels
//!              are not the fixed squid-axon NaV/KV layout of HHModel.
//!              Channel set described by ChannelMeta (data-driven) and
//!              consumed by hh_poly_step_slices — any channel expressible
//!              as gateHHrates/gateHHtauInf (m3h, n4, HCN, M, …) runs
//!              through the same SIMD+scalar engine. Runtime behind
//!              NeuroML hhCell with non-standard ionChannelHH; standard
//!              NaV/KV cells still prefer HHModel (NEURON-validated).
//!              Layout: column stride n, AlignedVec padded to 8.
//!
//! Module boundaries: common (shared constants + steady-state helper) |
//! hh (ConductancePopulation) | markov (MarkovPopulation).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

mod common;
mod hh;
mod markov;
#[cfg(test)]
mod tests;

pub(crate) use common::{steady_state_gates, SPIKE_THRESHOLD_MV};
pub use hh::ConductancePopulation;
pub use markov::{MarkovChannel, MarkovPopulation};
