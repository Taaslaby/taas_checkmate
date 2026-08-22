//! # ConductancePopulation — data-driven HH population (arbitrary channels)
//!
//! A [`Population`](crate::api::Population) for conductance-based cells whose
//! ion channels are **not** the fixed squid-axon NaV/KV layout baked into the
//! canonical [`HHModel`](crate::zoo::HHModel). The channel set is described by
//! [`ChannelMeta`](crate::kernels::hh_channels::ChannelMeta) (the data-driven representation consumed by the generic
//! engine [`hh_poly_step_slices`](crate::kernels::hh_channels::hh_poly_step_slices)), so any
//! channel expressible as `gateHHrates` / `gateHHtauInf` (m³h, n⁴, HCN, M, …)
//! runs through the same SIMD + scalar engine.
//!
//! This is the runtime behind NeuroML `hhCell` models whose `channelDensity`
//! references non-standard `ionChannelHH` components (see `api::neuroml`).
//! Standard NaV/KV cells still prefer `HHModel` (NEURON-validated fast path).
//!
//! # Layout
//!
//! Column stride is `n` (the logical population size), matching the stride the
//! [`hh_poly_step_slices`](crate::kernels::hh_channels::hh_poly_step_slices) engine expects. `AlignedVec` pads to a multiple of 8 for
//! the SIMD loads; the engine handles the scalar tail itself.
//!
//! ## Module boundaries (anti-monolith)
//!
//! | Concern | Module |
//! |---------|--------|
//! | Shared constants + steady-state helper | [`common`](self) |
//! | HH-engine population ([`ConductancePopulation`]) | [`hh`](self) |
//! | Kinetic-scheme population ([`MarkovPopulation`]) | [`markov`](self) |

mod common;
mod hh;
mod markov;
#[cfg(test)]
mod tests;

pub(crate) use common::{steady_state_gates, SPIKE_THRESHOLD_MV};
pub use hh::ConductancePopulation;
pub use markov::{MarkovChannel, MarkovPopulation};
