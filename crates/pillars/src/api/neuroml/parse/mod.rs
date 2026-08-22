//! # NeuroML2 parser (`api::neuroml::parse`)
//!
//! Turns a NeuroML2 (and/or LEMS) XML document into a small **intermediate
//! representation** ([`Ir`]) that is independent of both XML and the pillars
//! runtime. The compiler ([`super::compile`]) lowers this IR into a [`Network`].
//!
//! ## Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`ir`] | The intermediate representation: `Ir`, cells, synapses, channels, morphology, biophysics |
//! | [`units`] | Quantity parsing (`to_ms`/`to_mv`/`raw`…) + XML attribute helpers (`attr`/`child`…) |
//! | [`channels`] | `ionChannelHH`/`gateHHrates`/`gateHHtauInf`/`ionChannelKS`/`gateFractional` parsing |
//! | [`morphology`] | `morphology`/`segment`/`segmentGroup` parsing |
//! | [`biophysics`] | `channelDensity`/`membraneProperties`/`biophysicalProperties` parsing |
//! | [`standard`] | Standard-channel recognition + `resolve_channel`/`resolve_markov_channel` |
//! | [`parser`] | The main `parse()` entry point |
//!
//! [`Network`]: crate::api::network::Network

mod biophysics;
mod channels;
mod ir;
mod ir_validate;
mod prepass;
mod morphology;
mod parser;
mod standard;
#[cfg(test)]
mod tests;
mod units;

pub(crate) use ir::*;
pub(crate) use parser::*;
pub(crate) use standard::*;
pub(crate) use units::*;

// Shared imports re-exported so sibling submodules can `use super::*` (the
// `NeuromlError` type and LEMS component types live in the parent `neuroml`
// module, one level up from here).
pub(crate) use super::lems::ComponentType;
pub(crate) use super::NeuromlError;
