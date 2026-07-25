//! Karmac — SPAUN-inspired cognitive architecture modules.
//!
//! # Modules
//! - `gnw`: Global Neuronal Workspace (Baars/Dehaene) — attention broadcast
//! - `lod`: LOD — Carmack BSP-style neuron level-of-detail (HH→Izh→LIF→tanh)
//! - `basal_ganglia`: Action selection (winner-take-all routing)
//! - `working_memory`: Prefrontal cortex-like state retention
//! - `semantic_pointer`: SPA operations (bind/unbind/superpose via FFT)

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]

pub mod gnw;
pub mod lod;
pub mod basal_ganglia;
pub mod working_memory;
pub mod semantic_pointer;

pub use basal_ganglia::{BasalGanglia, BgConfig};
pub use working_memory::WorkingMemory;
pub use semantic_pointer::SemanticPointer;
