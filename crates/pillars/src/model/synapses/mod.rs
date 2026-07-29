//! # Synapses — SIMD-Accelerated Synaptic Dynamics
//!
//! Synaptic models that don't fit the standard point-neuron `Model` trait
//! because they operate on synapse state (not neuron state) or need extra
//! inputs (pre-synaptic spikes, conductances).
//!
//! Each module provides a single SIMD step function with scalar tail,
//! matching the performance of the original `kernels` implementations.

pub mod nmda;
pub mod tsodyks_markram;

pub use nmda::nmda_mg_block_step;
pub use tsodyks_markram::tsodyks_markram_step;
