//! # Sensors — SIMD-Accelerated Sensory Neuron Models
//!
//! Sensory neuron models that don't fit the standard point-neuron `Model` trait
//! because they need extra inputs beyond `I_syn` (load, stretch, etc.).
//!
//! Each module provides a single SIMD step function with scalar tail,
//! matching the performance of the original `kernels` implementations.

pub mod chordotonal;

pub use chordotonal::chordotonal_step;
