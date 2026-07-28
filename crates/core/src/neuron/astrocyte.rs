//! 🧫 Astrocyte & Gap-Junction Glial Wave Kernel
//!
//! Safe Rust wrapper around `taas_kernels::neuron::astrocyte::AstrocyteState`.
//!
//! The SIMD kernel is in `crates/kernels/src/neuron/astrocyte.rs`.
//! This module re-exports it with the same API for `taas-core` callers.
//!
//! # Components
//!
//! | Variable | Effect | τ |
//! |----------|--------|----|
//! | A(t) | Synaptic gain | ~1000 ms |
//! | Ca²⁺ | Volume transmission | ~1000 ms |
//! | V_gap | Gap junction coupling | ~100 ms |
//! | K_ext | Metabolic fatigue | ~500 ms |
//! | EAAT_occ | Glutamate clearance | ~10 ms |

pub use taas_kernels::neuron::astrocyte::AstrocyteState;
