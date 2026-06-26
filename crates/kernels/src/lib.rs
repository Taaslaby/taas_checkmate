//! Pure Rust SIMD kernels — replaces all ASM and FFI.
//! Use the [`safe`] module for safe slice-based wrappers.
//! Raw `unsafe` functions are available via internal submodules for crate-internal use.

pub mod attention;
pub mod decode;
pub mod measure;
pub mod neuron;
pub mod phase;
pub mod signal;
pub mod safe;

// Re-export safe wrappers as the primary public API
pub use safe::*;
// Re-export safe utility functions from neuron module
pub use neuron::{fast_exp, fast_exp_simd};

/// Wrap phase to [-π, π)
#[inline(always)]
pub fn wrap_phase(p: f32) -> f32 {
    let tau = core::f32::consts::TAU;
    p - tau * (p / tau).round()
}

/// Clamp amplitude with floor (совпадает с TriOscillator::step clamp)
#[inline(always)]
pub fn clamp_amp(a: f32) -> f32 {
    a.max(0.001).min(10.0)
}
