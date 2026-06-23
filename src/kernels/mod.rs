//! Pure Rust SIMD kernels — replaces all ASM and FFI.
//! All functions are `unsafe` for API parity with old FFI.
//! Uses `core::simd` or `wide` for explicit SIMD on supported platforms.

pub mod attention;
pub mod decode;
pub mod measure;
pub mod neuron;
pub mod stdp;
pub mod phase;
pub mod signal;

pub use attention::*;
pub use decode::*;
pub use measure::*;
pub use neuron::*;
pub use phase::*;

pub use signal::*;

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
