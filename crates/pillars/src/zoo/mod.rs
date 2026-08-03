//! # 🦁 Zoo — Library of Ready-Made Models
//!
//! Pre-built neuron models, synapses, sensors and glia.
//! Each model implements the [`Model`] trait.
//!
//! ## Boundary: Zoo vs Core
//!
//! | Layer | Responsibility |
//! |-------|----------------|
//! | `Model` trait | Interface contract (N_PARAMS, N_STATE, step) |
//! | Zoo models | Biological dynamics — written in SIMD-agnostic math |
//! | Core runner | SIMD dispatch, memory loading/saving |
//!
//! Models **never** touch raw pointers or memory layout.
//! They receive a [`SimdContext`], read/write state
//! and parameters through `.s()`, `.p()`, `.i_syn`, and produce `ctx.spike_mask()`.
//!
//! ## Module Index
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`neurons`] | Point neuron models (LIF, HH, Izhikevich, AdEx, GLIF, …) |
//! | [`synapses`] | Synaptic dynamics (NMDA, Tsodyks-Markram STP, CaPlasticity) |
//! | [`sensors`] | Bio-inspired sensors (chordotonal, stretch) |
//! | [`glia`] | Glial modulation (astrocyte, K⁺ buffering) |
//!
//! ## How to add a new model
//!
//! 1. Pick a file in `zoo/neurons/` or create a new one
//! 2. Implement `Model::step<S: SimdLane>(ctx)` — works for both f32x8 and f32
//! 3. Register in `zoo/neurons/mod.rs`
//! 4. Add tests following the `test_stress_silence` / `test_stress_explosion` pattern

pub mod neurons;
pub mod synapses;
pub mod sensors;
pub mod glia;

pub use neurons::*;
pub use synapses::*;
pub use sensors::*;
pub use glia::*;

use crate::core::SimdContext;
use crate::core::lane::SimdLane;

/// Core Model Trait — Zero-Cost Hardware Abstraction for Neural Dynamics.
///
/// Implemented by all zoo models. Single `step<S: SimdLane>(ctx)` handles
/// both 8-wide SIMD (f32x8 on AVX2) and scalar fallback (f32).
pub trait Model {
    const N_PARAMS: usize;
    const N_STATE: usize;
    const HAS_SPIKE: bool;
    /// # Inlining contract
    ///
    /// IMPLEMENTATIONS must annotate `step` with `#[inline(always)]`.
    /// `step` runs inside the hot SIMD loop in
    /// [`run_neuron`](crate::core::runner::run_neuron) /
    /// [`NeuralPopulation::step`](crate::core::NeuralPopulation::step).
    /// Without forced inlining LLVM's heuristics frequently emit `step` as a
    /// shared out-of-line function once a model has several call sites, which
    /// materializes the ~672-byte [`SimdContext`](crate::core::SimdContext) on
    /// the stack every chunk and costs **~2.5-3x throughput** (measured:
    /// LIF 356 ms → 125 ms for 138k×1000).
    ///
    /// (The attribute cannot live on the trait declaration itself —
    /// `#[inline]` on a required trait method is a no-op and now warns.)
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>);
}
