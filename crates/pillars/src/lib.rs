//! #  Pillars — Zero-Cost SIMD Engine for Neural Dynamics
//!
//! Every primitive compiles to **both** 8-wide SIMD (AVX2/NEON) and scalar
//! (any CPU) from the **same source code** — chosen at monomorphization time.
//!
//! ## The Five Layers
//!
//! ```text
//!   api/        Network orchestrator          Safe Rust, no raw pointers
//!   zoo/        25+ ready-made models         impl Model + impl ComputeNeuron
//!   ops/        Per-lane bio primitives       ohmic, gate, spike, plasticity (generic S)
//!   kernels/    Batch f32x8 microkernels      flat &[f32] in, SIMD loop + tail out
//!   core/       SIMD engine                   SimdLane trait → f32x8 | f32
//!   Hardware                                  AVX2 | NEON | scalar
//! ```
//!
//! ## Layer boundaries
//!
//! - **`api/`** — Safe Rust only. Orchestrates populations, sparse graphs, delays.
//! - **`zoo/`** — Ready-made models. Builds models from `ops/` + `core/` only.
//!   **Never** calls `kernels/` directly (batch kernels are a different
//!   execution paradigm, not a model-building tool).
//! - **`ops/`** — Per-lane generic primitives (`S: SimdLane`). The canonical
//!   public math layer for model authors.
//! - **`kernels/`** — Batch kernels: flat slices in, SIMD loop + scalar tail
//!   out. Standalone microkernels; do not build zoo models on them.
//! - **`core/`** — The engine: `SimdLane`, `SimdContext`, `NeuralPopulation`,
//!   runners. Where Rust generics become hardware instructions.
//!
//! **Rule of thumb:** one-way dependency `api → zoo → ops → core`,
//! with `kernels` as a sibling of `ops` (batch vs per-lane), never feeding `zoo`.
//! Each layer has a single responsibility. Code NEVER crosses more than one
//! layer boundary at a time.
//!
//! ## Compilation Pipeline
//!
//! ```text
//! User model:  let spike = v.cmp_ge(v_thresh);
//!                    ↓
//! SimdLane trait (generic over S: SimdLane)
//!                    ↓
//! ┌─────────────────┼─────────────────┐
//! │  S = f32x8      │  S = f32        │
//! │  _mm256_cmp_ps  │  ucomiss + setae│
//! │  (AVX2)         │  (scalar)       │
//! └─────────────────┴─────────────────┘
//! ```
//!
//! All 8-wide SIMD code is **emitted at compile time** by Rust's monomorphization.
//! No interpreters. No JIT. No runtime dispatch.
//!
//! ## Quick Start
//!
//! ```rust
//! use pillars::prelude::*;
//! let mut pop = NeuralPopulation::<LIFModel>::new(138_000);
//! pop.fill_param(0, 0.5);   // dt
//! pop.fill_param(3, -54.0); // v_thresh
//! pop.fill_state(0, -70.0); // v
//! pop.step(&vec![30.0; 138_000], &mut vec![0i32; 138_000]);
//! ```
//!
//! ## When to bypass the API
//!
//! For **absolute throughput** (realtime Drosophila), call
//! [`core::runner::run_neuron`] directly.
//! This skips the NeuralPopulation wrapper but still uses SIMD.
//! For **development** (correctness, debugging), use [`NeuralPopulation`].
//!
//! ## Three execution paradigms (all intentional)
//!
//! 1. **Compiled SIMD** — `core/` + `ops/` + `zoo/`: models monomorphize
//!    `S: SimdLane` into `f32x8`/`f32` at compile time. The main engine.
//! 2. **Program interpreter** — [`api::formula`]: OpNode graphs executed by
//!    `run_program` (used by the node editor UI).
//! 3. **Table lookup** — [`zoo::neurons::TriangulatedModel`]: LUT-inference
//!    instead of ODE integration (calibrated from a reference model).

#![deny(rustdoc::broken_intra_doc_links)]

pub mod api;
pub mod core;
pub mod kernels;
pub mod math;
pub mod ops;
pub mod prelude;
pub mod zoo;

pub use api::NeuralPopulation;
pub use pillars_macros::{dual_neuron, NeuronModel};

use wide::f32x8;

// All public API is re-exported through `prelude`.
// Use `use pillars::prelude::*;` in your code.

// ═══════════════════════════════════════════════════════════
//  Temperature (Q10) — NEURON convention
// ═══════════════════════════════════════════════════════════

/// Q10 temperature factor: `3^((T - 6.3) / 10)`.
/// At 6.3°C → 1.0, at 37°C → 29.
#[inline(always)]
pub fn q10_from_celsius_s(celsius: f32) -> f32 {
    (3.0_f32).powf((celsius - 6.3) / 10.0)
}

/// SIMD broadcast of Q10.
#[inline(always)]
pub fn q10_from_celsius(celsius: f32) -> f32x8 {
    f32x8::splat(q10_from_celsius_s(celsius))
}

/// τ_eff = τ / Q10 — kinetics accelerate at higher temperature.
#[inline(always)]
pub fn scale_tau_q10_s(tau: f32, celsius: f32) -> f32 {
    tau / q10_from_celsius_s(celsius)
}

/// SIMD broadcast of scaled tau.
#[inline(always)]
pub fn scale_tau_q10(tau: f32, celsius: f32) -> f32x8 {
    f32x8::splat(scale_tau_q10_s(tau, celsius))
}

// ═══════════════════════════════════════════════════════════
//  Drosophila E_rev Constants
// ═══════════════════════════════════════════════════════════

/// Chloride channels (GluCl, GABA_A): E_Cl = -70 mV.
pub const E_CL: f32 = -70.0;

/// Sodium channels (AMPA, nAChR): E_Na = 0 mV.
pub const E_NA: f32 = 0.0;

/// Potassium channels (delayed rectifier): E_K = -80 mV.
pub const E_K: f32 = -80.0;

/// Calcium channels (NMDA): E_Ca = 0 mV.
pub const E_CA: f32 = 0.0;

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use wide::f32x8;

    #[test]
    fn test_sigmoid_simd() {
        let v = f32x8::from([-65.0, -50.0, -35.0, -20.0, 0.0, 20.0, 40.0, 60.0]);
        let s = crate::math::sigmoid::sigmoid(v, -35.0, 5.0);
        let a = s.to_array();
        assert!((a[2] - 0.5).abs() < 0.01, "sigmoid at Vh={}", a[2]);
        assert!(a[0] < 0.01, "sigmoid at -65 ≈ 0: {0}", a[0]);
        assert!((a[7] - 1.0).abs() < 0.01, "sigmoid at 60 ≈ 1: {0}", a[7]);
    }

    #[test]
    fn test_q10() {
        assert!((q10_from_celsius_s(6.3) - 1.0).abs() < 0.001);
        assert!((q10_from_celsius_s(37.0) - 29.0).abs() < 1.0);
    }

    #[test]
    fn test_erev_invariants() {
        const _: () = assert!(E_CL < 0.0, "Cl⁻ must be hyperpolarising");
        assert!((E_CL - (-70.0)).abs() < 1.0);
    }
}
