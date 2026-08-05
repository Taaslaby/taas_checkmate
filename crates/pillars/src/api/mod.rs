//! # 🚀 API — High-Level Network Interface
//!
//! === BOUNDARY: Safe Rust ===
//! This is the **only layer** that users (Python, C++, or Rust script) should touch.
//! Every `unsafe` block, raw pointer, and SIMD intrinsic lives below this layer.
//!
//! ## Layers
//!
//! ```text
//!   API (this layer)    — Safe Rust, no unsafe visible
//!     │
//!   ZOO                 — Biological models (LIF, HH, STDP, …)
//!     │
//!   OPS                 — Per-lane bio primitives (ohmic, gate, plasticity)
//!     │
//!   CORE                — SimdLane, SimdContext, NeuralPopulation, runners
//!     │                    unsafe isolated behind safe wrappers
//!   KERNELS (sibling)   — batch f32x8 microkernels (never feed zoo)
//! ```
//!
//! Dependency direction is `api → zoo → ops → core`; `kernels/` is a sibling
//! of `ops/` (batch vs per-lane) and is not used to build zoo models.
//! See the crate-level docs in `lib.rs` for the full 5-layer scheme.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`graph`] | Sparse CSR adjacency for spike routing |
//! | [`network`] | Multi-population orchestrator with ring buffer delays |
//! | [`compact`] | Dense spike mask → active-index compaction |
//! | [`push`] | Event-driven scatter of active spikes into delay slots |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use pillars::prelude::*;
//!
//! let mut pop = NeuralPopulation::<LIFModel>::new(138_000);
//! pop.fill_param(0, 0.1);  // dt
//! pop.fill_param(1, 1.0/10.0);  // inv_tau_m
//! pop.fill_param(2, -70.0);  // v_rest
//! pop.fill_param(3, -54.0);  // v_thresh
//! pop.fill_param(4, -70.0);  // v_reset
//! pop.fill_param(5, 2.0);  // tau_ref
//! pop.fill_state(0, -70.0);  // v
//!
//! let i_in = vec![30.0; 138_000];
//! let mut spikes = vec![0i32; 138_000];
//! pop.step(&i_in, &mut spikes);
//! ```

pub mod builder;
pub mod compact;
pub mod delay;
pub mod formula;
pub mod graph;
pub mod network;
pub mod population;
pub mod push;

pub use crate::core::population::NeuralPopulation;
pub use builder::{
    AdExBuilder, HAAPBuilder, HHBuilder, HHPreset, IzhPattern, IzhikevichBuilder, LIFBuilder,
    ModelBuilder, UniversalBuilder, VSNBuilder,
};
pub use compact::{compact_spikes, compact_spikes_into};
pub use delay::DelayBuffer;
pub use formula::{
    run_program, validate_program, FormulaModel, OpKind, OpNode, I_SYN, MAX_FORMULA_MODELS, N_REGS,
    P0, S0, S1, S2, S3, TEMP0,
};
pub use graph::{SparseGraph, SynapseRow, MAX_DELAY_SLOTS};
pub use network::{Edge, Network};
pub use population::{Population, PopulationId};
