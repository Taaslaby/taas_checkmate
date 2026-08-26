//! -------------------------------------------------------------------------
//! Filename:    src/api/mod.rs
//! Class:       api
//! Purpose:     API — High-Level Network Interface
//! Description: Safe Rust facade over the engine. Only layer exposed to
//!              users (Python/C++/Rust script); all unsafe/SIMD isolated
//!              below in core/kernels/zoo.
//!
//! Layers: API (Safe Rust) → ZOO (LIF, HH, STDP, …) → OPS (per-lane bio
//! primitives) → CORE (SimdLane, NeuralPopulation, runners, unsafe behind
//! safe wrappers) → KERNELS sibling (batch f32x8 microkernels, never feed zoo).
//! Dependency direction: api → zoo → ops → core.
//!
//! Modules: graph (CSR adjacency) | network (multi-population orchestrator)
//! | compact (spike compaction) | push (event-driven scatter) | neurons
//! (EPropLIF, GLIF3/4, STP, …) | cpe (ChaosPatternEngine) | autotune
//! (E-Prop FR probe) | embodied (FlyWire neuromotor).
//!
//! Quick start:
//! ```rust,ignore
//! use pillars::prelude::*;
//! let mut pop = NeuralPopulation::<LIFModel>::new(138_000);
//! pop.fill_param(0, 0.1);  // dt
//! pop.fill_param(1, 1.0/10.0);  // inv_tau_m
//! pop.fill_param(2, -70.0);  // v_rest
//! pop.fill_param(3, -54.0);  // v_thresh
//! pop.fill_param(4, -70.0);  // v_reset
//! pop.fill_param(5, 2.0);  // tau_ref
//! pop.fill_state(0, -70.0);  // v
//! let i_in = vec![30.0; 138_000];
//! let mut spikes = vec![0i32; 138_000];
//! pop.step(&i_in, &mut spikes);
//! ```
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

pub mod autotune;
pub mod builder;
pub mod conductance;
pub mod cpe;
pub mod embodied;
pub mod formula;
pub mod human_hand;
pub mod net;
pub mod neuroml;
pub mod neurons;
pub mod trainer;
pub mod validate;

// Network wiring layer lives in `net/`; re-export its modules at the old
// flat paths (`api::graph`, `api::network`, …) so external callers keep
// their historical imports.
pub use net::{compact, delay, graph, network, population, push, synapse};

pub use crate::core::population::NeuralPopulation;
pub use builder::{
    AdExBuilder, BuildError, HAAPBuilder, HHBuilder, HHPreset, IzhPattern, IzhikevichBuilder,
    LIFBuilder, ModelBuilder, UniversalBuilder, VSNBuilder,
};
pub use compact::{compact_spikes, compact_spikes_into};
pub use conductance::{ConductancePopulation, MarkovChannel, MarkovPopulation};
pub use delay::DelayBuffer;
pub use formula::{
    run_program, validate_program, FormulaModel, OpKind, OpNode, I_SYN, MAX_FORMULA_MODELS, N_REGS,
    P0, S0, S1, S2, S3, TEMP0,
};
pub use graph::{SparseGraph, SynapseRow, MAX_DELAY_SLOTS};
pub use human_hand::{
    HumanHandProfile, HumanHandState, HumanMuscle, HumanMuscleSpec, HUMAN_SPECIFIC_TENSION_PA,
};
pub use network::{Edge, Network, SigmoidGradedSynapse};
pub use population::{Population, PopulationId};
pub use synapse::{PlasticityProfile, SynapseKinetics};
