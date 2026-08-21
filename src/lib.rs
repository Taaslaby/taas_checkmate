// Lint configuration — scoped to the root crate only (pillars has its own).
// - non_snake_case: physics notation (E_leak, I_soma, etc.) — pervasive.
// - needless_range_loop + manual_memcpy: SIMD stride loops on raw slices.
// - type_complexity + module_inception: compat re-exports and facade modules.
// - manual_clamp: bounds check on raw pointer math where .clamp() changes NaN behaviour.
#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::module_inception)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::manual_clamp)]

//! # taas-checkmate
//!
//! Pure Rust wavelet neural engine with SIMD acceleration, CUDA GPU support,
//! and FlyWire connectome simulation.
//!
//! ## Architecture
//!
//! ```text
//! Python (PyO3) ⇄ Rust safe wrappers ⇄ Pillars SIMD engine (wide::f32x8) / CUDA
//! ```
//!
//! The computational core of the crate is **[`pillars`]** — the zero-cost SIMD
//! engine for neural dynamics. Every neuron model, synapse, sensor and
//! plasticity rule lives in `pillars`, and the top-level API of this crate is
//! a direct re-export of it:
//!
//! ```rust
//! use taas_checkmate::{LIFModel, NeuralPopulation, run_neuron};
//! ```
//!
//! ## Feature gates
//!
//! | Feature    | Description |
//! |------------|-------------|
//! | default    | T3 oscillator engine (oscillator + PST + decoders) |
//! | train      | BPTT, SPIGRAD, connectome chase |
//! | connectome | Connectome LIF brain simulator |
//! | karmac     | Cognitive architecture (GNW, LOD, Basal) |
//! | cuda       | GPU acceleration (implies t3 + connectome) |
//! | python     | PyO3 bindings (implies t3 + connectome + train + karmac) |

// ═══════════════════════════════════════════════════════
//  🏛️ Pillars — the engine (primary API)
// ═══════════════════════════════════════════════════════
pub use pillars;
pub use pillars::prelude::*;

// ═══════════════════════════════════════════════════════
//  Internal crate re-exports
// ═══════════════════════════════════════════════════════
#[cfg(feature = "format")]
pub use taas_format;
pub use taas_train as train;

#[cfg(feature = "t3")]
pub use taas_t3 as t3;

#[cfg(feature = "connectome")]
pub use pillars_connectome as connectome;

#[cfg(feature = "cuda")]
pub use pillars_cuda::gpu as cuda;

// Cognitive architecture (GNW, LOD, BasalGanglia) — свернуто из отдельного крейта
#[cfg(feature = "karmac")]
pub mod karmac;

// ═══════════════════════════════════════════════════════
//  Main crate modules
// ═══════════════════════════════════════════════════════
pub mod iit;
pub mod io; // Integrated Information Theory (Φ)

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use train::rl;

// Корневые реэкспорты — напрямую из pillars.
pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use pillars::kernels::spike_entropy;
pub use pillars::api::neurons::{
    bsa_encode, bsa_kernel_hamming, compute_lfp, eprop_update, intrinsic_plasticity, phase_encode,
    population_rate_decode, probe_fr, stdp_homeostatic, stdp_update, step_adex,
    step_atlasov_neuromotor, step_fhn, step_gco, step_hr, step_izhikevich, step_lif, step_lif_at,
    step_saw, step_wc, suggest_config, surrogate_gradient, synaptic_normalization, AtlasovState,
    DelayBuffer, DopamineSTDP, EPropAdExMulti, EPropIzhikevich, EPropIzhikevichMulti, EPropLIF,
    EPropSaw, EPropTunerConfig, FrProbePoint, PoissonGenerator, ShortTermPlasticity, SornNetwork,
    TtfsEncoder,
};
pub use pillars::kernels::csr::{
    apply_stdp_ltd, compute_hamiltonian, propagate_spikes, rate_step, step_gco_full,
};
pub use pillars::kernels::measure::{
    band_energy, batch_coherence_sweep, interference, interference_mean, lazy_sincos_update,
    phase_coherence, winding_matrix,
};
pub use pillars::kernels::phase::{kuramoto_step_dense, kuramoto_step_sparse, t3_step_sparse};
pub use train::rl::{test_mdp_episode, ActorCriticAgent};

#[cfg(feature = "t3")]
pub use t3::{
    compute_pearson, normalize_features, pst_features, pst_linear_dim, pst_linear_features,
    Avx2Backend, Decoder, HologramEncoder, HybridDecoder, HybridSpiker, InverseKuramotoDecoder,
    KalmanDecoder, NlmsDecoder, NormalizedSpiker, PstControl, PstControlConfig, PstDecoder,
    RidgeSolver, ScalarBackend, SpikeStats, SpikeTrace, Spiker, StdpDecoder, T3Backend,
    TrajectoryBuffer, TriOscillator,
};

#[cfg(all(feature = "t3", feature = "cuda"))]
pub use t3::CudaBackend;
