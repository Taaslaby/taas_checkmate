#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::module_inception)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::misnamed_getters)]

//! # taas-checkmate
//!
//! Pure Rust wavelet neural engine with SIMD acceleration, CUDA GPU support,
//! and FlyWire connectome simulation.
//!
//! ## Architecture
//!
//! ```text
//! Python (PyO3) ⇄ Rust safe wrappers ⇄ SIMD kernels (wide::f32x8) / CUDA
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
//! | unity      | C ABI for Unity (implies connectome) |

// ─── Internal crate re-exports ─────────────────────────
pub use taas_core as core;
pub use taas_format;
pub use taas_iit as iit;
pub use taas_kernels as kernels;
pub use taas_train as train;

#[cfg(feature = "t3")]
pub use taas_t3 as t3;

#[cfg(feature = "connectome")]
pub use taas_connectome as connectome;

#[cfg(feature = "cuda")]
pub use cuda_kernels as cuda;

// ─── Main crate modules ────────────────────────────────
pub mod io;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "unity")]
pub mod unity;

// ─── Backward compatibility re-exports ──────────────────
pub use core::{koit, sparse};
pub use core::{kuramoto, neuron};
pub use train::rl;

pub use core::kuramoto::{
    band_energy, batch_coherence_sweep, interference, interference_mean, kuramoto_step_dense,
    kuramoto_step_sparse, lazy_sincos_update, phase_coherence, t3_step_sparse, winding_matrix,
};
pub use core::neuron::{
    bsa_encode, bsa_kernel_hamming, compute_lfp, eprop_update, intrinsic_plasticity, phase_encode,
    population_rate_decode, probe_fr, stdp_homeostatic, stdp_update, step_adex,
    step_atlasov_neuromotor, step_fhn, step_gco, step_hr, step_izhikevich, step_lif, step_lif_at,
    step_saw, step_wc, suggest_config, surrogate_gradient, synaptic_normalization, AtlasovState,
    DelayBuffer, DopamineSTDP, EPropAdExMulti, EPropIzhikevich, EPropIzhikevichMulti, EPropLIF,
    EPropSaw, EPropTunerConfig, FrProbePoint, PoissonGenerator, ShortTermPlasticity, SornNetwork,
    TtfsEncoder,
};
pub use core::sparse::{
    apply_stdp_ltd, compute_hamiltonian, propagate_spikes, rate_step, step_gco_full,
};
pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use taas_kernels::spike_entropy;
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
