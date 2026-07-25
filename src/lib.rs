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
pub use taas_kernels as kernels;
pub use taas_core as core;
pub use taas_train as train;    pub use taas_format;
pub use taas_iit as iit;

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
pub use core::{neuron, kuramoto};
pub use core::{sparse, koit};
pub use train::rl;

pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use core::kuramoto::{
    kuramoto_step_dense, kuramoto_step_sparse, t3_step_sparse,
    interference, interference_mean, phase_coherence,
    batch_coherence_sweep, lazy_sincos_update,
    band_energy, winding_matrix,
};
pub use taas_kernels::spike_entropy;
pub use core::neuron::{
    step_izhikevich, step_saw, step_psi_wave, step_lif, step_lif_at,
    step_adex, step_fhn, step_hr, step_atlasov_neuromotor, AtlasovState, step_wc,
    stdp_update, stdp_homeostatic, DelayBuffer,
    intrinsic_plasticity, synaptic_normalization, SornNetwork,
    bsa_encode, bsa_kernel_hamming, phase_encode,
    TtfsEncoder, population_rate_decode,
    EPropLIF, EPropIzhikevich, EPropIzhikevichMulti, EPropAdExMulti, EPropSaw,
    eprop_update, surrogate_gradient,
    probe_fr, suggest_config, FrProbePoint, EPropTunerConfig,
    PoissonGenerator, ShortTermPlasticity, DopamineSTDP, compute_lfp,
};
pub use train::rl::{
    ActorCriticAgent, test_mdp_episode,
};
pub use core::sparse::{
    propagate_spikes, compute_hamiltonian, step_psi_wave_full, rate_step, apply_stdp_ltd,
};

#[cfg(feature = "t3")]
pub use t3::{
    TriOscillator, NlmsDecoder, KalmanDecoder, InverseKuramotoDecoder,
    StdpDecoder, PstDecoder, HybridDecoder, RidgeSolver, TrajectoryBuffer,
    Decoder, T3Backend, Avx2Backend, ScalarBackend,
    pst_features, pst_linear_features, pst_linear_dim,
    compute_pearson, normalize_features,
    PstControl, PstControlConfig,
    Spiker, SpikeTrace, HologramEncoder, SpikeStats,
    NormalizedSpiker, HybridSpiker,
};

#[cfg(all(feature = "t3", feature = "cuda"))]
pub use t3::CudaBackend;
