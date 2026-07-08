#![allow(non_snake_case)]

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
//! | Feature | Description |
//! |---------|-------------|
//! | default | Core + T3 oscillator engine |
//! | t3      | T3 oscillator engine + decoders |
//! | python  | PyO3 Python bindings + Connectome |
//! | cuda    | CUDA GPU backend (implies python) |

// ─── Internal crate re-exports ─────────────────────────
pub use taas_kernels as kernels;
pub use taas_core as core;
pub use taas_train as train;
pub use taas_format;

#[cfg(feature = "t3")]
pub use taas_t3 as t3;

#[cfg(feature = "python")]
pub use taas_connectome as connectome;

#[cfg(all(feature = "python", feature = "cuda"))]
pub use cuda_kernels as cuda;

// ─── Main crate modules ────────────────────────────────
pub mod io;

#[cfg(feature = "python")]
pub mod python;

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
    step_adex, stdp_update, stdp_homeostatic, DelayBuffer,
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
    Sparker, SpikeTrace, HologramEncoder, SpikeStats,
    NormalizedSparker, HybridSparker,
};

#[cfg(all(feature = "t3", feature = "cuda"))]
pub use t3::CudaBackend;
