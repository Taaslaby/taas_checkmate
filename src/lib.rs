#![allow(non_snake_case)]

//! # taas-checkmate
//!
//! Pure Rust wavelet neural engine with `core::simd` SIMD acceleration.
//! No ASM, no C, no FFI — 100% safe Rust over explicit SIMD kernels.
//!
//! ## Architecture
//!
//! ```text
//! Python (PyO3) <-> Rust safe wrappers <-> pure Rust SIMD kernels
//! ```
//!
//! ## Modules
//!
//! | Module | Description | Feature |
//! |--------|-------------|---------|
//! | [`kernels`] | Raw SIMD-compatible loops (all models) | simd |
//! | [`io`] | PWM encoder, Linear decoder, gates, safety clamp | — |
//! | [`taas_format`] | .taas save/load format | — |
//! | [`core`] | Tensor, autograd (Var, backward) | autograd |
//! | [`models`] | neuron, nn (Dense, Softmax), kuramoto | — / autograd |
//! | [`ops`] | sparse (CSR), koit (XOR attention, RNG) | — |
//! | [`training`] | optim (SGD, Adam), loss, rl (Actor-Critic) | — / autograd |
//! | [`t3`] | T3 engine (TriOscillator, PST, decoders, Phase Gate, Cochlea) | — |
//! | [`bindings`] | PyO3 Python bindings | python |

/// IO encoders, decoders, gates, safety clamp
pub mod io;
/// Pure Rust SIMD kernels (izhikevich, lif, adex, saw, phase, stp, stdp)
pub use taas_kernels as kernels;
/// T3 engine: TriOscillator, PST features, decoders, phase gate, cochlea
pub use taas_t3 as t3;
/// .taas save/load format
pub mod taas_format;
/// Tensor and autograd (Var, backward, optimizers)
pub mod core;
/// Neuron models (Izhikevich, LIF, AdEx, SAW, STP, STDP, eprop) + NN layers + Kuramoto
pub mod models;
/// Sparse matrix ops (CSR) + Koit (XOR attention, RNG)
pub mod ops;
/// RL agent (Actor-Critic) + optimizers + loss functions
pub mod training;
/// PyO3 Python bindings
pub mod bindings;

// ─── Backward compatibility re-exports ─────────────────────
// These flatten modules: `taas_checkmate::neuron` = `models::neuron`.
// Remove once all external code uses canonical paths.
pub use models::{neuron, kuramoto};
pub use ops::{sparse, koit};
pub use training::rl;

#[cfg(feature = "autograd")]
pub use core::{tensor, autograd};
#[cfg(feature = "autograd")]
pub use models::nn;
#[cfg(feature = "autograd")]
pub use training::{optim, loss};

#[cfg(feature = "python")]
pub use bindings::python;

// Re-export most commonly used items
pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use models::kuramoto::{
    kuramoto_step_dense, kuramoto_step_sparse, t3_step_sparse,
    interference, interference_mean, phase_coherence,
    batch_coherence_sweep, lazy_sincos_update,
    band_energy, winding_matrix,
};
pub use models::neuron::{
    step_izhikevich, step_saw, step_psi_wave, step_lif, step_lif_at,
    step_adex, stdp_update, stdp_homeostatic, DelayBuffer,
    intrinsic_plasticity, synaptic_normalization, SornNetwork,
    bsa_encode, bsa_kernel_hamming, phase_encode,
    TtfsEncoder, population_rate_decode,
    EPropLIF, EPropIzhikevich, EPropIzhikevichMulti, EPropAdExMulti, EPropSaw, eprop_update, surrogate_gradient,
    probe_fr, suggest_config, FrProbePoint, EPropTunerConfig,
    PoissonGenerator, ShortTermPlasticity, DopamineSTDP, compute_lfp,
};
pub use training::rl::{
    ActorCriticAgent, test_mdp_episode,
};
pub use ops::sparse::{
    propagate_spikes, compute_hamiltonian, step_psi_wave_full, rate_step, apply_stdp_ltd,
};
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

#[cfg(feature = "cuda")]
pub use t3::CudaBackend;

#[cfg(feature = "autograd")]
pub use core::tensor::Tensor;
#[cfg(feature = "autograd")]
pub use models::nn::{Dense, Softmax, Dropout, BatchNorm, Activation, Layer};
#[cfg(feature = "autograd")]
pub use training::optim::{SGD, Adam};
#[cfg(feature = "autograd")]
pub use training::loss::{CrossEntropyLoss, MSELoss, accuracy};
#[cfg(feature = "autograd")]
pub use core::autograd::Var;
