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
//! | unity      | C ABI for Unity (implies connectome) |

// ═══════════════════════════════════════════════════════
//  🏛️ Pillars — the engine (primary API)
// ═══════════════════════════════════════════════════════
pub use pillars;
pub use pillars::prelude::*;

// ═══════════════════════════════════════════════════════
//  Internal crate re-exports
// ═══════════════════════════════════════════════════════
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
pub mod io; // Integrated Information Theory (Φ) — перенесено из taas-core (Фаза 10)

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "unity")]
pub mod unity;

// ═══════════════════════════════════════════════════════
//  Legacy bridges — stable aliases for pre-pillars crates
// ═══════════════════════════════════════════════════════
// Фаза 11: крейт `taas-kernels` удалён — compat-поверхность переехала на
// верхний уровень `pillars::kernels` (flat-реэкспорты) и в этот inline-модуль.
// Список ЯВНЫЙ (без glob): поверхность зафиксирована ровно по бывшему
// публичному API крейта — новые имена в pillars НЕ протекают сюда автоматически.
pub mod kernels {
    pub use pillars::kernels::io;
    pub use pillars::kernels::measure;
    pub use pillars::kernels::phase;
    pub use pillars::kernels::vnc;
    pub use pillars::kernels::{
        adam_step_simd, add_simd, band_energy, bcm_step, clamp_amp, delay_buffer_circular,
        delay_buffer_multi_tap, dopamine_traces_step, dopamine_weight_update, echo_tap, fast_exp,
        fast_exp_simd, fast_sincos_simd, fused_hidden_izhikevich_step, fused_output_lif_step,
        hebbian_update, homeostatic_scale, interference, interference_mean, kalman_update,
        kuramoto_update_viscous, lazy_sincos_update, linear_decoder, nlms_train, phase_coherence,
        phase_gate_apply, poisson_fused_hidden_step, pst_decode, relu_simd, ring_synthesis,
        safety_clamp, sgd_step_simd, sigmoid_gate, spike_entropy, stdp_homeostatic, stdp_update,
        stochastic_encoder, t3_complex_decoder, t3_drive_encoder, t3_inverse_kuramoto,
        t3_lms_train, t3_pst_features, t3_tact_generator, t3_update, thalamic_gate, wc_step,
        winding_matrix, wrap_phase,
    };
    /// Legacy alias: `kernels::neuron::hh` → `pillars::kernels::hh_channels`
    /// (старые тесты/бенчи; канон нейронов — `pillars::api::neurons` +
    /// `pillars::zoo::neurons`).
    pub mod neuron {
        pub use pillars::kernels::hh_channels as hh;
    }
}

pub use train::rl;

// Legacy-модульные алиасы (Фаза 10: крейт taas-core удалён) —
// прямые pillars-реэкспорты для Rust-тестов/бенчей.
pub mod neuron {
    pub use pillars::api::neurons::*;
}
pub mod sparse {
    pub use pillars::kernels::csr::*;
}
pub mod kuramoto {
    pub use pillars::kernels::decode::{
        adam_train, kalman_update, nlms_train, pst_decode, ring_synthesis, t3_lms_train,
    };
    pub use pillars::kernels::measure::{
        band_energy, batch_coherence_sweep, interference, lazy_sincos_update, winding_matrix,
    };
    pub use pillars::kernels::phase::{kuramoto_step_dense, kuramoto_step_sparse, t3_step_sparse};

    /// Mean interference — legacy slice-сигнатура (тесты).
    pub fn interference_mean(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]) {
        pillars::kernels::measure::interference_mean(&mut output[0], phases_a, phases_b);
    }
    /// Phase coherence — legacy slice-сигнатура (тесты).
    pub fn phase_coherence(
        output: &mut [f32],
        sin_buf: &[f32],
        cos_buf: &[f32],
        amplitudes: &[f32],
    ) {
        pillars::kernels::measure::phase_coherence(&mut output[0], sin_buf, cos_buf, amplitudes);
    }
}

// Корневые реэкспорты — напрямую из pillars (Фаза 9); `core` остаётся
// (крейт taas-core удалён — Фаза 10).
pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use kernels::spike_entropy;
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
