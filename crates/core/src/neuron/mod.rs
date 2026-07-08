//! Spiking neuron models — safe Rust wrappers over SIMD kernels.
//!
//! Sub-modules by model family:
//! - [`izhikevich`] — Izhikevich & E-Prop Izhikevich
//! - [`saw`] — SAW, ψ-wave & E-Prop SAW
//! - [`lif`] — LIF, LIF-AT & E-Prop LIF
//! - [`adex`] — AdEx & E-Prop AdEx
//! - [`stdp`] — STDP, homeostatic STDP & dopamine-modulated STDP
//! - [`stp`] — Short-Term Plasticity (Tsodyks-Markram) & LFP
//! - [`sorn`] — Self-Organizing Recurrent Network
//! - [`encoding`] — BSA, phase & TTFS spike encoding
//! - [`poisson`] — Poisson spike generator
//! - [`delay`] — Circular delay buffer
//! - [`stats`] — SpikeRaster, firing rates, ISI, CV
//! - [`eprop`] — Surrogate gradient & E-Prop weight update

pub mod izhikevich;
pub mod saw;
pub mod lif;
pub mod adex;
pub mod hh;
pub mod stdp;
pub mod stp;
pub mod sorn;
pub mod encoding;
pub mod poisson;
pub mod delay;
pub mod stats;
pub mod eprop;
pub mod autotune;
pub mod trainer;

// ─── Re-exports ──────────────────────────────────────────
pub use izhikevich::{step_izhikevich, EPropIzhikevich, EPropIzhikevichMulti};
pub use autotune::{probe_fr, suggest_config, FrProbePoint, EPropTunerConfig};
pub use trainer::EPropTrainer;
pub use saw::{step_saw, step_psi_wave, EPropSaw};
pub use lif::{step_lif, step_lif_at, EPropLIF};
pub use adex::{step_adex, EPropAdExMulti};
pub use hh::{HhLutConfig, HhState, HhParams, step_hh_lut, standard_config};
pub use stdp::{stdp_update, stdp_homeostatic, DopamineSTDP};
pub use stp::{ShortTermPlasticity, compute_lfp};
pub use sorn::{intrinsic_plasticity, synaptic_normalization, SornNetwork};
pub use encoding::{bsa_encode, bsa_kernel_hamming, phase_encode, TtfsEncoder};
pub use poisson::PoissonGenerator;
pub use delay::DelayBuffer;
pub use stats::{population_rate_decode, SpikeRaster, spike_raster_to_csv, firing_rates, mean_firing_rate, compute_isi, compute_cvs};
pub use eprop::{surrogate_gradient, eprop_update};

// ─── Constants ──────────────────────────────────────────

/// Current clamp limit for Izhikevich input (prevents runaway).
pub const I_CLAMP: f32 = 100.0;

/// Scale factor: 1/2^24, converts 24-bit random state to [0, 1)
pub const RNG_SCALE_24: f32 = 1.0 / 16777216.0;

/// Default refractory period (ms).
/// Shiu et al. 2024: 2.2ms for Drosophila LIF.
pub const DEFAULT_TAU_REF: f32 = 2.2;
