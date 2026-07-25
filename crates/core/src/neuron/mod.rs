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
//! - [`soma_dendrite`] — Somato-dendritic integration (LIF soma + cable dendrites)

// ═══════════════════════════════════════════════════════
//  🧠 Model Wrappers (safe API поверх SIMD ядер)
// ═══════════════════════════════════════════════════════
pub mod adex; // AdEx & E-Prop AdEx
pub mod atlasov_neuromotor;
pub mod fhn; // FitzHugh-Nagumo
pub mod hh; // LUT Hodgkin-Huxley
pub mod hr; // Hindmarsh-Rose
pub mod izhikevich; // Izhikevich & E-Prop Izhikevich
pub mod lif; // LIF, LIF-AT & E-Prop LIF
pub mod saw; // SAW, ψ-wave & E-Prop SAW
pub mod wc; // Wilson-Cowan population // Atlasov Neuromotor joint control

// ═══════════════════════════════════════════════════════
//  🔗 Plasticity & Learning (обучение и пластичность)
// ═══════════════════════════════════════════════════════
pub mod eprop; // Surrogate gradient & eprop_update
pub mod sorn; // Self-Organizing Recurrent Network
pub mod stdp; // STDP, homeostatic STDP & dopamine STDP
pub mod stp; // Short-Term Plasticity & LFP
pub mod trainer; // EPropTrainer (high-level training loop)

// ═══════════════════════════════════════════════════════
//  🧩 Multi-compartment & Analysis
// ═══════════════════════════════════════════════════════
pub mod autotune; // Automatic threshold/rate tuning
pub mod delay;
pub mod encoding; // BSA, phase & TTFS spike encoding
pub mod poisson; // Poisson spike generator
pub mod soma_dendrite; // Somato-dendritic integration
pub mod stats; // SpikeRaster, firing rates, ISI, CV // Circular delay buffer

// ─── Re-exports ──────────────────────────────────────────
pub use adex::{step_adex, EPropAdExMulti};
pub use atlasov_neuromotor::{step_atlasov_neuromotor, AtlasovState};
pub use autotune::{probe_fr, suggest_config, EPropTunerConfig, FrProbePoint};
pub use delay::DelayBuffer;
pub use encoding::{bsa_encode, bsa_kernel_hamming, phase_encode, TtfsEncoder};
pub use eprop::{eprop_update, surrogate_gradient};
pub use fhn::step_fhn;
pub use hh::{standard_config, step_hh_lut, HhLutConfig, HhParams, HhState};
pub use hr::step_hr;
pub use izhikevich::{step_izhikevich, EPropIzhikevich, EPropIzhikevichMulti};
pub use lif::{step_lif, step_lif_at, EPropLIF};
pub use poisson::PoissonGenerator;
pub use saw::{step_gco, step_saw, EPropSaw};
pub use soma_dendrite::{
    CableDendriteState, DendriteConfig, DendriteDecayFactors, SomatoDendriticNeuron,
};
pub use sorn::{intrinsic_plasticity, synaptic_normalization, SornNetwork};
pub use stats::{
    compute_cvs, compute_isi, firing_rates, mean_firing_rate, population_rate_decode,
    spike_raster_to_csv, SpikeRaster,
};
pub use stdp::{stdp_homeostatic, stdp_update, DopamineSTDP};
pub use stp::{compute_lfp, ShortTermPlasticity};
pub use trainer::EPropTrainer;
pub use wc::step_wc;

// ─── Constants ──────────────────────────────────────────

/// Current clamp limit for Izhikevich input (prevents runaway).
pub const I_CLAMP: f32 = 100.0;

/// Scale factor: 1/2^24, converts 24-bit random state to [0, 1)
pub const RNG_SCALE_24: f32 = 1.0 / 16777216.0;

/// Default refractory period (ms).
/// Shiu et al. 2024: 2.2ms for Drosophila LIF.
pub const DEFAULT_TAU_REF: f32 = 2.2;
