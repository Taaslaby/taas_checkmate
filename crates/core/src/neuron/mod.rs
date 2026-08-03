//! Spiking neuron models — wrappers around Pillars SIMD kernels + learning utilities.
//!
//! ## Architecture
//!
//! **Neuron models** (step functions) delegate to [`pillars::zoo`] for SIMD execution.
//! For new code, prefer using `pillars` directly:
//! ```rust
//! use pillars::prelude::*;
//!
//! // SoA layout: each parameter is a column of `n` floats (column stride = n).
//! // LIFModel params: [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref]
//! let n = 8;
//! let mut params = vec![0.0f32; n * 6];
//! params[..n].fill(0.5);
//! params[n..2 * n].fill(0.1);
//! params[2 * n..3 * n].fill(-70.0);
//! params[3 * n..4 * n].fill(-54.0);
//! params[4 * n..5 * n].fill(-70.0);
//! params[5 * n..6 * n].fill(2.0);
//!
//! let mut v = vec![-70.0f32; n];
//! let mut u = vec![0.0f32; n];
//! let mut h = vec![0.0f32; n];
//! let mut r = vec![0.0f32; n]; // refractory counter
//! let i_in = vec![30.0f32; n];
//! let mut spikes = vec![0i32; n];
//!
//! unsafe {
//!     run_neuron::<LIFModel>(
//!         v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
//!         r.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut(),
//!         i_in.as_ptr(), spikes.as_mut_ptr(),
//!         params.as_ptr(), n,
//!     );
//! }
//! assert!(v[0] > -70.0, "v should depolarize under current injection");
//! ```
//!
//! **Learning & Plasticity** (E-Prop, STDP, STP, SORN, BCM) remain in `core` —
//! they are not SIMD neuron models and don't fit the `pillars::Model` trait.
//!
//! **Utilities** (encoding, poisson, delay, stats, autotune) also remain in `core`.
//!
//! ## Pillars re-exports
//!
//! All neuron models from `pillars` are re-exported for convenience:
//! ```rust
//! use taas_core::neuron::{LIFModel, HHModel, run_neuron};
//! ```
//!
//! Sub-modules by category:
//! - Model wrappers: [`izhikevich`], [`saw`], [`lif`], [`adex`], [`hh`], etc.
//! - Plasticity: [`stdp`], [`stp`], [`sorn`], [`eprop`], [`bcm`], [`trainer`]
//! - Utility: [`encoding`], [`poisson`], [`delay`], [`stats`], [`autotune`], [`soma_dendrite`]

// ═══════════════════════════════════════════════════════
//  🧠 Model Wrappers (safe API поверх SIMD ядер)
// ═══════════════════════════════════════════════════════
pub mod adex; // AdEx & E-Prop AdEx
pub mod fhn; // FitzHugh-Nagumo
pub mod glif; // GLIF3/4 (Generalized LIF — Allen Institute)
pub mod haap; // HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting)
pub mod hr; // Hindmarsh-Rose
pub mod izhikevich; // Izhikevich & E-Prop Izhikevich
pub mod lif; // LIF, LIF-AT & E-Prop LIF
pub mod saw; // SAW, ψ-wave & E-Prop SAW
pub mod wc; // Wilson-Cowan population
pub mod atlasov_neuromotor; // Atlasov Neuromotor joint control
pub mod chordotonal; // Chordotonal sensory organ (load→rate)
pub mod astrocyte; // Astrocyte Ca²⁺ waves + Gap Junctions
pub mod embodied; // FlyWire Embodied Neuromotor Engine (поглощён из taas-embodied)

// ═══════════════════════════════════════════════════════
//  🔗 Plasticity & Learning (обучение и пластичность)
// ═══════════════════════════════════════════════════════
pub mod eprop; // Surrogate gradient & eprop_update
pub mod sorn; // Self-Organizing Recurrent Network
pub mod bcm; // BCM rule (Bienenstock-Cooper-Munro 1982) + Homeostatic Scaling
pub mod stdp; // STDP, homeostatic STDP & dopamine STDP
pub mod stp; // Short-Term Plasticity & LFP
pub mod trainer; // EPropTrainer (high-level training loop)

// ═══════════════════════════════════════════════════════
//  🧩 Multi-compartment & Analysis
// ═══════════════════════════════════════════════════════
pub mod autotune; // Automatic threshold/rate tuning
pub mod delay; // Circular delay buffer
pub mod encoding; // BSA, phase & TTFS spike encoding
pub mod poisson; // Poisson spike generator
pub mod soma_dendrite; // Somato-dendritic integration
pub mod stats; // SpikeRaster, firing rates, ISI, CV

// ─── Pillars Model Re-exports ────────────────────────────
pub use pillars::core::{run_neuron, SimdLane};
pub use pillars::zoo::{
    Model,
    LIFModel, IzhikevichModel, AdExModel, FHNModel,
    HAAPModel, HRModel, WCModel, MorrisLecarModel,
    GLIF3Model, GLIF4Model, HHModel, SAWModel,
    VSNModel, AstrocyteModel, CaPlasticityModel,
    AtlasovNeuromotorModel, Exp2SynModel,
};
#[cfg(feature = "rayon")]
pub use pillars::core::run_neuron_par;

// ─── Pillars Sensor/Synapse Re-exports ────────────────────
pub use pillars::zoo::sensors::chordotonal_step;
pub use pillars::zoo::synapses::{nmda_mg_block_step, tsodyks_markram_step};

// ─── Re-exports ──────────────────────────────────────────
pub use adex::{step_adex, EPropAdExMulti};
pub use atlasov_neuromotor::{
    step_atlasov_neuromotor, AtlasovState, AtlasovStateBuilder,
    BiomechanicsProfile, DrosophilaLegConfig, FiberPoolConfig,
};
pub use chordotonal::{ChordotonalBuilder, ChordotonalProfile, ChordotonalState};
pub use autotune::{probe_fr, suggest_config, EPropTunerConfig, FrProbePoint};
pub use delay::DelayBuffer;
pub use encoding::{bsa_encode, bsa_kernel_hamming, phase_encode, TtfsEncoder};
pub use eprop::{eprop_update, surrogate_gradient};
pub use fhn::step_fhn;
pub use haap::{step_haap, HaapProfile, HaapState, HaapStateBuilder};
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

/// Scale factor: 1/2^24, converts 24-bit random state to [0, 1)
pub const RNG_SCALE_24: f32 = 1.0 / 16777216.0;

/// Default refractory period (ms).
/// Shiu et al. 2024: 2.2ms for Drosophila LIF.
pub const DEFAULT_TAU_REF: f32 = 2.2;
