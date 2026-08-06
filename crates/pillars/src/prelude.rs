//! # Pillars Prelude — Zero-Cost SIMD Neural Dynamics
//!
//! One import brings in everything:
//!
//! ```rust
//! use pillars::prelude::*;
//! ```
//!
//! ## What's included (grouped by layer)
//!
//! - **core/ (engine)**: `SimdLane`, `SimdContext`, `Cond`, `ComputeNeuron`,
//!   `StepContext`, `NeuralPopulation`, `run_neuron*`
//! - **zoo/ (models)**: `Model` + point neurons, synapses, sensors, glia
//! - **api/ (orchestration)**: `Network`, `SparseGraph`, formula interpreter
//! - **math/**: sigmoid, vtrap, fast_exp (canonical per-lane primitives)
//! - **kernels/ (batch microkernels)**: signal, learning, dopamine, delay,
//!   phase, decode, measure, input, stats, dendrite, vnc
//! - **SIMD architecture**: `f32x8`, `SimdType`, `AlignedVec`, `pad_lanes`
//! - **Constants & Q10**: `E_CL`, `E_NA`, `E_K`, `E_CA`, `q10_from_celsius`

// ═══ Layer 1: api/ — network orchestration (safe) ═══════════
pub use crate::api::builder::{
    AdExBuilder, HAAPBuilder, HHBuilder, HHPreset, IzhPattern, IzhikevichBuilder, LIFBuilder,
    ModelBuilder, UniversalBuilder, VSNBuilder,
};
pub use crate::api::formula::{
    run_program, FormulaModel, OpKind, OpNode, I_SYN, MAX_FORMULA_MODELS, N_REGS, P0, S0, S1, S2,
    S3, TEMP0,
};
pub use crate::api::{Network, Population, PopulationId, SparseGraph};

// ═══ Layer 2: zoo/ — ready-made models ══════════════════════
pub use crate::zoo::sensors::chordotonal_step;
pub use crate::zoo::synapses::{nmda_mg_block_step, tsodyks_markram_step};
pub use crate::zoo::{
    AdExModel, AstrocyteModel, AtlasovNeuromotorModel, CaPlasticityModel, Exp2SynModel, FHNModel,
    GLIF3Model, GLIF4Model, HAAPModel, HHModel, HRModel, IzhikevichModel, LIFModel, LIFParams,
    LIFState, Model, MorrisLecarModel, SAWModel, VSNModel, WCModel,
};

// ═══ Layer 3: core/ — SIMD engine ══════════════════════════
#[cfg(feature = "rayon")]
pub use crate::core::run_neuron_par;
pub use crate::core::{
    run_neuron, run_neuron_ext, ComputeNeuron, Cond, NeuralPopulation, SimdContext, SimdLane,
    StepContext,
};

// ═══ Layer 4: math/ — canonical per-lane primitives ═════════
pub use crate::math::fast_exp::{fast_exp, fast_exp_simd};
pub use crate::math::sigmoid::{sigmoid, sigmoid_s};
pub use crate::math::vtrap::{vtrap, vtrap_s};

// ═══ SIMD architecture: types & buffers (core/) ═══════════════
pub use crate::core::{
    is_aligned, is_slice_aligned, num_chunks, pad_lanes, split_simd, split_simd_ro, tail_len,
    AlignedVec, SimdType, ALIGN, LANES,
};
pub use wide::f32x8;

// ═══ Constants & Q10 (crate root) ═══════════════════════════
pub use crate::{
    q10_from_celsius, q10_from_celsius_s, scale_tau_q10, scale_tau_q10_s, E_CA, E_CL, E_K, E_NA,
};

// ═══ Layer 5: kernels/ — batch f32x8 microkernels ═══════════
pub use crate::kernels::decode::{
    adam_train, kalman_update, nlms_train, pst_decode, ring_synthesis, t3_complex_decoder,
    t3_lms_train, t3_pst_features,
};
pub use crate::kernels::delay::{delay_buffer_circular, delay_buffer_multi_tap};
pub use crate::kernels::dendrite::{CableDendriteState, DendriteConfig, DendriteDecayFactors};
pub use crate::kernels::dopamine::{dopamine_traces_step, dopamine_weight_update};
pub use crate::kernels::hh_channels::{
    cat_c_alpha, cat_c_beta, k_n_alpha, k_n_beta, na_h_alpha, na_h_beta, na_m_alpha, na_m_beta,
    pacemaking_channels, standard_channels, steady_state_gates, ChannelSpec, GateSpec, RateFn,
};
pub use crate::kernels::input::{bsa_encode, phase_encode, PoissonGenerator, TtfsEncoder};
pub use crate::kernels::learning::{
    bcm_step, eprop_update, homeostatic_scale, stdp_update, surrogate_gradient, SornNetwork,
};
pub use crate::kernels::measure::{
    band_energy, batch_coherence_sweep, interference, interference_mean, lazy_sincos_update,
    phase_coherence, spike_entropy, winding_matrix,
};
pub use crate::kernels::phase::{
    kuramoto_prepare, kuramoto_update, kuramoto_update_viscous, t3_update,
};
pub use crate::kernels::signal::{
    adam_step_simd, add_simd, hebbian_update, linear_decoder, phase_gate_apply, relu_simd,
    safety_clamp, sgd_step_simd, sigmoid_gate, stochastic_encoder,
};
pub use crate::kernels::stats::{compute_cvs, firing_rates, SpikeRaster};
pub use crate::kernels::synapse::{
    build_receptor_types, nmda_mg_block, DecayFactors, ReceptorType, SynapticKinetics,
};
pub use crate::kernels::vnc::vnc_step_mapping;
pub use crate::kernels::{clamp_amp, fast_sincos_simd, wrap_phase};
