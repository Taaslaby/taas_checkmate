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
pub use crate::api::{Network, Population, PopulationId, SparseGraph};
pub use crate::api::formula::{OpKind, OpNode, run_program, FormulaModel, MAX_FORMULA_MODELS, N_REGS, S0, S1, S2, S3, P0, I_SYN, TEMP0};
pub use crate::api::builder::{IzhikevichBuilder, LIFBuilder, AdExBuilder, IzhPattern};

// ═══ Layer 2: zoo/ — ready-made models ══════════════════════
pub use crate::zoo::{
    Model,
    LIFModel, LIFState, LIFParams,
    IzhikevichModel, AdExModel, FHNModel,
    HAAPModel, HRModel, WCModel, MorrisLecarModel,
    GLIF3Model, GLIF4Model, HHModel, SAWModel,
    VSNModel, AstrocyteModel, CaPlasticityModel,
    AtlasovNeuromotorModel, Exp2SynModel,
};
pub use crate::zoo::sensors::chordotonal_step;
pub use crate::zoo::synapses::{nmda_mg_block_step, tsodyks_markram_step};

// ═══ Layer 3: core/ — SIMD engine ══════════════════════════
pub use crate::core::{
    ComputeNeuron, StepContext,
    SimdContext, SimdLane, Cond,
    NeuralPopulation, run_neuron, run_neuron_ext,
};
#[cfg(feature = "rayon")]
pub use crate::core::run_neuron_par;

// ═══ Layer 4: math/ — canonical per-lane primitives ═════════
pub use crate::math::sigmoid::{sigmoid, sigmoid_s};
pub use crate::math::vtrap::{vtrap, vtrap_s};
pub use crate::math::fast_exp::{fast_exp, fast_exp_simd};

// ═══ SIMD architecture: types & buffers (core/) ═══════════════
pub use crate::core::{
    LANES, ALIGN, SimdType, pad_lanes, num_chunks, tail_len,
    is_aligned, is_slice_aligned,
    AlignedVec, split_simd, split_simd_ro,
};
pub use wide::f32x8;

// ═══ Constants & Q10 (crate root) ═══════════════════════════
pub use crate::{
    q10_from_celsius, q10_from_celsius_s,
    scale_tau_q10, scale_tau_q10_s,
    E_CL, E_NA, E_K, E_CA,
};

// ═══ Layer 5: kernels/ — batch f32x8 microkernels ═══════════
pub use crate::kernels::{
    wrap_phase, clamp_amp, fast_sincos_simd,
};
pub use crate::kernels::signal::{
    safety_clamp, stochastic_encoder, linear_decoder,
    sigmoid_gate, phase_gate_apply, relu_simd, add_simd,
    sgd_step_simd, adam_step_simd, hebbian_update,
};
pub use crate::kernels::dopamine::{
    dopamine_traces_step, dopamine_weight_update,
};
pub use crate::kernels::delay::{delay_buffer_circular, delay_buffer_multi_tap};
pub use crate::kernels::phase::{kuramoto_update, kuramoto_update_viscous, t3_update, kuramoto_prepare};
pub use crate::kernels::decode::{
    t3_complex_decoder, t3_pst_features, t3_lms_train,
    nlms_train, kalman_update, adam_train, pst_decode, ring_synthesis,
};
pub use crate::kernels::measure::{
    interference, interference_mean, phase_coherence, band_energy, winding_matrix,
    lazy_sincos_update, spike_entropy, batch_coherence_sweep,
};
pub use crate::kernels::input::{PoissonGenerator, bsa_encode, phase_encode, TtfsEncoder};
pub use crate::kernels::learning::{
    surrogate_gradient, eprop_update, stdp_update, bcm_step, homeostatic_scale, SornNetwork,
};
pub use crate::kernels::synapse::{
    ReceptorType, DecayFactors, SynapticKinetics, nmda_mg_block, build_receptor_types,
};
pub use crate::kernels::hh_channels::{
    ChannelSpec, GateSpec, RateFn,
    standard_channels, pacemaking_channels, steady_state_gates,
    na_m_alpha, na_m_beta, na_h_alpha, na_h_beta, k_n_alpha, k_n_beta, cat_c_alpha, cat_c_beta,
};
pub use crate::kernels::stats::{SpikeRaster, firing_rates, compute_cvs};
pub use crate::kernels::dendrite::{CableDendriteState, DendriteConfig, DendriteDecayFactors};
pub use crate::kernels::vnc::vnc_step_mapping;
