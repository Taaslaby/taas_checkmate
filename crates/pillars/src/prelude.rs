//! # Pillars Prelude — Zero-Cost SIMD Neural Dynamics
//!
//! One import brings in everything:
//!
//! ```rust
//! use pillars::prelude::*;
//! ```
//!
//! ## What's included
//!
//! - **Model framework**: `Model`, `SimdContext`, `SimdLane`, `run_neuron`
//! - **17 neuron models**: LIF, Izhikevich, AdEx, HH, FHN, HAAP, HR, WC,
//!   Morris-Lecar, GLIF3, GLIF4, SAW, VSN, Astrocyte, CaPlasticity,
//!   Atlasov, Exp2Syn
//! - **Math primitives**: sigmoid, gate_integrate, ohmic, vtrap, exp, trig
//! - **Fused ops**: sigmoid_gate_ohmic, haap_step_fused, synaptic_sum
//! - **SIMD types**: `f32x8`, `SimdType`
//! - **Buffers**: `AlignedVec`, split_simd
//! - **Constants**: `E_CL`, `E_NA`, `E_K`, `E_CA`
//! - **Temperature**: `q10_from_celsius`, `scale_tau_q10`

// ─── Model Framework ─────────────────────────────────────────
pub use crate::model::{
    Model, SimdContext, SimdLane, run_neuron, run_neuron_ext,
    StorageBuffer, TensorBridge, EngineOrchestrator,
    LIFModel, IzhikevichModel, AdExModel, FHNModel,
    HAAPModel, HRModel, WCModel, MorrisLecarModel,
    GLIF3Model, GLIF4Model, HHModel, SAWModel,
    VSNModel, AstrocyteModel, CaPlasticityModel,
    AtlasovNeuromotorModel, Exp2SynModel,
};
#[cfg(feature = "rayon")]
pub use crate::model::run_neuron_par;

// ─── Math Primitives ────────────────────────────────────────
pub use crate::math::sigmoid::{sigmoid, sigmoid_s};
pub use crate::math::gate::{gate_integrate, gate_integrate_s, gate_integrate_q10};
pub use crate::math::ohmic::{ohmic, ohmic_s, ohmic_per_neuron, ohmic_per_neuron_full};
pub use crate::math::vtrap::{vtrap, vtrap_s};
pub use crate::math::fast_exp::{fast_exp, fast_exp_simd};

// ─── Fused Operations ────────────────────────────────────────
pub use crate::math::fused::{sigmoid_gate_ohmic, haap_step_fused, synaptic_sum};

// ─── SIMD Architecture ──────────────────────────────────────
pub use crate::arch::{
    LANES, ALIGN, SimdType, pad_lanes, num_chunks, tail_len,
    is_aligned, is_slice_aligned,
};
pub use crate::buf::{AlignedVec, split_simd, split_simd_ro};

// ─── Constants & Q10 ────────────────────────────────────────
pub use crate::{
    q10_from_celsius, q10_from_celsius_s,
    scale_tau_q10, scale_tau_q10_s,
    E_CL, E_NA, E_K, E_CA,
};

// ─── Sensors & Synapses ────────────────────────────────────
pub use crate::model::sensors::chordotonal_step;
pub use crate::model::synapses::{nmda_mg_block_step, tsodyks_markram_step};

// ─── SIMD Type ──────────────────────────────────────────────
pub use wide::f32x8;
