//! # 🔗 Synapses — SIMD-Accelerated Synaptic Dynamics
//!
//! Synaptic models that operate on synapse state (not neuron state).
//!
//! | Model | Description |
//! |-------|-------------|
//! | `NMDA` | Destexhe Mg²⁺ voltage-dependent block |
//! | `TsodyksMarkram` | Short-term plasticity (facilitation + depression) |
//! | `CaPlasticity` | Calcium-driven plasticity rule (weight update) |
//! | `Exp2Syn` | Dual-exponential conductance synapse (rise/decay) |

pub mod ca_plasticity;
pub mod exp2syn;
pub mod nmda;
pub mod tsodyks_markram;

pub use ca_plasticity::{
    CaPlasticityModel, CaPlasticityModelParamIdx, CaPlasticityModelStateIdx, CaPlasticityParams,
    CaPlasticityState,
};
pub use exp2syn::{
    exp2syn_factor, Exp2SynModel, Exp2SynModelParamIdx, Exp2SynModelStateIdx, Exp2SynParams,
    Exp2SynState,
};
pub use nmda::nmda_mg_block_step;
pub use tsodyks_markram::tsodyks_markram_step;
