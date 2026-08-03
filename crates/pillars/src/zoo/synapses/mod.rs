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

pub mod nmda;
pub mod tsodyks_markram;
pub mod ca_plasticity;
pub mod exp2syn;

pub use nmda::nmda_mg_block_step;
pub use tsodyks_markram::tsodyks_markram_step;
pub use ca_plasticity::{CaPlasticityModel, CaPlasticityState, CaPlasticityParams, CaPlastParamIdx};
pub use exp2syn::{Exp2SynModel, Exp2SynState, Exp2SynParams, Exp2SynParamIdx, exp2syn_factor};
