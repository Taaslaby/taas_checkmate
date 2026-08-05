//! # 🧠 Neurons — spiking & non-spiking point models
//!
//! All models implement the [`Model`](crate::zoo::Model) trait.
//! Use `run_neuron::<M>(...)` from [`core`](crate::core) to execute.
//!
//! ## Two API styles
//!
//! - **Typed** (`#[derive(NeuronModel)]` + `ComputeNeuron`): access fields
//!   directly via `state.v`, `params.v_thresh`. Preferred for new models.
//! - **ParamIdx enums**: kept for manual `fill_param` / slice access.
//!   Use `ctx.p_enum(ModelNameParamIdx::FieldName)` instead of raw `ctx.p(N)`.
//!
//! Synapses, glia and muscle live in [`super::synapses`] / [`super::glia`]
//! and are re-exported by [`super`] — neurons are **point bodies only**.

mod adex;
mod atlasov_neuromotor;
mod fhn;
mod glif3;
mod glif4;
mod haap;
mod hh;
mod hr;
mod izhikevich;
mod lif;
mod morris_lecar;
mod rate;
mod saw;
mod triangulated;
mod vsn;
mod wc;

/// CUDA C++ sources transpiled by `#[dual_neuron]` — NVRTC-ready.
/// Generated in `pillars` itself; no CUDA runtime needed to build it.
///
/// Every `#[derive(NeuronModel)]` model also generates `{Model}ParamIdx` /
/// `{Model}StateIdx` index enums (kill magic column numbers):
/// `pop.fill_param(LIFModelParamIdx::Dt.into(), 0.5)`.
pub use adex::ADEX_CUDA_SOURCE;
pub use adex::{
    AdExModel, AdExModelParamIdx, AdExModelStateIdx, AdExParamIdx, AdExParams, AdExState,
};
pub use atlasov_neuromotor::ATLASOV_NEUROMOTOR_CUDA_SOURCE;
pub use atlasov_neuromotor::{
    AtlasParamIdx, AtlasovNeuromotorModel, AtlasovNeuromotorModelParamIdx,
    AtlasovNeuromotorModelStateIdx, AtlasovNeuromotorParams, AtlasovNeuromotorState,
};
pub use fhn::FHNModel;
pub use fhn::{FHNModelParamIdx, FHNModelStateIdx};
pub use fhn::FHN_CUDA_SOURCE;
pub use glif3::GLIF3_CUDA_SOURCE;
pub use glif3::{GLIF3Model, GLIF3ModelParamIdx, GLIF3ModelStateIdx, GLIF3ParamIdx, GLIF3Params, GLIF3State};
pub use glif4::GLIF4_CUDA_SOURCE;
pub use glif4::{GLIF4Model, GLIF4ModelParamIdx, GLIF4ModelStateIdx, GLIF4ParamIdx, GLIF4Params, GLIF4State};
pub use haap::HAAP_CUDA_SOURCE;
pub use haap::{HAAPModel, HAAPModelParamIdx, HAAPModelStateIdx, HAAPParams, HAAPState, HAAParamIdx};
pub use hh::HH_CUDA_SOURCE;
pub use hh::{
    GateParams, HHModel, HHModelParamIdx, HHModelStateIdx, HHParamIdx, HHParams, HHState, GATE_H,
    GATE_M, GATE_N,
};
pub use hr::HRModel;
pub use hr::{HRModelParamIdx, HRModelStateIdx};
pub use hr::HR_CUDA_SOURCE;
pub use izhikevich::IZHIKEVICH_CUDA_SOURCE;
pub use izhikevich::{
    IzhParamIdx, IzhikevichModel, IzhikevichModelParamIdx, IzhikevichModelStateIdx,
    IzhikevichParams, IzhikevichState,
};
pub use lif::LIF_CUDA_SOURCE;
pub use lif::{LIFModel, LIFModelParamIdx, LIFModelStateIdx, LIFParams, LIFState};
pub use morris_lecar::MORRIS_LECAR_CUDA_SOURCE;
pub use morris_lecar::{
    MLParamIdx, MorrisLecarModel, MorrisLecarModelParamIdx, MorrisLecarModelStateIdx,
    MorrisLecarParams, MorrisLecarState,
};
pub use rate::RATE_CUDA_SOURCE;
pub use rate::{RateModel, RateModelParamIdx, RateModelStateIdx, RateParams, RateState};
pub use saw::SAWModel;
pub use saw::{SAWModelParamIdx, SAWModelStateIdx};
pub use saw::SAW_CUDA_SOURCE;
pub use triangulated::TriangulatedModel;
pub use vsn::VSN_CUDA_SOURCE;
pub use vsn::{VSNModel, VSNModelParamIdx, VSNModelStateIdx, VSNParamIdx, VSNParams, VSNState};
pub use wc::WCModel;
pub use wc::{WCModelParamIdx, WCModelStateIdx};
pub use wc::WC_CUDA_SOURCE;
