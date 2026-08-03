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

mod triangulated;
mod lif;
mod izhikevich;
mod adex;
mod fhn;
mod haap;
mod hr;
mod wc;
mod morris_lecar;
mod glif3;
mod glif4;
mod hh;
mod saw;
mod vsn;
mod atlasov_neuromotor;

pub use triangulated::TriangulatedModel;
pub use lif::{LIFModel, LIFState, LIFParams};
pub use izhikevich::{IzhikevichModel, IzhikevichState, IzhikevichParams, IzhParamIdx};
pub use adex::{AdExModel, AdExState, AdExParams, AdExParamIdx};
pub use fhn::FHNModel;
pub use haap::{HAAPModel, HAAPState, HAAPParams, HAAParamIdx};
pub use hr::HRModel;
pub use wc::WCModel;
pub use morris_lecar::{MorrisLecarModel, MorrisLecarState, MorrisLecarParams, MLParamIdx};
pub use glif3::{GLIF3Model, GLIF3State, GLIF3Params, GLIF3ParamIdx};
pub use glif4::{GLIF4Model, GLIF4State, GLIF4Params, GLIF4ParamIdx};
pub use hh::{HHModel, HHState, HHParams, HHParamIdx, GateParams, GATE_M, GATE_H, GATE_N};
pub use saw::SAWModel;
pub use vsn::{VSNModel, VSNState, VSNParams, VSNParamIdx};
pub use atlasov_neuromotor::{AtlasovNeuromotorModel, AtlasovNeuromotorState, AtlasovNeuromotorParams, AtlasParamIdx};
