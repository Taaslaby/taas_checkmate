//! # 🧫 Glia — Astrocyte & Neuro-Glial Modulation
//!
//! Hidden-state glial models that modulate synaptic transmission.

mod astrocyte;

pub use astrocyte::{
    AstroParamIdx, AstrocyteModel, AstrocyteModelParamIdx, AstrocyteModelStateIdx, AstrocyteParams,
    AstrocyteState,
};
