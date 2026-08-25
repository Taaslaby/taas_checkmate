//! Glia — Astrocyte & Neuro-Glial Modulation
//!
//! Hidden-state glial models that modulate synaptic transmission.

mod astrocyte;

pub use astrocyte::{
    AstrocyteModel, AstrocyteModelParamIdx, AstrocyteModelStateIdx, AstrocyteParams, AstrocyteState,
};
