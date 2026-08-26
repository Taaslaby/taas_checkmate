//! -------------------------------------------------------------------------
//! Class:       embodied
//! Purpose:     Embodiment — motor pipeline + neuromotor engine + reference models
//! Description: Fly motor pipeline (motoneuron spikes → NMJ delay → Neuromotor
//!              engine → joint drive), the Neuromotor muscle engine itself
//!              (Ca²⁺/Hill dynamics), and the human-hand reference profile
//!              for Neuromotor validation.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

pub mod fly;
pub mod human_hand;
pub mod neuromotor;

// Flat surface: `api::embodied::FlyEmbodiedState`, `api::embodied::HumanHandProfile`,
// `api::embodied::NeuromotorState` — one path for all embodiment types.
pub use fly::*;
pub use human_hand::*;
pub use neuromotor::*;
