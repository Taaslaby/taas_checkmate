//! -------------------------------------------------------------------------
//! Class:       learn
//! Purpose:     Learning — trainers, autotuners, ChaosPatternEngine
//! Description: E-prop trainer (BPTT-style surrogate), the E-Prop FR probe /
//!              autotuner, and the ChaosPatternEngine (contrastive
//!              dopamine-STDP). Grouped here to keep the api root flat.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

pub mod autotune;
pub mod cpe;
pub mod trainer;
