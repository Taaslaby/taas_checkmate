//! # Plasticity — Synaptic Plasticity Primitives
//!
//! | Module | Operation | Reference |
//! |--------|-----------|-----------|
//! | `stdp` | STDP + dopamine-modulated 3-factor | Bi & Poo 1998 |
//! | `stp` | Short-term plasticity (Tsodyks-Markram) | Tsodyks & Markram 1997 |
//! | `homeo` | Homeostatic scaling + rate estimation | Turrigiano 1999 |
//! | `trace` | Spike trace decay + add | — |

pub mod homeo;
pub mod stdp;
pub mod stp;
pub mod trace;

pub use homeo::*;
pub use stdp::*;
pub use stp::*;
pub use trace::*;
