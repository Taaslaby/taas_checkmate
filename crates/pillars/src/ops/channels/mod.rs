//! # Channels — Ion Channel Current Primitives
//!
//! Each channel is a function `I = gating(V) · g · (V - E_rev)`.
//! All use Ohm's law convention (subtracted from I_inj).

pub mod gap;
pub mod glu_cl;
pub mod hcn;
pub mod kir;
pub mod km;
pub mod leak;

pub use gap::*;
pub use glu_cl::*;
pub use hcn::*;
pub use kir::*;
pub use km::*;
pub use leak::*;
