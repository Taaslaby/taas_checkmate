//! # Physics — Biophysical Computation Primitives
//!
//! Operations for diffusion, ephaptic fields, mechanical force, etc.

pub mod diffusion;
pub mod field;
pub mod force;
pub mod heating;
pub mod metabolism;

pub use diffusion::*;
pub use field::*;
pub use force::*;
pub use heating::*;
pub use metabolism::*;
