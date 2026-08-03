//! # 🧮 Math — Canonical per-lane primitives
//!
//! Pure math with no biological naming: `sigmoid`, `vtrap`, `fast_exp`.
//! Biological primitives (ohmic, gate integration) live in [`crate::ops`].

pub mod fast_exp;
pub mod sigmoid;
pub mod vtrap;

pub use fast_exp::*;
pub use sigmoid::*;
pub use vtrap::*;
