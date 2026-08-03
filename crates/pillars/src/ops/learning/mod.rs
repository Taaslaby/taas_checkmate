//! # Learning — Learning Rule Primitives
//!
//! Operations for Hebbian learning, gradient optimization,
//! and surrogate gradient computation.

pub mod hebbian;
pub mod optimizer;
pub mod surrogate;

pub use hebbian::*;
pub use optimizer::*;
pub use surrogate::*;
