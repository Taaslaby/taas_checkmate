//! # 🔄 Phase Dynamics — Kuramoto & T3 Oscillators
//!
//! `f32x8` kernels for coupled phase oscillators. Both standard Kuramoto
//! and Stuart-Landau amplitude dynamics (T3 Triple-Threshold Theorem).
//!
//! ## Models
//!
//! | Kernel | Equation | Type |
//! |--------|----------|------|
//! | [`kuramoto_prepare`] | `φ ← wrap(φ)` + sin/cos | Phase wrap + trig in one pass |
//! | [`kuramoto_update`] | `φ ← φ + dt·(ω + C)` | Standard Kuramoto |
//! | [`kuramoto_update_viscous`] | `φ ← φ + dt·(ω + C + η·(⟨φ⟩−φ))` | Viscous damping |
//! | [`t3_update`] | `r ← r + dt·((α−r²)r + G·D − δ·r)` | Stuart-Landau amplitude |
//! | | `θ ← θ + dt·(ω + C + G·D/(r+ε))` | Kuramoto phase (with drive) |
//!
//! ## Kuramoto vs T3
//!
//! ```text
//! Standard Kuramoto:    φ̇ = ω + Σ W·sin(φⱼ − φᵢ)
//! T3 (Stuart-Landau):  ṙ = (α − r²)r + G·D − δ·r   (amplitude envelope)
//!                      θ̇ = ω + C + G·D/(r+ε)       (phase with gain)
//! ```
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | `kuramoto` | `kuramoto_prepare`, `kuramoto_update`, `kuramoto_update_viscous` |
//! | `t3` | `t3_update`, `t3_step_sparse` is in `sparse` |
//! | `gco` | `gco_step` |
//! | `sparse` | sparse helpers + `kuramoto_step_sparse`/`kuramoto_step_dense`/`t3_step_sparse` |
//! | `tests` | unit tests (test-only) |

mod gco;
mod kuramoto;
mod sparse;
mod t3;
#[cfg(test)]
mod tests;

pub use gco::*;
pub use kuramoto::*;
pub use sparse::*;
pub use t3::*;
