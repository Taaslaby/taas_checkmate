//! # Learning Microkernels — e-prop, STDP, BCM & SORN
//!
//! Batch kernels for SNN learning:
//! - [`surrogate_gradient`] — fast sigmoid surrogate (e-prop)
//! - [`eprop_update`] — standalone e-prop weight update `Δw = η·L_j·e_ji`
//! - [`stdp_update`] — batch STDP weight update on flat arrays (SIMD + tail)
//! - [`bcm_step`] — Bienenstock-Cooper-Munro (1982) rate plasticity
//!   with sliding threshold θ_M, plus [`homeostatic_scale`] (Turrigiano 1998)
//! - SORN (Self-Organizing Recurrent Network): intrinsic plasticity,
//!   synaptic normalization and the full [`SornNetwork`] loop.
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | `eprop` | `surrogate_gradient`, `eprop_update` |
//! | `stdp` | `stdp_update`, `stdp_homeostatic` |
//! | `bcm` | `bcm_step`, `homeostatic_scale` |
//! | `hebbian` | `hebbian_csr_step` |
//! | `sorn` | `intrinsic_plasticity`, `synaptic_normalization`, `SornNetwork` |
//! | `tests` | unit tests (test-only) |

mod bcm;
mod eprop;
mod hebbian;
mod sorn;
mod stdp;
#[cfg(test)]
mod tests;

pub use bcm::*;
pub use eprop::*;
pub use hebbian::*;
pub use sorn::*;
pub use stdp::*;
