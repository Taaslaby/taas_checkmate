//! Prelude — re-export all primitives for easy import.
//!
//! ```rust
//! use pillars::prelude::*;
//! ```

pub use crate::sigmoid::{sigmoid, sigmoid_s};
pub use crate::gate::{gate_integrate, gate_integrate_s};
pub use crate::ohmic::{ohmic, ohmic_s};
pub use crate::vtrap::{vtrap, vtrap_s};
pub use crate::{
    fast_exp_simd, fast_exp,
    q10_from_celsius, q10_from_celsius_s,
    scale_tau_q10, scale_tau_q10_s,
    E_CL, E_NA, E_K, E_CA,
};
pub use wide::f32x8;
