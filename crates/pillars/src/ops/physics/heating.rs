//! # Heating — Thermal and Temperature Effects
//!
//! ## `q10_factor`
//! ```text
//! Q10_factor = Q10^{(T - T_ref) / 10}
//! ```
//! Temperature coefficient for rate constants.
//! For time constants: `τ_corrected = τ / Q10_factor`.
//! For rate constants: `rate_corrected = rate · Q10_factor`.
//!
//! | Process | Q10 |
//! |---------|:---:|
//! | HH gate kinetics | 3.0 |
//! | Ion channel conductance | 1.2-1.5 |
//! | Synaptic transmission | 2-3 |
//! | Ca²⁺ buffering | 2.0 |

use crate::core::lane::SimdLane;

/// Compute Q10 temperature factor: `Q10^{(T - T_ref) / 10}`
#[inline(always)]
pub fn q10_factor<S: SimdLane>(q10: S, temp_current: f32, temp_ref: f32) -> S {
    let delta_t = temp_current - temp_ref;
    let exponent = q10.ln() * S::splat(delta_t / 10.0);
    exponent.exp()
}

/// Correct a time constant: `τ' = τ / Q10^{(T - T_ref) / 10}`
#[inline(always)]
pub fn correct_tau<S: SimdLane>(tau: S, q10: S, temp_current: f32, temp_ref: f32) -> S {
    tau / q10_factor(q10, temp_current, temp_ref)
}

/// Correct a rate constant: `rate' = rate · Q10^{(T - T_ref) / 10}`
#[inline(always)]
pub fn correct_rate<S: SimdLane>(rate: S, q10: S, temp_current: f32, temp_ref: f32) -> S {
    rate * q10_factor(q10, temp_current, temp_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_q10_at_reference() {
        let q10 = f32x8::splat(3.0);
        let f = q10_factor(q10, 6.3, 6.3); // same temp → factor = 1
        assert!((f.to_array()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_q10_10_degrees() {
        let q10 = f32x8::splat(3.0);
        let f = q10_factor(q10, 16.3, 6.3); // ΔT = 10 → factor = 3
        assert!(
            (f.to_array()[0] - 3.0).abs() < 0.01,
            "Q10=3 at ΔT=10: {:.4}",
            f.to_array()[0]
        );
    }
}
