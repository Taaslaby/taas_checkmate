//! # Pillars — Reusable SIMD Bio Primitives
//!
//! Строительные блоки (столпы) для нейромоделей.
//! Каждый примитив работает с `f32x8` (SIMD) И скалярным `f32`.
//!
//! ## Usage
//!
//! ```rust
//! use pillars::prelude::*;
//!
//! // SIMD: 8 нейронов за раз
//! let v = f32x8::splat(-65.0);
//! let u_inf = sigmoid(v, -35.0, 5.0);
//! let u_new = gate_integrate(u_inf, u_inf, 100.0, 0.05);
//! let i_ion = ohmic(0.1, -65.0, v) + ohmic(0.5, -80.0, v) * u_new;
//!
//! // Scalar: один нейрон
//! let v_s = -65.0_f32;
//! let u_inf_s = sigmoid_s(v_s, -35.0, 5.0);
//! ```
//!
//! ## Pillars Checker
//!
//! `#[pillars::check]` annotates neuron models and runs compile-time
//! biological invariant checks. Like borrow checker, but for neurobiology.
//!
//! ```ignore
//! #[pillars::check]
//! pub fn my_model(v: &mut [f32], ...) {
//!     // Компилятор проверяет:
//!     // - E_rev Cl⁻ == E_rev GABA (оба Cl⁻)
//!     // - gap junction bidirectional
//!     // - cnexp NOT on nonlinear ODEs
//! }
//! ```

use wide::f32x8;

pub mod prelude;
pub mod sigmoid;
pub mod gate;
pub mod ohmic;
pub mod vtrap;
pub mod fast_exp;

// Re-export core types for easy access
pub use sigmoid::{sigmoid, sigmoid_s};
pub use gate::{gate_integrate, gate_integrate_s};
pub use ohmic::{ohmic, ohmic_s};
pub use vtrap::{vtrap, vtrap_s};
pub use fast_exp::{fast_exp, fast_exp_simd};

// ═══════════════════════════════════════════════════════════
//  Temperature (Q10) — единый для всех primitives
// ═══════════════════════════════════════════════════════════

/// Compute Q10 factor from temperature in Celsius.
/// NEURON convention: Q10 = 3^((T - 6.3) / 10).
/// At 6.3°C: Q10 = 1.
/// At 37°C:  Q10 = 29.
#[inline(always)]
pub fn q10_from_celsius_s(celsius: f32) -> f32 {
    (3.0_f32).powf((celsius - 6.3) / 10.0)
}

/// SIMD version of Q10.
#[inline(always)]
pub fn q10_from_celsius(celsius: f32) -> f32x8 {
    f32x8::splat(q10_from_celsius_s(celsius))
}

/// Apply Q10 to a time constant: τ_eff = τ / Q10.
/// At high temperature, kinetics accelerate (τ decreases).
#[inline(always)]
pub fn scale_tau_q10_s(tau: f32, celsius: f32) -> f32 {
    tau / q10_from_celsius_s(celsius)
}

#[inline(always)]
pub fn scale_tau_q10(tau: f32, celsius: f32) -> f32x8 {
    f32x8::splat(scale_tau_q10_s(tau, celsius))
}

// ═══════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════

/// Drosophila E_rev for Chloride channels (GluCl, GABA_A).
/// Nernst: [Cl⁻]i = 10-15 mM → E_Cl = -70 mV.
pub const E_CL: f32 = -70.0;

/// Drosophila E_rev for Sodium channels (AMPA, nAChR).
pub const E_NA: f32 = 0.0;

/// Drosophila E_rev for Potassium channels (delayed rectifier).
pub const E_K: f32 = -80.0;

/// Drosophila E_rev for Calcium channels (NMDA).
pub const E_CA: f32 = 0.0;

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use wide::f32x8;
    use crate::prelude::*;

    #[test]
    fn test_sigmoid_simd() {
        let v = f32x8::from([-65.0, -50.0, -35.0, -20.0, 0.0, 20.0, 40.0, 60.0]);
        let s = sigmoid(v, -35.0, 5.0);
        let a = s.to_array();
        // At V=-35: sigmoid(0) = 0.5
        assert!((a[2] - 0.5).abs() < 0.01, "sigmoid at Vh={}", a[2]);
        // At V=-65: sigmoid(-30/5) = sigmoid(-6) ≈ 0.002
        assert!(a[0] < 0.01, "sigmoid at -65 should be near 0: {}", a[0]);
        // At V=60: sigmoid(95/5) = sigmoid(19) ≈ 1.0
        assert!((a[7] - 1.0).abs() < 0.01, "sigmoid at 60 should be near 1: {}", a[7]);
    }

    #[test]
    fn test_sigmoid_scalar() {
        let s = sigmoid_s(-35.0, -35.0, 5.0);
        assert!((s - 0.5).abs() < 0.01, "sigmoid_s at Vh: {}", s);
    }

    #[test]
    fn test_gate_integrate_simd() {
        // gate starting at 0.0, inf=1.0, tau=10ms, dt=1ms
        // expected: 1.0 + (0.0 - 1.0) * exp(-1/10) = 1 - 0.9048 = 0.0952
        let gate = f32x8::splat(0.0);
        let inf = f32x8::splat(1.0);
        let g = gate_integrate(gate, inf, 10.0, 1.0);
        let expected = 1.0 - (-1.0 / 10.0_f32).exp();
        assert!((g.to_array()[0] - expected).abs() < 0.001,
                "gate_integrate: got {}, expected {}", g.to_array()[0], expected);
    }

    #[test]
    fn test_ohmic_simd() {
        // g=1.0, E_rev=0, V=-65 → I = 1 * (0 - (-65)) = 65
        let v = f32x8::splat(-65.0);
        let i = ohmic(1.0, 0.0, v);
        assert!((i.to_array()[0] - 65.0).abs() < 0.01, "ohmic: {}", i.to_array()[0]);

        // Cl⁻: g=1.0, E_rev=-70, V=-65 → I = 1 * (-70 - (-65)) = -5 (гиперполяризация)
        let i_cl = ohmic(1.0, E_CL, v);
        assert!((i_cl.to_array()[0] - (-5.0)).abs() < 0.01, "ohmic Cl-: {}", i_cl.to_array()[0]);
    }

    #[test]
    fn test_vtrap_simd() {
        // vtrap(x, y) = x / (exp(x/y) - 1) when |x/y| > 1e-6, else y*(1 - x/(2*y))
        // x = 0 → vtrap = y
        use wide::f32x8;
        let x = f32x8::splat(0.0);
        let y = f32x8::splat(10.0);
        let r = vtrap(x, y);
        assert!((r.to_array()[0] - 10.0).abs() < 0.01, "vtrap(0,10) = {}", r.to_array()[0]);
    }

    #[test]
    fn test_q10() {
        let q = q10_from_celsius_s(6.3);
        assert!((q - 1.0).abs() < 0.001, "Q10 at 6.3°C should be 1: {}", q);

        let q37 = q10_from_celsius_s(37.0);
        assert!((q37 - 29.0).abs() < 1.0, "Q10 at 37°C should be ~29: {}", q37);
    }

    #[test]
    fn test_e_cl_negative() {
        // Cl⁻ каналы должны иметь отрицательный E_rev (гиперполяризация)
        assert!(E_CL < 0.0, "E_CL={} should be negative (hyperpolarizing)", E_CL);
        assert!((E_CL - (-70.0)).abs() < 1.0, "E_CL should be -70 for Drosophila");
    }

    // Verify pillars-macros compiles by applying #[check] to a test function
    // (only when checker feature is enabled)
    #[cfg(feature = "checker")]
    #[test]
    fn test_pillars_checker_compiles() {
        use pillars_macros::check;

        /// Dummy function to test #[check] proc-macro compiles
        #[check]
        fn dummy_step(_v: &mut [f32], _I: &[f32]) {}

        dummy_step(&mut [0.0], &[1.0]);
    }

    /// Without checker feature, just ensure crate-level E_rev invariants
    #[cfg(not(feature = "checker"))]
    #[test]
    fn test_pillars_erev_invariants() {
        assert!(E_CL < 0.0, "E_CL must be negative (Cl⁻)");
        assert!((E_CL - (-70.0)).abs() < 1.0, "E_CL ≈ -70 mV for Drosophila");
    }
}
