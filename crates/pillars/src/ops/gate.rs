//! # Gate — Gating Variable Dynamics
//!
//! ## `integrate_gate`
//! ```text
//! x' = x_inf + (x - x_inf) · exp(-dt · Q10 · (α+β))
//! ```
//! Exponential Euler — exact for linear gating kinetics.
//! Zero divisions: `inv_tau` precomputed at init.
//!
//! ## `sigmoid_gate`
//! ```text
//! x_∞ = 1 / (1 + exp(-(V - V_half) / k))
//! ```
//!
//! ## `forward_euler_gate`
//! ```text
//! x' = x + dt · (x_inf - x) · inv_tau
//! ```
//! Forward Euler — simpler, no exp per gate. Used in HAAP, Astrocyte.

use crate::core::lane::SimdLane;

/// Exponential Euler gate integration (exact, no truncation error).
///
/// `x' = x_inf + (x - x_inf) · exp(-dt · sum_rate)`
/// where `sum_rate = α + β = 1/τ`.
#[inline(always)]
pub fn integrate_gate<S: SimdLane>(x: S, x_inf: S, inv_tau: S, dt: S) -> S {
    let decay = (-dt * inv_tau).exp();
    x_inf + (x - x_inf) * decay
}

/// Sigmoid activation function: `1 / (1 + exp(-(v - v_half) / k))`
///
/// Thin wrapper around `S::sigmoid(v, v_half, k)` — uses scalar `v_half` and `k`
/// to avoid unnecessary SIMD splat operations. These are typically population-wide
/// constants, not per-neuron parameters.
///
/// For inverted sigmoid (e.g., HCN activation), pass negative k.
#[inline(always)]
pub fn sigmoid_gate<S: SimdLane>(v: S, v_half: f32, k: f32) -> S {
    S::sigmoid(v, v_half, k)
}

/// Forward Euler gate integration (simple, dt-limited).
///
/// `x' = x + dt · (x_inf - x) · inv_tau`
#[inline(always)]
pub fn forward_euler_gate<S: SimdLane>(x: S, x_inf: S, inv_tau: S, dt: S) -> S {
    x + dt * (x_inf - x) * inv_tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_integrate_gate_relaxation() {
        let x = f32x8::splat(0.5);
        let x_inf = f32x8::splat(0.1);
        let inv_tau = f32x8::splat(1.0);
        let dt = f32x8::splat(1.0);
        let x_new = integrate_gate(x, x_inf, inv_tau, dt);
        // decay = exp(-1) ≈ 0.3679, x' = 0.1 + 0.4·0.3679 ≈ 0.247
        let expected = 0.1 + 0.4 * (-1.0_f32).exp();
        assert!((x_new.to_array()[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_integrate_gate_exact_at_steady() {
        let x = f32x8::splat(0.1);
        let x_inf = f32x8::splat(0.1);
        let inv_tau = f32x8::splat(1.0);
        let dt = f32x8::splat(1.0);
        let x_new = integrate_gate(x, x_inf, inv_tau, dt);
        assert_eq!(x_new.to_array()[0], 0.1, "at steady state, no change");
    }

    #[test]
    fn test_sigmoid_gate_at_vhalf() {
        let v = f32x8::splat(0.0);
        let s = sigmoid_gate(v, 0.0, 5.0);
        assert!(
            (s.to_array()[0] - 0.5).abs() < 1e-6,
            "at V_half, sigmoid=0.5"
        );
    }

    #[test]
    fn test_sigmoid_gate_saturated() {
        let v = f32x8::splat(100.0);
        let s = sigmoid_gate(v, 0.0, 5.0);
        assert!(
            (s.to_array()[0] - 1.0).abs() < 0.01,
            "far above V_half, sigmoid≈1"
        );
    }

    #[test]
    fn test_sigmoid_gate_inverted() {
        let v = f32x8::splat(-100.0);
        // Inverted sigmoid (k negative) = activated at hyperpolarization
        let s = sigmoid_gate(v, -75.0, -5.5);
        assert!(
            s.to_array()[0] > 0.9,
            "inverted sigmoid at -100: {:.4}",
            s.to_array()[0]
        );
    }

    #[test]
    fn test_forward_euler_gate() {
        let x = f32x8::splat(0.0);
        let x_inf = f32x8::splat(1.0);
        let inv_tau = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let x_new = forward_euler_gate(x, x_inf, inv_tau, dt);
        // x' = 0 + 1*(1-0)*0.1 = 0.1
        assert!((x_new.to_array()[0] - 0.1).abs() < 1e-6);
    }
}
