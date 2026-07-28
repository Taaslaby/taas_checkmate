//! Gate integration — Exponential Euler for gating variables.
//!
//! Common form: `x = x_inf + (x_old - x_inf) * exp(-dt / tau)`
//!
//! This matches NEURON's `cnexp` method — exact for linear ODEs.
//! Q10 temperature scaling is built-in via `scale_tau_q10`.

use wide::f32x8;
use crate::fast_exp_simd;

/// SIMD gate integration: `x = x_inf + (x - x_inf) * exp(-dt / tau)`
/// Processes 8 neurons at once.
#[inline(always)]
pub fn gate_integrate(gate: f32x8, x_inf: f32x8, tau: f32, dt: f32) -> f32x8 {
    let tau_safe = tau.max(1e-6);  // prevent division by zero
    let decay = fast_exp_simd(-f32x8::splat(dt) / f32x8::splat(tau_safe));
    x_inf + (gate - x_inf) * decay
}

/// Scalar gate integration: `x = x_inf + (x - x_inf) * exp(-dt / tau)`
#[inline(always)]
pub fn gate_integrate_s(gate: f32, x_inf: f32, tau: f32, dt: f32) -> f32 {
    let tau_safe = tau.max(1e-6);
    let decay = (-dt / tau_safe).exp();
    x_inf + (gate - x_inf) * decay
}

/// Gate integration with Q10 temperature scaling.
/// tau_eff = tau / Q10(T), where Q10 = 3^((T-6.3)/10).
#[inline(always)]
pub fn gate_integrate_q10(gate: f32x8, x_inf: f32x8, tau: f32, dt: f32, celsius: f32) -> f32x8 {
    let q10 = f32x8::splat(crate::q10_from_celsius_s(celsius));
    let tau_eff = f32x8::splat(tau) / q10;
    let tau_safe = tau_eff.max(f32x8::splat(1e-6));
    let decay = fast_exp_simd(-f32x8::splat(dt) / tau_safe);
    x_inf + (gate - x_inf) * decay
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_gate_integrate_basics() {
        let gate = f32x8::splat(0.0);
        let inf = f32x8::splat(1.0);
        // tau=10, dt=1: decay = exp(-0.1) = 0.9048, gate=1+(0-1)*0.9048=0.0952
        let g = super::gate_integrate(gate, inf, 10.0, 1.0);
        let expected = 1.0 - (-1.0 / 10.0_f32).exp();
        assert!((g.to_array()[0] - expected).abs() < 0.001, "gate: {}", g.to_array()[0]);
    }

    #[test]
    fn test_gate_integrate_steady_state() {
        // If gate == x_inf, gate stays at x_inf
        let gate = f32x8::splat(0.5);
        let inf = f32x8::splat(0.5);
        let g = super::gate_integrate(gate, inf, 10.0, 1.0);
        assert!((g.to_array()[0] - 0.5).abs() < 0.001, "steady state: {}", g.to_array()[0]);
    }

    #[test]
    fn test_gate_integrate_scalar_matches_simd() {
        let g_s = super::gate_integrate_s(0.0, 1.0, 10.0, 1.0);
        let g_simd = super::gate_integrate(f32x8::splat(0.0), f32x8::splat(1.0), 10.0, 1.0);
        assert!((g_s - g_simd.to_array()[0]).abs() < 0.001, "scalar vs SIMD");
    }

    #[test]
    fn test_gate_integrate_q10_speedup() {
        // At 37°C, Q10 ≈ 29 → tau_eff ≈ 10/29 ≈ 0.34ms → much faster kinetics
        let gate = f32x8::splat(0.0);
        let inf = f32x8::splat(1.0);
        let g_37 = super::gate_integrate_q10(gate, inf, 10.0, 1.0, 37.0);
        let g_6 = super::gate_integrate_q10(gate, inf, 10.0, 1.0, 6.3);
        // At 37°C, gate approaches inf much faster than at 6.3°C
        assert!(g_37.to_array()[0] > g_6.to_array()[0],
                "Q10: 37°C={} should be > 6.3°C={}", g_37.to_array()[0], g_6.to_array()[0]);
    }
}
