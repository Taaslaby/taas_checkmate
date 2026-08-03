//! # Synapse — Synaptic Transmission
//!
//! ## `dual_exp_step`
//!
//! Uses **exponential Euler** for decay (stable for all tau values):
//! ```text
//! g1' = g1 · exp(-dt · τ₁⁻¹)
//! g2' = g2 · exp(-dt · τ₂⁻¹)
//! On spike: g1 += w, g2 += w
//! ```
//!
//! Exponential Euler is numerically stable for ANY dt/tau ratio.
//! Previously used Forward Euler which blows up when tau < dt
//! (e.g., tau1=0.1ms, dt=1ms → g' = g + 1·(-10·g) = -9g).
//!
//! Total conductance = g1 - g2 (peak ≈ 1 for normalized synapses).

use crate::core::lane::SimdLane;

/// Single step of dual-exponential synapse with exponential Euler decay.
///
/// Returns `(g1_new, g2_new)`.
///
/// # Spike format
/// `spike` может быть SIMD-маской (NaN от `cmp_ge`) или float (1.0).
/// Конвертируется в 1.0/0.0 через `spike_to_float` — безопасно для blend.
#[inline(always)]
pub fn dual_exp_step<S: SimdLane>(g1: S, g2: S, spike: S, weight: S, inv_tau1: S, inv_tau2: S, dt: S) -> (S, S) {
    // Exponential Euler: g' = g · exp(-dt · inv_tau)
    // Zero-division safe: inv_tau precomputed at init, exp(-dt*inv_tau) ∈ (0, 1]
    let g1_decay = g1 * (-dt * inv_tau1).exp();
    let g2_decay = g2 * (-dt * inv_tau2).exp();
    // Convert ANY spike format to 1.0/0.0 before arithmetic
    // ⚠️ NEVER use spike.cond().then() — float 1.0 has sign bit = 0,
    //    causing vblendvps to treat it as FALSE (only NaN has all bits set).
    let s = crate::ops::spike::spike_to_float(spike);
    let g1_new = g1_decay + s * weight;
    let g2_new = g2_decay + s * weight;
    (g1_new, g2_new)
}

/// Synaptic current from dual-exponential: `I_syn = scale · (g1 - g2) · (V - E_rev)`
///
/// Since g = g2 - g1 (conductance = decay - rise), this simplifies to:
/// `I_syn = scale · g · (E_rev - V)` (ADDING convention).
///
/// Convention: result should be **added** to I_inj in the membrane equation
/// (not subtracted, unlike `ohmic::ohmic_current`).
/// For excitatory synapses (E_rev > V), the current is positive (depolarizing).
/// For inhibitory synapses (E_rev < V), the current is negative (hyperpolarizing).
#[inline(always)]
pub fn dual_exp_current<S: SimdLane>(g1: S, g2: S, v: S, e_rev: S, scale: S) -> S {
    scale * (g1 - g2) * (v - e_rev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_dual_exp_decay() {
        let g1 = f32x8::splat(1.0);
        let g2 = f32x8::splat(0.0);
        let spike = f32x8::splat(0.0);
        let w = f32x8::splat(0.0);
        let inv_tau1 = f32x8::splat(1.0 / 5.0);
        let inv_tau2 = f32x8::splat(1.0 / 50.0);
        let dt = f32x8::splat(1.0);
        let (g1n, g2n) = dual_exp_step(g1, g2, spike, w, inv_tau1, inv_tau2, dt);
        // Exponential: 1 * exp(-1/5) ≈ 0.8187
        assert!((g1n.to_array()[0] - 0.8187).abs() < 0.01,
            "g1 should decay to ~0.819, got {:.4}", g1n.to_array()[0]);
        assert!(g2n.to_array()[0] < 0.001, "g2 should stay near 0");
    }

    #[test]
    fn test_dual_exp_spike_bump() {
        let g1 = f32x8::splat(0.0);
        let g2 = f32x8::splat(0.0);
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // true mask
        let w = f32x8::splat(0.5);
        let inv_tau1 = f32x8::splat(1.0 / 5.0);
        let inv_tau2 = f32x8::splat(1.0 / 50.0);
        let dt = f32x8::splat(1.0);
        let (g1n, g2n) = dual_exp_step(g1, g2, spike, w, inv_tau1, inv_tau2, dt);
        // With spike: g = 0 * exp + 0.5 = 0.5
        assert!((g1n.to_array()[0] - 0.5).abs() < 1e-6, "g1 should bump to w");
        assert!((g2n.to_array()[0] - 0.5).abs() < 1e-6, "g2 should bump to w");
    }

    #[test]
    fn test_dual_exp_fast_synapse_stability() {
        // tau1 = 0.1ms (very fast AMPA-like rise), dt = 1ms
        // Forward Euler would give: g1' = 1 + 1·(-1·10) = -9 (WRONG — negative!)
        // Exponential Euler: g1' = 1 · exp(-1·10) = 4.5e-5 (correct — near zero)
        let g1 = f32x8::splat(1.0);
        let g2 = f32x8::splat(0.0);
        let spike = f32x8::splat(0.0);
        let w = f32x8::splat(0.0);
        let inv_tau1 = f32x8::splat(10.0);  // 1/0.1ms
        let inv_tau2 = f32x8::splat(1.0 / 50.0);
        let dt = f32x8::splat(1.0);
        let (g1n, _) = dual_exp_step(g1, g2, spike, w, inv_tau1, inv_tau2, dt);
        assert!(g1n.to_array()[0] > 0.0,
            "Exp Euler: g1 should be positive with tau1=0.1, got {:.6}", g1n.to_array()[0]);
        assert!(g1n.to_array()[0] < 0.01,
            "Exp Euler: g1 should decay near 0 with tau1=0.1, got {:.6}", g1n.to_array()[0]);
    }

    #[test]
    fn test_dual_exp_current_sign() {
        let g1 = f32x8::splat(0.0);
        let g2 = f32x8::splat(1.0);  // g = g2 - g1 = 1.0
        let v = f32x8::splat(-65.0);
        let e_exc = f32x8::splat(0.0);   // excitatory
        let e_inh = f32x8::splat(-70.0); // inhibitory
        let scale = f32x8::splat(1.0);

        let i_exc = dual_exp_current(g1, g2, v, e_exc, scale);
        let i_inh = dual_exp_current(g1, g2, v, e_inh, scale);

        // This uses ADDING convention: (g1-g2)*(V-E) = g*(E-V)
        // Excitatory (E=0, V=-65): I = g*(0+65) > 0 → depolarizing when ADDED
        assert!(i_exc.to_array()[0] > 0.0,
            "excitatory current should be positive: {:.4}", i_exc.to_array()[0]);
        // Inhibitory (E=-70, V=-65): I = g*(-70+65) = g*(-5) < 0 → hyperpolarizing when ADDED
        assert!(i_inh.to_array()[0] < 0.0,
            "inhibitory current should be negative: {:.4}", i_inh.to_array()[0]);
    }
}
