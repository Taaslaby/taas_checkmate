//! # Ohmic — Ionic Current (Ohm's Law)
//!
//! ## Convention (NEURON)
//! ```text
//! I = g · (V - E_rev)
//! ```
//! All ionic currents use this form and are **subtracted** from I_inj:
//! ```text
//! dv/dt = (I_inj - I_ion) / C
//! ```
//!
//! Uses FMA: `g·V - g·E_rev` — one `vfmadd231ps` instruction.

use crate::core::lane::SimdLane;

/// Ohmic (conductance-based) current: `I = g · (V - E_rev)`
///
/// Computes ionic current using Ohm's law. Sign convention (NEURON):
/// - `I > 0` → **depolarizing** (inward cations or outward anions)
/// - `I < 0` → **hyperpolarizing** (outward cations or inward anions)
///
/// # Arguments
/// - `g`: conductance (mS/cm² or nS)
/// - `v`: membrane potential (mV)
/// - `e_rev`: reversal potential (mV)
///
/// # Examples
/// ```ignore
/// // Na⁺ leak: E_Na = 0 mV, V = -65 mV
/// let i_na = ohmic_current(g_na, v, E_NA);  // → negative (depolarizing)
///
/// // K⁺ leak: E_K = -80 mV, V = -65 mV
/// let i_k = ohmic_current(g_k, v, E_K);     // → positive (hyperpolarizing)
/// ```
///
/// # Performance
/// Single FMA instruction on AVX2: `g·V - g·E_rev` → `vfmadd231ps`
#[inline(always)]
pub fn ohmic_current<S: SimdLane>(g: S, v: S, e_rev: S) -> S {
    g.mul_add(v, -g * e_rev)
}

/// Gated ohmic current: `I = g_max · gating · (V - E_rev)`
///
/// Used for voltage-gated ion channels where conductance depends on
/// gate activation/inactivation states.
///
/// # Arguments
/// - `g_max`: maximum conductance (mS/cm²)
/// - `v`: membrane potential (mV)
/// - `e_rev`: reversal potential (mV)
/// - `gating`: combined gate factor (e.g., `m³·h` for Na⁺, `n⁴` for K⁺)
///
/// # Examples
/// ```ignore
/// // Hodgkin-Huxley Na⁺ channel: I_Na = g_Na · m³·h · (V - E_Na)
/// let gating_na = m.powi::<3>() * h;
/// let i_na = ohmic_gated(g_na_max, v, E_NA, gating_na);
///
/// // Hodgkin-Huxley K⁺ channel: I_K = g_K · n⁴ · (V - E_K)
/// let gating_k = n.powi::<4>();
/// let i_k = ohmic_gated(g_k_max, v, E_K, gating_k);
/// ```
///
/// # Performance
/// 2 FMA instructions: `g_eff = g_max·gating`, then `g_eff·V - g_eff·E_rev`
#[inline(always)]
pub fn ohmic_gated<S: SimdLane>(g_max: S, v: S, e_rev: S, gating: S) -> S {
    (g_max * gating).mul_add(v, -(g_max * gating) * e_rev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_ohmic_excitatory() {
        // I = g * (V - E): if V < E, current is negative → depolarizing
        let g = f32x8::splat(0.3);
        let v = f32x8::splat(-65.0);
        let e = f32x8::splat(0.0);    // E_rev=0mV (AMPA)
        let i = ohmic_current(g, v, e);
        // -65 * 0.3 = -19.5
        assert!((i.to_array()[0] - (-19.5)).abs() < 1e-6,
            "ohmic V=-65, E=0: got {:.4}", i.to_array()[0]);
    }

    #[test]
    fn test_ohmic_inhibitory() {
        let g = f32x8::splat(0.3);
        let v = f32x8::splat(-65.0);
        let e = f32x8::splat(-70.0);  // E_rev=-70mV (GABA)
        let i = ohmic_current(g, v, e);
        // (-65 - (-70)) * 0.3 = 5 * 0.3 = 1.5 (positive → hyperpolarizing)
        assert!((i.to_array()[0] - 1.5).abs() < 1e-6,
            "ohmic V=-65, E=-70: got {:.4}", i.to_array()[0]);
    }

    #[test]
    fn test_ohmic_gated() {
        let g = f32x8::splat(120.0);
        let v = f32x8::splat(-65.0);
        let e = f32x8::splat(50.0);
        let gating = f32x8::splat(0.05 * 0.05 * 0.05 * 0.6); // m³·h ≈ 0.000075
        let i = ohmic_gated(g, v, e, gating);
        // ~120 * 7.5e-5 * (-65 - 50) = 120 * 7.5e-5 * (-115) ≈ -1.035
        assert!(i.to_array()[0] < 0.0, "gated Na current should be negative: {:.6}", i.to_array()[0]);
    }
}
