//! Ohmic current — the fundamental synaptic/ionic current.
//!
//! `I = g * (E_rev - V)`
//!
//! Where:
//! - `g` = conductance (µS)
//! - `E_rev` = reversal potential (mV)
//! - `V` = membrane potential (mV)
//!
//! Positive I = depolarizing (inward), Negative I = hyperpolarizing (outward).

use wide::f32x8;

/// SIMD ohmic current: `I = g * (E_rev - V)`
/// Processes 8 neurons at once.
#[inline(always)]
pub fn ohmic(g: f32, e_rev: f32, v: f32x8) -> f32x8 {
    f32x8::splat(g) * (f32x8::splat(e_rev) - v)
}

/// Scalar ohmic current: `I = g * (E_rev - V)`
#[inline(always)]
pub fn ohmic_s(g: f32, e_rev: f32, v: f32) -> f32 {
    g * (e_rev - v)
}

/// SIMD ohmic with per-neuron conductance (e.g., from synaptic kinetics).
#[inline(always)]
pub fn ohmic_per_neuron(g: f32x8, e_rev: f32, v: f32x8) -> f32x8 {
    g * (f32x8::splat(e_rev) - v)
}

/// SIMD ohmic with per-neuron conductance AND per-neuron E_rev.
#[inline(always)]
pub fn ohmic_per_neuron_full(g: f32x8, e_rev: f32x8, v: f32x8) -> f32x8 {
    g * (e_rev - v)
}

#[cfg(test)]
mod tests {
    use wide::f32x8;

    #[test]
    fn test_ohmic_depolarizing() {
        // Na⁺: E_rev=0, V=-65 → I = g * (0 - (-65)) = 65g (depolarizing)
        let v = f32x8::splat(-65.0);
        let i = super::ohmic(1.0, 0.0, v);
        assert!((i.to_array()[0] - 65.0).abs() < 0.01, "depol: {}", i.to_array()[0]);
    }

    #[test]
    fn test_ohmic_hyperpolarizing() {
        // Cl⁻: E_rev=-70, V=-65 → I = g * (-70 - (-65)) = -5g (hyperpolarizing)
        let v = f32x8::splat(-65.0);
        let i = super::ohmic(1.0, crate::E_CL, v);
        assert!((i.to_array()[0] - (-5.0)).abs() < 0.01, "hyperpol: {}", i.to_array()[0]);
    }

    #[test]
    fn test_ohmic_zero_driving_force() {
        // At V = E_rev, I = 0
        let v = f32x8::splat(-70.0);
        let i = super::ohmic(1.0, -70.0, v);
        assert!(i.to_array()[0].abs() < 1e-6, "zero: {}", i.to_array()[0]);
    }

    #[test]
    fn test_ohmic_scalar_matches_simd() {
        let v = -65.0_f32;
        let i_s = super::ohmic_s(1.0, 0.0, v);
        let i_simd = super::ohmic(1.0, 0.0, f32x8::splat(v)).to_array()[0];
        assert!((i_s - i_simd).abs() < 0.001, "scalar vs SIMD");
    }
}
