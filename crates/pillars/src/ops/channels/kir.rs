//! # Kir — Inward Rectifier K⁺ Current
//!
//! ```text
//! I_Kir = g_Kir · (V - E_K) · block(V)
//! block(V) = 1 / (1 + exp((V - E_K + V_offset) / k))
//! ```
//!
//! Inward rectifiers pass K⁺ more easily inward than outward.
//! At rest (V ≈ E_K), Kir sets the resting membrane potential.
//! During depolarization, Kir channels close.
//!
//! Found in glial cells (K⁺ buffering), cardiac myocytes (I_K1),
//! and Drosophila muscle/glia.

use crate::core::lane::SimdLane;

/// Compute Kir current: `I = g · (V - E_K) · Boltzmann_block`
#[inline(always)]
pub fn kir_current<S: SimdLane>(v: S, g_max: S, e_rev: S, v_offset: S, k_slope: S) -> S {
    let block = S::one() / (S::one() + ((v - e_rev + v_offset) / k_slope).exp());
    (g_max * (v - e_rev)) * block
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_kir_resting() {
        // At V=E_K, block partially active
        let v = f32x8::splat(-80.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0);
        let off = f32x8::splat(10.0);
        let k = f32x8::splat(5.0);
        let i = kir_current(v, g, e, off, k);
        // V=E_K → V-E_K=0 → I=0
        assert!((i.to_array()[0]).abs() < 1e-6, "Kir at E_K should be 0: {:.6}", i.to_array()[0]);
    }

    #[test]
    fn test_kir_rectification() {
        let v = f32x8::splat(-100.0); // hyperpolarized → OPEN
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0);
        let off = f32x8::splat(10.0);
        let k = f32x8::splat(5.0);
        let i = kir_current(v, g, e, off, k);
        assert!(i.to_array()[0] < 0.0, "Kir at hyperpolarized should be inward (negative): {:.6}", i.to_array()[0]);
    }

    #[test]
    fn test_kir_scalar() {
        // Scalar f32 path — Kir at E_K
        let i = kir_current(-80.0f32, 1.0f32, -80.0f32, 10.0f32, 5.0f32);
        assert!((i).abs() < 1e-6, "Kir at E_K scalar should be 0: {:.6}", i);
        // Scalar hyperpolarized
        let i2 = kir_current(-100.0f32, 1.0f32, -80.0f32, 10.0f32, 5.0f32);
        assert!(i2 < 0.0, "Kir scalar hyperpolarized should be negative: {:.6}", i2);
    }

    #[test]
    fn test_kir_depolarized_block() {
        // At depolarized V, Kir channel should BLOCK (block ≈ 0)
        let v = f32x8::splat(-40.0); // depolarized
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0);
        let off = f32x8::splat(10.0);
        let k = f32x8::splat(5.0);
        let i = kir_current(v, g, e, off, k);
        // V-E_K = -40+80=40, so raw current would be 40, but block should suppress
        // block ≈ 1/(1+exp(40+10)/5) = 1/(1+exp(10)) ≈ 0
        assert!(i.to_array()[0].abs() < 0.01, "Kir at depolarized should block, got: {:.8}", i.to_array()[0]);
    }

    #[test]
    fn test_kir_block_sigmoid() {
        // Test the block function directly: at V-E_K+offset = 0 → block=0.5
        // V-E_K = -V_offset → block = 1/(1+exp(0/5)) = 0.5
        let v = f32x8::splat(-90.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0); // V-E_K = -10
        let off = f32x8::splat(10.0); // V-E_K+off = 0 → block = 0.5
        let k = f32x8::splat(5.0);
        let i = kir_current(v, g, e, off, k);
        // raw = 1*(-10-(-80)) = 1*(-10) = -10... wait
        // V-E_K = -90+80 = -10
        // raw current = 1.0 * (-10) = -10
        // block = 1/(1+exp((-10+10)/5)) = 1/(1+exp(0)) = 0.5
        // final = -10 * 0.5 = -5
        assert!((i.to_array()[0] - (-5.0)).abs() < 1e-4,
            "Kir at V-E_K=-offset should be half-blocked: {:.4}", i.to_array()[0]);
    }

    #[test]
    fn test_kir_zero_conductance() {
        let v = f32x8::splat(-100.0);
        let g = f32x8::splat(0.0);
        let e = f32x8::splat(-80.0);
        let off = f32x8::splat(10.0);
        let k = f32x8::splat(5.0);
        let i = kir_current(v, g, e, off, k);
        assert!((i.to_array()[0]).abs() < 1e-6, "zero conductance should give zero: {:.8}", i.to_array()[0]);
    }

    #[test]
    fn test_kir_all_neurons() {
        let v_vals = [-100.0, -90.0, -85.0, -80.0, -70.0, -60.0, -50.0, -40.0];
        let v = f32x8::new(v_vals);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(-80.0);
        let off = f32x8::splat(10.0);
        let k = f32x8::splat(5.0);
        let currents = kir_current(v, g, e, off, k).to_array();

        // At V=-100: V-E_K = -20, block=1/(1+exp(-10/5)) = 1/(1+exp(-2)) ≈ 0.88
        // I = -20 * 0.88 ≈ -17.6
        assert!(currents[0] < 0.0, "V=-100: inward current: {:.4}", currents[0]);
        // At V=-80 (E_K): should be 0
        assert!((currents[3]).abs() < 1e-6, "V=-80 (E_K): should be 0: {:.4}", currents[3]);
        // At V=-40: blocked
        assert!(currents[7].abs() < 0.01, "V=-40: should be blocked: {:.6}", currents[7]);
    }
}
