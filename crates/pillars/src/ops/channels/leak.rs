//! # Leak — Passive Leak Current
//!
//! ```text
//! I_leak = g_leak · (V - E_leak)
//! ```
//!
//! Simplest ion channel — no gating. Sets the resting membrane
//! potential and input resistance. Present in every neuron.

use crate::core::lane::SimdLane;

/// Leak current: `I = g · (V - E_rev)` — pure ohmic.
#[inline(always)]
pub fn leak_current<S: SimdLane>(v: S, g_leak: S, e_leak: S) -> S {
    g_leak * (v - e_leak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_leak_resting() {
        let v = f32x8::splat(-65.0);
        let g = f32x8::splat(0.3);
        let e = f32x8::splat(-65.0);
        let i = leak_current(v, g, e);
        assert!((i.to_array()[0]).abs() < 1e-6, "leak at E_leak should be 0");
    }

    #[test]
    fn test_leak_depolarized() {
        let v = f32x8::splat(-50.0);
        let g = f32x8::splat(0.3);
        let e = f32x8::splat(-65.0);
        let i = leak_current(v, g, e);
        // I = 0.3 * (-50 + 65) = 4.5
        assert!((i.to_array()[0] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_leak_scalar() {
        // Scalar (f32) path — should match exact math
        let i = leak_current(-50.0f32, 0.3f32, -65.0f32);
        assert!((i - 4.5).abs() < 1e-6, "scalar leak: {i}");
    }

    #[test]
    fn test_leak_extreme_values() {
        // Extreme depolarization
        let v = f32x8::splat(100.0);
        let g = f32x8::splat(10.0);
        let e = f32x8::splat(-90.0);
        let i = leak_current(v, g, e);
        // I = 10.0 * (100 - (-90)) = 1900
        assert!((i.to_array()[0] - 1900.0).abs() < 1e-3, "extreme depolarized: {}", i.to_array()[0]);

        // Extreme hyperpolarization
        let v2 = f32x8::splat(-120.0);
        let i2 = leak_current(v2, g, e);
        // I = 10.0 * (-120 - (-90)) = -300
        assert!((i2.to_array()[0] - (-300.0)).abs() < 1e-3, "extreme hyperpolarized: {}", i2.to_array()[0]);
    }

    #[test]
    fn test_leak_all_neurons_consistency() {
        let g = f32x8::splat(0.5);
        let e = f32x8::splat(-70.0);
        let v_vals = [-80.0, -70.0, -60.0, -50.0, -40.0, -30.0, 0.0, 20.0];
        let v = f32x8::new(v_vals);
        let currents = leak_current(v, g, e).to_array();
        for i in 0..8 {
            let expected = 0.5 * (v_vals[i] + 70.0);
            assert!((currents[i] - expected).abs() < 1e-5,
                "leak[{i}]: v={}, expected={}, got={}", v_vals[i], expected, currents[i]);
        }
    }

    #[test]
    fn test_leak_zero_conductance() {
        let g = f32x8::splat(0.0);
        let e = f32x8::splat(-65.0);
        let v = f32x8::splat(100.0);
        let i = leak_current(v, g, e);
        assert!((i.to_array()[0]).abs() < 1e-6, "zero conductance should give zero current: {}", i.to_array()[0]);
    }
}
