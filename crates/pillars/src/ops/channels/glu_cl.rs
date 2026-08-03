//! # GluCl — Glutamate-Gated Chloride Channel
//!
//! ```text
//! I_GluCl = g_max · [glu] · (V - E_Cl)
//! ```
//!
//! In Drosophila, glutamate is INHIBITORY via GluCl channels (Cl⁻ conductance).
//! This is OPPOSITE to mammals where glutamate is excitatory (AMPA/NMDA).
//!
//! E_Cl = -70 mV (typical). At V > -70: outward current = hyperpolarization = inhibition.
//! Target of ivermectin (agonist), fipronil (blocker).

use crate::core::lane::SimdLane;

/// GluCl current: `I = g_max · glu · (V - E_Cl)`
#[inline(always)]
pub fn glucl_current<S: SimdLane>(v: S, glu: S, g_max: S, e_cl: S) -> S {
    g_max * glu * (v - e_cl)
}

/// GluCl current with scalar params (convenience).
#[inline(always)]
pub fn glucl_current_scalar<S: SimdLane>(v: S, glu: S, g_max: f32, e_cl: f32) -> S {
    S::splat(g_max) * glu * (v - S::splat(e_cl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_glucl_inhibition() {
        let v = f32x8::splat(-60.0); // above E_Cl
        let glu = f32x8::splat(1.0);
        let g = f32x8::splat(0.5);
        let e = f32x8::splat(-70.0);
        let i = glucl_current(v, glu, g, e);
        // I = 0.5 * 1 * (-60 - (-70)) = 5 (positive → hyperpolarizing → inhibition)
        assert!((i.to_array()[0] - 5.0).abs() < 1e-6);
    }
}
