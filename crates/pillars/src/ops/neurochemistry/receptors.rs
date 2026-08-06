//! # Receptors — Neurotransmitter Receptor Currents
//!
//! ## NMDA
//! ```text
//! I_NMDA = g_max · [glu] · Mg_block(V) · (V - 0 mV)
//! Mg_block(V) = 1 / (1 + [Mg²⁺] · exp(-0.062 · V))
//! ```
//! Voltage-dependent Mg²⁺ block — molecular coincidence detector.
//!
//! ## GABA-A
//! ```text
//! I_GABA = g_max · [GABA] · (1 + bzd) · (1 + barb) · (V - E_Cl)
//! ```
//! Fast inhibitory, allosteric modulation by benzodiazepines/barbiturates.
//!
//! ## nAChR
//! ```text
//! I_nACh = g_max · [ACh] · (1 + nico_mod) · (V - 0 mV)
//! ```
//! Fast excitatory cation channel (Na⁺/K⁺/Ca²⁺).

use crate::core::lane::SimdLane;

/// NMDA current with Mg²⁺ block: `I = g_max · glu · Mg_block(V) · (V - E_rev)`
#[inline(always)]
pub fn nmda_current<S: SimdLane>(v: S, glu: S, g_max: S, e_rev: S, mg_conc: S) -> S {
    let mg_block = S::one() / (S::one() + mg_conc * (S::splat(-0.062) * v).exp());
    g_max * glu * mg_block * (v - e_rev)
}

/// GABA-A current: `I = g_max · [GABA] · (1 + bzd) · (1 + barb) · (V - E_Cl)`
/// E_Cl = -70 mV (default).
#[inline(always)]
pub fn gaba_a_current<S: SimdLane>(v: S, gaba: S, g_max: S, bzd_mod: S, barb_mod: S, e_cl: S) -> S {
    let g_mod = g_max * gaba * (S::one() + bzd_mod) * (S::one() + barb_mod);
    g_mod * (v - e_cl)
}

/// nAChR current: `I = g_max · [ACh] · (1 + nico_mod) · (V - 0 mV)`
#[inline(always)]
pub fn nachr_current<S: SimdLane>(v: S, ach: S, g_max: S, nico_mod: S) -> S {
    let g_eff = g_max * ach * (S::one() + nico_mod);
    g_eff * v // E_rev ≈ 0 mV
}

/// AMPA receptor current (fast excitatory): `I = g_max · [glu] · (V - 0 mV)`
#[inline(always)]
pub fn ampa_current<S: SimdLane>(v: S, glu: S, g_max: S) -> S {
    g_max * glu * v // E_rev ≈ 0 mV
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_nmda_mg_block_resting() {
        let v = f32x8::splat(-65.0);
        let glu = f32x8::splat(1.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(0.0);
        let mg = f32x8::splat(1.2);
        let i = nmda_current(v, glu, g, e, mg);
        // At -65mV: Mg_block ≈ 1/(1+1.2*exp(4.03)) ≈ 1/(1+68) ≈ 0.014
        // I ≈ 1*1*0.014*(-65) ≈ -0.94 (small inward current due to Mg block)
        assert!(
            i.to_array()[0] < 0.0,
            "NMDA at -65 should be weak inward: {:.6}",
            i.to_array()[0]
        );
    }

    #[test]
    fn test_nmda_depolarized() {
        let v = f32x8::splat(-10.0);
        let glu = f32x8::splat(1.0);
        let g = f32x8::splat(1.0);
        let e = f32x8::splat(0.0);
        let mg = f32x8::splat(1.2);
        let i = nmda_current(v, glu, g, e, mg);
        // At -10mV: Mg_block ≈ 1/(1+1.2*exp(0.62)) ≈ 1/(1+2.24) ≈ 0.31
        // I ≈ 1*1*0.31*(-10) ≈ -3.09
        assert!(
            i.to_array()[0] < -3.0,
            "NMDA at -10: {:.6}",
            i.to_array()[0]
        );
    }

    #[test]
    fn test_gaba_inhibition() {
        let v = f32x8::splat(-60.0);
        let gaba = f32x8::splat(1.0);
        let g = f32x8::splat(0.5);
        let bzd = f32x8::splat(0.0);
        let barb = f32x8::splat(0.0);
        let e = f32x8::splat(-70.0);
        let i = gaba_a_current(v, gaba, g, bzd, barb, e);
        // I = 0.5 * 1 * 1 * 1 * (-60 + 70) = 5 (positive → hyperpolarizing)
        assert!((i.to_array()[0] - 5.0).abs() < 1e-6);
    }
}
