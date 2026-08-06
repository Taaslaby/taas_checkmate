//! # Signaling — Intracellular Signaling Pathways
//!
//! ## NO/cGMP
//! ```text
//! d[NO]/dt = k_nNOS · Ca - k_deg · [NO]
//! d[cGMP]/dt = k_GC · [NO] - k_PDE · [cGMP]
//! mod = 1 + gain · cGMP / (cGMP + Kd)
//! ```
//! Retrograde NO signaling — modulates presynaptic release.
//!
//! ## CaMKII
//! ```text
//! d[CaMKII*]/dt = k_on · Ca · (1 - [CaMKII*]) - k_off · [CaMKII*]
//! ```
//! Ca²⁺/calmodulin-dependent kinase — molecular memory for LTP/LTD.
//!
//! ## Endocannabinoid (2-AG)
//! ```text
//! d[2-AG]/dt = k_prod · Ca² + k_mGluR · Gq - k_deg · [2-AG]
//! release_mod = 1 - gain · [2-AG] / ([2-AG] + Kd)
//! ```
//! Retrograde endocannabinoid signaling — short-term depression.

use crate::core::lane::SimdLane;

/// NO/cGMP signaling: returns (NO', cGMP', mod_factor).
#[inline(always)]
pub fn no_cgmp_step<S: SimdLane>(
    no: S,
    cgmp: S,
    ca: S,
    k_nnos: S,
    k_deg_no: S,
    k_gc: S,
    k_pde: S,
    gain: S,
    kd: S,
    dt: S,
) -> (S, S, S) {
    let new_no = (no + (k_nnos * ca - k_deg_no * no) * dt).max(S::zero());
    let new_cgmp = (cgmp + (k_gc * new_no - k_pde * cgmp) * dt).max(S::zero());
    let mod_factor = S::one() + gain * new_cgmp / (new_cgmp + kd);
    (new_no, new_cgmp, mod_factor)
}

/// CaMKII autophosphorylation: `CaMKII' = CaMKII + (k_on·Ca·(1-CaMKII) - k_off·CaMKII)·dt`
#[inline(always)]
pub fn camkii_step<S: SimdLane>(camkii: S, ca: S, k_on: S, k_off: S, dt: S) -> S {
    let one = S::one();
    let new = camkii + (k_on * ca * (one - camkii) - k_off * camkii) * dt;
    new.max(S::zero()).min(one)
}

/// Endocannabinoid (2-AG) signaling: returns (ecb', mod_factor).
#[inline(always)]
pub fn endocb_step<S: SimdLane>(
    ecb: S,
    ca: S,
    gq: S,
    k_prod: S,
    k_mglur: S,
    k_deg: S,
    gain: S,
    kd: S,
    dt: S,
) -> (S, S) {
    let new_ecb = (ecb + (k_prod * ca * ca + k_mglur * gq - k_deg * ecb) * dt).max(S::zero());
    let mod_val = S::one() - gain * new_ecb / (new_ecb + kd);
    (new_ecb, mod_val.max(S::one() - gain).min(S::one()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_no_cgmp_no_ca() {
        let no = f32x8::splat(0.0);
        let cgmp = f32x8::splat(0.0);
        let ca = f32x8::splat(0.0);
        let kn = f32x8::splat(0.1);
        let kd = f32x8::splat(0.1);
        let kgc = f32x8::splat(0.05);
        let kpde = f32x8::splat(0.05);
        let gain = f32x8::splat(0.5);
        let kd_val = f32x8::splat(0.5);
        let dt = f32x8::splat(1.0);
        let (new_no, _new_cgmp, mod_f) =
            no_cgmp_step(no, cgmp, ca, kn, kd, kgc, kpde, gain, kd_val, dt);
        assert!(new_no.to_array()[0].abs() < 1e-6, "no Ca → no NO");
        assert!((mod_f.to_array()[0] - 1.0).abs() < 1e-6, "no modulation");
    }
}
