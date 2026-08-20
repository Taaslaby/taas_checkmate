use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd};

use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  Receptor types
// ═══════════════════════════════════════════════════════════

/// Receptor type with predefined kinetics.
///
/// Stored as `u8` per synapse for compact storage.
///
/// # Drosophila-appropriate receptor types
///
/// | Code | Receptor | Effect | τ_decay | E_rev | NT |
/// |------|----------|--------|---------|-------|-----|
/// | 0 | AMPA | Fast exc (mammalian fallback) | 2ms | 0mV | default exc |
/// | 1 | NMDA | Slow exc (Ca²⁺, Mg²⁺ block) | 50ms | 0mV | — |
/// | 2 | GABA_A | Shunting inh (insect) | 10ms | -65mV | GABA |
/// | 3 | GABA_B | Slow inh (G-protein) | 200ms | -90mV | — |
/// | 4 | nAChR | Fast exc (insect ACh) | 3ms | 0mV | ACh |
/// | 5 | GluCl | Inhibitory Cl⁻ (insect GLUT) | 5ms | -70mV | GLUT |
/// | 6 | mAChR | Slow mod (GPCR, DA/SER/OCT) | 150ms | -90mV | DA/SER/OCT |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum ReceptorType {
    /// AMPA-like: fast excitatory, Na⁺ influx, τ=2ms, E_rev=0mV.
    AMPA = 0,
    /// NMDA-like: slow excitatory, Ca²⁺ influx, τ=50ms, E_rev=0mV.
    /// Has voltage-dependent Mg²⁺ block (see [`nmda_mg_block`]).
    NMDA = 1,
    /// GABA_A-like: shunting inhibition (insect), Cl⁻, τ=10ms, E_rev=-65mV.
    /// At V_rest=-65 the driving force is 0 → pure conductance shunt.
    GABA_A = 2,
    /// GABA_B-like: slow inhibitory, G-protein coupled, τ=200ms, E_rev=-90mV.
    GABA_B = 3,
    /// nAChR: fast excitatory (insect ACh), Na⁺/K⁺ influx, τ=3ms, E_rev=0mV.
    /// Main excitatory receptor in Drosophila CNS.
    nAChR = 4,
    /// GluCl: inhibitory Cl⁻ channel (insect glutamate), τ=5ms, E_rev=-70mV.
    GluCl = 5,
    /// mAChR-like: slow modulatory (DA/SER/OCT), G-protein coupled,
    /// τ=150ms, E_rev=-90mV.
    mAChR = 6,
}

impl ReceptorType {
    /// Number of defined receptor types.
    pub const COUNT: usize = 7;

    /// Decay time constant in milliseconds.
    #[inline(always)]
    pub fn tau_decay(self) -> f32 {
        match self {
            ReceptorType::AMPA => 2.0,
            ReceptorType::NMDA => 50.0,
            ReceptorType::GABA_A => 10.0,
            ReceptorType::GABA_B => 200.0,
            ReceptorType::nAChR => 3.0,
            ReceptorType::GluCl => 5.0,
            ReceptorType::mAChR => 150.0,
        }
    }

    /// Reversal potential in mV.
    #[inline(always)]
    pub fn e_rev(self) -> f32 {
        match self {
            ReceptorType::AMPA => 0.0,
            ReceptorType::NMDA => 0.0,
            ReceptorType::GABA_A => -65.0,
            ReceptorType::GABA_B => -90.0,
            ReceptorType::nAChR => 0.0,
            ReceptorType::GluCl => -70.0,
            ReceptorType::mAChR => -90.0,
        }
    }

    /// Convert from a stored `u8` discriminant; unknown values fall back to AMPA.
    #[inline(always)]
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => ReceptorType::AMPA,
            1 => ReceptorType::NMDA,
            2 => ReceptorType::GABA_A,
            3 => ReceptorType::GABA_B,
            4 => ReceptorType::nAChR,
            5 => ReceptorType::GluCl,
            6 => ReceptorType::mAChR,
            _ => ReceptorType::AMPA, // safe fallback
        }
    }

    /// Map an nt_type sign to a default receptor: `+1` → AMPA, `-1` → GABA_A.
    #[inline(always)]
    pub fn from_nt_type(nt: f32) -> Self {
        if nt > 0.0 {
            ReceptorType::AMPA
        } else {
            ReceptorType::GABA_A
        }
    }

    /// True if this receptor produces inward (depolarizing) current.
    #[inline(always)]
    pub fn is_excitatory(self) -> bool {
        matches!(
            self,
            ReceptorType::AMPA | ReceptorType::NMDA | ReceptorType::nAChR
        )
    }

    /// All receptor variants for iteration.
    pub fn all() -> [ReceptorType; 7] {
        [
            ReceptorType::AMPA,
            ReceptorType::NMDA,
            ReceptorType::GABA_A,
            ReceptorType::GABA_B,
            ReceptorType::nAChR,
            ReceptorType::GluCl,
            ReceptorType::mAChR,
        ]
    }
}

// ═══════════════════════════════════════════════════════════
//  NMDA voltage-dependent Mg²⁺ block
// ═══════════════════════════════════════════════════════════

/// Mg²⁺ block factor for the NMDA receptor: `B(V) = 1 / (1 + 0.33·exp(-0.06·V))`.
///
/// At resting (-65 mV): B ≈ 0.02 (blocked). At spike peak (+30 mV): B ≈ 0.95.
#[inline(always)]
pub fn nmda_mg_block(v: f32) -> f32 {
    1.0 / (1.0 + 0.33 * fast_exp((-0.06 * v).clamp(-10.0, 10.0)))
}

/// SIMD version of the Mg²⁺ block — bit-consistent with [`nmda_mg_block`]
/// via the shared [`fast_exp_poly_simd`] polynomial.
#[inline(always)]
pub(crate) fn nmda_mg_block_simd(v: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let factor = f32x8::splat(0.33);
    let arg = (f32x8::splat(-0.06) * v)
        .max(f32x8::splat(-10.0))
        .min(f32x8::splat(10.0));
    one / (one + factor * fast_exp_poly_simd(arg))
}
