//! # ⚡ Synapse Microkernels — Conductance-Based Kinetics (7 Receptors)
//!
//! Port of the legacy `crates/kernels/src/neuron/synapse.rs` — replaces
//! instant current injection with realistic conductance dynamics.
//!
//! ## Kinetics
//!
//! ```text
//!   g(t+dt) = g(t) · exp(-dt/τ)         ← exponential decay
//!   g(t)   += w · spike(t)              ← pre-spike jump
//!   I_syn   = Σ g_r · (E_rev_r − V)     ← ohmic current (NEURON convention)
//!             └─ NMDA: g · B(V)         ← Mg²⁺ voltage block
//! ```
//!
//! ## Drosophila Receptor Types
//!
//! | Code | Receptor | τ_decay | E_rev | Ion | Effect | NT |
//! |------|----------|---------|-------|-----|--------|-----|
//! | 0 | `AMPA` | 2 ms | 0 mV | Na⁺ | Fast excitatory (mammalian fallback) | Glu |
//! | 1 | `NMDA` | 50 ms | 0 mV | Ca²⁺ | Slow excitatory, Mg²⁺ voltage block | Glu |
//! | 2 | `GABA_A` | 10 ms | -65 mV | Cl⁻ | Shunting inhibition (insect) | GABA |
//! | 3 | `GABA_B` | 200 ms | -90 mV | G-protein | Slow inhibition (GPCR) | GABA |
//! | 4 | `nAChR` | 3 ms | 0 mV | Na⁺/K⁺ | Fast excitatory (main insect exc) | ACh |
//! | 5 | `GluCl` | 5 ms | -70 mV | Cl⁻ | Inhibitory Cl⁻ channel | Glutamate |
//! | 6 | `mAChR` | 150 ms | -90 mV | GPCR | Slow modulatory (DA/SER/OCT) | DA/5-HT/OA |
//!
//! ## Lifecycle (per simulation step)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────┐
//!   │  1. decay_all(factors)    g *= exp(-dt/τ)           │
//!   │  2. jump(neuron, type, w) g += w · pre_spike        │
//!   │  3. compute_currents(V)   I = Σ g·(E_rev − V)       │
//!   └──────────────────────────────────────────────────────┘
//!                                     │
//!                                     ∨
//!                          i_in[] → NeuralPopulation::step()
//! ```
//!
//! ## NMDA Mg²⁺ Block
//!
//! `B(V) = 1 / (1 + 0.33·exp(-0.06·V))`
//!
//! At rest (-65 mV): B ≈ 0.02 — channel is blocked.
//! At spike peak (+30 mV): B ≈ 0.95 — channel unblocks.
//!
//! ## Usage
//!
//! ```
//! use pillars::kernels::synapse::{SynapticKinetics, DecayFactors, ReceptorType};
//!
//! // 1. Allocate conductance buffers for 8 neurons
//! let mut kin = SynapticKinetics::new(8);
//!
//! // 2. Precompute decay factors (rebuild only when dt changes)
//! let factors = DecayFactors::new(0.1);  // dt = 0.1 ms
//!
//! // per-step loop:
//! //   3. Decay all conductances: g *= exp(-dt/τ)
//! //   4. Jump on pre-spike: g_post += w * spike
//! //   5. Compute synaptic current contribution
//! kin.decay_all(&factors);                   // SIMD: 7× f32x8 decay
//! kin.jump(0, ReceptorType::nAChR, 0.5);     // pre-spike → ACh conductance ↑
//! kin.jump(1, ReceptorType::GABA_A, 0.3);    // pre-spike → GABA conductance ↑
//!
//! let mut i_syn = vec![0.0f32; 8];
//! let v_memb = vec![-65.0f32; 8];
//! kin.compute_currents(&v_memb, &mut i_syn); // I_out += Σ g·(E_rev − V)
//! // i_syn[0] = 0.5 * (0 - (-65)) = 32.5 pA (depol via nAChR)
//! // i_syn[1] = 0.3 * (-65 - (-65)) = 0 pA (GABA_A shunt at rest)
//! ```
//!
//! ## References
//!
//! - Dayan & Abbott (2001) *Theoretical Neuroscience*, MIT Press.
//! - Wilson et al. (2018) "Drosophila olfactory bulb — fast ACh excitation".
//! - Liu & Wilson (2013) "GABA_A shunt inhibition in insect CNS".

use wide::f32x8;

use crate::core::util::{load8, store8};
use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd};

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
        matches!(self, ReceptorType::AMPA | ReceptorType::NMDA | ReceptorType::nAChR)
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
fn nmda_mg_block_simd(v: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let factor = f32x8::splat(0.33);
    let arg = (f32x8::splat(-0.06) * v).max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + factor * fast_exp_poly_simd(arg))
}

// ═══════════════════════════════════════════════════════════
//  Precomputed decay factors
// ═══════════════════════════════════════════════════════════

/// Precomputed decay factors `exp(-dt / τ)` for each receptor type.
///
/// Rebuilt only when `dt` changes; reused across steps.
#[derive(Debug, Clone, Copy)]
pub struct DecayFactors {
    pub ampa: f32,
    pub nmda: f32,
    pub gaba_a: f32,
    pub gaba_b: f32,
    pub nachr: f32,
    pub glucl: f32,
    pub machr: f32,
}

impl DecayFactors {
    /// Compute decay factors for a given `dt` (in ms).
    #[inline(always)]
    pub fn new(dt: f32) -> Self {
        Self {
            ampa: fast_exp(-dt / ReceptorType::AMPA.tau_decay()),
            nmda: fast_exp(-dt / ReceptorType::NMDA.tau_decay()),
            gaba_a: fast_exp(-dt / ReceptorType::GABA_A.tau_decay()),
            gaba_b: fast_exp(-dt / ReceptorType::GABA_B.tau_decay()),
            nachr: fast_exp(-dt / ReceptorType::nAChR.tau_decay()),
            glucl: fast_exp(-dt / ReceptorType::GluCl.tau_decay()),
            machr: fast_exp(-dt / ReceptorType::mAChR.tau_decay()),
        }
    }

    /// Decay factor for one receptor type.
    #[inline(always)]
    pub fn get(self, r: ReceptorType) -> f32 {
        match r {
            ReceptorType::AMPA => self.ampa,
            ReceptorType::NMDA => self.nmda,
            ReceptorType::GABA_A => self.gaba_a,
            ReceptorType::GABA_B => self.gaba_b,
            ReceptorType::nAChR => self.nachr,
            ReceptorType::GluCl => self.glucl,
            ReceptorType::mAChR => self.machr,
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Synaptic kinetics — conductance buffers
// ═══════════════════════════════════════════════════════════

/// Per-neuron conductance buffers for all 7 receptor types.
///
/// Lifecycle per simulation step:
/// 1. [`decay_all`](Self::decay_all) — exponential decay: `g *= exp(-dt/τ)`
/// 2. [`jump`](Self::jump) — pre-spike adds conductance
/// 3. [`compute_currents`](Self::compute_currents) — `I = Σ g·(E_rev − V)`
///
/// Buffers are padded to a multiple of 8 for SIMD access.
#[derive(Debug, Clone)]
pub struct SynapticKinetics {
    /// Actual number of neurons (unpadded).
    pub n: usize,
    /// Padded to %8.
    pub n_padded: usize,
    pub g_ampa: Vec<f32>,
    pub g_nmda: Vec<f32>,
    pub g_gaba_a: Vec<f32>,
    pub g_gaba_b: Vec<f32>,
    pub g_nachr: Vec<f32>,
    pub g_glucl: Vec<f32>,
    pub g_machr: Vec<f32>,
}

impl SynapticKinetics {
    /// Allocate conductance buffers for `n_neurons` (internally padded to %8).
    pub fn new(n_neurons: usize) -> Self {
        let n_padded = n_neurons.div_ceil(8) * 8;
        Self {
            n: n_neurons,
            n_padded,
            g_ampa: vec![0.0; n_padded],
            g_nmda: vec![0.0; n_padded],
            g_gaba_a: vec![0.0; n_padded],
            g_gaba_b: vec![0.0; n_padded],
            g_nachr: vec![0.0; n_padded],
            g_glucl: vec![0.0; n_padded],
            g_machr: vec![0.0; n_padded],
        }
    }

    /// Reset all conductances to zero.
    pub fn reset(&mut self) {
        self.g_ampa.fill(0.0);
        self.g_nmda.fill(0.0);
        self.g_gaba_a.fill(0.0);
        self.g_gaba_b.fill(0.0);
        self.g_nachr.fill(0.0);
        self.g_glucl.fill(0.0);
        self.g_machr.fill(0.0);
    }

    /// Apply exponential decay to all conductances: `g *= exp(-dt / τ)`.
    ///
    /// SIMD over 8-neuron lanes; padding elements decay too (they stay 0).
    pub fn decay_all(&mut self, factors: &DecayFactors) {
        let d = [
            f32x8::splat(factors.ampa),
            f32x8::splat(factors.nmda),
            f32x8::splat(factors.gaba_a),
            f32x8::splat(factors.gaba_b),
            f32x8::splat(factors.nachr),
            f32x8::splat(factors.glucl),
            f32x8::splat(factors.machr),
        ];
        let n8 = self.n_padded & !7;
        let bufs: [*mut f32; 7] = [
            self.g_ampa.as_mut_ptr(),
            self.g_nmda.as_mut_ptr(),
            self.g_gaba_a.as_mut_ptr(),
            self.g_gaba_b.as_mut_ptr(),
            self.g_nachr.as_mut_ptr(),
            self.g_glucl.as_mut_ptr(),
            self.g_machr.as_mut_ptr(),
        ];
        for off in (0..n8).step_by(8) {
            for (b, &f) in bufs.iter().zip(d.iter()) {
                unsafe {
                    store8(*b, off, load8(*b, off) * f);
                }
            }
        }
        for i in n8..self.n_padded {
            self.g_ampa[i] *= factors.ampa;
            self.g_nmda[i] *= factors.nmda;
            self.g_gaba_a[i] *= factors.gaba_a;
            self.g_gaba_b[i] *= factors.gaba_b;
            self.g_nachr[i] *= factors.nachr;
            self.g_glucl[i] *= factors.glucl;
            self.g_machr[i] *= factors.machr;
        }
    }

    /// Compute synaptic current: `I_out[i] += Σ g_r[i]·(E_rev_r − V[i])`.
    ///
    /// Uses the **NEURON/`ops::ohmic` convention** — the current is *added* to
    /// `i_out` and subtracted from the membrane equation downstream, so a
    /// positive term here depolarizes. The NMDA Mg²⁺ block modulates `g_nmda`
    /// in-place. SIMD lanes + scalar tail are bit-consistent; both paths read
    /// `E_rev` from [`ReceptorType::e_rev`] (single source of truth).
    pub fn compute_currents(&self, v: &[f32], i_out: &mut [f32]) {
        let n = self.n.min(v.len()).min(i_out.len());
        let n8 = n & !7;
        let block = nmda_mg_block_simd;
        let er: [_; 7] = ReceptorType::all().map(|r| f32x8::splat(r.e_rev()));
        let gbufs: [*const f32; 7] = [
            self.g_ampa.as_ptr(),
            self.g_nmda.as_ptr(),
            self.g_gaba_a.as_ptr(),
            self.g_gaba_b.as_ptr(),
            self.g_nachr.as_ptr(),
            self.g_glucl.as_ptr(),
            self.g_machr.as_ptr(),
        ];
        for off in (0..n8).step_by(8) {
            unsafe {
                let vi = load8(v.as_ptr(), off);
                let mut cur = load8(i_out.as_ptr(), off);
                for (r, (b, &e_r)) in ReceptorType::all().into_iter().zip(gbufs.iter().zip(er.iter())) {
                    let g = load8(*b, off);
                    // Mg²⁺ block applies only to the NMDA conductance.
                    let g_eff = if r == ReceptorType::NMDA { g * block(vi) } else { g };
                    cur += g_eff * (e_r - vi);
                }
                store8(i_out.as_mut_ptr(), off, cur);
            }
        }
        for i in n8..n {
            let vi = v[i];
            let b = nmda_mg_block(vi);
            let mut acc = 0.0f32;
            for r in ReceptorType::all() {
                let g = self.g_of(r, i);
                let g_eff = if r == ReceptorType::NMDA { g * b } else { g };
                acc += g_eff * (r.e_rev() - vi);
            }
            i_out[i] += acc;
        }
    }

    /// Borrow one conductance buffer by receptor type.
    #[inline(always)]
    fn g_of(&self, r: ReceptorType, i: usize) -> f32 {
        match r {
            ReceptorType::AMPA => self.g_ampa[i],
            ReceptorType::NMDA => self.g_nmda[i],
            ReceptorType::GABA_A => self.g_gaba_a[i],
            ReceptorType::GABA_B => self.g_gaba_b[i],
            ReceptorType::nAChR => self.g_nachr[i],
            ReceptorType::GluCl => self.g_glucl[i],
            ReceptorType::mAChR => self.g_machr[i],
        }
    }

    /// Jump a post-synaptic conductance by `amount` (triggered by a pre-spike).
    #[inline(always)]
    pub fn jump(&mut self, neuron: usize, receptor: ReceptorType, amount: f32) {
        if neuron >= self.n {
            return;
        }
        match receptor {
            ReceptorType::AMPA => self.g_ampa[neuron] += amount,
            ReceptorType::NMDA => self.g_nmda[neuron] += amount,
            ReceptorType::GABA_A => self.g_gaba_a[neuron] += amount,
            ReceptorType::GABA_B => self.g_gaba_b[neuron] += amount,
            ReceptorType::nAChR => self.g_nachr[neuron] += amount,
            ReceptorType::GluCl => self.g_glucl[neuron] += amount,
            ReceptorType::mAChR => self.g_machr[neuron] += amount,
        }
    }

    /// Total conductance for a neuron across all receptor types.
    pub fn total_conductance(&self, i: usize) -> f32 {
        if i >= self.n {
            return 0.0;
        }
        ReceptorType::all().into_iter().map(|r| self.g_of(r, i)).sum()
    }
}

// ═══════════════════════════════════════════════════════════
//  Build receptor array from nt_types (CSR)
// ═══════════════════════════════════════════════════════════

/// Build a per-synapse `receptor_types` array from per-neuron `nt_types` using
/// the CSR structure. Each synapse gets the receptor type of its pre-neuron's NT.
///
/// `nt > 0` → AMPA (exc), `nt ≤ 0` → GABA_A (inh).
pub fn build_receptor_types(row_ptr: &[i32], n_synapses: usize, nt_types: &[f32]) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = row_ptr.len().saturating_sub(1);
    let mut syn_idx = 0usize;
    for pre in 0..n {
        let nt = nt_types.get(pre).copied().unwrap_or(1.0);
        let rec = if nt > 0.0 { 0u8 } else { 2u8 }; // AMPA or GABA_A
        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = row_ptr.get(pre + 1).copied().unwrap_or(0).max(0) as usize;
        let end = end.min(n_synapses);
        for _ in start..end {
            if syn_idx < receptors.len() {
                receptors[syn_idx] = rec;
            }
            syn_idx += 1;
        }
    }
    receptors
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_factors_precision() {
        let factors = DecayFactors::new(0.5); // dt = 0.5 ms
        assert!((factors.ampa - 0.7788).abs() < 0.001, "AMPA decay: {}", factors.ampa);
    }

    #[test]
    fn test_decay_all_exponential() {
        let mut kin = SynapticKinetics::new(4);
        kin.g_ampa[0] = 1.0;
        kin.g_gaba_a[1] = 2.0;
        let factors = DecayFactors::new(2.0);
        kin.decay_all(&factors);
        assert!((kin.g_ampa[0] - 0.3679).abs() < 0.001, "AMPA: {}", kin.g_ampa[0]);
        assert!((kin.g_gaba_a[1] - 1.6375).abs() < 0.001, "GABA_A: {}", kin.g_gaba_a[1]);
    }

    #[test]
    fn test_jump_basic() {
        let mut kin = SynapticKinetics::new(4);
        kin.jump(0, ReceptorType::AMPA, 0.5);
        kin.jump(1, ReceptorType::GABA_A, 1.0);
        kin.jump(0, ReceptorType::NMDA, 0.3);
        assert_eq!(kin.g_ampa[0], 0.5);
        assert_eq!(kin.g_gaba_a[1], 1.0);
        assert_eq!(kin.g_nmda[0], 0.3);
        assert_eq!(kin.total_conductance(0), 0.8);
    }

    #[test]
    fn test_compute_currents_ohmic_direction() {
        // AMPA (E=0) at V=-65 → +6.5 (depol); GABA_A at rest → 0 (shunt)
        let mut kin = SynapticKinetics::new(2);
        let v = vec![-65.0, -65.0];
        let mut i_out = vec![0.0, 0.0];
        kin.g_ampa[0] = 0.1;
        kin.g_gaba_a[1] = 0.2;
        kin.compute_currents(&v, &mut i_out);
        assert!((i_out[0] - 6.5).abs() < 0.01, "AMPA I_syn: {}", i_out[0]);
        assert!(i_out[1].abs() < 0.01, "GABA_A shunt: {}", i_out[1]);
    }

    #[test]
    fn test_nmda_mg_block_voltage_profile() {
        let v_test = [-80.0, -65.0, -40.0, -20.0, 0.0, 20.0, 40.0];
        let mut kin = SynapticKinetics::new(7);
        for (i, _) in v_test.iter().enumerate() {
            kin.g_nmda[i] = 1.0;
        }
        let mut i_out = vec![0.0; 7];
        kin.compute_currents(&v_test, &mut i_out);
        assert!(i_out[0] < 5.0, "NMDA at -80 blocked: {}", i_out[0]);
        assert!(i_out[4].abs() < 1.0, "NMDA at 0: {}", i_out[4]);
        assert!(i_out[5] < -5.0, "NMDA at +20: {}", i_out[5]);
    }

    #[test]
    fn test_gabab_and_machr_inhibitory() {
        let mut kin = SynapticKinetics::new(2);
        let v = vec![-65.0, -65.0];
        let mut i_out = vec![0.0, 0.0];
        kin.g_gaba_b[0] = 1.0;
        kin.g_machr[1] = 1.0;
        kin.compute_currents(&v, &mut i_out);
        assert!((i_out[0] - (-25.0)).abs() < 0.1, "GABA_B: {}", i_out[0]);
        assert!((i_out[1] - (-25.0)).abs() < 0.1, "mAChR: {}", i_out[1]);
    }

    #[test]
    fn test_simd_tail_consistent() {
        // n=13 → 8 SIMD lanes + 5 scalar tail; identical state ⇒ identical current.
        // g_nmda is nonzero for ALL lanes, so the SIMD Mg²⁺-block path is
        // exercised against the scalar reference at both lane 4 and tail 12.
        let n = 13;
        let mut kin = SynapticKinetics::new(n);
        for i in 0..n {
            kin.g_ampa[i] = 0.1 * (i as f32 + 1.0);
            kin.g_nmda[i] = 0.05 * (i as f32 + 1.0);
            kin.g_gaba_a[i] = 0.02;
            kin.g_nachr[i] = 0.01;
        }
        let v: Vec<f32> = (0..n).map(|i| -70.0 + i as f32).collect();
        let mut i_out = vec![0.0f32; n];
        kin.compute_currents(&v, &mut i_out);
        for (idx, tol) in [(0usize, 1e-4), (4, 1e-4), (12, 1e-4)] {
            let vi = v[idx];
            let b = nmda_mg_block(vi);
            let expected = 0.1 * (idx as f32 + 1.0) * (0.0 - vi)
                + 0.05 * (idx as f32 + 1.0) * b * (0.0 - vi)
                + 0.02 * (-65.0 - vi)
                + 0.01 * (0.0 - vi);
            assert!(
                (i_out[idx] - expected).abs() < tol,
                "idx {idx}: got {} vs {expected}",
                i_out[idx]
            );
        }
    }

    #[test]
    fn test_build_receptor_types_csr() {
        let row_ptr = vec![0i32, 2, 4];
        let nt = vec![1.0f32, -1.0];
        let rec = build_receptor_types(&row_ptr, 4, &nt);
        assert_eq!(rec, vec![0u8, 0, 2, 2]);
    }
}
