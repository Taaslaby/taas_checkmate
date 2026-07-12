//! Synaptic kinetics — conductance-based synapse models with receptor types.
//!
//! # Replaces instant current injection with realistic conductance dynamics:
//!
//! | Receptor | Effect | τ_decay | E_rev | Ion |
//! |----------|--------|---------|-------|-----|
//! | AMPA     | Fast exc (mammalian) | 2 ms  | 0 mV  | Na⁺ |
//! | NMDA     | Slow exc | 50 ms | 0 mV  | Ca²⁺ (Mg²⁺ block) |
//! | GABA_A   | Fast inh | 10 ms | -70 mV | Cl⁻ |
//! | GABA_B   | Slow inh | 200 ms| -90 mV | G-protein |
//! | nAChR    | Fast exc (insect ACh) | 3 ms | 0 mV  | Na⁺/K⁺ |
//! | GluCl    | Fast inh (insect GLUT) | 5 ms | -70 mV | Cl⁻ |
//! | mAChR    | Slow mod (DA/SER/OCT) | 150 ms | -90 mV | GPCR |
//!
//! # Usage
//! ```ignore
//! let mut kin = SynapticKinetics::new(n);
//! let factors = DecayFactors::new(dt);
//! kin.decay_all(&factors);                         // g *= exp(-dt/τ)
//! kin.jump(neuron, ReceptorType::AMPA, weight);    // pre-spike → g↑
//! kin.compute_currents(&v, &mut I_syn);             // I = Σ g·(V - E_rev)
//! ```

#![allow(non_snake_case)]

use super::{load8, store8};
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
/// | 2 | GABA_A | Fast inh (Cl⁻) | 10ms | -70mV | GABA |
/// | 3 | GABA_B | Slow inh (G-protein) | 200ms | -90mV | — |
/// | 4 | nAChR | Fast exc (insect ACh) | 3ms | 0mV | ACh |
/// | 5 | GluCl | Fast inh (insect GLUT, Cl⁻) | 5ms | -70mV | GLUT |
/// | 6 | mAChR | Slow mod (GPCR, DA/SER/OCT) | 150ms | -90mV | DA/SER/OCT |
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum ReceptorType {
    /// AMPA-like: fast excitatory, Na⁺ influx, τ=2ms, E_rev=0mV (mammalian default)
    AMPA = 0,
    /// NMDA-like: slow excitatory, Ca²⁺ influx, τ=50ms, E_rev=0mV
    /// Has voltage-dependent Mg²⁺ block (see [`nmda_mg_block`]).
    NMDA = 1,
    /// GABA_A-like: fast inhibitory, Cl⁻ influx, τ=10ms, E_rev=-70mV
    GABA_A = 2,
    /// GABA_B-like: slow inhibitory, G-protein coupled, τ=200ms, E_rev=-90mV
    GABA_B = 3,
    /// nAChR: fast excitatory (insect ACh), Na⁺/K⁺ influx, τ=3ms, E_rev=0mV
    /// Main excitatory receptor in Drosophila CNS.
    nAChR = 4,
    /// GluCl: fast inhibitory (insect glutamate), Cl⁻ influx, τ=5ms, E_rev=-70mV
    /// Glutamate-gated chloride channel — inhibitory in arthropods!
    GluCl = 5,
    /// mAChR-like: slow modulatory (DA/SER/OCT), G-protein coupled, τ=150ms, E_rev=-90mV
    /// Muscarinic-like receptors for biogenic amines in insects.
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
            ReceptorType::GABA_A => -70.0,
            ReceptorType::GABA_B => -90.0,
            ReceptorType::nAChR => 0.0,
            ReceptorType::GluCl => -70.0,
            ReceptorType::mAChR => -90.0,
        }
    }

    /// Convert from stored `u8` discriminant.
    ///
    /// For unrecognized values, falls back to AMPA (safe default for exc).
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

    /// Map nt_type sign to default receptor (mammalian simple map):
    /// - `+1` (excitatory) → AMPA
    /// - `-1` (inhibitory) → GABA_A
    #[inline(always)]
    pub fn from_nt_type(nt: f32) -> Self {
        if nt > 0.0 { ReceptorType::AMPA } else { ReceptorType::GABA_A }
    }

    /// True if this receptor produces inward (depolarizing) current.
    /// Used to compute local E/I balance per neuron.
    #[inline(always)]
    pub fn is_excitatory(self) -> bool {
        matches!(self, ReceptorType::AMPA | ReceptorType::NMDA | ReceptorType::nAChR)
    }

    /// All receptor variants for iteration, including AMPA as default fallback.
    pub fn all() -> [ReceptorType; 7] {
        [ReceptorType::AMPA, ReceptorType::NMDA, ReceptorType::GABA_A, ReceptorType::GABA_B,
         ReceptorType::nAChR, ReceptorType::GluCl, ReceptorType::mAChR]
    }
}

// ═══════════════════════════════════════════════════════════
//  NMDA voltage-dependent Mg²⁺ block
// ═══════════════════════════════════════════════════════════

/// Mg²⁺ block factor for NMDA receptor: B(V) = 1 / (1 + 0.33·exp(-0.06·V)).
///
/// At resting (-65 mV): B ≈ 0.02 (blocked)
/// At spike peak (+30 mV): B ≈ 0.95 (open)
#[inline(always)]
pub fn nmda_mg_block(v: f32) -> f32 {
    1.0 / (1.0 + 0.33 * (-0.06 * v).exp())
}

/// SIMD version of Mg²⁺ block.
#[inline(always)]
unsafe fn nmda_mg_block_simd(v: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let factor = f32x8::splat(0.33);
    let exp_arg = f32x8::splat(-0.06) * v;
    // Clamp exp arg to safe range [-10, 10]
    let clamped = exp_arg.max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + factor * super::fast_exp_simd(clamped))
}

// ═══════════════════════════════════════════════════════════
//  Precomputed decay factors (reuse across steps)
// ═══════════════════════════════════════════════════════════

/// Precomputed decay factors `exp(-dt / τ)` for each receptor type.
/// Updated when `dt` changes.
#[derive(Debug, Clone)]
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
            ampa: (-dt / ReceptorType::AMPA.tau_decay()).exp(),
            nmda: (-dt / ReceptorType::NMDA.tau_decay()).exp(),
            gaba_a: (-dt / ReceptorType::GABA_A.tau_decay()).exp(),
            gaba_b: (-dt / ReceptorType::GABA_B.tau_decay()).exp(),
            nachr: (-dt / ReceptorType::nAChR.tau_decay()).exp(),
            glucl: (-dt / ReceptorType::GluCl.tau_decay()).exp(),
            machr: (-dt / ReceptorType::mAChR.tau_decay()).exp(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Synaptic kinetics — conductance buffers
// ═══════════════════════════════════════════════════════════

/// Per-neuron conductance buffers for all receptor types.
///
/// Lifecycle per simulation step:
/// 1. `decay_all(factors)` — экспоненциальный спад: `g *= exp(-dt/τ)`
/// 2. `compute_currents(v, I_out)` — ток из проводимостей
/// 3. `jump(post, type, weight)` — при спайке пре-нейрона
///
/// All buffers are padded to a multiple of 8 for SIMD access.
#[derive(Debug, Clone)]
pub struct SynapticKinetics {
    pub n: usize,        // actual number of neurons (unpadded)
    pub n_padded: usize, // padded to %8
    // Conductance buffers — one per receptor type, padded to n_padded
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
        let n_padded = (n_neurons + 7) / 8 * 8;
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
    /// Uses `f32x8` SIMD — 8 neurons per iteration.
    /// For any `padded` elements beyond `n`, the decay still applies (they stay 0).
    pub fn decay_all(&mut self, factors: &DecayFactors) {
        let d_ampa = f32x8::splat(factors.ampa);
        let d_nmda = f32x8::splat(factors.nmda);
        let d_gaba_a = f32x8::splat(factors.gaba_a);
        let d_gaba_b = f32x8::splat(factors.gaba_b);
        let d_nachr = f32x8::splat(factors.nachr);
        let d_glucl = f32x8::splat(factors.glucl);
        let d_machr = f32x8::splat(factors.machr);
        let n8 = self.n_padded & !7;

        unsafe {
            for off in (0..n8).step_by(8) {
                store8(self.g_ampa.as_mut_ptr(), off, load8(self.g_ampa.as_ptr(), off) * d_ampa);
                store8(self.g_nmda.as_mut_ptr(), off, load8(self.g_nmda.as_ptr(), off) * d_nmda);
                store8(self.g_gaba_a.as_mut_ptr(), off, load8(self.g_gaba_a.as_ptr(), off) * d_gaba_a);
                store8(self.g_gaba_b.as_mut_ptr(), off, load8(self.g_gaba_b.as_ptr(), off) * d_gaba_b);
                store8(self.g_nachr.as_mut_ptr(), off, load8(self.g_nachr.as_ptr(), off) * d_nachr);
                store8(self.g_glucl.as_mut_ptr(), off, load8(self.g_glucl.as_ptr(), off) * d_glucl);
                store8(self.g_machr.as_mut_ptr(), off, load8(self.g_machr.as_ptr(), off) * d_machr);
            }
        }

        // Scalar tail
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

    /// Compute synaptic current: `I_out[i] += Σ g_r[i] · (E_rev_r − V[i])`.
    ///
    /// Uses `f32x8` SIMD for the inner loop.
    /// For NMDA, the voltage-dependent Mg²⁺ block modulates `g_nmda` in-place.
    pub fn compute_currents(&self, v: &[f32], i_out: &mut [f32]) {
        let n = self.n.min(v.len()).min(i_out.len());
        let e_ampa = f32x8::splat(ReceptorType::AMPA.e_rev());     // 0
        let e_nmda = f32x8::splat(ReceptorType::NMDA.e_rev());     // 0
        let e_gaba_a = f32x8::splat(ReceptorType::GABA_A.e_rev()); // -70
        let e_gaba_b = f32x8::splat(ReceptorType::GABA_B.e_rev()); // -90
        let e_nachr = f32x8::splat(ReceptorType::nAChR.e_rev());   // 0
        let e_glucl = f32x8::splat(ReceptorType::GluCl.e_rev());   // -70
        let e_machr = f32x8::splat(ReceptorType::mAChR.e_rev());   // -90
        let n8 = n & !7;

        unsafe {
            for off in (0..n8).step_by(8) {
                let vi = load8(v.as_ptr(), off);
                let mut cur = load8(i_out.as_ptr(), off);

                // AMPA: g · (0 - V)
                cur = cur + load8(self.g_ampa.as_ptr(), off) * (e_ampa - vi);
                // NMDA: g · B(V) · (0 - V)  — Mg²⁺ block
                let g_nmda_v = load8(self.g_nmda.as_ptr(), off);
                cur = cur + g_nmda_v * nmda_mg_block_simd(vi) * (e_nmda - vi);
                // GABA_A: g · (-70 - V)
                cur = cur + load8(self.g_gaba_a.as_ptr(), off) * (e_gaba_a - vi);
                // GABA_B: g · (-90 - V)
                cur = cur + load8(self.g_gaba_b.as_ptr(), off) * (e_gaba_b - vi);
                // nAChR: g · (0 - V)  — fast excitatory (insect ACh)
                cur = cur + load8(self.g_nachr.as_ptr(), off) * (e_nachr - vi);
                // GluCl: g · (-70 - V) — fast inhibitory (insect GLUT, Cl⁻)
                cur = cur + load8(self.g_glucl.as_ptr(), off) * (e_glucl - vi);
                // mAChR: g · (-90 - V) — slow modulatory (DA/SER/OCT)
                cur = cur + load8(self.g_machr.as_ptr(), off) * (e_machr - vi);

                store8(i_out.as_mut_ptr(), off, cur);
            }
        }

        // Scalar tail
        for i in n8..n {
            let vi = v[i];
            i_out[i] += self.g_ampa[i] * (ReceptorType::AMPA.e_rev() - vi)
                      + self.g_nmda[i] * nmda_mg_block(vi) * (ReceptorType::NMDA.e_rev() - vi)
                      + self.g_gaba_a[i] * (ReceptorType::GABA_A.e_rev() - vi)
                      + self.g_gaba_b[i] * (ReceptorType::GABA_B.e_rev() - vi)
                      + self.g_nachr[i] * (ReceptorType::nAChR.e_rev() - vi)
                      + self.g_glucl[i] * (ReceptorType::GluCl.e_rev() - vi)
                      + self.g_machr[i] * (ReceptorType::mAChR.e_rev() - vi);
        }
    }

    /// Jump a post-synaptic conductance by `amount` (triggered by pre-spike).
    #[inline(always)]
    pub fn jump(&mut self, neuron: usize, receptor: ReceptorType, amount: f32) {
        if neuron >= self.n { return; }
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

    /// Read total conductance for a neuron across all receptor types.
    pub fn total_conductance(&self, i: usize) -> f32 {
        if i >= self.n { return 0.0; }
        self.g_ampa[i] + self.g_nmda[i] + self.g_gaba_a[i] + self.g_gaba_b[i]
            + self.g_nachr[i] + self.g_glucl[i] + self.g_machr[i]
    }
}

// ═══════════════════════════════════════════════════════════
//  Build receptor type array from nt_types
// ═══════════════════════════════════════════════════════════

/// Build per-synapse `receptor_types` array from per-neuron `nt_types` using
/// CSR structure. Each synapse gets the receptor type of its pre-neuron's NT.
///
/// `nt_to_receptor` maps nt value to a specific receptor (not just ±1).
/// If `None`, uses default: exc → AMPA, inh → GABA_A.
pub fn build_receptor_types(
    row_ptr: &[i32],
    n_synapses: usize,
    nt_types: &[f32],
) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = if row_ptr.len() > 1 { row_ptr.len() - 1 } else { 0 };

    let mut syn_idx = 0usize;
    for pre in 0..n {
        let nt = nt_types.get(pre).copied().unwrap_or(1.0);
        let rec = if nt > 0.0 { 0u8 } else { 2u8 }; // AMPA or GABA_A

        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(pre + 1).copied().unwrap_or(0) as usize).min(n_synapses);

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

    // ─── 1. Decay ─────────────────────────────────────

    #[test]
    fn test_decay_factors_precision() {
        let factors = DecayFactors::new(0.5); // dt=0.5ms
        // AMPA τ=2ms: exp(-0.5/2) = exp(-0.25) ≈ 0.7788
        assert!((factors.ampa - 0.7788).abs() < 0.001,
            "AMPA decay: {}", factors.ampa);
    }

    #[test]
    fn test_decay_all_zero_initial() {
        let mut kin = SynapticKinetics::new(8);
        let factors = DecayFactors::new(1.0);
        kin.decay_all(&factors);
        for i in 0..8 {
            assert_eq!(kin.g_ampa[i], 0.0);
            assert_eq!(kin.g_nmda[i], 0.0);
            assert_eq!(kin.g_gaba_a[i], 0.0);
            assert_eq!(kin.g_gaba_b[i], 0.0);
        }
    }

    #[test]
    fn test_decay_all_exponential() {
        let mut kin = SynapticKinetics::new(4);
        kin.g_ampa[0] = 1.0;
        kin.g_gaba_a[1] = 2.0;

        let factors = DecayFactors::new(2.0); // dt=2ms
        // AMPA τ=2ms: 1.0 * exp(-2/2) = 1.0 * 0.3679 = 0.3679
        // GABA_A τ=10ms: 2.0 * exp(-2/10) = 2.0 * 0.8187 = 1.6375
        kin.decay_all(&factors);

        assert!((kin.g_ampa[0] - 0.3679).abs() < 0.001,
            "AMPA decay: {}", kin.g_ampa[0]);
        assert!((kin.g_gaba_a[1] - 1.6375).abs() < 0.001,
            "GABA_A decay: {}", kin.g_gaba_a[1]);
    }

    // ─── 2. Jump ──────────────────────────────────────

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

    // ─── 3. Compute currents ─────────────────────────

    #[test]
    fn test_compute_currents_basic() {
        let mut kin = SynapticKinetics::new(2);
        let v = vec![-65.0, -65.0];
        let mut i_out = vec![0.0, 0.0];

        kin.g_ampa[0] = 0.1;  // AMPA: 0.1 * (0 - (-65)) = 6.5 (depol)
        kin.g_gaba_a[1] = 0.2; // GABA_A: 0.2 * (-70 - (-65)) = -1.0 (hyperpol)

        kin.compute_currents(&v, &mut i_out);

        assert!((i_out[0] - 6.5).abs() < 0.01,
            "neuron 0 AMPA I_syn: {}", i_out[0]);
        assert!((i_out[1] - (-1.0)).abs() < 0.01,
            "neuron 1 GABA_A I_syn: {}", i_out[1]);
    }

    #[test]
    fn test_compute_currents_nmda_block() {
        let mut kin = SynapticKinetics::new(2);
        kin.g_nmda[0] = 1.0;
        kin.g_nmda[1] = 1.0;

        let v = vec![-65.0, 30.0];
        let mut i_out = vec![0.0, 0.0];

        kin.compute_currents(&v, &mut i_out);

        // At -65mV: B(V) ≈ 0.024, I ≈ 1.0 * 0.024 * (0 - (-65)) = 1.56 (depol)
        // At +30mV: B(V) ≈ 0.95, I ≈ 1.0 * 0.95 * (0 - 30) = -28.5 (hyperpol, driving force reversed)
        let mag_neg = i_out[1].abs(); // magnitude at depolarized V
        assert!(mag_neg > 5.0,
            "NMDA Mg block: at 30mV current magnitude should be large, got {}", i_out[1]);
    }

    // ─── 4. Receptor type utilities ──────────────────

    #[test]
    fn test_receptor_type_defaults() {
        assert_eq!(ReceptorType::from_nt_type(1.0), ReceptorType::AMPA);
        assert_eq!(ReceptorType::from_nt_type(-1.0), ReceptorType::GABA_A);
        assert_eq!(ReceptorType::from_u8(0), ReceptorType::AMPA);
        assert_eq!(ReceptorType::from_u8(2), ReceptorType::GABA_A);
    }

    #[test]
    fn test_nmda_mg_block_values() {
        let b_m80 = nmda_mg_block(-80.0);
        assert!(b_m80 < 0.03, "Mg block at -80mV: {}", b_m80);

        let b_0 = nmda_mg_block(0.0);
        assert!((b_0 - 0.75).abs() < 0.05, "Mg block at 0mV: {}", b_0);

        let b_p30 = nmda_mg_block(30.0);
        assert!(b_p30 > 0.90, "Mg block at +30mV: {}", b_p30);
    }
}
