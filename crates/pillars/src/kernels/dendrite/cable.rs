#![allow(non_snake_case)] // R_a/R_m/C_m, V_soma, dV_dt — физическая нотация

//! # Cable Dendrite Microkernels — Compartmentalized Dendrites
//!
//! Each neuron has `k` compartments connected axially (`R_a`). All
//! compartments have membrane resistance (`R_m`) to the soma.
//!
//! ## Cable equation (Forward Euler)
//! ```text
//! C_m·dV_k/dt = (V_{k−1} − V_k)/R_a   ← axial current from previous
//!              + (V_{k+1} − V_k)/R_a   ← axial current to next
//!              + (V_soma − V_k)/R_m    ← electrotonic current to soma
//!              + Σ g_r[k]·(E_rev_r − V_k)  ← local synaptic current
//!
//! I_dend += (V_k − V_soma)/R_m          ← total current into soma
//! ```
//!
//! ## Layout (column-major, stride = n_padded)
//! ```text
//! buf[comp * n_padded + nrn] = compartment comp, neuron nrn
//! ```
//! This lets us load 8 neurons of one compartment with a single `f32x8`.

use wide::f32x8;

use crate::core::util::{load8, store8};
use crate::math::fast_exp::{fast_exp, fast_exp_poly_simd};

// ═══════════════════════════════════════════════════════════
//  NMDA Mg²⁺ block
// ═══════════════════════════════════════════════════════════

/// Mg²⁺ block factor: `B(V) = 1 / (1 + 0.33·exp(−0.06·V))`.
/// At resting (−65 mV): B ≈ 0.02 (blocked). At spike peak (+30 mV): B ≈ 0.95.
///
/// Argument clamped to [−10, 10] — bit-identical to the SIMD path.
#[inline(always)]
fn nmda_mg_block(v: f32) -> f32 {
    1.0 / (1.0 + 0.33 * fast_exp((-0.06 * v).clamp(-10.0, 10.0)))
}

#[inline(always)]
fn nmda_mg_block_simd(v: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let factor = f32x8::splat(0.33);
    let exp_arg = f32x8::splat(-0.06) * v;
    let clamped = exp_arg.max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + factor * fast_exp_poly_simd(clamped))
}

// ═══════════════════════════════════════════════════════════
//  Receptor reversal potentials (mV)
// ═══════════════════════════════════════════════════════════

const E_AMPA: f32 = 0.0;
const E_NMDA: f32 = 0.0;
const E_GABA_A: f32 = -65.0;
const E_GABA_B: f32 = -90.0;
const E_NACHR: f32 = 0.0;
const E_GLUCL: f32 = -70.0; // Drosophila Cl⁻ channel — inhibitory
const E_MACHR: f32 = -90.0;

use super::config::{ActiveDendriteChannels, DendriteConfig, DendriteDecayFactors};

/// Состояние компартментализованного дендритного кабеля для всей популяции.
///
/// All buffers are padded to %8 for SIMD (stride = n_padded).
///
/// # Пример сомато-дендритного цикла
/// ```ignore
/// dend.decay_conductances(&factors);   // g *= exp(-dt/τ)
/// dend.cable_step(&V_soma, dt);        // cable dynamics → I_dend
/// // I_dend[i] now holds the dendritic current into soma
/// // ... propagate spikes → jump conductances on compartments
/// ```
#[derive(Debug, Clone)]
pub struct CableDendriteState {
    pub n: usize,
    pub k: usize,
    pub n_padded: usize,
    pub config: DendriteConfig,
    /// V_dend[n_padded × k] — column-major
    pub V_dend: Vec<f32>,
    /// Conductances of all 7 types [n_padded × k] each
    pub g_ampa: Vec<f32>,
    pub g_nmda: Vec<f32>,
    pub g_gaba_a: Vec<f32>,
    pub g_gaba_b: Vec<f32>,
    pub g_nachr: Vec<f32>,
    pub g_glucl: Vec<f32>,
    pub g_machr: Vec<f32>,
    /// `I_dend[n_padded]` — accumulator of current into soma (overwritten each step)
    pub I_dend: Vec<f32>,
    /// Active HH/Ca²⁺ channels in dendrites (optional)
    pub active_channels: Option<ActiveDendriteChannels>,
}

impl CableDendriteState {
    /// Create a new dendritic tree. V_dend starts at V_rest = −65 mV.
    pub fn new(n: usize, config: &DendriteConfig) -> Self {
        let n_padded = n.div_ceil(8) * 8;
        let total = n_padded * config.n_compartments;
        Self {
            n,
            k: config.n_compartments,
            n_padded,
            config: config.clone(),
            V_dend: vec![-65.0; total],
            g_ampa: vec![0.0; total],
            g_nmda: vec![0.0; total],
            g_gaba_a: vec![0.0; total],
            g_gaba_b: vec![0.0; total],
            g_nachr: vec![0.0; total],
            g_glucl: vec![0.0; total],
            g_machr: vec![0.0; total],
            I_dend: vec![0.0; n_padded],
            active_channels: None,
        }
    }

    /// Enable active ionic channels in dendrites (Na⁺, K⁺, Ca²⁺).
    pub fn enable_active_channels(&mut self, g_na: f32, g_kv: f32, g_ca: f32) {
        let total = self.n_padded * self.k;
        self.active_channels = Some(ActiveDendriteChannels::new(total, g_na, g_kv, g_ca));
    }

    /// Step cable dynamics (alias for backwards compatibility).
    pub fn step(&mut self, V_soma: &[f32], dt: f32) {
        self.cable_step(V_soma, dt);
    }

    /// Step cable dynamics: update V_dend and accumulate I_dend.
    ///
    /// # Panics
    /// If `V_soma.len() < n`.
    pub fn cable_step(&mut self, V_soma: &[f32], dt: f32) {
        assert!(
            V_soma.len() >= self.n,
            "cable_step: V_soma too short ({} < {})",
            V_soma.len(),
            self.n
        );

        self.I_dend.fill(0.0);
        let n = self.n;
        let n_padded = self.n_padded;
        let k = self.k;
        let R_a = self.config.R_a;
        let R_m = self.config.R_m;
        let C_m = self.config.C_m;

        let inv_Rm = f32x8::splat(1.0 / R_m);
        let inv_Cm = f32x8::splat(1.0 / C_m);
        let dt_s = f32x8::splat(dt);
        let zero = f32x8::splat(0.0);
        let has_axial = k > 1 && R_a > 0.1;
        let inv_Ra = if has_axial {
            f32x8::splat(1.0 / R_a)
        } else {
            zero
        };

        let e_ampa = f32x8::splat(E_AMPA);
        let e_nmda = f32x8::splat(E_NMDA);
        let e_gaba_a = f32x8::splat(E_GABA_A);
        let e_gaba_b = f32x8::splat(E_GABA_B);
        let e_nachr = f32x8::splat(E_NACHR);
        let e_glucl = f32x8::splat(E_GLUCL);
        let e_machr = f32x8::splat(E_MACHR);

        let n8 = n & !7;

        for comp in 0..k {
            let base = comp * n_padded;

            for off in (0..n8).step_by(8) {
                let idx = base + off;
                unsafe {
                    let vd = load8(self.V_dend.as_ptr(), idx);
                    let vs = load8(V_soma.as_ptr(), off);

                    let mut i_syn = zero
                        + load8(self.g_ampa.as_ptr(), idx) * (e_ampa - vd)
                        + load8(self.g_nmda.as_ptr(), idx) * nmda_mg_block_simd(vd) * (e_nmda - vd)
                        + load8(self.g_gaba_a.as_ptr(), idx) * (e_gaba_a - vd)
                        + load8(self.g_gaba_b.as_ptr(), idx) * (e_gaba_b - vd)
                        + load8(self.g_nachr.as_ptr(), idx) * (e_nachr - vd)
                        + load8(self.g_glucl.as_ptr(), idx) * (e_glucl - vd)
                        + load8(self.g_machr.as_ptr(), idx) * (e_machr - vd);

                    let mut dV_dt = (vs - vd) * inv_Rm;

                    if has_axial {
                        if comp > 0 {
                            let v_prev = load8(self.V_dend.as_ptr(), (comp - 1) * n_padded + off);
                            i_syn += (v_prev - vd) * inv_Ra;
                        } else {
                            i_syn += (vs - vd) * inv_Ra;
                        }
                        if comp < k - 1 {
                            let v_next = load8(self.V_dend.as_ptr(), (comp + 1) * n_padded + off);
                            i_syn += (v_next - vd) * inv_Ra;
                        }
                    }

                    dV_dt += i_syn * inv_Cm;
                    let new_vd = vd + dV_dt * dt_s;
                    store8(self.V_dend.as_mut_ptr(), idx, new_vd);

                    let mut i_to_soma = (vd - vs) * inv_Rm;
                    if has_axial && comp == 0 {
                        i_to_soma += (vd - vs) * inv_Ra;
                    }

                    let mut i_dend = load8(self.I_dend.as_ptr(), off);
                    i_dend += i_to_soma;
                    store8(self.I_dend.as_mut_ptr(), off, i_dend);
                }
            }

            // Scalar tail (n % 8 != 0)
            for i in n8..n {
                let idx = base + i;
                let vd = self.V_dend[idx];
                let vs = V_soma[i];

                let i_syn = self.g_ampa[idx] * (E_AMPA - vd)
                    + self.g_nmda[idx] * nmda_mg_block(vd) * (E_NMDA - vd)
                    + self.g_gaba_a[idx] * (E_GABA_A - vd)
                    + self.g_gaba_b[idx] * (E_GABA_B - vd)
                    + self.g_nachr[idx] * (E_NACHR - vd)
                    + self.g_glucl[idx] * (E_GLUCL - vd)
                    + self.g_machr[idx] * (E_MACHR - vd);

                let mut dV_dt = (vs - vd) / R_m + i_syn / C_m;

                if has_axial {
                    let inv_Ra_scalar = 1.0 / R_a;
                    if comp > 0 {
                        let v_prev = self.V_dend[(comp - 1) * n_padded + i];
                        dV_dt += (v_prev - vd) * inv_Ra_scalar / C_m;
                    } else {
                        dV_dt += (vs - vd) * inv_Ra_scalar / C_m;
                    }
                    if comp < k - 1 {
                        let v_next = self.V_dend[(comp + 1) * n_padded + i];
                        dV_dt += (v_next - vd) * inv_Ra_scalar / C_m;
                    }
                }

                self.V_dend[idx] = vd + dV_dt * dt;

                let mut i_to_soma = (vd - vs) / R_m;
                if has_axial && comp == 0 {
                    i_to_soma += (vd - vs) / R_a;
                }
                self.I_dend[i] += i_to_soma;
            }
        }

        if let Some(ref mut active) = self.active_channels {
            active.step(&mut self.V_dend, self.config.C_m, dt);
        }
    }

    /// Exponential decay of conductances: `g *= exp(-dt/τ)`.
    pub fn decay_conductances(&mut self, factors: &DendriteDecayFactors) {
        let d0 = f32x8::splat(factors.decays[0]);
        let d1 = f32x8::splat(factors.decays[1]);
        let d2 = f32x8::splat(factors.decays[2]);
        let d3 = f32x8::splat(factors.decays[3]);
        let d4 = f32x8::splat(factors.decays[4]);
        let d5 = f32x8::splat(factors.decays[5]);
        let d6 = f32x8::splat(factors.decays[6]);

        let total = self.n_padded * self.k;
        let n8 = total & !7;

        unsafe {
            for off in (0..n8).step_by(8) {
                store8(
                    self.g_ampa.as_mut_ptr(),
                    off,
                    load8(self.g_ampa.as_ptr(), off) * d0,
                );
                store8(
                    self.g_nmda.as_mut_ptr(),
                    off,
                    load8(self.g_nmda.as_ptr(), off) * d1,
                );
                store8(
                    self.g_gaba_a.as_mut_ptr(),
                    off,
                    load8(self.g_gaba_a.as_ptr(), off) * d2,
                );
                store8(
                    self.g_gaba_b.as_mut_ptr(),
                    off,
                    load8(self.g_gaba_b.as_ptr(), off) * d3,
                );
                store8(
                    self.g_nachr.as_mut_ptr(),
                    off,
                    load8(self.g_nachr.as_ptr(), off) * d4,
                );
                store8(
                    self.g_glucl.as_mut_ptr(),
                    off,
                    load8(self.g_glucl.as_ptr(), off) * d5,
                );
                store8(
                    self.g_machr.as_mut_ptr(),
                    off,
                    load8(self.g_machr.as_ptr(), off) * d6,
                );
            }
        }

        for i in n8..total {
            self.g_ampa[i] *= factors.decays[0];
            self.g_nmda[i] *= factors.decays[1];
            self.g_gaba_a[i] *= factors.decays[2];
            self.g_gaba_b[i] *= factors.decays[3];
            self.g_nachr[i] *= factors.decays[4];
            self.g_glucl[i] *= factors.decays[5];
            self.g_machr[i] *= factors.decays[6];
        }
    }

    /// Reset all conductances to 0.
    pub fn reset_conductances(&mut self) {
        let total = self.n_padded * self.k;
        self.g_ampa[..total].fill(0.0);
        self.g_nmda[..total].fill(0.0);
        self.g_gaba_a[..total].fill(0.0);
        self.g_gaba_b[..total].fill(0.0);
        self.g_nachr[..total].fill(0.0);
        self.g_glucl[..total].fill(0.0);
        self.g_machr[..total].fill(0.0);
    }

    // ─── Conductance setters ─────────────────────────────

    /// Jump conductance by `amount` on compartment `comp` of neuron `nrn`.
    pub fn jump_by_index(&mut self, nrn: usize, comp: usize, receptor: u8, amount: f32) {
        match receptor {
            0 => self.jump_ampa(nrn, comp, amount),
            1 => self.jump_nmda(nrn, comp, amount),
            2 => self.jump_gaba_a(nrn, comp, amount),
            3 => self.jump_gaba_b(nrn, comp, amount),
            4 => self.jump_nachr(nrn, comp, amount),
            5 => self.jump_glucl(nrn, comp, amount),
            6 => self.jump_machr(nrn, comp, amount),
            _ => {}
        }
    }

    pub fn jump_ampa(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_ampa[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_nmda(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_nmda[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_gaba_a(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_gaba_a[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_gaba_b(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_gaba_b[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_nachr(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_nachr[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_glucl(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_glucl[comp * self.n_padded + nrn] += amount;
        }
    }
    pub fn jump_machr(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_machr[comp * self.n_padded + nrn] += amount;
        }
    }

    // ─── Getters ─────────────────────────────────────────

    /// Total conductance of all types for compartment `comp` of neuron `nrn`.
    pub fn total_conductance(&self, nrn: usize, comp: usize) -> f32 {
        if nrn >= self.n || comp >= self.k {
            return 0.0;
        }
        let idx = comp * self.n_padded + nrn;
        self.g_ampa[idx]
            + self.g_nmda[idx]
            + self.g_gaba_a[idx]
            + self.g_gaba_b[idx]
            + self.g_nachr[idx]
            + self.g_glucl[idx]
            + self.g_machr[idx]
    }

    /// Potential of compartment `comp` of neuron `nrn`.
    pub fn v_dend(&self, nrn: usize, comp: usize) -> f32 {
        if nrn >= self.n || comp >= self.k {
            return 0.0;
        }
        self.V_dend[comp * self.n_padded + nrn]
    }

    /// Dendritic current of neuron `nrn` (after cable_step).
    pub fn dend_current(&self, nrn: usize) -> f32 {
        if nrn >= self.n {
            0.0
        } else {
            self.I_dend[nrn]
        }
    }

    /// Set V_dend of a compartment (for tests).
    pub fn set_v_dend(&mut self, nrn: usize, comp: usize, value: f32) {
        if nrn < self.n && comp < self.k {
            self.V_dend[comp * self.n_padded + nrn] = value;
        }
    }
}

/// Устаревший тип — используйте [`CableDendriteState`] вместо него.
pub type DendriteState = CableDendriteState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendrite_basic_ampa_epsps() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(8, &config);
        let V_soma = vec![-65.0f32; 8];
        dend.jump_ampa(0, 0, 0.1);
        let factors = DendriteDecayFactors::new(0.5);
        for _ in 0..20 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        assert!(
            dend.dend_current(0) > 0.0,
            "AMPA should depolarize, got {}",
            dend.dend_current(0)
        );
        assert!(dend.dend_current(1).abs() < 1e-4, "no-input neuron ~0");
    }

    #[test]
    fn test_dendrite_gaba_shunt() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend_only = CableDendriteState::new(4, &config);
        let mut dend_shunt = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        dend_only.jump_ampa(0, 0, 0.2);
        dend_shunt.jump_ampa(0, 0, 0.2);
        dend_shunt.jump_gaba_a(0, 0, 0.4);
        for _ in 0..20 {
            dend_only.decay_conductances(&factors);
            dend_shunt.decay_conductances(&factors);
            dend_only.cable_step(&V_soma, 0.5);
            dend_shunt.cable_step(&V_soma, 0.5);
        }
        let i_only = dend_only.dend_current(0);
        let i_shunt = dend_shunt.dend_current(0);
        assert!(
            i_shunt < i_only,
            "GABA_A should shunt: only={i_only:.6}, shunt={i_shunt:.6}"
        );
    }

    #[test]
    fn test_dendrite_simd_vs_scalar_tail() {
        let config = DendriteConfig {
            n_compartments: 3,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(9, &config);
        let V_soma = vec![-65.0f32; 9];
        let factors = DendriteDecayFactors::new(0.5);
        for i in 0..9 {
            dend.jump_ampa(i, 0, 0.3);
            dend.jump_ampa(i, 2, 0.5);
        }
        for _ in 0..15 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let diff = (dend.dend_current(0) - dend.dend_current(8)).abs();
        assert!(diff < 1e-5, "SIMD vs scalar tail mismatch: {diff:.10}");
        for comp in 0..3 {
            let vd_diff = (dend.v_dend(0, comp) - dend.v_dend(8, comp)).abs();
            assert!(
                vd_diff < 1e-5,
                "V_dend[comp={comp}] mismatch: {vd_diff:.10}"
            );
        }
    }

    #[test]
    fn test_dendrite_proximal_vs_distal_ampa() {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut prox = CableDendriteState::new(4, &config);
        let mut dist = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        prox.jump_ampa(0, 0, 0.3);
        dist.jump_ampa(0, 3, 0.3);
        for _ in 0..20 {
            prox.decay_conductances(&factors);
            dist.decay_conductances(&factors);
            prox.cable_step(&V_soma, 0.5);
            dist.cable_step(&V_soma, 0.5);
        }
        let i_prox = prox.dend_current(0);
        let i_dist = dist.dend_current(0);
        assert!(
            i_prox > i_dist,
            "Proximal should win: prox={i_prox:.6}, dist={i_dist:.6}"
        );
    }

    #[test]
    fn test_dendrite_conductance_decay() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        dend.jump_ampa(0, 0, 1.0);
        dend.jump_nmda(1, 0, 1.0);
        let factors = DendriteDecayFactors::new(2.0);
        dend.decay_conductances(&factors);
        assert!(
            (dend.g_ampa[0] - 0.3679).abs() < 0.001,
            "AMPA decay: {}",
            dend.g_ampa[0]
        );
        assert!(
            (dend.g_nmda[1] - 0.9608).abs() < 0.001,
            "NMDA decay: {}",
            dend.g_nmda[1]
        );
    }

    #[test]
    fn test_dendrite_all_7_receptors() {
        let config = DendriteConfig {
            n_compartments: 1,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(8, &config);
        let V_soma = vec![-65.0f32; 8];
        let factors = DendriteDecayFactors::new(0.5);
        dend.jump_ampa(0, 0, 0.5);
        dend.jump_nmda(1, 0, 0.5);
        dend.jump_gaba_a(2, 0, 0.5);
        dend.jump_gaba_b(3, 0, 0.5);
        dend.jump_nachr(4, 0, 0.5);
        dend.jump_glucl(5, 0, 0.5);
        dend.jump_machr(6, 0, 0.5);
        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        assert!(dend.dend_current(4) > 0.0, "nAChR should be excitatory");
        assert!(
            dend.dend_current(3) < 0.0,
            "GABA_B should be hyperpolarizing"
        );
    }

    #[test]
    fn test_dendrite_cable_steady_state() {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.5,
        };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-60.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        for _ in 0..500 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let vd0 = dend.v_dend(0, 0);
        let vd3 = dend.v_dend(0, 3);
        assert!((vd0 - (-60.0)).abs() < 1.0, "V_dend[0] → -60, got {vd0:.4}");
        assert!((vd3 - (-60.0)).abs() < 1.0, "V_dend[3] → -60, got {vd3:.4}");
        assert!(dend.dend_current(0).abs() < 0.02, "steady state I_dend ≈ 0");
    }

    #[test]
    fn test_dendrite_axial_voltage_gradient() {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: 500.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        dend.jump_ampa(0, 0, 0.5);
        for _ in 0..30 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let v0 = dend.v_dend(0, 0);
        let v1 = dend.v_dend(0, 1);
        let v2 = dend.v_dend(0, 2);
        let v3 = dend.v_dend(0, 3);
        assert!(v0 > v1, "V should decay: v0={v0:.4} > v1={v1:.4}");
        assert!(v1 > v2, "V should decay: v1={v1:.4} > v2={v2:.4}");
        assert!(v2 >= v3, "V should decay: v2={v2:.4} >= v3={v3:.4}");
    }

    #[test]
    fn test_dendrite_high_axial_resistance_isolates() {
        let config = DendriteConfig {
            n_compartments: 3,
            R_a: 10000.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        dend.jump_ampa(0, 0, 0.5);
        for _ in 0..20 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let v0 = dend.v_dend(0, 0);
        let v2 = dend.v_dend(0, 2);
        assert!(
            (v0 - v2).abs() > 1.0,
            "High R_a isolates: ΔV={:.4}",
            (v0 - v2).abs()
        );
    }

    #[test]
    fn test_dendrite_low_axial_resistance_near_equal() {
        let dt = 0.05;
        let config = DendriteConfig {
            n_compartments: 3,
            R_a: 1.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(dt);
        dend.jump_ampa(0, 0, 0.5);
        for _ in 0..200 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, dt);
        }
        let v0 = dend.v_dend(0, 0);
        let v1 = dend.v_dend(0, 1);
        let v2 = dend.v_dend(0, 2);
        let max_diff = ((v0 - v1).abs()).max((v0 - v2).abs());
        assert!(max_diff < 0.02, "Low R_a equalizes: max_diff={max_diff:.8}");
    }

    #[test]
    fn test_dendrite_nmda_mg_block() {
        let config = DendriteConfig {
            n_compartments: 1,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        for i in 0..4 {
            dend.jump_nmda(i, 0, 0.5);
        }
        let V_soma = vec![-80.0, -65.0, -40.0, 0.0];
        let factors = DendriteDecayFactors::new(0.5);
        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let i0 = dend.dend_current(0);
        let i3 = dend.dend_current(3);
        assert!(
            i0.abs() < i3.abs() || i0.abs() < 5.0,
            "NMDA at -80 blocked: {i0:.6}"
        );
    }

    #[test]
    fn test_dendrite_membrane_time_constant() {
        let config = DendriteConfig {
            n_compartments: 1,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 1.0,
        };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-50.0f32; 4];
        let factors = DendriteDecayFactors::new(5.0);
        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 5.0);
        }
        let vd = dend.v_dend(0, 0);
        assert!(vd > -63.0 && vd < -57.0, "0.5τ relaxation: {vd:.4}");
    }

    #[test]
    fn test_dendrite_reset_conductances() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        dend.jump_ampa(0, 0, 0.5);
        dend.jump_nmda(1, 1, 0.3);
        dend.jump_gaba_a(2, 1, 0.2);
        assert!(dend.g_ampa[0] > 0.0);
        dend.reset_conductances();
        let total = dend.n_padded * dend.k;
        for i in 0..total {
            assert_eq!(dend.g_ampa[i], 0.0, "g_ampa[{i}]");
            assert_eq!(dend.g_nmda[i], 0.0);
            assert_eq!(dend.g_gaba_a[i], 0.0);
            assert_eq!(dend.g_gaba_b[i], 0.0);
            assert_eq!(dend.g_nachr[i], 0.0);
            assert_eq!(dend.g_glucl[i], 0.0);
            assert_eq!(dend.g_machr[i], 0.0);
        }
    }

    #[test]
    fn test_dendrite_jump_by_index() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        dend.jump_by_index(0, 0, 0, 0.5); // AMPA
        dend.jump_by_index(1, 1, 4, 0.3); // nAChR
        dend.jump_by_index(2, 0, 2, 0.2); // GABA_A
        assert!((dend.g_ampa[0] - 0.5).abs() < 1e-6);
        assert!((dend.g_nachr[dend.n_padded + 1] - 0.3).abs() < 1e-6);
        assert!((dend.g_gaba_a[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_active_dendritic_channels() {
        let config = DendriteConfig {
            n_compartments: 2,
            R_a: 100.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut dend = CableDendriteState::new(4, &config);
        dend.enable_active_channels(10.0, 5.0, 2.0);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);
        dend.jump_ampa(0, 0, 1.5);
        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }
        let v0 = dend.v_dend(0, 0);
        assert!(
            v0 > -50.0,
            "active Na⁺ should boost depolarization: {v0:.2} mV"
        );
    }

    #[test]
    fn test_dendrite_config_helpers() {
        let config = DendriteConfig::default();
        assert!((config.tau() - 10.0).abs() < 1e-5, "τ = R_m·C_m = 10ms");
        let lambda = config.electrotonic_length();
        assert!(lambda > 0.0 && lambda < 1.0, "λ in (0,1) for finite R_a");
        assert!(
            DendriteConfig::compartments_for_length(1.0) >= 10,
            "≥10 compartments per λ"
        );
    }
}
