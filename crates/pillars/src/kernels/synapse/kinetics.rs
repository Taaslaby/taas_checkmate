use super::receptor::{nmda_mg_block, nmda_mg_block_simd, ReceptorType};
use crate::core::util::{load8, store8};
use crate::math::fast_exp::fast_exp;
use wide::f32x8;

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
                for (r, (b, &e_r)) in ReceptorType::all()
                    .into_iter()
                    .zip(gbufs.iter().zip(er.iter()))
                {
                    let g = load8(*b, off);
                    // Mg²⁺ block applies only to the NMDA conductance.
                    let g_eff = if r == ReceptorType::NMDA {
                        g * block(vi)
                    } else {
                        g
                    };
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
        ReceptorType::all()
            .into_iter()
            .map(|r| self.g_of(r, i))
            .sum()
    }
}
