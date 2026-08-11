#![allow(non_snake_case)] // I_soma, E_leak, V_thresh — физическая нотация

//! # 🧠 Stateful Neuron Populations — канон pillars
//!
//! Высокоуровневые stateful-классы поверх `run_neuron` / `kernels` /
//! `ops` — единый источник правды для API и python. Внутри каждый класс
//! делегирует в pillars-ядро (`run_neuron::<Model>`, `kernels::eprop`,
//! `ops::plasticity::stp`, `kernels::dendrite`, `zoo::sensors::chordotonal`).
//!
//! | Класс | Движок | Описание |
//! |-------|--------|----------|
//! | [`EPropLIF`] | `kernels::eprop::eprop_lif_step` | LIF + eligibility traces (Bellec 2020) |
//! | [`Glif3State`] / [`Glif4State`] | `run_neuron::<GLIF3/4Model>` | Allen Institute GLIF3/4 |
//! | [`ShortTermPlasticity`] | `ops::plasticity::stp::stp_step` | Tsodyks-Markram |
//! | [`SomatoDendriticNeuron`] | `run_neuron::<LIFModel>` + `kernels::dendrite` | LIF soma + cable dendrites |
//! | [`ChordotonalState`] | `zoo::sensors::chordotonal_step` | Drosophila mechanoreceptor |
//! | [`PoissonGenerator`] | `kernels::input` | реэкспорт |
//! | [`SornNetwork`] | `kernels::learning` | реэкспорт |
//! | [`AstrocyteState`] | `kernels::astrocyte` | реэкспорт |
//!
//! ## Подмодули
//!
//! | Подмодуль | Классы | Ядро |
//! |-----------|--------|------|
//! | [`eprop`] | EPropIzhikevich/Multi, EPropSaw, EPropAdExMulti | `kernels::eprop` |
//! | [`haap`] | HaapState/Profile/Builder + step_haap | `run_neuron::<HAAPModel>` |
//! | [`atlasov`] | AtlasovState + профили/конфиги + step | `run_neuron::<AtlasovNeuromotorModel>` |
//! | [`bcm`] | BcmSynapse, HomeostaticScale | `kernels::learning::{bcm_step, homeostatic_scale}` |
//! | [`hr`] | step_hr (repacking-адаптер) | `run_neuron::<HRModel>` |
//! | [`lif`] | step_lif / step_lif_at | `run_neuron::<LIFModel/LIFAtModel>` |
//! | [`izhikevich`] | step_izhikevich / step_izhikevich_dual | `run_neuron::<IzhikevichModel / IzhikevichDualModel>` |
//! | [`saw`] | step_saw / step_gco | `run_neuron::<SAWModel>` / `kernels::phase::gco_step` |
//! | [`adex`] | step_adex | `run_neuron::<AdExModel>` |
//! | [`fhn`] | step_fhn | `run_neuron::<FHNModel>` |
//! | [`wc`] | step_wc | `kernels::wc::wc_step` |
//!
//! Также в модуле: [`DopamineSTDP`] (`kernels::dopamine`), [`DelayBuffer`] (NMJ-задержки),
//! реэкспорты encoding ([`TtfsEncoder`], bsa/phase) и stats ([`SpikeRaster`], firing_rates, ISI, CV).

#[cfg(feature = "rayon")]
use rayon::prelude::*;
use wide::f32x8;

use crate::core::lane::SimdLane;
use crate::core::util::{load8, load8_i32, store8};
use crate::ops::plasticity::stp as pillars_stp;

// ─── Подмодули ─────────────────────────────────────────────

pub mod adex;
pub mod atlasov;
pub mod bcm;
pub mod eprop;
pub mod fhn;
pub mod haap;
pub mod hr;
pub mod izhikevich;
pub mod lif;
pub mod saw;
pub mod wc;

pub use adex::step_adex;
pub use atlasov::{
    step_atlasov_neuromotor, AtlasovState, AtlasovStateBuilder, BiomechanicsProfile,
    DrosophilaLegConfig, FiberPoolConfig,
};
pub use bcm::{BcmSynapse, HomeostaticScale};
pub use eprop::{EPropAdExMulti, EPropIzhikevich, EPropIzhikevichMulti, EPropSaw};
pub use fhn::step_fhn;
pub use haap::{step_haap, HaapProfile, HaapState, HaapStateBuilder};
pub use hr::step_hr;
pub use izhikevich::{step_izhikevich, step_izhikevich_dual};
pub use lif::{step_lif, step_lif_at};
pub use saw::{step_gco, step_saw};
pub use wc::step_wc;

// ─── Реэкспорты kernels-канона ─────────────────────────────

pub use crate::api::autotune::{probe_fr, suggest_config, EPropTunerConfig, FrProbePoint};
pub use crate::kernels::astrocyte::AstrocyteState;
pub use crate::kernels::dendrite::{CableDendriteState, DendriteConfig, DendriteDecayFactors};
pub use crate::kernels::input::{
    bsa_encode, bsa_kernel_hamming, phase_encode, PoissonGenerator, TtfsEncoder,
};
pub use crate::kernels::learning::{
    eprop_update, intrinsic_plasticity, stdp_homeostatic, stdp_update, surrogate_gradient,
    synaptic_normalization, SornNetwork,
};
pub use crate::kernels::stats::{
    compute_cvs, compute_isi, firing_rates, mean_firing_rate, population_rate_decode,
    spike_raster_to_csv, SpikeRaster,
};

// ─── Константы ─────────────────────────────────────────────

/// Default refractory period (ms).
/// Shiu et al. 2024: 2.2ms for Drosophila LIF.
pub const DEFAULT_TAU_REF: f32 = 2.2;

// ═══════════════════════════════════════════════════════════
//  E-Prop LIF — Eligibility Propagation (online SNN learning)
// ═══════════════════════════════════════════════════════════

/// E-Prop LIF neuron with eligibility traces + refractory period.
///
/// Based on: Bellec et al. "A solution to the learning dilemma for
/// recurrent networks of spiking neurons" (Nature Comms 2020)
///
/// Refractory period matches LIF semantics: during refractory,
/// v/eligibility/psi are frozen, refractory counter decrements.
pub struct EPropLIF {
    pub v: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub psi: Vec<f32>,
    pub refractory: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_m: f32,
    pub tau_e: f32,
    pub tau_ref: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub tau_surrogate: f32,
}

impl EPropLIF {
    pub fn new(n: usize, dt: f32, tau_m: f32, tau_e: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            v: vec![-65.0; n_pad],
            eligibility: vec![0.0; n_pad],
            psi: vec![0.0; n_pad],
            refractory: vec![0.0; n_pad],
            n: n_pad,
            dt,
            tau_m,
            tau_e,
            tau_ref: 3.0,
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -70.0,
            tau_surrogate: 2.0,
        }
    }

    /// One step of LIF dynamics + eligibility trace update + refractory.
    /// Uses SIMD `f32x8` kernel.
    ///
    /// Returns: number of output spikes
    pub fn step(&mut self, I: &[f32], spikes: &mut [i32], pre_spikes: &[i32]) -> usize {
        let n = self.n.min(I.len()).min(spikes.len()).min(pre_spikes.len());
        let n8 = n & !7;
        if n8 == 0 {
            return 0;
        }

        #[cfg(feature = "rayon")]
        let total: usize = self.v[..n8]
            .par_chunks_exact_mut(8)
            .zip(self.eligibility[..n8].par_chunks_exact_mut(8))
            .zip(self.psi[..n8].par_chunks_exact_mut(8))
            .zip(I[..n8].par_chunks_exact(8))
            .zip(spikes[..n8].par_chunks_exact_mut(8))
            .zip(pre_spikes[..n8].par_chunks_exact(8))
            .zip(self.refractory[..n8].par_chunks_exact_mut(8))
            .map(|((((((v8, e8), ps8), I8), sp8), pre8), refr8)| {
                crate::kernels::eprop::eprop_lif_step(
                    v8,
                    e8,
                    ps8,
                    I8,
                    sp8,
                    pre8,
                    refr8,
                    self.dt,
                    self.tau_m,
                    self.tau_e,
                    self.tau_ref,
                    self.v_rest,
                    self.v_thresh,
                    self.v_reset,
                    self.tau_surrogate,
                )
            })
            .sum();
        #[cfg(not(feature = "rayon"))]
        let total: usize = {
            let mut total = 0usize;
            for i in (0..n8).step_by(8) {
                total += crate::kernels::eprop::eprop_lif_step(
                    &mut self.v[i..i + 8],
                    &mut self.eligibility[i..i + 8],
                    &mut self.psi[i..i + 8],
                    &I[i..i + 8],
                    &mut spikes[i..i + 8],
                    &pre_spikes[i..i + 8],
                    &mut self.refractory[i..i + 8],
                    self.dt,
                    self.tau_m,
                    self.tau_e,
                    self.tau_ref,
                    self.v_rest,
                    self.v_thresh,
                    self.v_reset,
                    self.tau_surrogate,
                );
            }
            total
        };

        // Scalar tail (n & 7)
        let mut tail_count = 0;
        if n8 < n {
            let dt = self.dt;
            let tau_m = self.tau_m;
            let tau_e = self.tau_e;
            let tau_ref = self.tau_ref;
            let v_rest = self.v_rest;
            let v_thresh = self.v_thresh;
            let v_reset = self.v_reset;
            let tau_sg = self.tau_surrogate;
            for i in n8..n {
                if self.refractory[i] > 0.0 {
                    self.refractory[i] = (self.refractory[i] - dt).max(0.0);
                    spikes[i] = 0;
                    continue;
                }
                let decay_m = (-dt / tau_m).exp();
                self.v[i] = v_rest + (self.v[i] - v_rest) * decay_m + I[i] * (1.0 - decay_m);
                self.psi[i] = surrogate_gradient(self.v[i], v_thresh, tau_sg);
                let z_pre = if pre_spikes[i] != 0 { 1.0 } else { 0.0 };
                let e_decay = (-dt / tau_e).exp();
                self.eligibility[i] =
                    self.eligibility[i] * e_decay + self.psi[i] * z_pre * tau_e * (1.0 - e_decay);
                if self.v[i] >= v_thresh {
                    spikes[i] = 1;
                    tail_count += 1;
                    self.v[i] = v_reset;
                    self.refractory[i] = tau_ref;
                } else {
                    spikes[i] = 0;
                }
            }
        }

        total + tail_count
    }

    /// Apply E-Prop weight update: Δw = η · L_j · e_ji
    pub fn update_weights(&self, weights: &mut [f32], learning_signals: &[f32], lr: f32) {
        let n = self.n.min(weights.len()).min(learning_signals.len());
        let chunks = n / 8;
        if chunks > 0 {
            #[cfg(feature = "rayon")]
            weights[..chunks * 8]
                .par_chunks_exact_mut(8)
                .zip(self.eligibility[..chunks * 8].par_chunks_exact(8))
                .zip(learning_signals[..chunks * 8].par_chunks_exact(8))
                .for_each(|((w8, e8), l8)| {
                    for i in 0..8 {
                        w8[i] += lr * l8[i] * e8[i];
                    }
                });
            #[cfg(not(feature = "rayon"))]
            for i in 0..chunks * 8 {
                weights[i] += lr * learning_signals[i] * self.eligibility[i];
            }
        }
        for i in chunks * 8..n {
            weights[i] += lr * learning_signals[i] * self.eligibility[i];
        }
    }

    pub fn reset(&mut self) {
        self.v.fill(-65.0);
        self.eligibility.fill(0.0);
        self.psi.fill(0.0);
        self.refractory.fill(0.0);
    }
}

// ═══════════════════════════════════════════════════════════
//  GLIF3 / GLIF4 (Allen Institute)
// ═══════════════════════════════════════════════════════════

/// 1/x для x > 0, иначе 0 — сторож легаси-семантики τ ≤ 0 → «без эволюции».
#[inline(always)]
fn inv_or_zero(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / x
    } else {
        0.0
    }
}

/// Repack legacy 8-col GLIF3 params → GLIF3Model 9-col (кэшируется в state).
fn pack_glif3(params: &[f32], tau_ref: f32, n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 9];
    for i in 0..n {
        p[i] = params[i];
        for c in 1..5 {
            p[c * n + i] = params[c * n + i];
        }
        p[5 * n + i] = inv_or_zero(params[5 * n + i]);
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = params[7 * n + i];
        p[8 * n + i] = tau_ref;
    }
    p
}

/// Repack legacy 10-col GLIF4 params → GLIF4Model 11-col (кэшируется в state).
fn pack_glif4(params: &[f32], tau_ref: f32, n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 11];
    for i in 0..n {
        p[i] = params[i];
        for c in 1..5 {
            p[c * n + i] = params[c * n + i];
        }
        p[5 * n + i] = inv_or_zero(params[5 * n + i]);
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = params[7 * n + i];
        p[8 * n + i] = inv_or_zero(params[8 * n + i]);
        p[9 * n + i] = params[9 * n + i];
        p[10 * n + i] = tau_ref;
    }
    p
}

/// GLIF3 state — LIF with after-spike current (I_ahp).
#[derive(Debug, Clone)]
pub struct Glif3State {
    pub n: usize,
    /// Membrane potential `[n]` (mV)
    pub v: Vec<f32>,
    /// After-spike conductance `[n]`
    pub g_ahp: Vec<f32>,
    /// Total input current `[n]`
    pub Ii: Vec<f32>,
    /// Spike output `[n]`, 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters `[n]`
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
    /// Кэш pillars-лейаута (9-col GLIF3Model) — пересобирается при set_params.
    packed: Vec<f32>,
}

impl Glif3State {
    /// Create a new GLIF3 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
        }
        let packed = pack_glif3(&params, tau_ref, n);
        Self {
            n,
            v: vec![E_leak; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
            packed,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        let n = self.n;
        unsafe {
            crate::core::runner::run_neuron::<crate::zoo::neurons::GLIF3Model>(
                self.v.as_mut_ptr(),
                self.g_ahp.as_mut_ptr(),
                self.refractory.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.Ii.as_ptr(),
                self.spikes.as_mut_ptr(),
                self.packed.as_ptr(),
                n,
            );
        }
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak at column 2
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's parameters.
    pub fn set_params(
        &mut self,
        idx: usize,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
    ) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
            self.packed = pack_glif3(&self.params, self.tau_ref, n);
        }
    }
}

/// GLIF4 state — GLIF3 + threshold adaptation (θ).
#[derive(Debug, Clone)]
pub struct Glif4State {
    pub n: usize,
    /// Membrane potential `[n]` (mV)
    pub v: Vec<f32>,
    /// Threshold offset `[n]` (effective threshold = v_thresh_0 + θ)
    pub theta: Vec<f32>,
    /// After-spike conductance `[n]`
    pub g_ahp: Vec<f32>,
    /// Total input current `[n]`
    pub Ii: Vec<f32>,
    /// Spike output `[n]`, 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters `[n]`
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh_0, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
    /// Кэш pillars-лейаута (11-col GLIF4Model) — пересобирается при set_params.
    packed: Vec<f32>,
}

impl Glif4State {
    /// Create a new GLIF4 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 10];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh_0;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
            params[8 * n + j] = tau_theta;
            params[9 * n + j] = delta_theta;
        }
        let packed = pack_glif4(&params, tau_ref, n);
        Self {
            n,
            v: vec![E_leak; n],
            theta: vec![0.0; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
            packed,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        let n = self.n;
        unsafe {
            crate::core::runner::run_neuron::<crate::zoo::neurons::GLIF4Model>(
                self.v.as_mut_ptr(),
                self.theta.as_mut_ptr(),
                self.g_ahp.as_mut_ptr(),
                self.refractory.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.Ii.as_ptr(),
                self.spikes.as_mut_ptr(),
                self.packed.as_ptr(),
                n,
            );
        }
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak
        self.theta.fill(0.0);
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's GLIF4 parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn set_params(
        &mut self,
        idx: usize,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
    ) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh_0;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
            self.params[8 * n + idx] = tau_theta;
            self.params[9 * n + idx] = delta_theta;
            self.packed = pack_glif4(&self.params, self.tau_ref, n);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Short-Term Plasticity (Tsodyks-Markram) + LFP proxy
// ═══════════════════════════════════════════════════════════

/// Short-Term Plasticity synapse state per connection.
///
/// Based on: Tsodyks & Markram (1997) "The neural code between
/// neocortical pyramidal neurons depends on neurotransmitter release
/// probability" — PNAS.
///
/// Канон — `crate::ops::plasticity::stp::stp_step` (lane-функция):
/// Exponential Euler, SIMD-цикл + scalar tail — тот же lane-канон на f32
/// (нейрон 0 == нейрон tail бит-в-бит).
pub struct ShortTermPlasticity {
    pub x: Vec<f32>,
    pub u: Vec<f32>,
    pub tau_d: Vec<f32>,
    pub tau_f: Vec<f32>,
    pub u0: Vec<f32>,
    pub a: Vec<f32>,
    pub n: usize,
    pub dt: f32,
}

impl ShortTermPlasticity {
    /// Create STP with n synapses.
    pub fn new(n: usize, dt: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            x: vec![1.0; n_pad],
            u: vec![0.2; n_pad],
            tau_d: vec![200.0; n_pad],
            tau_f: vec![600.0; n_pad],
            u0: vec![0.2; n_pad],
            a: vec![1.0; n_pad],
            n: n_pad,
            dt,
        }
    }

    /// Create STP with heterogeneous (randomized) parameters.
    pub fn new_heterogeneous(n: usize, dt: f32, seed: u32) -> Self {
        let mut rng = FastRng::new(seed);
        let n_pad = n.div_ceil(8) * 8;
        let mut stp = Self::new(n_pad, dt);
        for i in 0..n_pad {
            stp.tau_d[i] = 100.0 + rng.next_f32() * 400.0;
            stp.tau_f[i] = 300.0 + rng.next_f32() * 700.0;
            stp.u0[i] = 0.05 + rng.next_f32() * 0.45;
            stp.a[i] = 0.5 + rng.next_f32() * 1.5;
        }
        stp
    }

    /// One step: update STP state and compute effective synaptic current.
    pub fn step(&mut self, pre_spikes: &[i32], I_out: &mut [f32]) {
        let n = self.n.min(pre_spikes.len()).min(I_out.len());
        if n == 0 {
            return;
        }
        let (x, u) = (&mut self.x[..n], &mut self.u[..n]);
        let (tau_d, tau_f, u0, a) = (
            &self.tau_d[..n],
            &self.tau_f[..n],
            &self.u0[..n],
            &self.a[..n],
        );
        let i_out = &mut I_out[..n];
        let pre = &pre_spikes[..n];

        let n8 = n & !7;
        let dt_s = f32x8::splat(self.dt);
        let zero_s = f32x8::splat(0.0);
        let one_s = f32x8::splat(1.0);

        for off in (0..n8).step_by(8) {
            unsafe {
                let uu = load8(u.as_ptr(), off);
                let xx = load8(x.as_ptr(), off);
                let sp = load8_i32(pre.as_ptr(), off);
                let td = load8(tau_d.as_ptr(), off).max(f32x8::splat(1e-10));
                let tf = load8(tau_f.as_ptr(), off).max(f32x8::splat(1e-10));
                let u0v = load8(u0.as_ptr(), off);
                let av = load8(a.as_ptr(), off);
                let (un, xn) = pillars_stp::stp_step(uu, xx, sp, u0v, one_s / td, one_s / tf, dt_s);
                // s = 1 где pre_spike, иначе 0; I_out += s·a·u_new·x_new
                let s = sp.cmp_gt(zero_s).blend(one_s, zero_s);
                let mut io = load8(i_out.as_ptr(), off);
                io += s * av * un * xn;
                store8(u.as_mut_ptr(), off, un);
                store8(x.as_mut_ptr(), off, xn);
                store8(i_out.as_mut_ptr(), off, io);
            }
        }
        // Скалярный tail — тот же lane-канон на f32.
        for i in n8..n {
            let s = if pre[i] != 0 { 1.0f32 } else { 0.0f32 };
            let (un, xn) = pillars_stp::stp_step(
                u[i],
                x[i],
                s,
                u0[i],
                1.0 / tau_d[i].max(1e-10),
                1.0 / tau_f[i].max(1e-10),
                self.dt,
            );
            u[i] = un;
            x[i] = xn;
            i_out[i] += s * a[i] * un * xn;
        }
    }

    pub fn reset(&mut self) {
        self.x.fill(1.0);
        self.u
            .iter_mut()
            .zip(self.u0.iter())
            .for_each(|(u, &u0)| *u = u0);
    }
}

/// Fast LCG RNG for initialization.
struct FastRng {
    state: u32,
}

impl FastRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 * crate::kernels::RNG_SCALE_24
    }
}

/// LFP proxy: aggregate synaptic activity into local field potential.
pub fn compute_lfp(
    spikes: &[i32],
    electrode_map: &[i32],
    lfp_out: &mut [f32],
    n_electrodes: usize,
) {
    let n_out = n_electrodes.min(lfp_out.len());
    lfp_out[..n_out].fill(0.0);
    let n = spikes.len().min(electrode_map.len());
    for i in 0..n {
        if spikes[i] != 0 {
            let e = electrode_map[i] as usize;
            if e < n_electrodes {
                lfp_out[e] += 1.0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Somato-Dendritic Integration — LIF soma + cable dendrites
// ═══════════════════════════════════════════════════════════

/// Somato-dendritic нейронная популяция.
///
/// Объединяет LIF сому с `CableDendriteState` (cable-compartment дендриты).
/// Поддерживает 7 типов рецепторов: AMPA, NMDA, GABA_A/B, nAChR, GluCl, mAChR.
#[derive(Debug)]
pub struct SomatoDendriticNeuron {
    /// Количество нейронов (кратно 8 для SIMD)
    pub n: usize,
    /// Соматический потенциал `[n]`
    pub v: Vec<f32>,
    /// Соматический ток `[n]` = I_ext + I_dend
    pub I_soma: Vec<f32>,
    /// Рефрактерные счётчики `[n]`
    pub refractory: Vec<f32>,
    /// Выходные спайки `[n]`, 0/1
    pub spikes: Vec<i32>,
    /// Дендритная популяция
    pub dendrite: CableDendriteState,
    /// Decay factors (предвычислены из dt)
    decay: DendriteDecayFactors,
    /// Параметры LIF [6 × n], column-major: [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref]
    /// (pillars LIFModel канон)
    params: Vec<f32>,
    /// Рефрактерный период (ms)
    pub tau_ref: f32,
}

impl SomatoDendriticNeuron {
    /// Создать популяцию из `n_neurons` нейронов.
    ///
    /// # Parameters
    /// - `n_neurons`: число нейронов (округляется вверх до кратного 8)
    /// - `config`: параметры дендритного кабеля
    /// - `dt`: шаг интегрирования (ms)
    /// - `tau_m`: мембранная постоянная (ms)
    /// - `v_rest`, `v_thresh`, `v_reset`: мембранные уровни (mV)
    /// - `tau_ref`: рефрактерный период (ms)
    pub fn new(
        n_neurons: usize,
        config: &DendriteConfig,
        dt: f32,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;

        // LIF params column-major [dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref]
        let mut params = vec![0.0f32; n * 6];
        for i in 0..n {
            params[i] = dt;
            let tm = if tau_m > 0.0 { tau_m } else { 1e-6 };
            params[n + i] = 1.0 / tm;
            params[2 * n + i] = v_rest;
            params[3 * n + i] = v_thresh;
            params[4 * n + i] = v_reset;
            params[5 * n + i] = tau_ref;
        }

        Self {
            n,
            v: vec![v_rest; n],
            I_soma: vec![0.0; n],
            refractory: vec![0.0; n],
            spikes: vec![0; n],
            dendrite: CableDendriteState::new(n, config),
            decay: DendriteDecayFactors::new(dt),
            params,
            tau_ref,
        }
    }

    /// Шаг симуляции.
    ///
    /// 1. Decay дендритных проводимостей
    /// 2. cable_step → I_dend
    /// 3. LIF step с I_total = I_ext + I_dend
    pub fn step(&mut self, I_ext: &[f32], dt: f32) {
        assert!(I_ext.len() >= self.n, "I_ext слишком короткий");

        // 1. Decay conductances
        self.dendrite.decay_conductances(&self.decay);

        // 2. Cable step: V_dend + I_dend
        self.dendrite.cable_step(&self.v, dt);

        // 3. Combine currents: I_soma = I_ext + I_dend
        for (i, s) in self.I_soma.iter_mut().enumerate().take(self.n) {
            *s = I_ext[i] + self.dendrite.I_dend[i];
        }

        // 4. LIF step for all neurons (pillars LIFModel канон)
        unsafe {
            crate::core::runner::run_neuron::<crate::zoo::neurons::LIFModel>(
                self.v.as_mut_ptr(),
                self.refractory.as_mut_ptr(), // state 1 (refractory)
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.I_soma.as_ptr(),
                self.spikes.as_mut_ptr(),
                self.params.as_ptr(),
                self.n,
            );
        }
    }

    /// Обновить LIF параметры нейрона `nrn`.
    pub fn set_lif_params(
        &mut self,
        nrn: usize,
        tau_m: f32,
        v_rest: f32,
        v_thresh: f32,
        v_reset: f32,
    ) {
        if nrn < self.n {
            let tm = if tau_m > 0.0 { tau_m } else { 1e-6 };
            self.params[self.n + nrn] = 1.0 / tm;
            self.params[2 * self.n + nrn] = v_rest;
            self.params[3 * self.n + nrn] = v_thresh;
            self.params[4 * self.n + nrn] = v_reset;
        }
    }

    // ─── Conductance jumps (forwarded to dendrite) ───────

    /// AMPA на компартмент `comp` нейрона `nrn`.
    #[inline]
    pub fn jump_ampa(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_ampa(nrn, comp, g);
    }
    /// NMDA на компартмент.
    #[inline]
    pub fn jump_nmda(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_nmda(nrn, comp, g);
    }
    /// GABA_A на компартмент.
    #[inline]
    pub fn jump_gaba_a(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_gaba_a(nrn, comp, g);
    }
    /// GABA_B на компартмент.
    #[inline]
    pub fn jump_gaba_b(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_gaba_b(nrn, comp, g);
    }
    /// nAChR (ацетилхолин — возбуждение в ЦНС насекомых) на компартмент.
    #[inline]
    pub fn jump_nachr(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_nachr(nrn, comp, g);
    }
    /// GluCl (глутамат-управляемые хлорные каналы — торможение у насекомых) на компартмент.
    #[inline]
    pub fn jump_glucl(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_glucl(nrn, comp, g);
    }
    /// mAChR (мускариновый — модуляция) на компартмент.
    #[inline]
    pub fn jump_machr(&mut self, nrn: usize, comp: usize, g: f32) {
        self.dendrite.jump_machr(nrn, comp, g);
    }

    // ─── Getters ─────────────────────────────────────────

    /// Потенциал компартмента `comp` нейрона `nrn`.
    #[inline]
    pub fn v_dend(&self, nrn: usize, comp: usize) -> f32 {
        self.dendrite.v_dend(nrn, comp)
    }
    /// Дендритный ток нейрона `nrn`.
    #[inline]
    pub fn dend_current(&self, nrn: usize) -> f32 {
        self.dendrite.dend_current(nrn)
    }
}

// ═══════════════════════════════════════════════════════════
//  Chordotonal Sensory Organ (Drosophila mechanoreceptor)
// ═══════════════════════════════════════════════════════════

/// Chordotonal sensory organ state — load/stretch → firing rate.
#[derive(Debug, Clone)]
pub struct ChordotonalState {
    pub n: usize,
    /// Receptor potential (mV), state variable
    pub v_sens: Vec<f32>,
    /// Adaptation variable, state variable
    pub adapt: Vec<f32>,
    /// Output firing rates (Hz)
    pub rates: Vec<f32>,
    /// Per-channel params (9 × n_padded)
    pub params: Vec<f32>,
    n_padded: usize,
    dt: f32,
}

/// Physiological profile for chordotonal sensor tuning.
#[derive(Debug, Clone, Copy)]
pub struct ChordotonalProfile {
    pub tau_sens: f32,
    pub tau_adapt: f32,
    pub b_adapt: f32,
    pub load_gain: f32,
    pub stretch_gain: f32,
    pub baseline: f32,
    pub threshold: f32,
    pub max_rate: f32,
}

impl ChordotonalProfile {
    /// Rapid-adapting phasic sensor — fires at onset of load, then decays.
    /// Typical for chordotonal organs detecting velocity.
    pub const fn phasic() -> Self {
        Self {
            tau_sens: 2.0,
            tau_adapt: 30.0,
            b_adapt: 0.8,
            load_gain: 5.0,
            stretch_gain: 2.0,
            baseline: -3.0,
            threshold: 0.0,
            max_rate: 200.0,
        }
    }

    /// Slow-adapting tonic sensor — steady firing proportional to load.
    /// Typical for campaniform sensilla detecting sustained load.
    pub const fn tonic() -> Self {
        Self {
            tau_sens: 10.0,
            tau_adapt: 500.0,
            b_adapt: 0.1,
            load_gain: 3.0,
            stretch_gain: 1.0,
            baseline: -2.0,
            threshold: 0.0,
            max_rate: 100.0,
        }
    }

    /// Mixed phasic-tonic (biological chordotonal response).
    /// Default for Drosophila leg chordotonal organs.
    pub const fn mixed() -> Self {
        Self {
            tau_sens: 5.0,
            tau_adapt: 100.0,
            b_adapt: 0.3,
            load_gain: 3.0,
            stretch_gain: 1.5,
            baseline: -2.0,
            threshold: 0.0,
            max_rate: 200.0,
        }
    }
}

impl Default for ChordotonalProfile {
    fn default() -> Self {
        Self::mixed()
    }
}

/// Builder for `ChordotonalState`.
#[derive(Debug, Clone)]
pub struct ChordotonalBuilder {
    n: usize,
    profile: ChordotonalProfile,
    dt: f32,
}

impl ChordotonalBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            profile: ChordotonalProfile::default(),
            dt: 1.0,
        }
    }

    pub fn with_profile(mut self, profile: ChordotonalProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn build(self) -> ChordotonalState {
        let n_padded = self.n.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n_padded * 9];
        let p = &self.profile;

        for i in 0..n_padded {
            params[i] = self.dt;
            params[n_padded + i] = p.tau_sens;
            params[2 * n_padded + i] = p.tau_adapt;
            params[3 * n_padded + i] = p.b_adapt;
            params[4 * n_padded + i] = p.load_gain;
            params[5 * n_padded + i] = p.stretch_gain;
            params[6 * n_padded + i] = p.baseline;
            params[7 * n_padded + i] = p.threshold;
            params[8 * n_padded + i] = p.max_rate;
        }

        ChordotonalState {
            n: self.n,
            v_sens: vec![0.0; n_padded],
            adapt: vec![0.0; n_padded],
            rates: vec![0.0; n_padded],
            params,
            n_padded,
            dt: self.dt,
        }
    }
}

impl ChordotonalState {
    /// Create new chordotonal sensor state with default (mixed) profile.
    pub fn new(n: usize) -> Self {
        ChordotonalBuilder::new(n).build()
    }

    /// Step sensor: convert loads/stretches to firing rates.
    ///
    /// * `loads` — contact forces, length `n` (normalized 0..1)
    /// * `stretches` — joint angle deviations (normalized -1..1)
    /// * `d_loads` — optional velocity (derivative of load), length `n`
    pub fn step(&mut self, loads: &[f32], stretches: &[f32], d_loads: Option<&[f32]>) {
        assert!(loads.len() >= self.n, "loads too short");
        assert!(stretches.len() >= self.n, "stretches too short");

        let dt_old = self.params[0];
        let dt_new = self.dt;
        if (dt_old - dt_new).abs() > 1e-6 {
            for i in 0..self.n_padded {
                self.params[i] = dt_new;
            }
            self.dt = dt_new;
        }

        // Pad inputs to n_padded
        let mut loads_padded = loads.to_vec();
        loads_padded.resize(self.n_padded, 0.0);
        let mut stretches_padded = stretches.to_vec();
        stretches_padded.resize(self.n_padded, 0.0);

        let (_d_loads_padded, d_loads_ptr) = if let Some(dl) = d_loads {
            let mut padded = dl.to_vec();
            padded.resize(self.n_padded, 0.0);
            let ptr = padded.as_ptr();
            (padded, ptr)
        } else {
            (vec![], std::ptr::null())
        };

        unsafe {
            crate::zoo::sensors::chordotonal_step(
                self.v_sens.as_mut_ptr(),
                self.adapt.as_mut_ptr(),
                self.rates.as_mut_ptr(),
                loads_padded.as_ptr(),
                stretches_padded.as_ptr(),
                d_loads_ptr,
                self.params.as_ptr(),
                self.n_padded,
            );
        }
    }

    /// Get firing rates for the first `n` sensors.
    pub fn firing_rates(&self) -> &[f32] {
        &self.rates[..self.n]
    }
}

// ═══════════════════════════════════════════════════════════
//  Dopamine-Modulated STDP — 3-factor plasticity for RL
// ═══════════════════════════════════════════════════════════

/// Dopamine-modulated STDP synapse with eligibility traces.
///
/// Based on: Izhikevich (2007) "Solving the distal reward problem
/// through linkage of STDP and dopamine signaling" — Cerebral Cortex.
///
/// Канон — `crate::kernels::dopamine` (SIMD + scalar tail внутри канона).
pub struct DopamineSTDP {
    pub trace_pre: Vec<f32>,
    pub trace_post: Vec<f32>,
    pub eligibility: Vec<f32>,
    pub n: usize,
    pub dt: f32,
    pub tau_pre: f32,
    pub tau_post: f32,
    pub tau_e: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub w_min: f32,
    pub w_max: f32,
}

impl DopamineSTDP {
    /// Create new DopamineSTDP with `n` synapses (padded to multiple of 8).
    ///
    /// Default params (from Izhikevich 2007):
    /// - τ_pre = 20 ms, τ_post = 20 ms, τ_e = 1000 ms
    /// - A_ltp = 0.1, A_ltd = 0.12, w_min = 0.0, w_max = 1.0
    pub fn new(n: usize, dt: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            trace_pre: vec![0.0; n_pad],
            trace_post: vec![0.0; n_pad],
            eligibility: vec![0.0; n_pad],
            n: n_pad,
            dt,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 1000.0,
            a_ltp: 0.1,
            a_ltd: 0.12,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// One step: update pre/post traces and eligibility trace.
    pub fn step(&mut self, pre_spikes: &[i32], post_spikes: &[i32]) {
        let n = self.n.min(pre_spikes.len()).min(post_spikes.len());
        if n == 0 {
            return;
        }
        crate::kernels::dopamine::dopamine_traces_step(
            &mut self.trace_pre[..n],
            &mut self.trace_post[..n],
            &mut self.eligibility[..n],
            &pre_spikes[..n],
            &post_spikes[..n],
            self.dt,
            self.tau_pre,
            self.tau_post,
            self.tau_e,
            self.a_ltp,
            self.a_ltd,
        );
    }

    /// Apply dopamine-modulated weight update: Δw = η · D · e.
    pub fn apply_dopamine(&self, weights: &mut [f32], dopamine: f32, lr: f32) {
        let n = self.n.min(weights.len());
        if n == 0 {
            return;
        }
        crate::kernels::dopamine::dopamine_weight_update(
            &mut weights[..n],
            &self.eligibility[..n],
            dopamine,
            lr,
            self.w_min,
            self.w_max,
        );
    }

    pub fn clear_eligibility(&mut self) {
        self.eligibility.fill(0.0);
    }

    pub fn reset(&mut self) {
        self.trace_pre.fill(0.0);
        self.trace_post.fill(0.0);
        self.eligibility.fill(0.0);
    }
}

// ═══════════════════════════════════════════════════════════
//  Circular delay buffer
// ═══════════════════════════════════════════════════════════

/// Circular delay buffer with SIMD read.
///
/// Отдельный от `api::delay::DelayBuffer` (сетевой ring buffer) — этот
/// класс обслуживает NMJ-задержки и tap-чтение (push/read/read_multi).
#[derive(Debug, Clone)]
pub struct DelayBuffer {
    buf: Vec<f32>,
    n: usize,
    pos: usize,
}

impl DelayBuffer {
    pub fn new(n: usize) -> Self {
        let padded = n.div_ceil(8) * 8;
        Self {
            buf: vec![0.0f32; padded],
            n,
            pos: 0,
        }
    }

    pub fn push(&mut self, data: &[f32]) {
        for (i, &v) in data.iter().enumerate() {
            let idx = (self.pos + i) % self.n;
            self.buf[idx] = v;
        }
        self.pos = (self.pos + data.len()) % self.n;
    }

    pub fn read(&self, delay: usize) -> Vec<f32> {
        let n_out = self.n.min(8);
        let mut out = vec![0.0f32; n_out];
        let read_pos = (self.pos + self.n - delay % self.n) % self.n;
        for (i, o) in out.iter_mut().enumerate() {
            let idx = (read_pos + i) % self.n;
            *o = self.buf[idx];
        }
        out
    }

    pub fn read_multi(&self, delays: &[i32]) -> Vec<f32> {
        let n_taps = delays.len().min(self.n).min(8);
        let mut out = vec![0.0f32; n_taps];
        let wrapped: Vec<i32> = delays
            .iter()
            .map(|&d| {
                let abs_d = d.unsigned_abs() as usize % self.n;
                if d < 0 {
                    -(abs_d as i32)
                } else {
                    abs_d as i32
                }
            })
            .collect();
        crate::kernels::delay::delay_buffer_multi_tap(&mut out, &self.buf[..self.n], &wrapped);
        out
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glif3_state_basic() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF3 state: should fire with I=30");
        assert!(state.g_ahp[0] > 0.0, "GLIF3 state: g_ahp should accumulate");
    }

    #[test]
    fn glif4_state_basic() {
        let mut state = Glif4State::new(
            8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 20.0, 3.0, 2.0,
        );
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF4 state: should fire with I=30");
        assert!(state.theta[0] > 0.0, "GLIF4 state: theta should accumulate");
        assert!(state.g_ahp[0] > 0.0, "GLIF4 state: g_ahp should accumulate");
    }

    #[test]
    fn glif3_state_reset() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        for _ in 0..100 {
            state.step(Some(&i_ext));
        }
        assert!(state.g_ahp[0] > 0.0);

        state.reset();
        assert_eq!(state.v[0], -65.0);
        assert_eq!(state.g_ahp[0], 0.0);
        assert_eq!(state.refractory[0], 0.0);
    }

    #[test]
    fn eprop_lif_fires_with_input() {
        let mut pop = EPropLIF::new(8, 1.0, 10.0, 20.0);
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let pre = vec![1i32; 8];

        let mut total = 0usize;
        for _ in 0..200 {
            total += pop.step(&i_ext, &mut spikes, &pre);
        }
        assert!(total > 0, "EPropLIF should fire with I=30");
        assert!(
            pop.eligibility.iter().any(|&e| e > 0.0),
            "eligibility should accumulate"
        );
    }

    #[test]
    fn eprop_update_weights() {
        let pop = EPropLIF::new(8, 1.0, 10.0, 20.0);
        let mut w = vec![0.5f32; 8];
        let l = vec![1.0f32; 8];
        pop.update_weights(&mut w, &l, 0.1);
        assert!((w[0] - 0.5).abs() < 1e-6, "no eligibility → no change");
    }

    #[test]
    fn test_soma_dendrite_ampa_drives_spike() {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: 10.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut pop = SomatoDendriticNeuron::new(8, &config, 0.5, 20.0, -65.0, -54.0, -65.0, 3.0);
        pop.set_lif_params(0, 20.0, -65.0, -60.0, -65.0);
        let i_ext = vec![0.0f32; 8];

        let mut spike_count = 0i32;
        for _ in 0..300 {
            pop.jump_ampa(0, 0, 0.20);
            pop.step(&i_ext, 0.1);
            spike_count += pop.spikes[0];
        }
        assert!(
            spike_count > 0,
            "Нейрон 0 должен спайковать от дендритного AMPA, got {spike_count}"
        );
        assert_eq!(pop.spikes[1], 0, "Нейрон 1 без входа не должен спайковать");
    }

    #[test]
    fn test_chordotonal_state_basic() {
        let mut s = ChordotonalState::new(8);
        let loads = vec![0.5f32; 8];
        let stretches = vec![0.0f32; 8];
        s.step(&loads, &stretches, None);
        assert!(s.rates[0] > 0.0, "Load should produce firing rate");
    }

    #[test]
    fn test_stp_step_updates_state() {
        let mut stp = ShortTermPlasticity::new(8, 1.0);
        let pre = vec![1i32; 8];
        let mut out = vec![0.0f32; 8];
        stp.step(&pre, &mut out);
        assert!(out[0] > 0.0, "spike should produce current");
        assert!(stp.x[0] < 1.0, "x should deplete after spike");
    }

    #[test]
    fn dopamine_stdp_basic() {
        let mut stdp = DopamineSTDP::new(8, 1.0);
        // Co-activity with simultaneous pre+post: ltd (0.12) > ltp (0.1) →
        // отрицательный eligibility — канон dopamine_traces_step.
        let pre = vec![1i32; 8];
        let post = vec![1i32; 8];
        stdp.step(&pre, &post);
        assert!(
            stdp.eligibility[0] != 0.0,
            "co-activity should create eligibility"
        );

        let mut w = vec![0.5f32; 8];
        stdp.apply_dopamine(&mut w, 1.0, 0.1);
        assert!(w[0] != 0.5, "dopamine should update weight");
    }

    #[test]
    fn delay_buffer_push_read() {
        let mut buf = DelayBuffer::new(8);
        buf.push(&[1.0f32; 8]);
        let delayed = buf.read(2);
        assert!(
            (delayed[0] - 1.0).abs() < 1e-6,
            "read(2) should return pushed data"
        );
    }

    #[test]
    fn ttfs_encoder_basic() {
        let mut enc = TtfsEncoder::new(4, 10);
        let mut spikes = vec![0i32; 4];
        assert!(
            enc.encode_step(&[1.0f32; 4], &mut spikes),
            "value=1 → fire at step 0"
        );
        assert_eq!(spikes[0], 1);
    }

    #[test]
    fn spike_raster_record() {
        let mut raster = SpikeRaster::new();
        let spikes = [1i32, 0, 0, 1];
        raster.record(0, &spikes, 4);
        let (dense, steps) = raster.to_dense(4);
        assert_eq!(steps, 1);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[3], 1);
        assert_eq!(raster.total_spikes(), 2);
    }
}
