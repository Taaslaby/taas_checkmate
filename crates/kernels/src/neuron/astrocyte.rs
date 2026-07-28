//! 🧫 Astrocyte & Gap-Junction Glial Wave Kernel
//!
//! Полная модель астроцита для VNC closed-loop симуляции.
//!
//! ## Компоненты
//!
//! | Компонент | Переменная | Эффект | τ |
//! |-----------|-----------|--------|---|
//! | Медленная активность | `A(t)` | Synaptic gain: `g_syn × (1 + α·A)` | ~1000 ms |
//! | Ca²⁺ волна | `Ca` | Volume gain: `g_syn × (1 + μ·Ca)` | ~1000 ms |
//! | Gap Junction | `V_gap` | Электрическая связь: `I_gap = g_gap · Σ(V_j − V_i)` | — |
//! | K⁺ буферизация | `K_ext` | Метаболическая усталость: `V_th += κ·(K_ext − K_0)` | ~500 ms |
//! | EAAT транспорт | `EAAT_occ` | Глутаматный клиренс: снижает `g_glucl` при спайках | ~10 ms |
//!
//! ## Физиология
//!
//! Астроциты в VNC Drosophila выполняют три ключевые функции:
//!
//! 1. **Система шумоподавления** — EAAT транспортеры откачивают избыток глутамата
//!    из синаптической щели, превращая непрерывный шум в чистые пачки спайков.
//!
//! 2. **Метаболический переключатель походки** — K⁺ spatial buffering: после
//!    длительной активности (400+ Hz) [K⁺]_o растёт, V_thresh ползёт вверх,
//!    и муха переключается с трипода на волновую походку (wave gait).
//!
//! 3. **Глобальный тонус** — Ca²⁺ волна через gap junctions меняет объёмную
//!    нейромодуляцию, синхронизируя тонус всех сегментов VNC.

#![allow(non_snake_case)]

use super::{load8, store8};
use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  AstrocyteState
// ═══════════════════════════════════════════════════════════

/// Полное состояние сети астроцитов.
///
/// Все буферы длины `n_padded` (выровнены до %8 для SIMD).
///
/// # Жизненный цикл (один шаг симуляции)
///
/// 1. `astro.step(spikes, &gap_pairs, dt)` — обновить всю кинетику
/// 2. `astro.apply_to_kinetics(&mut kin, dt)` — промодулировать синаптические
///    проводимости (EAAT клиренс + volume gain + synaptic gain)
/// 3. `astro.modulate_v_thresh(&mut v_thresh)` — применить K⁺ усталость к порогам
#[derive(Debug, Clone)]
pub struct AstrocyteState {
    // ─── Geometry ─────────────────────────────────────
    pub n: usize,
    pub n_padded: usize,

    // ─── State variables ──────────────────────────────
    /// Медленная активность A(t) — synaptic gain modulation
    pub a: Vec<f32>,
    /// Intracellular Ca²⁺ — gap junction wave propagation
    pub ca: Vec<f32>,
    /// Gap junction membrane potential (mV)
    pub v_gap: Vec<f32>,
    /// Extracellular [K⁺] (mM) — metabolic fatigue
    pub k_ext: Vec<f32>,
    /// EAAT transporter occupancy [0..1] — glutamate clearance
    pub eaat_occ: Vec<f32>,

    // ─── Time constants ───────────────────────────────
    /// τ for A(t) slow activity (ms). Default: 1000 (1 second)
    pub tau_astro: f32,
    /// τ for Ca²⁺ decay (ms). Default: 1000 (1 second)
    pub tau_ca: f32,
    /// τ for K⁺ clearance (ms). Default: 500 (0.5 second)
    pub tau_k: f32,
    /// τ for EAAT recovery (ms). Default: 10 (10 ms — fast)
    pub tau_eaat: f32,
    /// τ for V_gap relaxation back to baseline (ms). Default: 100
    pub tau_gap: f32,

    // ─── Gains ────────────────────────────────────────
    /// Synaptic gain modulation strength: `gain = 1 + α·A`
    pub alpha: f32,
    /// Volume transmission gain: `gain *= 1 + μ·Ca`
    pub mu: f32,
    /// Gap junction conductance
    pub g_gap: f32,
    /// EAAT clearance strength per step: `g_glucl *= (1 − ε · EAAT_occ · dt)`
    pub eaat_strength: f32,

    // ─── K⁺ fatigue ──────────────────────────────────
    /// Baseline extracellular [K⁺] (mM). Default: 4.0
    pub k_baseline: f32,
    /// V_thresh shift per [K⁺] delta (mV/mM). Default: 2.0
    pub k_sensitivity: f32,
    /// [K⁺] released per spike (mM / spike). Default: 0.05
    pub k_per_spike: f32,

    // ─── Ca²⁺ dynamics ───────────────────────────────
    /// Ca²⁺ influx per pre-synaptic spike. Default: 1.0
    pub ca_per_spike: f32,
    /// Ca²⁺ baseline. Default: 0.1
    pub ca_baseline: f32,
}

impl AstrocyteState {
    /// Create new astrocyte state for `n_neurons`.
    ///
    /// Default parameters (Drosophila VNC):
    /// - τ_astro = 1000 ms (slow activity)
    /// - α = 0.1 (10% gain modulation per unit A)
    /// - All other params use biological defaults
    #[inline]
    pub fn new(n_neurons: usize, tau_astro: f32, alpha: f32) -> Self {
        let n_padded = n_neurons.div_ceil(8) * 8;
        Self {
            n: n_neurons,
            n_padded,
            a: vec![0.0; n_padded],
            ca: vec![0.1; n_padded],
            v_gap: vec![-70.0; n_padded],
            k_ext: vec![4.0; n_padded],
            eaat_occ: vec![0.0; n_padded],
            tau_astro,
            tau_ca: 1000.0,
            tau_k: 500.0,
            tau_eaat: 10.0,
            tau_gap: 100.0,
            alpha,
            mu: 0.5,
            g_gap: 0.2,
            eaat_strength: 0.5,
            k_baseline: 4.0,
            k_sensitivity: 2.0,
            k_per_spike: 0.05,
            ca_per_spike: 1.0,
            ca_baseline: 0.1,
        }
    }

    /// Reset all state variables to their initial values.
    #[inline]
    pub fn reset(&mut self) {
        self.a.fill(0.0);
        self.ca.fill(0.1);
        self.v_gap.fill(-70.0);
        self.k_ext.fill(4.0);
        self.eaat_occ.fill(0.0);
    }

    /// Full dynamics step — updates all 5 state variables.
    ///
    /// Uses `f32x8` SIMD for the per-neuron loop, scalar for gap junction coupling.
    ///
    /// # Dynamics (Forward Euler)
    ///
    /// **A(t)** — slow activity:
    /// ```text
    /// A += dt · (−A / τ_astro + spikes · ca_per_spike)
    /// ```
    ///
    /// **Ca²⁺** — integrates spikes + gap junction drive:
    /// ```text
    /// Ca += dt · (−(Ca − Ca_base) / τ_ca + spikes · ca_per_spike)
    /// ```
    ///
    /// **V_gap** — gap junction electrical coupling:
    /// ```text
    /// V_gap[i] += dt · (−(V_gap[i] − V_rest) / τ_gap + g_gap · Σ(V_gap[j] − V_gap[i]))
    /// ```
    ///
    /// **K_ext** — potassium accumulation + clearance:
    /// ```text
    /// K_ext = K_baseline + (K_ext − K_baseline) · exp(−dt / τ_K) + spikes · K_per_spike
    /// ```
    ///
    /// **EAAT_occ** — transporter saturation:
    /// ```text
    /// EAAT_occ += dt · (−EAAT_occ / τ_EAAT + spikes · 0.01)
    /// EAAT_occ = clamp(EAAT_occ, 0, 1)
    /// ```
    ///
    /// # Args
    /// * `spikes` — per-neuron spike indicator (0 or 1), length ≥ n
    /// * `gap_pairs` — list of (i, j) gap junction connections
    /// * `dt` — time step in milliseconds
    pub fn step(&mut self, spikes: &[i32], gap_pairs: &[(usize, usize)], dt: f32) {
        let n = self.n.min(spikes.len());
        let n8 = n & !7;

        // Precompute decay factors
        let decay_a = f32x8::splat((-dt / self.tau_astro.max(1.0)).exp());
        let decay_ca = f32x8::splat((-dt / self.tau_ca.max(1.0)).exp());
        let decay_k = f32x8::splat((-dt / self.tau_k.max(1.0)).exp());
        let decay_eaat = f32x8::splat((-dt / self.tau_eaat.max(1.0)).exp());
        let decay_gap = f32x8::splat((-dt / self.tau_gap.max(1.0)).exp());

        let influx_ca = f32x8::splat(dt * self.ca_per_spike);
        let influx_k = f32x8::splat(self.k_per_spike);
        let eaat_influx = f32x8::splat(0.01 * dt);
        let ca_base = f32x8::splat(self.ca_baseline);
        let k_base = f32x8::splat(self.k_baseline);
        let v_rest = f32x8::splat(-70.0);
        let zero = f32x8::splat(0.0);
        let one = f32x8::splat(1.0);

        unsafe {
            for off in (0..n8).step_by(8) {
                // Load spikes as f32
                let sp = f32x8::new([
                    spikes[off] as f32,
                    spikes[off + 1] as f32,
                    spikes[off + 2] as f32,
                    spikes[off + 3] as f32,
                    spikes[off + 4] as f32,
                    spikes[off + 5] as f32,
                    spikes[off + 6] as f32,
                    spikes[off + 7] as f32,
                ]);

                // === A(t): slow activity ===
                let a_cur = load8(self.a.as_ptr(), off);
                let a_new = a_cur * decay_a + sp * influx_ca;
                store8(self.a.as_mut_ptr(), off, a_new.max(zero).min(f32x8::splat(10.0)));

                // === Ca²⁺: integrate spikes, decay to baseline ===
                let ca_cur = load8(self.ca.as_ptr(), off);
                let ca_decay = ca_cur * decay_ca;
                let ca_trend = ca_base * (one - decay_ca);
                let ca_new = ca_decay + ca_trend + sp * influx_ca;
                store8(self.ca.as_mut_ptr(), off, ca_new.max(zero).min(f32x8::splat(5.0)));

                // === K_ext: integrate spikes, decay to baseline ===
                let k_cur = load8(self.k_ext.as_ptr(), off);
                let k_new = k_base + (k_cur - k_base) * decay_k + sp * influx_k;
                store8(self.k_ext.as_mut_ptr(), off, k_new.max(zero).min(f32x8::splat(20.0)));

                // === EAAT_occ: integrate spikes, fast decay ===
                let eaat_cur = load8(self.eaat_occ.as_ptr(), off);
                let eaat_new = eaat_cur * decay_eaat + sp * eaat_influx;
                store8(self.eaat_occ.as_mut_ptr(), off, eaat_new.max(zero).min(one));

                // === V_gap: relaxation to baseline ===
                let vg_cur = load8(self.v_gap.as_ptr(), off);
                let vg_relax = v_rest + (vg_cur - v_rest) * decay_gap;
                store8(self.v_gap.as_mut_ptr(), off, vg_relax); // gap junction coupling added below
            }
        }

        // Scalar tail (n % 8 != 0)
        for i in n8..n {
            let sp = spikes[i] as f32;

            // A(t)
            let a_cur = self.a[i];
            self.a[i] = (a_cur * (-dt / self.tau_astro.max(1.0)).exp()
                + sp * dt * self.ca_per_spike)
                .max(0.0)
                .min(10.0);

            // Ca²⁺
            let ca_cur = self.ca[i];
            let dc = (-dt / self.tau_ca.max(1.0)).exp();
            self.ca[i] = (ca_cur * dc + self.ca_baseline * (1.0 - dc) + sp * dt * self.ca_per_spike)
                .max(0.0)
                .min(5.0);

            // K_ext
            let k_cur = self.k_ext[i];
            let dk = (-dt / self.tau_k.max(1.0)).exp();
            self.k_ext[i] = (self.k_baseline + (k_cur - self.k_baseline) * dk + sp * self.k_per_spike)
                .max(0.0)
                .min(20.0);

            // EAAT_occ
            let e_cur = self.eaat_occ[i];
            let de = (-dt / self.tau_eaat.max(1.0)).exp();
            self.eaat_occ[i] = (e_cur * de + sp * 0.01 * dt).max(0.0).min(1.0);

            // V_gap relaxation
            let vg_cur = self.v_gap[i];
            let dg = (-dt / self.tau_gap.max(1.0)).exp();
            self.v_gap[i] = -70.0 + (vg_cur + 70.0) * dg;
        }

        // === Gap Junction coupling (V_gap) ===
        // Scalar loop over sparse gap junction pairs
        if self.g_gap > 0.0 && !gap_pairs.is_empty() {
            let mut i_gap = vec![0.0f32; n];
            for &(i, j) in gap_pairs {
                if i < n && j < n {
                    let diff = self.v_gap[j] - self.v_gap[i];
                    let current = self.g_gap * diff;
                    i_gap[i] += current;
                    i_gap[j] -= current;
                }
            }
            for i in 0..n {
                self.v_gap[i] += i_gap[i] * dt * 0.1; // 0.1 = scale factor
            }
        }
    }

    /// Compute total modulation gain for all neurons: `gain[i] = (1 + α·A[i]) · (1 + μ·Ca[i])`.
    ///
    /// Returns Vec<f32> of length `n_padded` (padded with 1.0).
    #[inline]
    pub fn compute_gain(&self) -> Vec<f32> {
        let mut gain = vec![1.0; self.n_padded];
        let alpha_s = f32x8::splat(self.alpha);
        let mu_s = f32x8::splat(self.mu);
        let one = f32x8::splat(1.0);
        let n8 = self.n & !7;

        unsafe {
            for off in (0..n8).step_by(8) {
                let a_v = load8(self.a.as_ptr(), off);
                let ca_v = load8(self.ca.as_ptr(), off);
                let g = (one + a_v * alpha_s) * (one + ca_v * mu_s);
                store8(gain.as_mut_ptr(), off, g);
            }
        }
        for i in n8..self.n {
            gain[i] = (1.0 + self.a[i] * self.alpha) * (1.0 + self.ca[i] * self.mu);
        }

        gain
    }

    /// Apply K⁺-induced metabolic fatigue: `V_thresh[i] += k_sensitivity · (K_ext[i] − K_baseline)`.
    ///
    /// Modifies `v_thresh` in-place (must be length ≥ n).
    #[inline]
    pub fn modulate_v_thresh(&self, v_thresh: &mut [f32]) {
        let n = self.n.min(v_thresh.len());
        let n8 = n & !7;
        let sens = f32x8::splat(self.k_sensitivity);
        let k_base = f32x8::splat(self.k_baseline);

        unsafe {
            for off in (0..n8).step_by(8) {
                let k = load8(self.k_ext.as_ptr(), off);
                let shift = (k - k_base) * sens;
                let vt = load8(v_thresh.as_ptr(), off);
                store8(v_thresh.as_mut_ptr(), off, vt + shift);
            }
        }
        for i in n8..n {
            v_thresh[i] += self.k_sensitivity * (self.k_ext[i] - self.k_baseline);
        }
    }

    /// Apply astrocyte modulation directly to `Ii` (input current) — volume transmission.
    ///
    /// Reduces global excitability: `Ii[i] *= (1 + μ·Ca[i])⁻¹` (inhibitory volume transmission).
    /// Or, if `excitatory=true`: `Ii[i] *= (1 + μ·Ca[i])` (excitatory).
    #[inline]
    pub fn modulate_Ii(&self, Ii: &mut [f32], excitatory: bool) {
        let n = self.n.min(Ii.len());
        let n8 = n & !7;
        let mu_s = f32x8::splat(self.mu);
        let one = f32x8::splat(1.0);

        unsafe {
            for off in (0..n8).step_by(8) {
                let ca_v = load8(self.ca.as_ptr(), off);
                let gain = if excitatory {
                    one + ca_v * mu_s
                } else {
                    let g = one + ca_v * mu_s;
                    one / g.max(f32x8::splat(0.1))
                };
                let ii = load8(Ii.as_ptr(), off);
                store8(Ii.as_mut_ptr(), off, ii * gain);
            }
        }
        for i in n8..n {
            let vol_gain = self.ca[i] * self.mu;
            if excitatory {
                Ii[i] *= 1.0 + vol_gain;
            } else {
                Ii[i] /= (1.0 + vol_gain).max(0.1);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Astrocyte modulation helpers for SynapticKinetics
// ═══════════════════════════════════════════════════════════
//
// These are public methods that synapse.rs delegates to.
// Keep method signatures stable.

impl AstrocyteState {
    /// Apply EAAT glutamate clearance to GluCl conductances:
    /// `g_glucl[i] *= (1 − eaat_strength · EAAT_occ[i] · dt)`
    ///
    /// Uses `f32x8` SIMD. Clamps gain to [0, 1].
    #[inline]
    pub fn apply_eaat_clearance(&self, g_glucl: &mut [f32], dt: f32) {
        let n = self.n.min(g_glucl.len());
        let n8 = n & !7;
        let strength = f32x8::splat((self.eaat_strength * dt).min(1.0));
        let one = f32x8::splat(1.0);

        unsafe {
            for off in (0..n8).step_by(8) {
                let eaat = load8(self.eaat_occ.as_ptr(), off);
                let g_cur = load8(g_glucl.as_ptr(), off);
                let gain = (one - eaat * strength).max(f32x8::splat(0.0));
                store8(g_glucl.as_mut_ptr(), off, g_cur * gain);
            }
        }
        for i in n8..n {
            let clearance = (1.0 - self.eaat_strength * self.eaat_occ[i] * dt).max(0.0);
            g_glucl[i] *= clearance;
        }
    }

    /// Apply synaptic gain modulation to all 7 conductance arrays:
    /// `g_r[i] *= (1 + α·A[i]) · (1 + μ·Ca[i])`
    ///
    /// Like `compute_gain()` but applies directly to pre-allocated arrays.
    /// Internal — used by `apply_to_kinetics`.
    #[inline]
    pub(crate) fn apply_synaptic_gain_raw(
        &self,
        g_ampa: &mut [f32],
        g_nmda: &mut [f32],
        g_gaba_a: &mut [f32],
        g_gaba_b: &mut [f32],
        g_nachr: &mut [f32],
        g_glucl: &mut [f32],
        g_machr: &mut [f32],
    ) {
        let n = self.n.min(g_ampa.len());
        let n8 = n & !7;
        let alpha_s = f32x8::splat(self.alpha);
        let mu_s = f32x8::splat(self.mu);
        let one = f32x8::splat(1.0);

        unsafe {
            for off in (0..n8).step_by(8) {
                let a_v = load8(self.a.as_ptr(), off);
                let ca_v = load8(self.ca.as_ptr(), off);
                let gain = (one + a_v * alpha_s) * (one + ca_v * mu_s);

                store8(g_ampa.as_mut_ptr(), off, load8(g_ampa.as_ptr(), off) * gain);
                store8(g_nmda.as_mut_ptr(), off, load8(g_nmda.as_ptr(), off) * gain);
                store8(g_gaba_a.as_mut_ptr(), off, load8(g_gaba_a.as_ptr(), off) * gain);
                store8(g_gaba_b.as_mut_ptr(), off, load8(g_gaba_b.as_ptr(), off) * gain);
                store8(g_nachr.as_mut_ptr(), off, load8(g_nachr.as_ptr(), off) * gain);
                store8(g_glucl.as_mut_ptr(), off, load8(g_glucl.as_ptr(), off) * gain);
                store8(g_machr.as_mut_ptr(), off, load8(g_machr.as_ptr(), off) * gain);
            }
        }
        for i in n8..n {
            let gain = (1.0 + self.a[i] * self.alpha) * (1.0 + self.ca[i] * self.mu);
            g_ampa[i] *= gain;
            g_nmda[i] *= gain;
            g_gaba_a[i] *= gain;
            g_gaba_b[i] *= gain;
            g_nachr[i] *= gain;
            g_glucl[i] *= gain;
            g_machr[i] *= gain;
        }
    }

    /// Full modulation: EAAT clearance + synaptic gain.
    ///
    /// Intended to be called once per simulation step, after `step()` and before
    /// synaptic current computation.
    ///
    /// # Lifecycle
    ///
    /// ```ignore
    /// astro.step(spikes, &gap_pairs, dt);
    /// astro.apply_to_kinetics(&mut kin, dt);
    /// kin.decay_all(&factors);
    /// // ... propagate spikes ...
    /// kin.compute_currents(&v, &mut I_syn);
    /// ```
    #[inline]
    pub fn apply_to_kinetics(
        &self,
        kin: &mut crate::neuron::synapse::SynapticKinetics,
        dt: f32,
    ) {
        let n = self.n.min(kin.n);
        if n == 0 {
            return;
        }
        // Step 1: EAAT clearance on GluCl
        self.apply_eaat_clearance(&mut kin.g_glucl, dt);
        // Step 2: Synaptic gain on all receptors
        self.apply_synaptic_gain_raw(
            &mut kin.g_ampa,
            &mut kin.g_nmda,
            &mut kin.g_gaba_a,
            &mut kin.g_gaba_b,
            &mut kin.g_nachr,
            &mut kin.g_glucl,
            &mut kin.g_machr,
        );
    }
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── 1. Basic lifecycle ──────────────────────────

    #[test]
    fn test_astrocyte_new_and_reset() {
        let astro = AstrocyteState::new(8, 1000.0, 0.1);
        assert_eq!(astro.n, 8);
        assert_eq!(astro.n_padded, 8);
        assert!(astro.a.iter().all(|&v| v == 0.0));
        assert!(astro.ca.iter().all(|&v| (v - 0.1).abs() < 1e-6));
        assert!(astro.k_ext.iter().all(|&v| (v - 4.0).abs() < 1e-6));
        assert!(astro.eaat_occ.iter().all(|&v| v == 0.0));
        assert!(astro.v_gap.iter().all(|&v| (v + 70.0).abs() < 1e-6));

        let mut astro = astro;
        astro.a[0] = 5.0;
        astro.k_ext[0] = 10.0;
        astro.reset();
        assert_eq!(astro.a[0], 0.0);
        assert!((astro.k_ext[0] - 4.0).abs() < 1e-6);
    }

    // ─── 2. A(t) integrates spikes slowly ────────────

    #[test]
    fn test_astrocyte_a_slow_integration() {
        let mut astro = AstrocyteState::new(8, 1000.0, 0.1);
        let spikes = vec![1i32; 8];
        let gap_pairs = vec![];

        for _ in 0..10 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        assert!(astro.a[0] > 5.0, "A should accumulate: {}", astro.a[0]);

        // Stop spikes — A should decay slowly
        let spikes = vec![0i32; 8];
        for _ in 0..10 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }
        assert!(astro.a[0] < 10.0, "A should decay: {}", astro.a[0]);
        assert!(astro.a[0] > 1.0, "A should still be >0: {}", astro.a[0]);
    }

    // ─── 3. Ca²⁺ integrates spikes ──────────────────

    #[test]
    fn test_astrocyte_ca_integration() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        let spikes = vec![1i32; 4];
        let gap_pairs = vec![];

        for _ in 0..5 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        assert!(astro.ca[0] > 0.5, "Ca should rise: {}", astro.ca[0]);

        // Stop spikes — Ca should decay towards baseline
        let spikes = vec![0i32; 4];
        for _ in 0..5 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }
        // After 5ms with τ_ca=1000ms: minimal decay, should still be > 0.5
        assert!(astro.ca[0] > 0.5, "Ca should remain elevated: {}", astro.ca[0]);
    }

    // ─── 4. K_ext integrates spikes ──────────────────

    #[test]
    fn test_astrocyte_k_ext_integration() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        astro.k_per_spike = 0.5; // Strong K⁺ release for test
        astro.tau_k = 100.0;
        let spikes = vec![1i32; 4];
        let gap_pairs = vec![];

        for _ in 0..5 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        assert!(astro.k_ext[0] > 4.5, "K should rise: {}", astro.k_ext[0]);

        // Stop spikes — K should decay towards baseline
        let spikes = vec![0i32; 4];
        for _ in 0..10 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }
        // After 10ms with τ_k=100ms: K should have decayed by ~10%
        // Initial was ~7.0 (after 5 spikes with k_per_spike=0.5)
        // Should be lower but still above baseline
        assert!(
            astro.k_ext[0] < 7.0,
            "K should decay below peak: {}",
            astro.k_ext[0]
        );
        assert!(
            astro.k_ext[0] > 3.0,
            "K should still be above 0: {}",
            astro.k_ext[0]
        );
    }

    // ─── 5. EAAT integrates spikes ──────────────────

    #[test]
    fn test_astrocyte_eaat_integration() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        astro.tau_eaat = 100.0; // Longer decay for test
        let spikes = vec![1i32; 4];
        let gap_pairs = vec![];

        // Single spike
        astro.step(&spikes, &gap_pairs, 1.0);
        assert!(astro.eaat_occ[0] > 0.0, "EAAT should rise: {}", astro.eaat_occ[0]);

        // Stop spikes — EAAT should decay
        let spikes = vec![0i32; 4];
        for _ in 0..5 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }
        assert!(
            astro.eaat_occ[0] < 0.01,
            "EAAT should decay to near 0: {}",
            astro.eaat_occ[0]
        );
    }

    // ─── 6. Gap junction coupling ────────────────────

    #[test]
    fn test_astrocyte_gap_junction_and_ca_wave() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        astro.g_gap = 0.2;
        let gap_pairs = vec![(0, 1), (1, 2), (2, 3)];

        // Depolarize first node
        astro.v_gap[0] = -30.0;
        let spikes = vec![1, 0, 0, 0];
        let empty_gap = vec![];

        // V_gap relaxation without gap coupling first
        for _ in 0..50 {
            astro.step(&spikes, &empty_gap, 1.0);
        }

        // With gap coupling — wave should propagate
        for _ in 0..50 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        // V_gap[1] should have depolarized from the wave
        assert!(astro.v_gap[1] > -70.0, "Gap wave should propagate");

        // Ca²⁺ should have risen from spike
        assert!(astro.ca[0] > 0.5, "Ca should rise from spike: {}", astro.ca[0]);
    }

    // ─── 7. Compute gain ────────────────────────────

    #[test]
    fn test_astrocyte_compute_gain() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.5);
        // Manually set A and Ca
        astro.a[0] = 0.0;
        astro.a[1] = 1.0;
        astro.a[2] = 2.0;
        astro.ca[0] = 0.1;
        astro.ca[1] = 0.5;
        astro.ca[2] = 1.0;

        let gain = astro.compute_gain();

        // Neuron 0: A=0, Ca=0.1 → gain = 1.0 * (1 + 0.5*0.1) = 1.05
        assert!((gain[0] - 1.05).abs() < 1e-6, "gain[0]={}", gain[0]);
        // Neuron 1: A=1, α=0.5 → (1+0.5) = 1.5, Ca=0.5, μ=0.5 → (1+0.25) = 1.25, total = 1.5*1.25 = 1.875
        assert!((gain[1] - 1.875).abs() < 1e-6, "gain[1]={}", gain[1]);
        // Neuron 2: A=2 → (1+1.0)=2.0, Ca=1.0 → (1+0.5)=1.5, total=3.0
        assert!((gain[2] - 3.0).abs() < 1e-6, "gain[2]={}", gain[2]);
    }

    // ─── 8. Modulate V_thresh ──────────────────────────

    #[test]
    fn test_astrocyte_modulate_v_thresh() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        astro.k_ext[0] = 10.0; // Elevated K+
        astro.k_sensitivity = 2.0; // 2 mV/mM

        let mut v_thresh = vec![-50.0f32; 4];
        astro.modulate_v_thresh(&mut v_thresh);

        // Neuron 0: K=10, baseline=4, diff=6, shift=12 → V_th = -50 + 12 = -38
        assert!((v_thresh[0] - (-38.0)).abs() < 1e-6, "v_thresh[0]={}", v_thresh[0]);
        // Neuron 1: K=4, diff=0, shift=0 → V_th = -50
        assert!((v_thresh[1] - (-50.0)).abs() < 1e-6, "v_thresh[1]={}", v_thresh[1]);
    }

    // ─── 9. EAAT clearance ──────────────────────────

    #[test]
    fn test_astrocyte_eaat_clearance() {
        let mut astro = AstrocyteState::new(4, 1000.0, 0.1);
        let mut g_glucl = vec![1.0f32; 4];
        astro.eaat_occ[0] = 0.8; // 80% occupancy
        astro.eaat_strength = 0.5;

        // dt=1ms: clearance = 1 - 0.5 * 0.8 * 1 = 0.6
        astro.apply_eaat_clearance(&mut g_glucl, 1.0);
        assert!((g_glucl[0] - 0.6).abs() < 0.01, "g_glucl[0]={}", g_glucl[0]);

        // Neuron with no EAAT: unchanged
        assert!((g_glucl[1] - 1.0).abs() < 0.01, "g_glucl[1]={}", g_glucl[1]);
    }

    // ─── 10. Integration: apply_to_kinetics ────────────

    #[test]
    fn test_astrocyte_apply_to_kinetics_integration() {
        use crate::neuron::synapse::SynapticKinetics;

        let mut astro = AstrocyteState::new(8, 1000.0, 0.1);
        let mut kin = SynapticKinetics::new(8);
        kin.g_ampa[0] = 1.0;
        kin.g_glucl[1] = 0.5;

        // Drive spikes to activate astrocyte
        let spikes = vec![1i32; 8];
        let gap_pairs = vec![];
        for _ in 0..50 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        // Apply full modulation (EAAT + gain)
        astro.apply_to_kinetics(&mut kin, 1.0);

        // AMPA should be amplified (gain > 1)
        assert!(
            kin.g_ampa[0] > 1.0,
            "AMPA g[0] should be amplified: {}",
            kin.g_ampa[0]
        );

        // GluCl should be amplified AND possibly EAAT-cleared
        assert!(
            kin.g_glucl[1] >= 0.0,
            "GluCl g[1] should stay >= 0: {}",
            kin.g_glucl[1]
        );

        println!(
            "Integration: AMPA g[0]={:.6} (was 1.0), GluCl g[1]={:.6} (was 0.5)",
            kin.g_ampa[0], kin.g_glucl[1]
        );
    }

    // ─── 11. SIMD vs scalar tail ──────────────────────

    #[test]
    fn test_astrocyte_simd_vs_scalar_tail() {
        let mut astro = AstrocyteState::new(9, 1000.0, 0.1);
        let spikes = vec![1i32; 9];
        let gap_pairs = vec![];

        for _ in 0..100 {
            astro.step(&spikes, &gap_pairs, 1.0);
        }

        // Neuron 0 (SIMD) vs Neuron 8 (scalar tail) should match
        for (name, arr) in [
            ("a", &astro.a),
            ("ca", &astro.ca),
            ("k_ext", &astro.k_ext),
            ("eaat_occ", &astro.eaat_occ),
        ] {
            let diff = (arr[0] - arr[8]).abs();
            assert!(
                diff < 1e-5,
                "SIMD vs scalar mismatch {name}: arr[0]={}, arr[8]={}, diff={}",
                arr[0],
                arr[8],
                diff
            );
        }
    }
}
