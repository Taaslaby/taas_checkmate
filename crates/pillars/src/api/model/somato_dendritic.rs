//! Neuron population classes.

use crate::kernels::dendrite::{CableDendriteState, DendriteConfig, DendriteDecayFactors};

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
        crate::core::runner::run_neuron_slices::<crate::zoo::neurons::LIFModel>(
            crate::core::runner::NeuronState {
                v: Some(&mut self.v),
                u: Some(&mut self.refractory), // state 1 (refractory)
                ..Default::default()
            },
            &self.I_soma,
            &mut self.spikes,
            &self.params,
        );
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
