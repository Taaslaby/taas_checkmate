//! Somato-Dendritic Integration — LIF soma + cable dendrites.
//!
//! Объединяет `step_lif` с `CableDendriteState` в единый биофизически
//! точный цикл симуляции.
//!
//! # Цикл симуляции (каждый шаг)
//! ```text
//! 1. decay_conductances()   — g_r *= exp(-dt/τ_r)
//! 2. cable_step()           → I_dend[n]
//! 3. step_lif(I_soma + I_dend)  → v[n], spikes[n]
//! 4. jump_conductances()    — доставить входящие спайки в компартменты
//! ```
//!
//! # Пример
//! ```ignore
//! let mut pop = SomatoDendriticNeuron::new(
//!     64, &DendriteConfig { n_compartments: 4, R_a: 200.0, R_m: 100.0, C_m: 0.1 },
//!     0.5, 20.0, -65.0, -54.0, -65.0, 3.0,
//! );
//! pop.jump_ampa(0, 3, 0.1); // дистальный AMPA
//! pop.step(&I_ext, 0.5);
//! ```

pub use taas_kernels::neuron::dendrite::{
    CableDendriteState, DendriteConfig, DendriteDecayFactors,
};

use crate::kernels;

/// Somato-dendritic нейронная популяция.
///
/// Объединяет LIF сому с `CableDendriteState` (cable-compartment дендриты).
/// Поддерживает 7 типов рецепторов: AMPA, NMDA, GABA_A/B, nAChR, GluCl, mAChR.
#[derive(Debug)]
pub struct SomatoDendriticNeuron {
    /// Количество нейронов (кратно 8 для SIMD)
    pub n: usize,
    /// Соматический потенциал [n]
    pub v: Vec<f32>,
    /// Соматический ток [n] = I_ext + I_dend
    pub I_soma: Vec<f32>,
    /// Рефрактерные счётчики [n]
    pub refractory: Vec<f32>,
    /// Выходные спайки [n], 0/1
    pub spikes: Vec<i32>,
    /// Дендритная популяция
    pub dendrite: CableDendriteState,
    /// Decay factors (предвычислены из dt)
    decay: DendriteDecayFactors,
    /// Параметры LIF [5 × n], column-major: [dt, tau_m, v_rest, v_thresh, v_reset]
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

        // LIF params column-major [dt, tau_m, v_rest, v_thresh, v_reset]
        let mut params = vec![0.0f32; n * 5];
        for i in 0..n {
            params[i] = dt;
            params[n + i] = tau_m;
            params[2 * n + i] = v_rest;
            params[3 * n + i] = v_thresh;
            params[4 * n + i] = v_reset;
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
        for i in 0..self.n {
            self.I_soma[i] = I_ext[i] + self.dendrite.I_dend[i];
        }

        // 4. LIF step for all neurons
        kernels::lif_step(
            &mut self.v,
            &mut self.I_soma,
            &mut self.spikes,
            &self.params,
            &mut self.refractory,
            self.tau_ref,
            None,
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
            self.params[self.n + nrn] = tau_m;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pop(n: usize, r_a: f32) -> SomatoDendriticNeuron {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: r_a,
            R_m: 100.0,
            C_m: 0.1,
        };
        SomatoDendriticNeuron::new(n, &config, 0.5, 20.0, -65.0, -54.0, -65.0, 3.0)
    }

    #[test]
    fn test_soma_dendrite_ampa_drives_spike() {
        let mut pop = make_pop(8, 10.0);
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
    fn test_soma_dendrite_proximal_stronger_than_distal() {
        let i_ext = vec![0.0f32; 8];
        let mut prox = make_pop(8, 200.0);
        let mut dist = make_pop(8, 200.0);

        for _ in 0..100 {
            prox.jump_ampa(0, 0, 0.05); // компартмент 0 (проксимальный)
            dist.jump_ampa(0, 3, 0.05); // компартмент 3 (дистальный)
            prox.step(&i_ext, 0.5);
            dist.step(&i_ext, 0.5);
        }

        let i_p = prox.dend_current(0);
        let i_d = dist.dend_current(0);
        assert!(
            i_p > i_d,
            "Проксимальный > дистальный: prox={i_p:.6}, dist={i_d:.6}"
        );
    }

    #[test]
    fn test_soma_dendrite_gaba_shunts_ampa() {
        let i_ext = vec![0.0f32; 8];
        let mut ampa_only = make_pop(8, 10.0);
        let mut ampa_gaba = make_pop(8, 10.0);
        ampa_only.set_lif_params(0, 20.0, -65.0, -60.0, -65.0);
        ampa_gaba.set_lif_params(0, 20.0, -65.0, -60.0, -65.0);

        let mut spikes_only = 0i32;
        let mut spikes_gaba = 0i32;
        for _ in 0..400 {
            ampa_only.jump_ampa(0, 0, 0.20);
            ampa_gaba.jump_ampa(0, 0, 0.20);
            ampa_gaba.jump_gaba_a(0, 0, 0.40); // GABA шунт
            ampa_only.step(&i_ext, 0.1);
            ampa_gaba.step(&i_ext, 0.1);
            spikes_only += ampa_only.spikes[0];
            spikes_gaba += ampa_gaba.spikes[0];
        }
        assert!(
            spikes_only > spikes_gaba,
            "GABA должна уменьшить число спайков: only={spikes_only}, gaba={spikes_gaba}"
        );
    }
}
