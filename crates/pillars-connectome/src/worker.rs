//! # AlphaLifWorker — переиспользуемый воркер для симуляции мозга.
//!
//! Инкапсулирует модель **Shiu et al. 2024** («A Drosophila computational
//! brain model», Nature) на движке pillars-connectome: LIF + alpha-синапс с
//! аксональной задержкой, Poisson-стимуляция прямо в мембранный потенциал,
//! рефрактерность per-neuron (0 для стимулируемых GRN).
//!
//! Это тот же путь, что `BrainRunner::run_lif_alpha` (см. runner.rs), но
//! обёрнутый в готовый к использованию объект:
//!
//! ```no_run
//! use pillars_connectome::{AlphaLifWorker, ShiuParams};
//!
//! # fn main() -> Result<(), String> {
//! let mut worker = AlphaLifWorker::from_taas("flywire.taas", ShiuParams::flywire())?;
//! worker.set_sugar_grn(&[0, 1, 2]);   // стимуляция сахарных GRN 200 Hz
//! let out = worker.run_seconds(1.0);  // 1 s мозга
//! println!("{} spikes, {} active", out.total_spikes, out.active_neurons);
//! # Ok(())
//! # }
//! ```
//!
//! Метрики считаются на лету — без материализации гигантского spike-буфера
//! (`n_steps × n_neurons`), поэтому можно гонять минуты мозга на 138K
//! нейронах, не упираясь в память.

use std::time::Instant;

use crate::data::ConnectomeData;
use crate::runner::{BrainRunner, NeuronConfig};

/// Параметры модели Шу (brian2 `default_params` / `run_pytorch.py MODEL_PARAMS`).
///
/// | Параметр | Значение | Смысл |
/// |----------|----------|-------|
/// | `dt_ms` | 0.1 | шаг симуляции |
/// | `v_rest` / `v_reset` | −52 мВ | потенциал покоя / сброса |
/// | `v_thresh` | −45 мВ | порог спайка |
/// | `tau_mem` | 20 мс | мембранная постоянная |
/// | `tau_ref` | 2.2 мс | рефрактерность (0 для стимулируемых GRN) |
/// | `tau_syn` | 5 мс | затухание alpha-синапса |
/// | `t_delay` | 1.8 мс | аксональная задержка |
/// | `poisson_rate` | 200 Гц | частота стимуляции GRN |
/// | `poisson_weight` | 68.75 мВ | скачок v на Poisson-событие (= w_syn·f_poi) |
#[derive(Debug, Clone)]
pub struct ShiuParams {
    pub dt_ms: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub tau_mem: f32,
    pub tau_ref: f32,
    pub tau_syn: f32,
    pub t_delay: f32,
    pub poisson_rate: f32,
    pub poisson_weight: f32,
}

impl ShiuParams {
    /// Точные параметры из `Drosophila_brain_model/model.py` (Sugar GRNs 200 Hz).
    pub fn flywire() -> Self {
        Self {
            dt_ms: 0.1,
            v_rest: -52.0,
            v_thresh: -45.0,
            v_reset: -52.0,
            tau_mem: 20.0,
            tau_ref: 2.2,
            tau_syn: 5.0,
            t_delay: 1.8,
            poisson_rate: 200.0,
            poisson_weight: 0.275 * 250.0, // w_syn · f_poi = 68.75 мВ
        }
    }
}

/// Результат прогона воркера.
#[derive(Debug, Clone, Copy, Default)]
pub struct AlphaLifOutput {
    /// Общее число спайков за прогон.
    pub total_spikes: usize,
    /// Сколько нейронов спайкнуло хотя бы раз.
    pub active_neurons: usize,
    /// Время симуляции мозга (мс).
    pub brain_ms: f32,
    /// Реальное время (с), потраченное на прогон.
    pub wall_seconds: f64,
}

impl AlphaLifOutput {
    /// Во сколько раз быстрее реального времени работает симуляция
    /// (>1 — быстрее реального времени; 0.1 — в 10 раз медленнее).
    pub fn realtime_ratio(&self) -> f64 {
        let brain_s = self.brain_ms as f64 / 1000.0;
        if self.wall_seconds > 0.0 {
            brain_s / self.wall_seconds
        } else {
            f64::INFINITY
        }
    }
}

/// Воркер: alpha-LIF модель Шу на произвольном коннектоме.
///
/// Собирает `BrainRunner` с `NeuronConfig::LifAlpha`, настраивает
/// рефрактерность и Poisson-стимуляцию, и гоняет симуляцию **чанками**
/// (`chunk_steps` шагов на вызов), считая спайки и активные нейроны на лету.
///
/// Состояние сохраняется между вызовами `run_seconds` — можно продолжать
/// симуляцию шаг за шагом (режим «always-on» worker).
pub struct AlphaLifWorker {
    runner: BrainRunner,
    params: ShiuParams,
    /// Шагов на один чанк (память: chunk_steps × n_neurons × 4 байта).
    chunk_steps: usize,
    /// Стимулируемые нейроны (рефрактерность 0 + Poisson-вход).
    sugar_grn: Vec<usize>,
    /// Суммарное время мозга, уже отсимулированное (мс).
    brain_ms: f32,
}

impl AlphaLifWorker {
    /// Создать воркер из коннектома и параметров.
    ///
    /// Не требует receptor_types — модель Шу чисто возбуждающе-тормозная
    /// по знаку весов (`Excitatory × Connectivity` уже вшит в `.taas`).
    pub fn new(connectome: ConnectomeData, params: ShiuParams) -> Result<Self, String> {
        let config = NeuronConfig::LifAlpha {
            dt_ms: params.dt_ms,
            tau_ref: params.tau_ref,
            tau_m: params.tau_mem,
            v_rest: params.v_rest,
            v_thresh: params.v_thresh,
            v_reset: params.v_reset,
            tau_syn: params.tau_syn,
            t_delay: params.t_delay,
        };
        let runner = BrainRunner::new(connectome, config)?;
        // Чанк 0.1 s — буфер 1000×138K×4 ≈ 0.55 GB, комфортно.
        let chunk_steps = ((100.0 / params.dt_ms).round() as usize).max(1);
        Ok(Self {
            runner,
            params,
            chunk_steps,
            sugar_grn: Vec::new(),
            brain_ms: 0.0,
        })
    }

    /// Удобный конструктор: загрузить `.taas` с диска и собрать воркер.
    pub fn from_taas(path: &str, params: ShiuParams) -> Result<Self, String> {
        let f = std::fs::File::open(path)
            .map_err(|e| format!("cannot open {path}: {e}"))?;
        let conn = ConnectomeData::load_baked(&mut std::io::BufReader::new(f))
            .map_err(|e| format!("bad .taas: {e}"))?;
        Self::new(conn, params)
    }

    /// Задать стимулируемые GRN: рефрактерность 0 + Poisson-вход `poisson_rate`
    /// с весом `poisson_weight` (скачок прямо в v — как `target_var='v'` у Шу).
    pub fn set_sugar_grn(&mut self, indices: &[usize]) {
        self.runner.set_alpha_refrac_ms(indices, 0.0);
        self.runner
            .add_poisson_group(indices, self.params.poisson_rate, self.params.poisson_weight);
        self.sugar_grn = indices.to_vec();
    }

    /// Прогнать `steps` шагов мозга (чанками, как `run_seconds`), с опциональным
    /// внешним током, посчитать метрики на лету.
    ///
    /// `external_i` — ток (мВ), добавляемый прямо в v на каждом шаге, длиной
    /// `n_neurons()` или пустой срез (без внешнего входа).
    ///
    /// Состояние (v, g, delay-буфер, рефрактерность) сохраняется — повторный
    /// вызов продолжает симуляцию. Активные нейроны — **объединение** по всем
    /// чанкам (нейрон активен, если спайкнул хотя бы раз за `steps`).
    pub fn run_steps(&mut self, steps: usize, external_i: &[f32]) -> AlphaLifOutput {
        let n = self.runner.n_neurons();
        let t0 = Instant::now();
        let mut total_spikes = 0usize;
        let mut active = vec![false; n];

        let mut remaining = steps;
        while remaining > 0 {
            let chunk = remaining.min(self.chunk_steps);
            // Считаем метрики на лету — без материализации растра (n_steps × n),
            // то есть без fill 554 MB + скан 554 MB на каждый 0.1 s-чанк.
            let (chunk_total, chunk_active) = self.runner.run_count(chunk, external_i);
            total_spikes += chunk_total;
            for i in 0..n {
                active[i] |= chunk_active[i];
            }
            remaining -= chunk;
        }

        let wall = t0.elapsed().as_secs_f64();
        let brain_ms = steps as f32 * self.params.dt_ms;
        self.brain_ms += brain_ms;
        AlphaLifOutput {
            total_spikes,
            active_neurons: active.iter().filter(|&&a| a).count(),
            brain_ms,
            wall_seconds: wall,
        }
    }

    /// Прогнать `seconds` секунд мозга, посчитать метрики на лету.
    ///
    /// Тонкая обёртка над [`Self::run_steps`] — переводит секунды в шаги.
    pub fn run_seconds(&mut self, seconds: f32) -> AlphaLifOutput {
        let n_steps_total = (seconds * 1000.0 / self.params.dt_ms).round() as usize;
        self.run_steps(n_steps_total, &[])
    }

    /// Число нейронов в коннектоме.
    pub fn n_neurons(&self) -> usize {
        self.runner.n_neurons()
    }

    /// Число стимулируемых GRN (0, если `set_sugar_grn` не вызывался).
    pub fn n_sugar_grn(&self) -> usize {
        self.sugar_grn.len()
    }

    /// Мгновенные мембранные потенциалы (для отладки/визуализации).
    pub fn v(&self) -> &[f32] {
        self.runner.get_v()
    }

    /// Сбросить состояние (v→v_rest, g=0, буферы занулены) — новая симуляция.
    pub fn reset(&mut self) {
        self.runner.reset();
        self.brain_ms = 0.0;
    }
}

/// Проверка соответствия модели Шу на цепочке 0→1→2: сигнал должен дойти
/// до нейрона 2 с задержкой ≈ 2·(t_delay + время нарастания g).
#[cfg(test)]
mod tests {
    use super::*;

    fn chain_connectome() -> ConnectomeData {
        ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1i32, 2],
            vec![500.0f32, 500.0],
            vec![],
            vec![],
        )
    }

    #[test]
    fn worker_runs_chain_with_delay() {
        let mut params = ShiuParams::flywire();
        params.t_delay = 1.0; // 10 шагов
        let mut worker = AlphaLifWorker::new(chain_connectome(), params).unwrap();
        // Нейрон 0 подгоняется постоянным током 100 мВ (спайкает каждый шаг),
        // сигнал идёт 0→1→2 через alpha-синапс с задержкой. 100 шагов = 10 мс:
        // нужно ≥ 2·t_delay + время нарастания alpha-синапса (τ_syn = 5 мс),
        // на 30 шагах (3 мс) нейрон 2 физически не успевает.
        let ext = vec![100.0f32, 0.0, 0.0];
        let out = worker.run_steps(100, &ext);
        assert!(out.total_spikes > 0);
        // Neuron 2 должен спайкнуть (цепь 0→1→2 с весом 500 мВ).
        assert!(out.active_neurons >= 3, "chain 0→1→2 should light up: {out:?}");
    }

    #[test]
    fn worker_outputs_realtime_ratio() {
        let mut worker = AlphaLifWorker::new(chain_connectome(), ShiuParams::flywire()).unwrap();
        let out = worker.run_seconds(0.1);
        assert!(out.realtime_ratio() > 0.0);
        assert!(out.brain_ms > 0.0);
    }
}
