//! # HeteroRunner — настоящий гетерогенный мозг
//!
//! ОДИН общий коннектом (CSR + nt_types), в котором разные группы нейронов
//! живут на РАЗНЫХ моделях динамики одновременно. В отличие от
//! [`BrainRunner`](crate::BrainRunner) (одна модель на всю сеть), здесь:
//!
//! ```text
//!         ┌────────────────  ОБЩИЙ CSR-коннектом  ────────────────┐
//!         │                                                       │
//!   группа Alpha  ──спайки──►  общий queue ──► CSR propagate ──► i_in
//!   группа HH     ──спайки──►        ▲                              │
//!   группа HAAP   ──спайки──►        │                              ▼
//!   группа Izh    ──спайки──►   (все модели)            i_in[глобальные]
//!   группа Rate   ──спайки──►                              │  (след. шаг)
//!                                                          ▼
//!                                         каждая группа берёт СВОЁ из i_in
//! ```
//!
//! Каждый шаг:
//! 1. каждая группа степает СВОЮ динамику (dynamics-only, без внутренней
//!    propagation) со своим входом: `i_in[глобальные индексы] + external`;
//! 2. локальные спайки разлетаются в общий глобальный вектор спайков;
//! 3. общий queue → **одна** CSR-propagation через ВЕСЬ коннектом → `i_in`
//!    для следующего шага.
//!
//! Так спайк LIF-нейрона реально достигает HH/HAAP/Izh/Rate-нейрона и
//! наоборот — через одни и те же синапсы. Это и есть «настоящий
//! гетерогенный мозг», которого раньше не было (mono-runner прогонял всю
//! сеть через одну модель).
//!
//! ## Единый такт
//!
//! Все группы делят ОДИН `dt_ms` (общая временна́я линия). Если в мозгу есть
//! группа с жёстким шагом (HAAP/HH требуют ~0.05 мс), глобальный `dt_ms`
//! должен быть не крупнее — иначе эти модели численно нестабильны. Группы,
//! которым мелкий шаг не нужен (alpha/Izh/Rate), просто степают чаще.
//!
//! ## Соглашение о входе
//!
//! - `external_I` и Poisson — «драйв» (в масштабе LIF-тока);
//! - `i_in` (результат propagation) — рекуррентный вход (weight × nt, уже со
//!   знаком тормозного пресинапса), с задержкой 1 шаг (как в mono HH/HAAP);
//! - alpha-группа: драйв идёт **прямо в `v`** (как в `run_lif_alpha`),
//!   рекуррентный — в `i_syn` (проводимость);
//! - HH/HAAP/Izh/Rate: `вход = масштаб·(драйв + рекуррентный)`.

use crate::data::ConnectomeData;
use crate::propagate::{push_spikes_to_queue, record_spikes};
use crate::runner::{alpha_lif_population, NeuronConfig, PoissonGroup};
use crate::sim::{
    haap_population, hh_population_pacemaking, hh_population_standard, step_izhikevich_pillars,
};
use crate::types::SpikeQueue;
use pillars::core::NeuralPopulation;
use pillars::kernels::csr::ValidatedCsrGraph;
use pillars::zoo::{
    AlphaLifModel, AlphaLifModelStateIdx, HAAPModel, HHModel, TransformerModel,
    TransformerModelParamIdx, TransformerModelStateIdx,
};
use rand::{Rng, SeedableRng};

/// Одна модель динамики для одной группы нейронов.
///
/// Каждая группа владеет своим состоянием (популяция/массивы) и своими
/// глобальными индексами нейронов в общем коннектоме.
enum GroupEngine {
    /// LIF + alpha-синапс (Shiu et al. 2024). Драйв → `v`, рекуррент → `i_syn`.
    ///
    /// Аксональная задержка: как в mono `run_lif_alpha`, рекуррент идёт
    /// через кольцо из `delay_steps+1` слотов (t_delay→шаги). Без этого
    /// hetero отдаёт рекуррент за 1 шаг вместо ~4 — на динамическом входе
    /// петли сходятся иначе и могут коллапсировать в тишину.
    Alpha {
        indices: Vec<usize>,
        pop: NeuralPopulation<AlphaLifModel>,
        /// Входной масштаб драйва (1.0 = как в mono LifAlpha).
        drive_scale: f32,
        /// Масштаб рекуррентного входа (синаптические веса общие для всех
        /// моделей; чувствительность к ним у каждой модели своя).
        rec_scale: f32,
        /// Число шагов задержки аксонального кольца.
        delay_steps: usize,
        /// Кольцо задержки: (delay_steps+1) × k (k = indices.len()).
        delay_ring: Vec<f32>,
        /// Позиция записи в кольцо.
        delay_pos: usize,
    },
    /// Hodgkin-Huxley (standard или pacemaking).
    Hh {
        indices: Vec<usize>,
        pop: NeuralPopulation<HHModel>,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// HAAP (M-current + h-current), current-based.
    Haap {
        indices: Vec<usize>,
        pop: NeuralPopulation<HAAPModel>,
        /// Персистентный синаптический ток (затухает с τ_syn).
        syn: Vec<f32>,
        decay: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// Izhikevich (regular spiking).
    Izh {
        indices: Vec<usize>,
        v: Vec<f32>,
        u: Vec<f32>,
        trace: Vec<f32>,
        refr: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
        dt_ms: f32,
        tau_ref: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// Rate model (continuous firing rates).
    Rate {
        indices: Vec<usize>,
        v: Vec<f32>,
        tau: f32,
        dt_ms: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// SurgeN — нейрон-трансформер (pillars zoo `TransformerModel`):
    /// temporal self-attention над 8 слотами памяти, leaky-интегратор → спайк.
    /// Вход = масштаб·(драйв + рекуррент), как у HH/Izh; задержка 1 шаг.
    Transformer {
        indices: Vec<usize>,
        pop: NeuralPopulation<TransformerModel>,
        drive_scale: f32,
        rec_scale: f32,
    },
}

impl GroupEngine {
    fn indices(&self) -> &[usize] {
        match self {
            GroupEngine::Alpha { indices, .. }
            | GroupEngine::Hh { indices, .. }
            | GroupEngine::Haap { indices, .. }
            | GroupEngine::Izh { indices, .. }
            | GroupEngine::Rate { indices, .. }
            | GroupEngine::Transformer { indices, .. } => indices,
        }
    }

    /// Dynamics-only step: читает `drive` (external+poisson, глобальный) и
    /// `rec` (рекуррентный вход, глобальный), пишет спайки в `spikes_out`
    /// (глобальный вектор, по глобальным индексам группы).
    fn step(
        &mut self,
        drive: &[f32],
        rec: &[f32],
        spikes_out: &mut [i32],
    ) {
        let _ = rec;
        let _ = spikes_out;
        self.step_inner(drive, rec, spikes_out);
    }

    /// Отдать текущий слой рекуррента в группу (после propagation).
    /// Alpha-группы складывают его в своё кольцо задержки; остальные — no-op
    /// (они и так читают `rec` на следующем шаге).
    fn deliver(&mut self, rec: &[f32]) {
        match self {
            GroupEngine::Alpha {
                indices,
                delay_steps,
                delay_ring,
                delay_pos,
                ..
            } => {
                let k = indices.len();
                let size = *delay_steps + 1;
                let slot = &mut delay_ring[*delay_pos * k.max(1)..(*delay_pos + 1) * k.max(1)];
                for (j, &g) in indices.iter().enumerate() {
                    slot[j] = rec.get(g).copied().unwrap_or(0.0);
                }
                *delay_pos = (*delay_pos + 1) % size;
            }
            _ => {}
        }
    }

    fn step_inner(
        &mut self,
        drive: &[f32],
        rec: &[f32],
        spikes_out: &mut [i32],
    ) {
        match self {
            GroupEngine::Alpha {
                indices,
                pop,
                drive_scale,
                rec_scale,
                delay_steps,
                delay_ring,
                delay_pos,
            } => {
                let n = indices.len();
                let mut v_slice = vec![0.0f32; n];
                let mut i_syn = vec![0.0f32; n];
                let _ = delay_steps;
                let read_pos = *delay_pos;
                for (k, &g) in indices.iter().enumerate() {
                    // Драйв → v (как в run_lif_alpha), масштабированный.
                    v_slice[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale;
                    // Рекуррент → i_syn: читаем ЗАДЕРЖАННЫЙ слот кольца
                    // (как mono run_lif_alpha, а не 1-шаговый rec).
                    i_syn[k] = delay_ring[read_pos * n + k] * *rec_scale;
                }
                // Переносим накопленный v популяции + этот шаг драйва.
                {
                    let v_col = pop.state_slice_mut(AlphaLifModelStateIdx::v.into());
                    for (k, &g) in indices.iter().enumerate() {
                        // v популяции живёт в локальных индексах группы.
                        let _ = g;
                        v_col[k] += v_slice[k];
                    }
                }
                let mut spikes = vec![0i32; n];
                pop.step(&i_syn, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Hh {
                indices,
                pop,
                drive_scale,
                rec_scale,
            } => {
                let mut i_in = vec![0.0f32; indices.len()];
                for (k, &g) in indices.iter().enumerate() {
                    i_in[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                }
                let mut spikes = vec![0i32; indices.len()];
                pop.step(&i_in, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Haap {
                indices,
                pop,
                syn,
                decay,
                drive_scale,
                rec_scale,
            } => {
                let n = indices.len();
                let mut i_in = vec![0.0f32; n];
                for (k, &g) in indices.iter().enumerate() {
                    // Персистентный синаптический ток: decay + новый рекуррент.
                    syn[k] = syn[k] * *decay + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                    i_in[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale + syn[k];
                }
                let mut spikes = vec![0i32; n];
                pop.step(&i_in, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Izh {
                indices,
                v,
                u,
                trace,
                refr,
                a,
                b,
                c,
                d,
                dt_ms,
                tau_ref,
                drive_scale,
                rec_scale,
            } => {
                // SIMD-шаг работает с паддленными массивами; пишем драйв только
                // в первые n (реальные нейроны), хвост — нули.
                let kp = v.len();
                let mut ii = vec![0.0f32; kp];
                for (k, &g) in indices.iter().enumerate() {
                    ii[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                }
                let mut spikes = vec![0i32; kp];
                step_izhikevich_pillars(
                    v, u, trace, a, b, &mut ii, *dt_ms, 0.99, &mut spikes, c, d, refr, *tau_ref,
                );
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Rate {
                indices,
                v,
                tau,
                dt_ms,
                drive_scale,
                rec_scale,
            } => {
                let dt = *dt_ms / 1000.0;
                let decay = (-dt / tau.max(1e-6)).exp();
                let inv_decay = 1.0 - decay;
                for (k, &g) in indices.iter().enumerate() {
                    let input = (drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale)
                        .clamp(0.0, 1.0);
                    v[k] = v[k] * decay + input * inv_decay;
                    if v[k] > 0.5 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Transformer {
                indices,
                pop,
                drive_scale,
                rec_scale,
            } => {
                // Вход = масштаб·(драйв + рекуррент), как у HH/Izh. Модель сама
                // нормализует query и держит память последних 8 входов.
                let mut i_in = vec![0.0f32; indices.len()];
                for (k, &g) in indices.iter().enumerate() {
                    i_in[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                }
                let mut spikes = vec![0i32; indices.len()];
                pop.step(&i_in, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
        }
    }
}

/// Описание одной группы нейронов для [`HeteroRunner`].
#[derive(Debug, Clone)]
pub struct HeteroGroup {
    /// Модель и параметры группы.
    pub config: NeuronConfig,
    /// Глобальные индексы нейронов этой группы в общем коннектоме.
    pub indices: Vec<usize>,
    /// Масштаб входного драйва (external + Poisson) этой группы.
    /// LIF-alpha: 1.0; HAAP/HH: ~8.0 (см. kailo HAAP_INPUT_SCALE).
    pub drive_scale: f32,
    /// Масштаб рекуррентного входа (синаптические веса общие).
    /// LIF-alpha: 1.0; HAAP: 0.25 (см. kailo HAAP_SYN_SCALE).
    pub rec_scale: f32,
}

/// Настоящий гетерогенный мозг: один коннектом, много моделей сразу.
///
/// Конструктор [`HeteroRunner::new`] принимает полный коннектом и список
/// групп. Группы покрывают ВСЕ нейроны (каждый ровно один раз) — иначе
/// непокрытые нейроны молчат (их состояние не степается).
///
/// # Снапшоты (save/load)
///
/// [`HeteroRunner::snapshot`] возвращает полный слепок состояния
/// (коннектом с текущими весами, все группы, счётчик шагов), а
/// [`HeteroRunner::restore`] восстанавливает его на месте. Снапшот — это
/// чистые данные (`Vec`), их можно сериализовать через крейт `taas-format`
/// (`save_hetero_snapshot`/`load_hetero_snapshot`).
///
/// ⚠️ RNG Пуассоновских групп не сериализуется (у `StdRng` нет API чтения
/// состояния) — после `restore` стохастика начинается с начального сида, а
/// не продолжается с точки снапшота. Для детерминированной симуляции без
/// Пуассона снапшот/restore — бит-в-бит. 
pub struct HeteroRunner {
    /// RNG Пуассоновских групп: создаётся из фиксированного сида (см.
    /// `StdRng::from_seed` в `new`). Сид хранится в снапшоте и
    /// восстанавливается при `restore`.
    pub(crate) poisson_seed: [u8; 32],

    /// Общий коннектом (публичный — пластичность kailo читает values).
    pub connectome: ConnectomeData,
    csr_graph: ValidatedCsrGraph,
    /// Конфиги групп (для reset/перестроения engines).
    specs: Vec<HeteroGroup>,
    groups: Vec<GroupEngine>,
    // Общие буферы (глобальные, размер n).
    drive: Vec<f32>,   // external + poisson на этот шаг
    rec: Vec<f32>,     // рекуррентный вход (результат propagation)
    spikes: Vec<i32>,  // глобальные спайки шага
    queue: SpikeQueue,
    spike_buffer: Vec<i32>,
    step_counter: usize,
    n: usize,
    dt_ms: f32,
    poisson_groups: Vec<PoissonGroup>,
    rng: rand::rngs::StdRng,
}

impl HeteroRunner {
    /// Создать гетерогенный мозг.
    ///
    /// # Ошибки
    /// - пустой коннектом;
    /// - пустой список групп;
    /// - группа покрывает нейрон за пределами коннектома;
    /// - нейрон покрыт дважды (группы пересекаются).
    pub fn new(
        connectome: ConnectomeData,
        groups: Vec<HeteroGroup>,
        dt_ms: f32,
    ) -> Result<Self, String> {
        let n = connectome.n_neurons;
        if n == 0 {
            return Err("HeteroRunner: connectome has 0 neurons".into());
        }
        if groups.is_empty() {
            return Err("HeteroRunner: no groups".into());
        }
        let csr_graph =
            ValidatedCsrGraph::from_slices(&connectome.row_ptr, &connectome.col_indices, n);

        // Проверка покрытия: каждый нейрон ровно один раз.
        let mut covered = vec![false; n];
        let mut engines = Vec::with_capacity(groups.len());
        let mut specs = Vec::with_capacity(groups.len());
        for g in groups {
            if g.indices.is_empty() {
                continue; // пустые группы пропускаем
            }
            for &idx in &g.indices {
                if idx >= n {
                    return Err(format!("HeteroRunner: group index {idx} >= n={n}"));
                }
                if covered[idx] {
                    return Err(format!("HeteroRunner: neuron {idx} covered twice"));
                }
                covered[idx] = true;
            }
            let eng = build_engine(&g, dt_ms)?;
            specs.push(g.clone());
            engines.push(eng);
        }
        if engines.is_empty() {
            return Err("HeteroRunner: no non-empty groups".into());
        }

        let poisson_seed = [0u8; 32];
        let rng = rand::rngs::StdRng::from_seed(poisson_seed);
        let queue_cap = n.max(4).next_power_of_two();
        Ok(Self {
            connectome,
            csr_graph,
            specs,
            groups: engines,
            drive: vec![0.0f32; n],
            rec: vec![0.0f32; n],
            spikes: vec![0i32; n],
            queue: SpikeQueue::new(queue_cap),
            spike_buffer: vec![0i32; n * 64],
            step_counter: 0,
            n,
            dt_ms,
            poisson_groups: Vec::new(),
            rng,
            poisson_seed,
        })
    }

    /// Добавить Poisson-группу (глобальные индексы).
    pub fn add_poisson_group(&mut self, indices: &[usize], rate_hz: f32, weight: f32) {
        let indices = indices
            .iter()
            .filter(|&&i| i < self.n)
            .copied()
            .collect();
        self.poisson_groups.push(PoissonGroup {
            indices,
            rate_hz,
            weight,
        });
    }

    /// Очистить Poisson-группы.
    pub fn clear_poisson_groups(&mut self) {
        self.poisson_groups.clear();
    }

    /// Число нейронов.
    pub fn n_neurons(&self) -> usize {
        self.n
    }

    /// Общий шаг интеграции (мс).
    pub fn dt_ms(&self) -> f32 {
        self.dt_ms
    }

    /// Всего шагов просимулировано.
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Глобальные индексы каждой группы (для диагностики/observatory).
    pub fn group_indices(&self) -> Vec<&[usize]> {
        self.groups.iter().map(|g| g.indices()).collect()
    }

    /// Экспорт данных групп для pillars-cuda `CompiledNetwork::new`.
    ///
    /// Возвращает Vec<(source, kernel_name, n, ns, np, states, params)> для
    /// каждой непустой группы — ровно то, что нужно для `NetworkPop`.
    /// Экспорт данных групп для pillars-cuda `CompiledNetwork::new`.
    ///
    /// Returns Vec of (source, kernel_name, n, ns, np, states_flat, params_flat)
    /// for each NeuralPopulation-backed group (Alpha/Hh/Haap/Transformer).
    /// Izh and Rate use raw vectors and are skipped (CPU fallback).
    pub fn export_pops_for_gpu(&self) -> Vec<(&'static str, &'static str, usize, usize, usize, Vec<f32>, Vec<f32>)> {
        use pillars::zoo::neurons::*;
        use pillars::zoo::Model;
        macro_rules! export_pop {
            ($pop:expr, $indices:expr, $source:expr, $kernel:expr, $M:ty) => {{
                let n = $indices.len();
                let stride = n.div_ceil(8) * 8;
                let ns = <$M as Model>::N_STATE;
                let np = <$M as Model>::N_PARAMS;
                let mut states = Vec::with_capacity(ns * stride);
                for c in 0..ns {
                    let col = $pop.state_slice(c);
                    states.extend_from_slice(col);
                    states.resize(states.len() + (stride - col.len()), 0.0);
                }
                let mut params = Vec::with_capacity(np * stride);
                for c in 0..np {
                    let col = $pop.param_slice(c);
                    params.extend_from_slice(col);
                    params.resize(params.len() + (stride - col.len()), 0.0);
                }
                ($source, $kernel, n, ns, np, states, params)
            }};
        }
        self.groups.iter().filter_map(|g| match g {
            GroupEngine::Alpha { pop, indices, .. } =>
                Some(export_pop!(pop, indices, LIF_CUDA_SOURCE, "lif_kernel", AlphaLifModel)),
            GroupEngine::Hh { pop, indices, .. } =>
                Some(export_pop!(pop, indices, HH_CUDA_SOURCE, "hh_kernel", HHModel)),
            GroupEngine::Haap { pop, indices, .. } =>
                Some(export_pop!(pop, indices, HAAP_CUDA_SOURCE, "haap_kernel", HAAPModel)),
            GroupEngine::Transformer { pop, indices, .. } =>
                Some(export_pop!(pop, indices, TRANSFORMER_CUDA_SOURCE, "transformer_kernel", TransformerModel)),
            _ => None, // Izh/Rate use raw vectors — no NeuralPopulation to export
        }).collect()
    }

    /// Запустить `n_steps` шагов. `external_I` — глобальный драйв `[n]`.
    ///
    /// Возвращает spike buffer `[n_steps × n]` (0/1).
    pub fn run(&mut self, n_steps: usize, external_I: &[f32]) -> &[i32] {
        let needed = n_steps * self.n;
        if self.spike_buffer.len() < needed {
            self.spike_buffer.resize(needed, 0);
        }
        self.spike_buffer[..needed].fill(0);

        let n = self.n;
        for step in 0..n_steps {
            // 1. Драйв = external + poisson (глобальный).
            self.drive[..n].fill(0.0);
            for (i, &e) in external_I.iter().enumerate().take(n) {
                self.drive[i] += e;
            }
            inject_poisson_into(&self.poisson_groups, &mut self.rng, &mut self.drive, self.dt_ms);

            // 2. Каждая группа степает СВОЮ динамику (dynamics-only).
            self.spikes[..n].fill(0);
            for g in self.groups.iter_mut() {
                g.step(&self.drive, &self.rec, &mut self.spikes);
            }

            // 3. Общий queue + record.
            push_spikes_to_queue(&self.spikes, n, &mut self.queue);
            record_spikes(&self.spikes, n, &mut self.spike_buffer, step);

            // 4. ОДНА propagation через ВЕСЬ коннектом → rec (след. шаг).
            self.rec[..n].fill(0.0);
            crate::propagate::csr_propagate_from_queue_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.queue,
                &mut self.rec,
                1.0,
            );

            // 5. Раздать rec в группы (alpha → кольцо задержки).
            for g in self.groups.iter_mut() {
                g.deliver(&self.rec);
            }
        }

        self.step_counter += n_steps;
        &self.spike_buffer[..needed]
    }

    /// Сбросить состояние всех групп (популяции/массивы → дефолт).
    pub fn reset(&mut self) {
        let specs = self.specs.clone();
        let dt = self.dt_ms;
        let mut engines = Vec::with_capacity(specs.len());
        for spec in &specs {
            match build_engine(spec, dt) {
                Ok(eng) => engines.push(eng),
                Err(_) => continue,
            }
        }
        self.groups = engines;
        self.spike_buffer.fill(0);
        self.step_counter = 0;
        self.rec.fill(0.0);
        self.drive.fill(0.0);
        self.spikes.fill(0);
    }
}

/// Построить engine группы по конфигу.
fn build_engine(g: &HeteroGroup, dt_ms: f32) -> Result<GroupEngine, String> {
    let k = g.indices.len();
    let scale = if g.drive_scale > 0.0 {
        g.drive_scale
    } else {
        1.0
    };
    let rscale = if g.rec_scale > 0.0 {
        g.rec_scale
    } else {
        1.0
    };
    let engine = match &g.config {
        NeuronConfig::LifAlpha {
            t_delay,
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_syn,
            ..
        } => {
            let pop = alpha_lif_population(
                k, dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, *tau_syn,
            );
            // Аксональная задержка (как mono run_lif_alpha): кольцо из
            // (delay_steps+1) слотов по k нейронов. Ширина — ровно на один
            // шаг больше задержки, чтобы читаемый слот был из прошлого.
            let delay_steps = (*t_delay / dt_ms.max(1e-6)).round().max(0.0) as usize;
            let size = delay_steps + 1;
            GroupEngine::Alpha {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
                delay_steps,
                delay_ring: vec![0.0; size * k.max(1)],
                delay_pos: 0,
            }
        }
        NeuronConfig::Lif {
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            ..
        }
        | NeuronConfig::LifSynaptic {
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            ..
        } => {
            // Lif/LifSynaptic — как alpha (conductance-подобный вход) для простоты.
            let pop = alpha_lif_population(
                k, dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, 5.0,
            );
            GroupEngine::Alpha {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
                delay_steps: 0,
                delay_ring: vec![0.0; k.max(1)],
                delay_pos: 0,
            }
        }
        NeuronConfig::Hh { use_pacemaking, .. } | NeuronConfig::HhSynaptic { use_pacemaking, .. } => {
            let pop = if *use_pacemaking {
                hh_population_pacemaking(k, dt_ms)
            } else {
                hh_population_standard(k, dt_ms)
            };
            GroupEngine::Hh {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::Haap { .. } => {
            let pop = haap_population(k, dt_ms);
            let decay = 0.0; // мгновенный ток (как mono run_haap с τ_syn=0)
            GroupEngine::Haap {
                indices: g.indices.clone(),
                pop,
                syn: vec![0.0; k],
                decay,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::HaapSynaptic { tau_syn, .. } => {
            let pop = haap_population(k, dt_ms);
            let decay = if *tau_syn > 0.0 {
                (-dt_ms / tau_syn).exp()
            } else {
                0.0
            };
            GroupEngine::Haap {
                indices: g.indices.clone(),
                pop,
                syn: vec![0.0; k],
                decay,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::Izh {
            dt_ms: idt,
            tau_ref,
            a,
            b,
            c,
            d,
        }
        | NeuronConfig::IzhSynaptic {
            dt_ms: idt,
            tau_ref,
            a,
            b,
            c,
            d,
        } => {
            let _ = idt; // используем глобальный dt_ms
            // SIMD-ядро Izh требует кратные 8 длины — паддим, как mono runner.
            let kp = k.div_ceil(8) * 8;
            GroupEngine::Izh {
                indices: g.indices.clone(),
                v: vec![-65.0; kp],
                u: vec![-13.0; kp],
                trace: vec![0.0; kp],
                refr: vec![0.0; kp],
                a: vec![*a; kp],
                b: vec![*b; kp],
                c: vec![*c; kp],
                d: vec![*d; kp],
                dt_ms,
                tau_ref: *tau_ref,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::IzhDual { .. } | NeuronConfig::IzhEprop { .. } => {
            return Err("HeteroRunner: IzhDual/IzhEprop не поддержаны (используйте Izh)".into());
        }
        NeuronConfig::Rate { tau, .. } => GroupEngine::Rate {
            indices: g.indices.clone(),
            v: vec![0.0; k],
            tau: *tau,
            dt_ms,
            drive_scale: scale,
            rec_scale: rscale,
        },
        NeuronConfig::Transformer {
            dt_ms: tdt,
            decay,
            gain,
            inv_tau_acc,
            thresh,
            k_scale,
            eta_k,
            eta_v,
            w_base,
            w_span,
            v_base,
            v_span,
        } => {
            let _ = tdt; // модель использует ctx.dt() (глобальный dt_ms)
            let pop = transformer_population(
                k,
                dt_ms,
                *decay,
                *gain,
                *inv_tau_acc,
                *thresh,
                *k_scale,
                *eta_k,
                *eta_v,
                *w_base,
                *w_span,
                *v_base,
                *v_span,
            );
            GroupEngine::Transformer {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::Gco { .. } => {
            return Err("HeteroRunner: Gco не поддержан (используйте Rate)".into());
        }
    };
    Ok(engine)
}

/// Собрать популяцию SurgeN-трансформеров (pillars zoo `TransformerModel`).
/// Веса QKV стартуют на базе (равномерное внимание); обучение — по eta.
fn transformer_population(
    k: usize,
    dt_ms: f32,
    decay: f32,
    gain: f32,
    inv_tau_acc: f32,
    thresh: f32,
    k_scale: f32,
    eta_k: f32,
    eta_v: f32,
    w_base: f32,
    w_span: f32,
    v_base: f32,
    v_span: f32,
) -> NeuralPopulation<TransformerModel> {
    let mut pop = NeuralPopulation::<TransformerModel>::new(k);
    pop.fill_param(TransformerModelParamIdx::dt.into(), dt_ms);
    pop.fill_param(TransformerModelParamIdx::decay.into(), decay);
    pop.fill_param(TransformerModelParamIdx::gain.into(), gain);
    pop.fill_param(TransformerModelParamIdx::inv_tau_acc.into(), inv_tau_acc);
    pop.fill_param(TransformerModelParamIdx::thresh.into(), thresh);
    pop.fill_param(TransformerModelParamIdx::k_scale.into(), k_scale);
    pop.fill_param(TransformerModelParamIdx::eta_k.into(), eta_k);
    pop.fill_param(TransformerModelParamIdx::eta_v.into(), eta_v);
    pop.fill_param(TransformerModelParamIdx::w_base.into(), w_base);
    pop.fill_param(TransformerModelParamIdx::w_span.into(), w_span);
    pop.fill_param(TransformerModelParamIdx::v_base.into(), v_base);
    pop.fill_param(TransformerModelParamIdx::v_span.into(), v_span);
    // Старт: равномерное внимание (все ключи/значения на базе).
    for i in 0..8 {
        pop.fill_state(TransformerModelStateIdx::w0 as usize + i, w_base);
        pop.fill_state(TransformerModelStateIdx::v0 as usize + i, v_base);
    }
    pop
}

/// Inject Poisson-group events into an arbitrary buffer (общая логика).
fn inject_poisson_into(
    groups: &[PoissonGroup],
    rng: &mut rand::rngs::StdRng,
    buf: &mut [f32],
    dt_ms: f32,
) {
    for pg in groups {
        if pg.rate_hz <= 0.0 || pg.weight == 0.0 {
            continue;
        }
        let p = pg.rate_hz * dt_ms / 1000.0;
        if p <= 0.0 {
            continue;
        }
        if p >= 1.0 {
            for &idx in &pg.indices {
                if idx < buf.len() {
                    buf[idx] += pg.weight;
                }
            }
        } else {
            for &idx in &pg.indices {
                if idx < buf.len() && rng.gen::<f32>() < p {
                    buf[idx] += pg.weight;
                }
            }
        }
    }
}

// ─── Снапшоты состояния гетерогенного мозга ───────────────────────

/// Полное состояние одной группы гетерогенного мозга (чистые данные).
///
/// Параметры/состояния моделей хранятся в SoA-колонках (как
/// `NeuralPopulation`), длина колонок — SIMD-паддинг (кратна 8). Для
/// Izh/Rate — отдельные массивы. Снапшот не сериализуем сам по себе;
/// сериализация — в крейте `taas-format`.
#[derive(Debug, Clone)]
pub enum HeteroGroupState {
    /// LIF + alpha-синапс (Shiu et al. 2024).
    Alpha {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        /// N_PARAMS колонки (dt, time_factor, v_rest, v_thresh, v_reset,
        /// refr_steps, syn_decay).
        params: Vec<Vec<f32>>,
        /// N_STATE колонки (v, g, refr_cnt).
        state: Vec<Vec<f32>>,
        delay_ring: Vec<f32>,
        /// Число слотов задержки (steps_delay + 1).
        delay_size: usize,
        delay_pos: usize,
    },
    /// Hodgkin-Huxley (standard или pacemaking).
    Hh {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
    },
    /// HAAP (M-ток + h-ток).
    Haap {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
        /// Персистентный синаптический ток.
        syn: Vec<f32>,
        decay: f32,
    },
    /// Izhikevich (regular spiking).
    Izh {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        v: Vec<f32>,
        u: Vec<f32>,
        trace: Vec<f32>,
        refr: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
        dt_ms: f32,
        tau_ref: f32,
    },
    /// Rate model.
    Rate {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        v: Vec<f32>,
        tau: f32,
        dt_ms: f32,
    },
    /// SurgeN — нейрон-трансформер (pillars zoo `TransformerModel`).
    Transformer {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
    },
}

impl HeteroGroupState {
    /// Индексы нейронов группы (глобальные).
    pub fn indices(&self) -> &[usize] {
        match self {
            HeteroGroupState::Alpha { indices, .. }
            | HeteroGroupState::Hh { indices, .. }
            | HeteroGroupState::Haap { indices, .. }
            | HeteroGroupState::Izh { indices, .. }
            | HeteroGroupState::Rate { indices, .. }
            | HeteroGroupState::Transformer { indices, .. } => indices,
        }
    }

    /// Имя типа группы (для диагностики/формата).
    pub fn kind(&self) -> &'static str {
        match self {
            HeteroGroupState::Alpha { .. } => "alpha",
            HeteroGroupState::Hh { .. } => "hh",
            HeteroGroupState::Haap { .. } => "haap",
            HeteroGroupState::Izh { .. } => "izh",
            HeteroGroupState::Rate { .. } => "rate",
            HeteroGroupState::Transformer { .. } => "transformer",
        }
    }
}

/// Полный слепок состояния [HeteroRunner].
#[derive(Debug, Clone)]
pub struct HeteroSnapshot {
    pub connectome: ConnectomeData,
    pub step_counter: usize,
    pub dt_ms: f32,
    /// Сид RNG Пуассоновских групп (см. док [`HeteroRunner`]).
    pub poisson_seed: [u8; 32],
    pub groups: Vec<HeteroGroupState>,
}

/// Колонки параметров/состояния модели (SoA, как `NeuralPopulation`).
fn pop_columns<M: pillars::zoo::Model + 'static>(
    pop: &pillars::NeuralPopulation<M>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let params = (0..M::N_PARAMS).map(|c| pop.param_slice(c).to_vec()).collect();
    let state = (0..M::N_STATE).map(|c| pop.state_slice(c).to_vec()).collect();
    (params, state)
}

/// Пересобрать популяцию из снапшот-колонок.
fn pop_from_columns<M: pillars::zoo::Model + 'static>(
    params: &[Vec<f32>],
    state: &[Vec<f32>],
    n: usize,
    kind: &str,
) -> Result<pillars::NeuralPopulation<M>, String> {
    if params.len() != M::N_PARAMS || state.len() != M::N_STATE {
        return Err(format!(
            "hetero restore: {kind}: колонок {}/{} != {}/{}",
            params.len(),
            state.len(),
            M::N_PARAMS,
            M::N_STATE
        ));
    }
    let mut pop = pillars::NeuralPopulation::<M>::new(n);
    for (c, col) in params.iter().enumerate() {
        let sl = pop.param_slice_mut(c);
        if col.len() > sl.len() {
            return Err(format!("hetero restore: {kind} p_{c} длина {}", col.len()));
        }
        sl[..col.len()].copy_from_slice(col);
    }
    for (c, col) in state.iter().enumerate() {
        let sl = pop.state_slice_mut(c);
        if col.len() > sl.len() {
            return Err(format!("hetero restore: {kind} s_{c} длина {}", col.len()));
        }
        sl[..col.len()].copy_from_slice(col);
    }
    Ok(pop)
}

impl HeteroRunner {
    /// Полный слепок текущего состояния (для save/load через taas-format).
    ///
    /// Веса `connectome.values` — текущие (после пластичности), так что
    /// restore возвращает ровно то же состояние, включая пластику.
    pub fn snapshot(&self) -> HeteroSnapshot {
        let groups = self
            .groups
            .iter()
            .map(|g| match g {
                GroupEngine::Alpha {
                    indices,
                    pop,
                    drive_scale,
                    rec_scale,
                    delay_steps,
                    delay_ring,
                    delay_pos,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Alpha {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                        delay_ring: delay_ring.clone(),
                        delay_size: delay_steps + 1,
                        delay_pos: *delay_pos,
                    }
                }
                GroupEngine::Hh {
                    indices,
                    pop,
                    drive_scale,
                    rec_scale,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Hh {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                    }
                }
                GroupEngine::Haap {
                    indices,
                    pop,
                    syn,
                    decay,
                    drive_scale,
                    rec_scale,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Haap {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                        syn: syn.clone(),
                        decay: *decay,
                    }
                }
                GroupEngine::Izh {
                    indices,
                    v,
                    u,
                    trace,
                    refr,
                    a,
                    b,
                    c,
                    d,
                    dt_ms,
                    tau_ref,
                    drive_scale,
                    rec_scale,
                } => HeteroGroupState::Izh {
                    indices: indices.clone(),
                    drive_scale: *drive_scale,
                    rec_scale: *rec_scale,
                    v: v.clone(),
                    u: u.clone(),
                    trace: trace.clone(),
                    refr: refr.clone(),
                    a: a.clone(),
                    b: b.clone(),
                    c: c.clone(),
                    d: d.clone(),
                    dt_ms: *dt_ms,
                    tau_ref: *tau_ref,
                },
                GroupEngine::Rate {
                    indices,
                    v,
                    tau,
                    dt_ms,
                    drive_scale,
                    rec_scale,
                } => HeteroGroupState::Rate {
                    indices: indices.clone(),
                    drive_scale: *drive_scale,
                    rec_scale: *rec_scale,
                    v: v.clone(),
                    tau: *tau,
                    dt_ms: *dt_ms,
                },
                GroupEngine::Transformer {
                    indices,
                    pop,
                    drive_scale,
                    rec_scale,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Transformer {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                    }
                }
            })
            .collect();
        HeteroSnapshot {
            connectome: self.connectome.clone(),
            step_counter: self.step_counter,
            dt_ms: self.dt_ms,
            poisson_seed: self.poisson_seed,
            groups,
        }
    }

    /// Восстановить состояние ровно из снапшота.
    ///
    /// Пересобирает группы из снапшота (параметры+состояние), заменяет
    /// коннектом (в.ч. веса после пластичности) и сбрасывает очереди.
    /// Пуассон после `restore` стартует с `poisson_seed` (см. док модуля).
    pub fn restore(&mut self, snap: &HeteroSnapshot) -> Result<(), String> {
        let n = snap.connectome.n_neurons;
        if n != self.n {
            return Err(format!(
                "hetero restore: n_neurons {} != {}",
                n, self.n
            ));
        }
        if snap.dt_ms != self.dt_ms {
            return Err(format!(
                "hetero restore: dt_ms {} != {}",
                snap.dt_ms, self.dt_ms
            ));
        }

        // Валидируем покрытие группами как в `new`.
        let mut covered = vec![false; n];
        let mut engines = Vec::with_capacity(snap.groups.len());
        let mut specs = Vec::with_capacity(snap.groups.len());
        for gs in &snap.groups {
            if gs.indices().is_empty() {
                continue;
            }
            for &idx in gs.indices() {
                if idx >= n {
                    return Err(format!("hetero restore: index {idx} >= n={n}"));
                }
                if covered[idx] {
                    return Err(format!("hetero restore: neuron {idx} covered twice"));
                }
                covered[idx] = true;
            }
            let (eng, spec) = engine_from_state(gs, snap.dt_ms)?;
            specs.push(spec);
            engines.push(eng);
        }

        // Какие-то нейроны не покрыты — отклоняем (как в new).
        if let Some(idx) = covered.iter().position(|&c| !c) {
            return Err(format!("hetero restore: neuron {idx} is not covered"));
        }

        self.connectome = snap.connectome.clone();
        self.csr_graph = ValidatedCsrGraph::from_slices(
            &self.connectome.row_ptr,
            &self.connectome.col_indices,
            n,
        );
        self.specs = specs;
        self.groups = engines;
        self.step_counter = snap.step_counter;
        self.poisson_seed = snap.poisson_seed;
        self.rng = rand::rngs::StdRng::from_seed(snap.poisson_seed);
        self.queue.clear();
        self.rec.fill(0.0);
        self.drive.fill(0.0);
        self.spikes.fill(0);
        self.spike_buffer.fill(0);
        Ok(())
    }
}

/// Пересобрать один GroupEngine из снапшот-состояния группы.
fn engine_from_state(
    gs: &HeteroGroupState,
    dt_ms: f32,
) -> Result<(GroupEngine, HeteroGroup), String> {
    let k = gs.indices().len();
    let spec = HeteroGroup {
        config: match gs {
            // Конфиг нужен только для specs; сам engine строим прямыми полями.
            HeteroGroupState::Alpha { .. } => NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref: 2.2,
                tau_m: 20.0,
                v_rest: -52.0,
                v_thresh: -45.0,
                v_reset: -52.0,
                tau_syn: 5.0,
                t_delay: 1.8,
            },
            HeteroGroupState::Hh { .. } => NeuronConfig::Hh {
                dt_ms,
                use_pacemaking: false,
            },
            HeteroGroupState::Haap { .. } => NeuronConfig::HaapSynaptic {
                dt_ms,
                tau_syn: 12.0,
            },
            HeteroGroupState::Izh { .. } => NeuronConfig::Izh {
                dt_ms,
                tau_ref: 2.0,
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
            HeteroGroupState::Rate { .. } => NeuronConfig::Rate { dt_ms, tau: 0.0 },
            HeteroGroupState::Transformer { .. } => {
                NeuronConfig::Transformer {
                    dt_ms,
                    decay: 0.95,
                    gain: 1.0,
                    inv_tau_acc: 0.5,
                    thresh: 0.35,
                    k_scale: 0.5,
                    eta_k: 0.0,
                    eta_v: 0.0,
                    w_base: 1.0,
                    w_span: 2.0,
                    v_base: 1.0,
                    v_span: 2.0,
                }
            }
        },
        indices: gs.indices().to_vec(),
        drive_scale: gs_drive(gs),
        rec_scale: gs_rec(gs),
    };
    let eng = match gs {
        HeteroGroupState::Alpha {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
            delay_ring,
            delay_size,
            delay_pos,
        } => {
            let pop = pop_from_columns::<pillars::zoo::AlphaLifModel>(params, state, k, "alpha")?;
            GroupEngine::Alpha {
                indices: indices.clone(),
                pop,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
                delay_steps: delay_size.saturating_sub(1),
                delay_ring: delay_ring.clone(),
                delay_pos: *delay_pos % (*delay_size).max(1),
            }
        }
        HeteroGroupState::Hh {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
        } => {
            let pop = pop_from_columns::<pillars::zoo::HHModel>(params, state, k, "hh")?;
            GroupEngine::Hh {
                indices: indices.clone(),
                pop,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
            }
        }
        HeteroGroupState::Haap {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
            syn,
            decay,
        } => {
            let pop = pop_from_columns::<pillars::zoo::HAAPModel>(params, state, k, "haap")?;
            GroupEngine::Haap {
                indices: indices.clone(),
                pop,
                syn: syn.clone(),
                decay: *decay,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
            }
        }
        HeteroGroupState::Izh {
            indices,
            drive_scale,
            rec_scale,
            v,
            u,
            trace,
            refr,
            a,
            b,
            c,
            d,
            dt_ms,
            tau_ref,
        } => GroupEngine::Izh {
            indices: indices.clone(),
            v: v.clone(),
            u: u.clone(),
            trace: trace.clone(),
            refr: refr.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            d: d.clone(),
            dt_ms: *dt_ms,
            tau_ref: *tau_ref,
            drive_scale: *drive_scale,
            rec_scale: *rec_scale,
        },
        HeteroGroupState::Rate {
            indices,
            drive_scale,
            rec_scale,
            v,
            tau,
            dt_ms,
        } => GroupEngine::Rate {
            indices: indices.clone(),
            v: v.clone(),
            tau: *tau,
            dt_ms: *dt_ms,
            drive_scale: *drive_scale,
            rec_scale: *rec_scale,
        },
        HeteroGroupState::Transformer {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
        } => {
            let pop = pop_from_columns::<pillars::zoo::TransformerModel>(
                params,
                state,
                k,
                "transformer",
            )?;
            GroupEngine::Transformer {
                indices: indices.clone(),
                pop,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
            }
        }
    };
    Ok((eng, spec))
}

fn gs_drive(gs: &HeteroGroupState) -> f32 {
    match gs {
        HeteroGroupState::Alpha { drive_scale, .. }
        | HeteroGroupState::Hh { drive_scale, .. }
        | HeteroGroupState::Haap { drive_scale, .. }
        | HeteroGroupState::Izh { drive_scale, .. }
        | HeteroGroupState::Rate { drive_scale, .. }
        | HeteroGroupState::Transformer { drive_scale, .. } => *drive_scale,
    }
}

fn gs_rec(gs: &HeteroGroupState) -> f32 {
    match gs {
        HeteroGroupState::Alpha { rec_scale, .. }
        | HeteroGroupState::Hh { rec_scale, .. }
        | HeteroGroupState::Haap { rec_scale, .. }
        | HeteroGroupState::Izh { rec_scale, .. }
        | HeteroGroupState::Rate { rec_scale, .. }
        | HeteroGroupState::Transformer { rec_scale, .. } => *rec_scale,
    }
}

// ─── Тесты ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConnectomeData;

    /// 4 нейрона: 0→1, 1→2, 2→3, 3→0 (кольцо). Веса 0.8/0.5/0.3/0.7.
    fn ring() -> ConnectomeData {
        ConnectomeData::new(
            vec![0i32, 1, 2, 3, 4],
            vec![1, 2, 3, 0],
            vec![0.8f32, 0.5, 0.3, 0.7],
            vec![1.0, 1.0, -1.0, 1.0],
            vec![("a", vec![0, 1]), ("b", vec![2, 3])],
        )
    }

    /// Гетерогенный мозг: цепочка 0(alpha)→1(Izh)→2(alpha) с сильными весами
    /// + Rate-нейрон 3, живущий рядом. Спайк alpha-0 пробивает Izh-1, тот —
    /// alpha-2: сигнал пересекает границы моделей в одном коннектоме.
    #[test]
    fn heterogeneous_mixed_models_share_connectome() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::izh(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 0.001),
                indices: vec![3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        // Rate-3 получает собственный драйв (как сенсорный канал популяции).
        let ext = vec![30.0f32, 0.0, 0.0, 1.0];
        let spikes = runner.run(300, &ext).to_vec();

        let s0: i32 = spikes.iter().step_by(4).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(4).copied().sum();
        let s2: i32 = spikes.iter().skip(2).step_by(4).copied().sum();
        let s3: i32 = spikes.iter().skip(3).step_by(4).copied().sum();
        assert!(s0 > 0, "alpha 0 should fire, got {s0}");
        assert!(s1 > 0, "Izh 1 should fire from alpha 0, got {s1}");
        assert!(s2 > 0, "alpha 2 must fire from Izh 1 (cross-model), got {s2}");
        assert!(s3 > 0, "Rate 3 should fire from its own drive, got {s3}");
    }

    /// Спайк из alpha-группы реально доходит до HH-группы (тонкий шаг dt,
    /// HH численно нестабилен при 0.5): цепочка 0(alpha)→1(HH)→2(alpha).
    #[test]
    fn cross_model_propagation_alpha_hh_alpha() {
        // 0 → 1 → 2, вес 40.0: alpha-нейрон 2 приходит в равновесие
        // v = v_rest + g, а порог на 7 мВ выше v_rest — g=40 пробивает.
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.05),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::hh(0.05),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.05).unwrap();
        let ext = vec![30.0f32, 0.0, 0.0];
        let spikes = runner.run(10000, &ext).to_vec();

        let s0: i32 = spikes.iter().step_by(3).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(3).copied().sum();
        let s2: i32 = spikes.iter().skip(2).step_by(3).copied().sum();
        assert!(s0 > 0, "drive neuron 0 should fire, got {s0}");
        assert!(s1 > 0, "HH neuron 1 should fire from alpha, got {s1}");
        assert!(
            s2 > 0,
            "alpha neuron 2 must fire from HH neuron 1 (cross-model), got {s2}"
        );
    }

    /// Спайк из alpha-группы реально доходит до Transformer-группы и из неё
    /// — в следующий alpha: цепочка 0(alpha)→1(transformer)→2(alpha).
    /// Трансформер держит память входов и спайкает на устойчивом сигнале.
    #[test]
    fn cross_model_propagation_alpha_transformer_alpha() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::transformer(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        let ext = vec![30.0f32, 0.0, 0.0];
        let spikes = runner.run(3000, &ext).to_vec();

        let s0: i32 = spikes.iter().step_by(3).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(3).copied().sum();
        let s2: i32 = spikes.iter().skip(2).step_by(3).copied().sum();
        assert!(s0 > 0, "drive neuron 0 should fire, got {s0}");
        assert!(
            s1 > 0,
            "transformer neuron 1 should fire from alpha drive, got {s1}"
        );
        assert!(
            s2 > 0,
            "alpha neuron 2 must fire from transformer neuron 1 (cross-model), got {s2}"
        );
    }

    /// Round-trip снапшота: прогон с `restore` на полпути бит-в-бит
    /// совпадает с непрерывным прогоном (без Пуассона). Состояния всех
    /// групп (incl. delay ring и персистентных токов) восстанавливаются.
    #[test]
    fn snapshot_restore_roundtrip_bitwise() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::izh(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 0.001),
                indices: vec![3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let ext = vec![30.0f32, 0.0, 0.0, 1.0];

        // Контроль: непрерывный прогон 300 шагов.
        let mut a = HeteroRunner::new(conn.clone(), groups.clone(), 0.5).unwrap();
        let full = a.run(300, &ext).to_vec();

        // Снапшот на полпути: 200 шагов → снимок → restore → ещё 100.
        let mut b = HeteroRunner::new(conn, groups, 0.5).unwrap();
        let first = b.run(200, &ext).to_vec();
        let snap = b.snapshot();
        // Проверяем, что веса после пластичности/симуляции = как в контроле.
        assert_eq!(snap.connectome.values, a.connectome.values);
        assert_eq!(snap.step_counter, 200);
        b.restore(&snap).unwrap();
        let second = b.run(100, &ext).to_vec();

        // Склейка = непрерывный прогон.
        assert_eq!(first.len() + second.len(), full.len());
        for i in 0..first.len() {
            assert_eq!(first[i], full[i], "first segment mismatch at {i}");
        }
        for (j, &s) in second.iter().enumerate() {
            assert_eq!(s, full[200 * 4 + j], "second segment mismatch at {j}");
        }
        assert_eq!(b.step_counter(), 300);
    }

    /// Пустое покрытие нейрона в снапшоте — ошибка restore.
    #[test]
    fn restore_rejects_uncovered() {
        let conn = ring();
        let groups = vec![HeteroGroup {
            config: crate::BrainConfig::lif_alpha(0.5),
            indices: vec![0, 1, 2, 3],
            drive_scale: 1.0,
            rec_scale: 1.0,
        }];
        let mut r = HeteroRunner::new(conn, groups, 0.5).unwrap();
        r.run(10, &[1.0; 4]);
        let snap = r.snapshot();
        // Ломаем покрытие: убираем нейрон 3 из группы.
        let mut bad = snap.clone();
        if let HeteroGroupState::Alpha { indices, .. } = &mut bad.groups[0] {
            indices.pop();
        }
        assert!(r.restore(&bad).is_err());
    }

    /// Пустое покрытие нейрона — ошибка.
    #[test]
    fn overlapping_groups_rejected() {
        let conn = ring();
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 10.0),
                indices: vec![1, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        assert!(HeteroRunner::new(conn, groups, 0.5).is_err());
    }

    /// Пустой коннектом — ошибка.
    #[test]
    fn empty_connectome_rejected() {
        let conn = ConnectomeData::new(vec![0i32], vec![], vec![], vec![], vec![]);
        let groups = vec![HeteroGroup {
            config: crate::BrainConfig::lif_alpha(0.5),
            indices: vec![0],
            drive_scale: 1.0,
            rec_scale: 1.0,
        }];
        assert!(HeteroRunner::new(conn, groups, 0.5).is_err());
    }

    /// Poisson-стимуляция в гетерогенном мозге.
    #[test]
    fn poisson_drives_hetero_brain() {
        let conn = ring();
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 10.0),
                indices: vec![2, 3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        runner.add_poisson_group(&[0, 1], 500.0, 50.0);
        let spikes = runner.run(100, &[0.0; 4]).to_vec();
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "poisson-only should fire neurons, got {total}");
    }

    /// HAAP-группа с персистентным синаптическим током (τ_syn > 0).
    #[test]
    fn haap_group_runs_with_synaptic_decay() {
        // 0(alpha) → 1(haap), вес 40 (HAAP-порог ~40 мВ от покоя −65).
        let conn = ConnectomeData::new(
            vec![0i32, 1, 1],
            vec![1],
            vec![40.0f32],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.05),
                indices: vec![0],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::haap_synaptic(0.05, 12.0),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.05).unwrap();
        let ext = vec![30.0f32, 0.0];
        let spikes = runner.run(2000, &ext).to_vec();
        let s0: i32 = spikes.iter().step_by(2).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(2).copied().sum();
        assert!(s0 > 0, "alpha drive should fire, got {s0}");
        assert!(s1 > 0, "HAAP should fire from alpha input, got {s1}");
    }
}
