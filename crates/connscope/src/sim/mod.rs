//! Симуляция локального контура + растормаживание + живой поток.
//!
//! Модели нейронов — плагины из [`models`]: в коде хранится **индекс** в
//! [`models::MODELS`], а механика (имя/конфиг/настройка runner'а) живёт в
//! файле плагина. Новый нейрон = один файл в `models/` + строка в реестре.

pub mod models;
pub mod neuroml;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

use crate::core::nt::Nt;
use crate::core::pharma::{Drug, Target};
use crate::io::data::MancData;

// ── Фармакология ────────────────────────────────────────────

/// Колонки параметров pillars-моделей (порядок `HHModelParamIdx` /
/// `HAAPModelParamIdx`, генерируется макросом `NeuronModel` из порядка полей):
///
/// HH:   0=dt, 1=g_Na, 2=E_Na, 3=g_K, 4=E_K, 5=g_Leak, 6=E_Leak, 7=q10,
///       8=v_thresh, 9=inv_cm, 10=g_CaT, 11=E_CaT
/// HAAP: 0=dt, 1=inv_cm, 2=g_l, 3=e_l, 4=g_m, 5=e_k, 6=g_h, 7=e_h, …
const HH_COL_G_NA: usize = 1;
const HH_COL_G_K: usize = 3;
const HH_COL_G_LEAK: usize = 5;
const HAAP_COL_G_L: usize = 2;
const HAAP_COL_G_M: usize = 4;
const HAAP_COL_G_H: usize = 6;

/// Доза препарата для прогона: препарат + концентрация (µM).
#[derive(Debug, Clone)]
pub struct PharmaDose {
    pub drug: Drug,
    pub conc_um: f32,
}

impl PharmaDose {}

/// Фармакокинетика: доза + период полувыведения в **шагах симуляции**
/// (шаг = `dt_ms`). Однокомпартментная PK: болюс `C0` в `t=0`,
/// экспоненциальный клиренс `C(t) = C0·2^(−t/T½)`.
///
/// `half_life_steps <= 0` — концентрация постоянна (классический случай).
#[derive(Debug, Clone)]
pub struct PharmaPk {
    pub dose: PharmaDose,
    pub half_life_steps: f32,
}

impl PharmaPk {
    /// Постоянная концентрация (без кинетики).
    pub fn constant(dose: PharmaDose) -> Self {
        Self {
            dose,
            half_life_steps: 0.0,
        }
    }

    /// Концентрация в µM на шаге `t`.
    pub fn conc_at(&self, t: usize) -> f32 {
        if self.half_life_steps <= 0.0 {
            return self.dose.conc_um;
        }
        self.dose.conc_um
            * (-(t as f32) * std::f32::consts::LN_2 / self.half_life_steps).exp()
    }

    /// Множитель мишени на шаге `t` (1.0 = не действует).
    pub fn scale_at(&self, t: usize) -> f32 {
        self.dose.drug.scale(self.conc_at(t))
    }
}

/// Честная причина, почему мишень недоступна модели.
fn channel_missing_msg(target: Target) -> &'static str {
    match target {
        Target::NaV => "NaV-канал есть только в HH",
        Target::KV => "KV-канал есть только в HH",
        Target::CaT => "CaT-ток есть только в pacemaking-HH",
        Target::Leak => "leak-канал есть только в HH/HAAP",
        Target::Hcn => "h-ток есть только в HAAP",
        Target::M => "M-ток есть только в HAAP",
        _ => "мишень недоступна",
    }
}

/// Есть ли у модели проводимость-мишень препарата.
/// Канальные мишени живут только в HH (NaV/KV/Leak) и HAAP (M/Hcn/Leak);
/// рецепторные (GABA_A/nAChR/GluCl) масштабируют синапсы — подходят любой модели.
pub fn pharma_target_fit(target: Target, model: usize) -> Result<(), &'static str> {
    if !target.is_channel() {
        return Ok(());
    }
    match models::MODELS.get(model) {
        Some(m) if m.has_channel(target) => Ok(()),
        _ => Err(channel_missing_msg(target)),
    }
}

/// Кэш «дозы на базе» для идемпотентного пересчёта на каждом чанке:
/// первый вызов захватывает оригиналы (веса синапсов или колонку
/// проводимости), дальше каждый чанк ставит база·scale(t) — накопления нет.
enum PharmaBaseline {
    /// Канальная мишень: колонка проводимости + её оригинал (без препарата).
    Channel {
        col: usize,
        is_haap: bool,
        label: &'static str,
        base: Vec<f32>,
        name: String,
    },
    /// Рецепторная мишень: код НТ пресинаптиков + оригинал весов синапсов.
    Receptor {
        code: u8,
        nt_name: &'static str,
        base: Vec<f32>,
        name: String,
        count: usize,
    },
}

/// Применить фарму к runner'у на чанке `t` (идемпотентно: каждый вызов
/// ставит база·scale(t), накопления нет — кинетика не «складывается»).
///
/// * **канальные мишени** — масштабирование проводимостей HH/HAAP
///   (`g_Na`, `g_K`, `g_Leak` / `g_m`, `g_h`, `g_l`);
/// * **рецепторные мишени** — масштабирование весов синапсов, чей
///   пресинаптический нейрон имеет соответствующий НТ (GABA_A→GABA,
///   nAChR→ACh, GluCl→GLUT).
///
/// `nt_codes` — коды НТ всех нейронов (`MancData::nt`); `local_of` —
/// отображение локальный → глобальный индекс для подграфа (None — полная сеть).
///
/// Первый вызов захватывает базу и возвращает статус; повторные — пустую
/// строку. Ошибка (мишень недоступна модели / нет таких синапсов) — `Err`
/// со статусом: прогон продолжается без препарата.
fn apply_pharma_chunk(
    runner: &mut BrainRunner,
    nt_codes: &[u8],
    local_of: Option<&[usize]>,
    pk: &PharmaPk,
    t: usize,
    baseline: &mut Option<PharmaBaseline>,
) -> Result<String, String> {
    let scale = pk.scale_at(t);
    if baseline.is_none() && (scale - 1.0).abs() < 1e-4 {
        // Доза не действует с первого шага (и не начнёт — концентрация
        // только убывает): препарат просто не участвует в прогоне.
        return Ok(String::new());
    }
    if baseline.is_none() {
        // ── первый вызов: проверка мишени + захват базы ──
        let d = &pk.dose.drug;
        if d.target.is_channel() {
            let is_hh = matches!(
                runner.config,
                NeuronConfig::Hh { .. } | NeuronConfig::HhSynaptic { .. }
            );
            let is_haap = matches!(
                runner.config,
                NeuronConfig::Haap { .. } | NeuronConfig::HaapSynaptic { .. }
            );
            let present = match d.target {
                Target::NaV | Target::KV => is_hh,
                Target::CaT => false, // стандартный HH: g_CaT = 0, тока нет
                Target::Leak => is_hh || is_haap,
                Target::Hcn | Target::M => is_haap,
                _ => false,
            };
            if !present {
                return Err(format!("⚠ {}: {}", d.name, channel_missing_msg(d.target)));
            }
            // Leak есть и в HH, и в HAAP — берём ту колонку, что реально есть.
            let (col, is_haap, label, base) = match d.target {
                Target::NaV => (
                    HH_COL_G_NA,
                    false,
                    "g_Na",
                    runner.get_hh_param_column(HH_COL_G_NA),
                ),
                Target::KV => (
                    HH_COL_G_K,
                    false,
                    "g_K",
                    runner.get_hh_param_column(HH_COL_G_K),
                ),
                Target::Leak => {
                    let hh_base = runner.get_hh_param_column(HH_COL_G_LEAK);
                    if !hh_base.is_empty() {
                        (HH_COL_G_LEAK, false, "g_Leak", hh_base)
                    } else {
                        (HAAP_COL_G_L, true, "g_l", runner.get_haap_param_column(HAAP_COL_G_L))
                    }
                }
                Target::Hcn => (
                    HAAP_COL_G_H,
                    true,
                    "g_h",
                    runner.get_haap_param_column(HAAP_COL_G_H),
                ),
                Target::M => (
                    HAAP_COL_G_M,
                    true,
                    "g_m",
                    runner.get_haap_param_column(HAAP_COL_G_M),
                ),
                _ => return Ok(String::new()),
            };
            if base.is_empty() {
                return Err(format!("⚠ {}: {}", d.name, channel_missing_msg(d.target)));
            }
            *baseline = Some(PharmaBaseline::Channel {
                col,
                is_haap,
                label,
                base,
                name: d.name.clone(),
            });
        } else {
            let Some(nt) = d.target.receptor_nt() else {
                return Ok(String::new());
            };
            let code = nt as u8;
            let n = runner.n_neurons();
            let row_ptr = &runner.connectome.row_ptr;
            let mut count = 0usize;
            for pre in 0..n {
                let global = local_of.map(|m| m[pre]).unwrap_or(pre);
                if nt_codes.get(global).copied() == Some(code) {
                    count += row_ptr[pre + 1].max(0) as usize - row_ptr[pre].max(0) as usize;
                }
            }
            if count == 0 {
                return Err(format!(
                    "⚠ {}: в данных нет нейронов с {} (синапсы не тронуты)",
                    d.name,
                    nt.name()
                ));
            }
            *baseline = Some(PharmaBaseline::Receptor {
                code,
                nt_name: nt.name(),
                base: runner.connectome.values.clone(),
                name: d.name.clone(),
                count,
            });
        }
    }
    // ── применение: база·scale(t) ──
    let note = match baseline.as_ref().unwrap() {
        PharmaBaseline::Channel {
            col,
            is_haap,
            label,
            base,
            name,
        } => {
            let scaled: Vec<f32> = base.iter().map(|&x| x * scale).collect();
            let n = if *is_haap {
                runner.set_haap_param_column(*col, &scaled)
            } else {
                runner.set_hh_param_column(*col, &scaled)
            };
            if t == 0 {
                format!("{name}: {label} → ×{scale:.2} ({n} нейронов)")
            } else {
                String::new()
            }
        }
        PharmaBaseline::Receptor {
            code,
            nt_name,
            base,
            name,
            count,
        } => {
            runner.connectome.values.copy_from_slice(base);
            let n = runner.n_neurons();
            let (row_ptr, values) = (&runner.connectome.row_ptr, &mut runner.connectome.values);
            for pre in 0..n {
                let global = local_of.map(|m| m[pre]).unwrap_or(pre);
                if nt_codes.get(global).copied() != Some(*code) {
                    continue;
                }
                let lo = row_ptr[pre].max(0) as usize;
                let hi = row_ptr[pre + 1].max(0) as usize;
                let hi = hi.min(values.len());
                for v in &mut values[lo..hi] {
                    *v *= scale;
                }
            }
            if t == 0 {
                format!("{name}: {nt_name} → ×{scale:.2} ({count} синапсов)")
            } else {
                String::new()
            }
        }
    };
    if t == 0 && pk.half_life_steps > 0.0 && !note.is_empty() {
        Ok(format!("{note} · T½={} шагов", pk.half_life_steps as i64))
    } else {
        Ok(note)
    }
}

/// Индуцированный CSR-подграф на `nodes` (в порядке nodes) + remap.
pub fn induced_subgraph(data: &MancData, nodes: &[usize]) -> (Vec<usize>, ConnectomeData) {
    let mut idx: Vec<usize> = nodes.to_vec();
    idx.sort_unstable();
    idx.dedup();
    let remap: HashMap<usize, usize> = idx.iter().enumerate().map(|(k, &g)| (g, k)).collect();
    let k = idx.len();
    let mut rp = vec![0i32; k + 1];
    let mut ci: Vec<i32> = Vec::new();
    let mut val: Vec<f32> = Vec::new();
    let mut nt: Vec<f32> = Vec::with_capacity(k);
    for &g in &idx {
        let lo = data.row_ptr[g].max(0) as usize;
        let hi = data.row_ptr[g + 1].max(0) as usize;
        for e in lo..hi.min(data.s) {
            let t = data.col[e] as usize;
            if let Some(&lt) = remap.get(&t) {
                ci.push(lt as i32);
                val.push(data.val[e]);
            }
        }
        rp[idx.iter().position(|&x| x == g).unwrap() + 1] = ci.len() as i32;
        nt.push(data.nt_types[g]);
    }
    let conn = ConnectomeData::new(rp, ci, val, nt, vec![]);
    (idx, conn)
}

/// Результат симуляции: растр `[steps·k]`, частоты `[k]`, V-трейсы `[chunks·k]`.
pub struct SimResult {
    pub idx: Vec<usize>,        // глобальные индексы (локальный порядок)
    pub raster: Vec<i32>,       // steps * k (0/1)
    pub rates: Vec<f32>,        // k
    pub vtraces: Vec<Vec<f32>>, // n_chunks × k
    pub steps: usize,
    pub dt_ms: f32,
    pub chunk: usize,
    pub drive_local: Option<usize>,
    /// Полное имя модели (для строк статуса).
    pub model: &'static str,
    /// Статус применённого препарата (пусто — препарата не было).
    pub pharma_note: String,
}

pub fn run_simulation(
    data: &MancData,
    nodes: &[usize],
    drive: usize,
    model: usize,
    bias: f32,
    steps: usize,
    pulse_frac: f32,
    dt_ms: f32,
    g_h: f32,
    pk: Option<PharmaPk>,
) -> Result<SimResult, String> {
    let (idx, conn) = induced_subgraph(data, nodes);
    if idx.is_empty() {
        return Err("пустой контур".into());
    }
    let m = models::MODELS[model];
    let mut runner = BrainRunner::new(conn, m.config(dt_ms))?;
    m.tune(&mut runner, idx.len(), g_h);
    let drive_local = idx.iter().position(|&g| g == drive);
    let chunk = 10usize;
    let mut raster = Vec::with_capacity(steps * idx.len());
    let mut vtraces: Vec<Vec<f32>> = Vec::new();
    let pulse_end = (steps as f32 * pulse_frac) as usize;
    let mut baseline: Option<PharmaBaseline> = None;
    let mut pharma_note = String::new();
    for t in (0..steps).step_by(chunk) {
        if let Some(p) = pk.as_ref() {
            match apply_pharma_chunk(&mut runner, &data.nt, Some(&idx), p, t, &mut baseline) {
                Ok(n) if !n.is_empty() => pharma_note = n,
                Err(e) => pharma_note = e,
                _ => {}
            }
        }
        let n = chunk.min(steps - t);
        let mut ext = vec![0.0f32; idx.len()];
        if let Some(d) = drive_local {
            if t < pulse_end {
                ext[d] = bias;
            }
        }
        let buf = runner.run(n, &ext);
        raster.extend_from_slice(buf);
        vtraces.push(runner.get_v().to_vec());
    }
    let k = idx.len();
    let mut rates = vec![0.0f32; k];
    let duration_s = steps as f32 * dt_ms / 1000.0;
    if duration_s > 0.0 {
        for step in 0..steps {
            let base = step * k;
            for i in 0..k {
                rates[i] += raster[base + i] as f32;
            }
        }
        for r in &mut rates {
            *r /= duration_s;
        }
    }
    Ok(SimResult {
        idx,
        raster,
        rates,
        vtraces,
        steps,
        dt_ms,
        chunk,
        drive_local,
        model: m.name(),
        pharma_note,
    })
}

/// Растормаживание: src →⊣ mid →⊣ y (торможение тормоза = растормаживание y).
pub fn find_disinhibition(
    data: &MancData,
    src: usize,
    top: usize,
) -> Vec<(usize, usize, f32, f32)> {
    let mut chains: Vec<(usize, usize, f32, f32)> = Vec::new();
    let lo = data.row_ptr[src].max(0) as usize;
    let hi = data.row_ptr[src + 1].max(0) as usize;
    for e in lo..hi.min(data.s) {
        let mid = data.col[e] as usize;
        if mid >= data.n || !Nt::from_code(data.nt[mid]).inhibitory() {
            continue;
        }
        let mlo = data.row_ptr[mid].max(0) as usize;
        let mhi = data.row_ptr[mid + 1].max(0) as usize;
        for e2 in mlo..mhi.min(data.s) {
            let y = data.col[e2] as usize;
            if y != src {
                chains.push((mid, y, data.val[e], data.val[e2]));
            }
        }
    }
    chains.sort_by(|a, b| {
        (b.2 * b.3)
            .partial_cmp(&(a.2 * a.3))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    chains.truncate(top);
    chains
}

/// Результат симуляции всей сети: популяционная активность по шагам,
/// разбивка по классам (индекс = код класса 0..7) и доля сработавших.
pub struct FullSimResult {
    pub steps: usize,
    pub dt_ms: f32,
    pub drive: usize,
    /// Полное имя модели (для строк статуса).
    pub model: &'static str,
    /// спайков на шаг (все классы)
    pub activity: Vec<f32>,
    /// спайков на шаг по классам
    pub class_activity: Vec<[f32; 8]>,
    /// кумулятивная доля сработавших нейронов [0..1] по шагам
    pub activated: Vec<f32>,
    pub total_spikes: i64,
    pub active_neurons: usize,
    /// первый шаг срабатывания каждого нейрона (u32::MAX = не сработал)
    pub first_fire: Vec<u32>,
    /// сколько раз сработал каждый нейрон
    pub spike_counts: Vec<u32>,
    /// времена спайков каждого нейрона (шаги, по возрастанию) — для
    /// ISI-гистограммы и фазового анализа моторов (антифаза)
    pub spike_times: Vec<Vec<u32>>,
    /// Статус применённого препарата (пусто — препарата не было).
    pub pharma_note: String,
}

/// Итог симуляции: локальный контур (ego) или полная сеть.
pub enum SimOutcome {
    Local(SimResult),
    Full(FullSimResult),
}

/// Полный прогон всей сети: стимул в `drive`, считаем популяционную активность
/// по шагам и разбивку по классам — видно, как сигнал идёт по мозгу.
pub fn run_full_sim(
    data: &MancData,
    drive: usize,
    model: usize,
    bias: f32,
    steps: usize,
    pulse_frac: f32,
    dt_ms: f32,
    g_h: f32,
    pk: Option<PharmaPk>,
) -> Result<FullSimResult, String> {
    let n = data.n;
    if n == 0 {
        return Err("нет данных".into());
    }
    if drive >= n {
        return Err(format!("драйв #{drive} вне сети ({n})"));
    }
    let conn = ConnectomeData::new(
        data.row_ptr.clone(),
        data.col.clone(),
        data.val.clone(),
        data.nt_types.clone(),
        vec![],
    );
    let m = models::MODELS[model];
    let mut runner = BrainRunner::new(conn, m.config(dt_ms))?;
    m.tune(&mut runner, n, g_h);
    let chunk = 10usize;
    let pulse_end = (steps as f32 * pulse_frac) as usize;
    let mut fired = vec![false; n];
    let mut activated_count = 0usize;
    let mut first_fire = vec![u32::MAX; n];
    let mut spike_counts = vec![0u32; n];
    let mut spike_times: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut activity = vec![0.0f32; steps];
    let mut class_activity = vec![[0.0f32; 8]; steps];
    let mut activated = vec![0.0f32; steps];
    let mut total_spikes: i64 = 0;
    let mut baseline: Option<PharmaBaseline> = None;
    let mut pharma_note = String::new();
    for t in (0..steps).step_by(chunk) {
        if let Some(p) = pk.as_ref() {
            match apply_pharma_chunk(&mut runner, &data.nt, None, p, t, &mut baseline) {
                Ok(n) if !n.is_empty() => pharma_note = n,
                Err(e) => pharma_note = e,
                _ => {}
            }
        }
        let ns = chunk.min(steps - t);
        let mut ext = vec![0.0f32; n];
        if t < pulse_end {
            ext[drive] = bias;
        }
        let buf = runner.run(ns, &ext);
        for s in 0..ns {
            let step = t + s;
            let base = s * n;
            let mut cnt = 0.0f32;
            for i in 0..n {
                if buf[base + i] != 0 {
                    cnt += 1.0;
                    total_spikes += 1;
                    if !fired[i] {
                        fired[i] = true;
                        activated_count += 1;
                        first_fire[i] = step as u32;
                    }
                    spike_counts[i] += 1;
                    spike_times[i].push(step as u32);
                    let k = (data.klass[i] as usize).min(7);
                    class_activity[step][k] += 1.0;
                }
            }
            activity[step] = cnt;
            activated[step] = activated_count as f32 / n as f32;
        }
    }
    Ok(FullSimResult {
        steps,
        dt_ms,
        drive,
        model: m.name(),
        activity,
        class_activity,
        activated,
        total_spikes,
        active_neurons: activated_count,
        first_fire,
        spike_counts,
        spike_times,
        pharma_note,
    })
}

// ── живой сигнал: потоковая симуляция ────────────────────────────

/// Событие потоковой симуляции: на каждом шаге — список сработавших
/// нейронов (глобальные индексы), в конце — итоговый результат.
pub enum StreamMsg {
    Fired { step: usize, fired: Vec<usize> },
    Done(Box<FullSimResult>),
}

/// Управление фоновым потоком симуляции: приёмник событий + флаг отмены.
pub struct SimStream {
    pub rx: Receiver<StreamMsg>,
    pub cancel: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

impl SimStream {
    /// Останавливает поток (флаг отмены + join, чтобы не оставить висящий поток).
    pub fn stop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Потоковый полный прогон всей сети: та же математика, что в `run_full_sim`,
/// но шаги **потоком** — на каждый шаг в канал уходит список сработавших
/// нейронов, а UI анимирует по ним «свечение» в 3D (см. `app::sim_ui`).
///
/// Поток бежит на максимальной скорости — темп показа задаёт UI
/// (`live_speed` шагов/с): события копятся в канале и потребляются
/// дозированно. Отмена — `SimStream::stop` / флаг `cancel`.
pub fn run_full_sim_stream(
    data: &MancData,
    drive: usize,
    model: usize,
    bias: f32,
    steps: usize,
    pulse_frac: f32,
    dt_ms: f32,
    g_h: f32,
    pk: Option<PharmaPk>,
) -> Result<SimStream, String> {
    let n = data.n;
    if n == 0 {
        return Err("нет данных".into());
    }
    if drive >= n {
        return Err(format!("драйв #{drive} вне сети ({n})"));
    }
    let conn = ConnectomeData::new(
        data.row_ptr.clone(),
        data.col.clone(),
        data.val.clone(),
        data.nt_types.clone(),
        vec![],
    );
    let klass = data.klass.clone();
    let nt_codes = data.nt.clone();
    let m = models::MODELS[model];
    let config = m.config(dt_ms);
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let c2 = cancel.clone();
    let handle = thread::Builder::new()
        .name("sim_stream".into())
        .spawn(move || {
            let mut runner = match BrainRunner::new(conn, config) {
                Ok(r) => r,
                Err(_) => return,
            };
            m.tune(&mut runner, n, g_h);
            let mut baseline: Option<PharmaBaseline> = None;
            let mut pharma_note = String::new();
            let chunk = 10usize;
            let pulse_end = (steps as f32 * pulse_frac) as usize;
            let mut fired = vec![false; n];
            let mut activated_count = 0usize;
            let mut first_fire = vec![u32::MAX; n];
            let mut spike_counts = vec![0u32; n];
            let mut spike_times: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut activity = vec![0.0f32; steps];
            let mut class_activity = vec![[0.0f32; 8]; steps];
            let mut activated = vec![0.0f32; steps];
            let mut total_spikes: i64 = 0;
            'outer: for t in (0..steps).step_by(chunk) {
                if c2.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(p) = pk.as_ref() {
                    match apply_pharma_chunk(&mut runner, &nt_codes, None, p, t, &mut baseline) {
                        Ok(n) if !n.is_empty() => pharma_note = n,
                        Err(e) => pharma_note = e,
                        _ => {}
                    }
                }
                let ns = chunk.min(steps - t);
                let mut ext = vec![0.0f32; n];
                if t < pulse_end {
                    ext[drive] = bias;
                }
                let buf = runner.run(ns, &ext);
                for s in 0..ns {
                    let step = t + s;
                    let base = s * n;
                    let mut cnt = 0.0f32;
                    let mut fired_list: Vec<usize> = Vec::new();
                    for i in 0..n {
                        if buf[base + i] != 0 {
                            cnt += 1.0;
                            total_spikes += 1;
                            if !fired[i] {
                                fired[i] = true;
                                activated_count += 1;
                                first_fire[i] = step as u32;
                            }
                            spike_counts[i] += 1;
                            spike_times[i].push(step as u32);
                            let k = (klass[i] as usize).min(7);
                            class_activity[step][k] += 1.0;
                            fired_list.push(i);
                        }
                    }
                    activity[step] = cnt;
                    activated[step] = activated_count as f32 / n as f32;
                    if tx
                        .send(StreamMsg::Fired {
                            step,
                            fired: fired_list,
                        })
                        .is_err()
                    {
                        break 'outer; // приёмник закрыт (окно закрылось)
                    }
                }
            }
            let res = FullSimResult {
                steps,
                dt_ms,
                drive,
                model: m.name(),
                activity,
                class_activity,
                activated,
                total_spikes,
                active_neurons: activated_count,
                first_fire,
                spike_counts,
                spike_times,
                pharma_note,
            };
            let _ = tx.send(StreamMsg::Done(Box::new(res)));
        })
        .map_err(|e| format!("не удалось запустить поток: {e}"))?;
    Ok(SimStream {
        rx,
        cancel,
        handle: Some(handle),
    })
}

// ── Тесты ─────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        apply_pharma_chunk, find_disinhibition, induced_subgraph, pharma_target_fit, run_full_sim,
        run_full_sim_stream, run_simulation, PharmaBaseline, PharmaDose, PharmaPk, StreamMsg,
    };
    use crate::core::pharma::{Drug, Effect, Target};
    use crate::io::data::{tiny_data, MancData};

    /// Тестовый препарат: блокатор NaV с IC50 = 0.01 µM → при 0.01 µM scale = 0.5.
    fn ttx() -> Drug {
        Drug {
            name: "TTX".into(),
            target: Target::NaV,
            effect: Effect::Block,
            potency_um: 0.01,
            hill: 1.0,
            emax: 0.0,
            mechanism: "тест".into(),
            reference: "тест".into(),
            note: None,
        }
    }

    fn dose(drug: Drug, conc_um: f32) -> PharmaDose {
        PharmaDose { drug, conc_um }
    }

    /// Постоянная кинетика (классический случай) из дозы.
    fn const_pk(drug: Drug, conc_um: f32) -> PharmaPk {
        PharmaPk::constant(dose(drug, conc_um))
    }

    /// Канальный препарат на HH: колонка g_Na масштабируется в runner'е.
    #[test]
    fn pharma_scales_hh_g_na() {
        let d = tiny_data();
        let r = run_full_sim(&d, 0, 2, 20.0, 50, 1.0, 0.5, 0.3, Some(const_pk(ttx(), 0.01)))
            .unwrap(); // 2 = HH
        assert!(r.pharma_note.contains("g_Na"), "note: {}", r.pharma_note);
        assert!(r.pharma_note.contains("×0.50"), "note: {}", r.pharma_note);
    }

    /// Канальный препарат на LIF: честный отказ, прогон не падает.
    #[test]
    fn pharma_channel_rejected_on_lif() {
        let d = tiny_data();
        let r = run_full_sim(&d, 0, 0, 20.0, 50, 1.0, 0.5, 0.3, Some(const_pk(ttx(), 0.01)))
            .unwrap(); // 0 = LIF
        assert!(r.pharma_note.starts_with("⚠"), "note: {}", r.pharma_note);
        assert!(r.pharma_note.contains("NaV"), "note: {}", r.pharma_note);
    }

    /// Рецепторный препарат: масштабируются синапсы от GABA-пресинаптиков.
    #[test]
    fn pharma_receptor_scales_gaba_synapses() {
        let mut d = tiny_data();
        d.nt[0] = 2; // GABA
        d.nt[1] = 1; // ACh
        d.nt[2] = 3; // GLUT
        let picro = Drug {
            name: "пикротоксин".into(),
            target: Target::GabaA,
            effect: Effect::Antagonize,
            potency_um: 1.0,
            hill: 1.0,
            emax: 0.0,
            mechanism: "тест".into(),
            reference: "тест".into(),
            note: None,
        };
        // Локальный контур: GABA(0)→ACh(1) идёт через синапс 0→1.
        let r = run_simulation(
            &d,
            &[0, 1, 2],
            0,
            0,
            20.0,
            200,
            1.0,
            0.5,
            0.3,
            Some(const_pk(picro, 1.0)),
        )
        .unwrap(); // IC50 → ×0.50
        assert!(r.pharma_note.contains("GABA"), "note: {}", r.pharma_note);
        assert!(r.pharma_note.contains("×0.50"), "note: {}", r.pharma_note);
        assert!(r.pharma_note.contains("1 синапс"), "note: {}", r.pharma_note);
    }

    /// Нулевая концентрация = препарат не действует (note пустой).
    #[test]
    fn pharma_inactive_dose_is_noop() {
        let d = tiny_data();
        let r = run_full_sim(&d, 0, 2, 20.0, 50, 1.0, 0.5, 0.3, Some(const_pk(ttx(), 0.0)))
            .unwrap();
        assert!(r.pharma_note.is_empty(), "note: {}", r.pharma_note);
    }

    /// Однокомпартментная PK: через T½ шагов концентрация ровно C0/2.
    #[test]
    fn pk_conc_halves_after_half_life() {
        let pk = PharmaPk {
            dose: dose(ttx(), 10.0),
            half_life_steps: 100.0,
        };
        assert!((pk.conc_at(0) - 10.0).abs() < 1e-4);
        assert!((pk.conc_at(100) - 5.0).abs() < 1e-3, "C(T½) = C0/2");
        assert!((pk.conc_at(200) - 2.5).abs() < 1e-3, "через 2×T½ — C0/4");
        assert!(pk.conc_at(1000) < 0.01, "клиренс почти полный");
        // Постоянная кинетика не убывает.
        let c = PharmaPk::constant(dose(ttx(), 10.0));
        assert_eq!(c.conc_at(0), 10.0);
        assert_eq!(c.conc_at(999), 10.0);
    }

    /// Идемпотентность: повторное применение с тем же scale не накапливает
    /// (g_Na = база·scale), а scale=1.0 восстанавливает базу.
    #[test]
    fn pharma_chunk_is_idempotent_and_restores() {
        let d = tiny_data();
        let mut runner = pillars_connectome::BrainRunner::new(
            pillars_connectome::ConnectomeData::new(
                d.row_ptr.clone(),
                d.col.clone(),
                d.val.clone(),
                d.nt_types.clone(),
                vec![],
            ),
            pillars_connectome::BrainConfig::hh(0.5),
        )
        .unwrap();
        let g0 = runner.get_hh_param_column(1)[0]; // g_Na базы
        let pk = PharmaPk {
            dose: dose(ttx(), 0.01), // scale = 0.5
            half_life_steps: 0.0,
        };
        let mut baseline: Option<PharmaBaseline> = None;
        apply_pharma_chunk(&mut runner, &d.nt, None, &pk, 0, &mut baseline).unwrap();
        let g1 = runner.get_hh_param_column(1)[0];
        assert!((g1 - g0 * 0.5).abs() < 1e-3, "база·0.5: {g0} → {g1}");
        apply_pharma_chunk(&mut runner, &d.nt, None, &pk, 10, &mut baseline).unwrap();
        let g2 = runner.get_hh_param_column(1)[0];
        assert!((g2 - g1).abs() < 1e-4, "повтор не накапливает: {g1} vs {g2}");
        // scale = 1.0 (концентрация выклиренсилась) → база восстановилась.
        let pk1 = PharmaPk {
            dose: dose(ttx(), 0.0),
            half_life_steps: 0.0,
        };
        // scale_at(0) с conc 0 = 1.0, но база уже захвачена — применяется база·1.
        let mut b2: Option<PharmaBaseline> = baseline.take();
        apply_pharma_chunk(&mut runner, &d.nt, None, &pk1, 20, &mut b2).unwrap();
        let g3 = runner.get_hh_param_column(1)[0];
        assert!((g3 - g0).abs() < 1e-3, "восстановление базы: {g0} vs {g3}");
    }

    /// Совместимость мишени с моделью: каналы — только HH/HAAP, рецепторы — все.
    #[test]
    fn pharma_target_fit_matrix() {
        assert!(pharma_target_fit(Target::NaV, 0).is_err()); // LIF
        assert!(pharma_target_fit(Target::NaV, 1).is_err()); // Izh
        assert!(pharma_target_fit(Target::NaV, 2).is_ok()); // HH
        assert!(pharma_target_fit(Target::NaV, 3).is_err()); // HAAP
        assert!(pharma_target_fit(Target::Hcn, 3).is_ok()); // HAAP
        assert!(pharma_target_fit(Target::M, 3).is_ok()); // HAAP
        assert!(pharma_target_fit(Target::Leak, 2).is_ok()); // HH
        assert!(pharma_target_fit(Target::GabaA, 0).is_ok()); // рецептор — любая
    }

    #[test]
    fn test_induced_subgraph() {
        let d = tiny_data();
        let (idx, conn) = induced_subgraph(&d, &[0, 1, 2]);
        assert_eq!(idx, vec![0, 1, 2]);
        assert_eq!(conn.n_neurons, 3);
        assert_eq!(conn.n_synapses, 3);
        // 0→1, 1→2, 2→0 all kept
        assert_eq!(conn.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_find_disinhibition() {
        // 0 (inhibitory GABA) → 1 (inhibitory) → 2
        let mut d = tiny_data();
        d.nt[0] = 2; // GABA (торм.)
        d.nt[1] = 3; // GLUT (торм.)
        d.nt[2] = 1; // ACh
                     // edges: 0→1, 1→2, 2→0
        let chains = find_disinhibition(&d, 0, 10);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].0, 1); // mid
        assert_eq!(chains[0].1, 2); // y
    }

    #[test]
    fn test_run_simulation_lif() {
        let d = tiny_data();
        let r = run_simulation(&d, &[0, 1, 2], 0, 0, 20.0, 200, 1.0, 0.5, 0.3, None).unwrap(); // 0 = LIF
        assert_eq!(r.idx, vec![0, 1, 2]);
        assert_eq!(r.raster.len(), 200 * 3);
        assert!(!r.vtraces.is_empty());
        assert_eq!(r.rates.len(), 3);
    }

    /// Потоковая симуляция: шаги приходят по каналу, в конце — итог.
    #[test]
    fn test_stream_full_sim_lif() {
        let d = tiny_data();
        let mut st = run_full_sim_stream(&d, 0, 0, 20.0, 100, 1.0, 0.5, 0.3, None).unwrap(); // 0 = LIF
        let mut last_step = 0usize;
        let mut done = false;
        while let Ok(msg) = st.rx.recv() {
            match msg {
                StreamMsg::Fired { step, .. } => last_step = step + 1,
                StreamMsg::Done(r) => {
                    assert_eq!(r.steps, 100);
                    assert!(r.total_spikes >= 0);
                    done = true;
                    break;
                }
            }
        }
        assert!(done, "поток должен завершиться событием Done");
        assert!(last_step > 0, "должны прийти шаги с вспышками");
        // отмена не паникует на завершённом потоке
        st.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        st.stop();
    }

    #[test]
    fn test_run_full_sim_lif() {
        let d = tiny_data();
        let r = run_full_sim(&d, 0, 0, 20.0, 100, 1.0, 0.5, 0.3, None).unwrap(); // 0 = LIF
        assert_eq!(r.steps, 100);
        assert_eq!(r.activity.len(), 100);
        assert_eq!(r.class_activity.len(), 100);
        assert_eq!(r.activated.len(), 100);
        assert!(r.active_neurons <= d.n);
        // сумма по классам == общая активность каждого шага
        for step in 0..r.steps {
            let sum: f32 = r.class_activity[step].iter().sum();
            assert!((sum - r.activity[step]).abs() < 0.5);
        }
        // доля активированных не убывает
        for step in 1..r.steps {
            assert!(r.activated[step] >= r.activated[step - 1]);
        }
    }

    /// Замер скорости полного прогона VNC на реальном графе (запуск вручную: --ignored).
    #[test]
    #[ignore]
    fn test_full_vnc_real_speed() {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../resources/examples/data/manc_viz.bin");
        if !p.exists() {
            eprintln!("skip: bin не найден");
            return;
        }
        let d = MancData::load(&p).unwrap();
        let t0 = std::time::Instant::now();
        let r = run_full_sim(&d, 158, 0, 30.0, 200, 0.6, 0.5, 0.3, None).unwrap(); // 0 = LIF
        eprintln!(
            "full sim 200 шагов: {:?}, спайков={}, активных={}",
            t0.elapsed(),
            r.total_spikes,
            r.active_neurons
        );
    }
}
