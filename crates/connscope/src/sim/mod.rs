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

use pillars_connectome::{BrainRunner, ConnectomeData};

use crate::core::nt::Nt;
use crate::io::data::MancData;

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
    for t in (0..steps).step_by(chunk) {
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
    for t in (0..steps).step_by(chunk) {
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
        find_disinhibition, induced_subgraph, run_full_sim, run_full_sim_stream, run_simulation,
        StreamMsg,
    };
    use crate::io::data::{tiny_data, MancData};

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
        let r = run_simulation(&d, &[0, 1, 2], 0, 0, 20.0, 200, 1.0, 0.5, 0.3).unwrap(); // 0 = LIF
        assert_eq!(r.idx, vec![0, 1, 2]);
        assert_eq!(r.raster.len(), 200 * 3);
        assert!(!r.vtraces.is_empty());
        assert_eq!(r.rates.len(), 3);
    }

    /// Потоковая симуляция: шаги приходят по каналу, в конце — итог.
    #[test]
    fn test_stream_full_sim_lif() {
        let d = tiny_data();
        let mut st = run_full_sim_stream(&d, 0, 0, 20.0, 100, 1.0, 0.5, 0.3).unwrap(); // 0 = LIF
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
        let r = run_full_sim(&d, 0, 0, 20.0, 100, 1.0, 0.5, 0.3).unwrap(); // 0 = LIF
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
        let r = run_full_sim(&d, 158, 0, 30.0, 200, 0.6, 0.5, 0.3).unwrap(); // 0 = LIF
        eprintln!(
            "full sim 200 шагов: {:?}, спайков={}, активных={}",
            t0.elapsed(),
            r.total_spikes,
            r.active_neurons
        );
    }
}
