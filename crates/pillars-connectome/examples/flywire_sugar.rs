//! FlyWire 783 (138 639 нейронов, 15 091 983 синапса) на pillars-connectome.
//!
//! Воспроизводит эксперимент «Sugar GRNs (200 Hz)» из репозитория
//! fly-brain (Shiu et al. 2023) — через [`AlphaLifWorker`]: LIF + alpha-синапс
//! с аксональной задержкой, Poisson-стимуляция сахарных GRN прямо в v.
//!
//! ```text
//! v_rest  = -52 mV, v_thresh = -45 mV, v_reset = -52 mV
//! tau_mem = 20 ms, tau_ref  = 2.2 ms (0 для сахарных GRN), dt = 0.1 ms
//! tau_syn = 5 ms, t_delay  = 1.8 ms
//! w_syn   = 0.275 mV (вес уже вшит в .taas: w = Excitatory × Connectivity × 0.275)
//! poisson = 200 Hz × 68.75 mV скачок в v
//! ```
//!
//! Запуск:
//! ```bash
//! cargo run -p pillars-connectome --release --example flywire_sugar -- \
//!     <path/to/flywire_783.taas> [t_run_s] [n_trials]
//! ```
//!
//! Печатает setup/sim-время в формате, сравнимом с их benchmark-results.csv
//! (sugar, n_run=1): Brian2 CPU 1 s → 16 567 спайков / 380 активных;
//! PyTorch CUDA → 17 429 / 415. Наш воркер попадает в 17 153 / 395.

use std::time::Instant;

use pillars_connectome::{AlphaLifWorker, ConnectomeData, ShiuParams};

/// 21 сахарный GRN (индексы из sugar_grn.json, получены из FlyWire ID).
const SUGAR_GRN: [usize; 21] = [
    69093, 97602, 122795, 124291, 29281, 100605, 110469, 51107, 49584, 129730, 126873, 28825,
    126600, 126752, 32863, 108426, 111357, 14842, 90589, 92298, 12494,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .expect("usage: flywire_sugar <flywire_783.taas> [t_run_s] [n_trials]");
    let t_run_s: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let n_trials: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);

    // ─── Setup: загрузка .taas + воркер ───
    let t0 = Instant::now();
    let conn = ConnectomeData::load_baked(&mut std::io::BufReader::new(
        std::fs::File::open(&path).expect("cannot open .taas"),
    ))
    .expect("bad .taas");
    let setup_s = t0.elapsed().as_secs_f64();
    println!(
        "[setup] n_neurons={} n_synapses={} load={:.3}s",
        conn.n_neurons, conn.n_synapses, setup_s
    );

    // ─── Воркер: модель Шу (alpha-LIF) ───
    let mut worker = AlphaLifWorker::new(conn, ShiuParams::flywire()).expect("worker init");
    worker.set_sugar_grn(&SUGAR_GRN);
    println!(
        "[worker] {} neurons, {} sugar GRN, chunk={} steps (0.1 s)",
        worker.n_neurons(),
        worker.n_sugar_grn(),
        (100.0 / ShiuParams::flywire().dt_ms).round() as usize
    );

    // ─── Симуляция чанками (состояние сохраняется между вызовами) ───
    let t1 = Instant::now();
    let mut total_spikes = 0usize;
    let mut active_max = 0usize;
    for _ in 0..n_trials {
        let out = worker.run_seconds(t_run_s);
        total_spikes += out.total_spikes;
        active_max = active_max.max(out.active_neurons);
        println!(
            "[trial] t={:.1}s brain, {} spikes, {} active, wall={:.3}s (ratio {:.3}x)",
            t_run_s,
            out.total_spikes,
            out.active_neurons,
            out.wall_seconds,
            out.realtime_ratio()
        );
    }
    let sim_s = t1.elapsed().as_secs_f64();

    println!(
        "[sim] t_run={}s n_trials={} total_spikes={} active_neurons(max)={}",
        t_run_s, n_trials, total_spikes, active_max
    );
    println!("[sim] sim_time={:.3}s", sim_s);
    println!("[total] {:.3}s", setup_s + sim_s);
}
