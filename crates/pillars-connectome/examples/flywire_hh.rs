//! FlyWire 783 (138 639 нейронов, 15 091 983 синапса) на pillars-connectome — **HH**.
//!
//! Тот же сценарий «Sugar GRNs (200 Hz)», что в flywire_sugar.rs (LIF), но на
//! настоящем Hodgkin-Huxley с биологической физиологией:
//!
//! ```text
//! g_Na = 120 mS/cm², E_Na = +50 mV   (быстрый натрий)
//! g_K  =  36 mS/cm², E_K  = −77 mV   (задержанный калий)
//! g_L  =   0.3 mS/cm², E_L = −65 mV  (утечка)
//! dt = 0.01 ms (HH численно стабилен только на мелком шаге)
//! ```
//!
//! Это параметры твоего атласа (`generate_model_params.py`: HAAP → HH),
//! совпадающие с `hh_population_standard` в pillars — той самой моделью,
//! что проходит parity-тесты против старого LUT-ядра.
//!
//! Проводимостные синапсы (`HhSynaptic`, τ_syn = 5 мс) — как в LIF-модели Шу,
//! но нейроны — настоящий HH. Без проводимости спайки умирают за один шаг.
//!
//! Запуск:
//! ```bash
//! cargo run -p pillars-connectome --release --example flywire_hh -- \
//!     resources/data/flywire_783.taas [t_run_ms] [n_trials]
//! ```

use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

/// 21 сахарный GRN (индексы из sugar_grn.json, получены из FlyWire ID).
const SUGAR_GRN: [usize; 21] = [
    69093, 97602, 122795, 124291, 29281, 100605, 110469, 51107, 49584, 129730, 126873, 28825,
    126600, 126752, 32863, 108426, 111357, 14842, 90589, 92298, 12494,
];

const DT_MS: f32 = 0.01;
/// HH спайкает при токе ≳ 10 мкА/см² (parity-тест: I=15 → спайки).
/// Держим на GRN постоянный ток 20 мкА/см² — как оптиогенетическая стимуляция.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "resources/data/flywire_783.taas".into());
    let t_run_ms: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(50.0);
    let n_trials: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);

    // ─── Setup: загрузка .taas ───
    let t0 = Instant::now();
    let f = File::open(&path).expect("cannot open .taas");
    let conn = ConnectomeData::load_baked(&mut BufReader::new(f)).expect("bad .taas");
    let setup_s = t0.elapsed().as_secs_f64();
    println!(
        "[setup] n_neurons={} n_synapses={} load={:.3}s",
        conn.n_neurons, conn.n_synapses, setup_s
    );

    // ─── Конфиг HH + проводимостные синапсы ───
    let mut conn = conn;
    conn.build_receptor_types();
    let config = NeuronConfig::HhSynaptic {
        dt_ms: DT_MS,
        use_pacemaking: false,
    };

    // ─── Симуляция ───
    let n_steps = (t_run_ms / DT_MS) as usize;
    let t1 = Instant::now();
    let mut runner = BrainRunner::new(conn, config).expect("runner init");
    // HH — нелинейный интегратор: одиночные редкие пуассоновские импульсы не
    // зажигают спайк, нужен постоянный удерживающий ток на GRN (как оптиогенетика).
    let mut ext = vec![0.0f32; runner.n_neurons()];
    for &g in &SUGAR_GRN {
        ext[g] = 20.0;
    }

    let nn = runner.n_neurons();
    let mut total_spikes = 0usize;
    let mut active = std::collections::HashSet::new();
    for _ in 0..n_trials {
        let spikes = runner.run(n_steps, &ext).to_vec(); // снимок буфера
        for &s in spikes.iter() {
            if s != 0 {
                total_spikes += 1;
            }
        }
        // активные нейроны — по последнему trial
        let n_steps_here = spikes.len() / nn;
        for step in 0..n_steps_here {
            let off = step * nn;
            for i in 0..nn {
                if spikes[off + i] != 0 {
                    active.insert(i);
                }
            }
        }
    }
    let sim_s = t1.elapsed().as_secs_f64();

    let sim_ms_of_brain = t_run_ms * n_trials as f32;
    let realtime_ratio = sim_ms_of_brain as f64 / (sim_s * 1000.0);

    println!("[sim] model=HH dt={DT_MS}ms t_run={t_run_ms}ms n_trials={n_trials} n_steps={n_steps}");
    println!(
        "[sim] sim_time={:.3}s total_spikes={} active_neurons={}",
        sim_s, total_spikes, active.len()
    );
    println!(
        "[sim] realtime_ratio={:.4}x (мозг работает в {:.1} раз быстрее реального времени)",
        realtime_ratio,
        1.0 / realtime_ratio.max(1e-9)
    );
    println!("[total] {:.3}s", setup_s + sim_s);
}
