//! 🧠 bake_run.rs — оживляет любой .taas коннектом: LIF или HH, со стимуляцией.
//!
//! Usage:
//!     cargo run -p pillars-connectome --release --example bake_run -- \
//!         resources/data/fafb_783.taas lif 50
//!         resources/data/manc_vnc.taas hh 30
//!         resources/data/manc_vnc.taas hh 30 resources/data/manc_vnc_hh_physio.hhparams
//!         resources/data/c_elegans_pharynx.taas lif 200
//!
//! 4-й аргумент (опционально) — .hhparams: per-neuron литературная физиология
//! (7 столбцов f32: g_na, g_k, g_l, e_na, e_k, e_l, inv_cm).
//!
//! Считает: total spikes, активные нейроны, спайки по суперклассам.

use pillars::zoo::HHModelParamIdx;
use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

const SENSORY_OR_EXC: [f32; 11] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
const LIF_STIM: f32 = 40.0;
const HH_STIM: f32 = 20.0;

/// Загрузить .hhparams (magic 0x48485031, n_neurons u64, 7·n f32 столбцов SoA).
fn load_hhparams(path: &str, n: usize) -> Vec<Vec<f32>> {
    let data = std::fs::read(path).expect("cannot read .hhparams");
    assert!(data.len() >= 12, "truncated .hhparams header");
    let mut b = [0u8; 12];
    b.copy_from_slice(&data[..12]);
    let magic = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let n_stored = u64::from_le_bytes(b[4..12].try_into().unwrap()) as usize;
    assert_eq!(magic, 0x4848_5031, "bad .hhparams magic");
    assert_eq!(n_stored, n, "n_neurons mismatch: file={n_stored} brain={n}");
    assert_eq!(data.len(), 12 + 7 * n * 4, "truncated .hhparams body");
    let mut cols = Vec::with_capacity(7);
    let mut off = 12;
    for _ in 0..7 {
        let mut v = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(off) as *const f32, v.as_mut_ptr(), n);
        }
        off += n * 4;
        cols.push(v);
    }
    cols
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: bake_run <file.taas> <lif|hh> <ms> [physio.hhparams]");
    let model = args.get(2).map(|s| s.as_str()).unwrap_or("lif");
    let ms: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let physio_path = args.get(4).map(|s| s.as_str());

    let f = std::fs::File::open(path).expect("cannot open .taas");
    let mut conn = ConnectomeData::load_baked(&mut std::io::BufReader::new(f)).expect("bad .taas");
    let n = conn.n_neurons;
    println!(
        "== {} | {} нейронов, {} синапсов, receptor_types: {} ==",
        path,
        n,
        conn.n_synapses,
        conn.receptor_types.len()
    );
    conn.build_receptor_types();
    let nt = conn.nt_types.clone();

    let mut runner = match model {
        "hh" => BrainRunner::new(
            conn,
            NeuronConfig::HhSynaptic {
                dt_ms: 0.05,
                use_pacemaking: false,
            },
        ),
        _ => BrainRunner::new(
            conn,
            NeuronConfig::LifSynaptic {
                dt_ms: 0.1,
                tau_ref: 2.2,
                tau_m: 20.0,
                v_rest: -52.0,
                v_thresh: -45.0,
                v_reset: -52.0,
            },
        ),
    }
    .expect("BrainRunner init");

    // per-neuron литературная физиология (только для HH)
    if model == "hh" {
        if let Some(p) = physio_path {
            let cols = load_hhparams(p, runner.n_neurons());
            // колонки .hhparams: g_na, g_k, g_l, e_na, e_k, e_l, inv_cm
            let map: [(usize, usize); 7] = [
                (0, HHModelParamIdx::g_Na.into()),
                (1, HHModelParamIdx::g_K.into()),
                (2, HHModelParamIdx::g_Leak.into()),
                (3, HHModelParamIdx::E_Na.into()),
                (4, HHModelParamIdx::E_K.into()),
                (5, HHModelParamIdx::E_Leak.into()),
                (6, HHModelParamIdx::inv_cm.into()),
            ];
            for (src, dst) in map {
                let nw = runner.set_hh_param_column(dst, &cols[src]);
                assert_eq!(nw, runner.n_neurons());
            }
            println!("  применена литературная физиология из {p}");
        } else {
            println!("  физиология: стандартный HH (один набор на всех)");
        }
    }

    let dt_ms = if model == "hh" { 0.05f32 } else { 0.1 };
    let n_steps = ((ms as f32) / dt_ms) as usize;
    let stim = if model == "hh" { HH_STIM } else { LIF_STIM };

    // стимуляция: постоянный ток на сенсорные/возбуждающие классы (nt_types 1..10)
    let stim_ids: Vec<usize> = nt
        .iter()
        .enumerate()
        .filter(|(_, &t)| SENSORY_OR_EXC.contains(&t) && t > 0.0 && t != 4.0) // не моторные
        .map(|(i, _)| i)
        .take(3000)
        .collect();
    println!("стимулируем {} нейронов (ток {})…", stim_ids.len(), stim);

    let mut ext = vec![0.0f32; n];
    for &i in &stim_ids {
        ext[i] = stim;
    }

    let t0 = std::time::Instant::now();
    let spikes = runner.run(n_steps, &ext).to_vec();
    let elapsed = t0.elapsed();

    // статистика
    let mut total = 0u64;
    let mut active = 0u32;
    let mut by_class = vec![0u64; 11];
    for step in 0..n_steps {
        for i in 0..n {
            if spikes[step * n + i] != 0 {
                total += 1;
                let c = nt[i] as usize;
                if c < 11 {
                    by_class[c] += 1;
                }
            }
        }
    }
    for i in 0..n {
        let mut has = false;
        for step in 0..n_steps {
            if spikes[step * n + i] != 0 {
                has = true;
                break;
            }
        }
        if has {
            active += 1;
        }
    }

    let ms_sim = n_steps as f64 * dt_ms as f64;
    println!("─── результат ───");
    println!(
        "симуляция: {:.0} мс мозга ({n_steps} шагов) за {:.2} с",
        ms_sim,
        elapsed.as_secs_f64()
    );
    println!("total spikes: {total}");
    println!(
        "активных нейронов: {active}/{n} ({:.1}%)",
        100.0 * active as f64 / n as f64
    );
    let names = [
        "unk", "optic", "sensory", "central", "motor", "desc", "asc", "endocr", "visProj",
        "visCent", "sensAsc",
    ];
    for (c, &cnt) in by_class.iter().enumerate() {
        if cnt > 0 {
            println!("  class {}({:>7}): {cnt} спайков", c, names[c]);
        }
    }
    println!(
        "скорость: {:.1}M нейрон-шагов/с",
        n as f64 * n_steps as f64 / elapsed.as_secs_f64() / 1e6
    );
}
