//! FlyWire 783 (138 639 нейронов, 15 091 983 синапса) — GPU vs CPU.
//!
//! Тот же эксперимент «Sugar GRNs (200 Hz)», но на GPU
//! (`pillars_cuda::AlphaLifGpu`), с CPU-воркером рядом для сравнения.
//!
//! ```bash
//! cargo run -p pillars-connectome --features cuda --release \
//!     --example flywire_sugar_gpu -- resources/data/flywire_783.taas 1.0
//! ```
//!
//! Нейрон и Poisson bit-identical (тот же `StdRng`, тот же `alphalif_step`);
//! scatter — атомарный push (O(fired_edges)), как в эталонных GPU-симуляторах
//! (PyTorch/NEST/GeNN), поэтому спайк-счёты совпадают в пределах ~1 ulp.

use std::time::Instant;

use pillars_connectome::{AlphaLifWorker, ConnectomeData, ShiuParams};
use pillars_cuda::{AlphaLifGpu, AlphaLifParams};

/// 21 сахарный GRN (из resources/data/sugar_grn.json).
const SUGAR_GRN: [usize; 21] = [
    69093, 97602, 122795, 124291, 29281, 100605, 110469, 51107, 49584, 129730, 126873, 28825,
    126600, 126752, 32863, 108426, 111357, 14842, 90589, 92298, 12494,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .expect("usage: flywire_sugar_gpu <flywire_783.taas> [t_run_s]");
    let t_run_s: f32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1.0);

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

    // ── GPU воркер (CSC gather scatter + alphalif_step, bit-exact) ──
    let t1 = Instant::now();
    let mut gpu = AlphaLifGpu::new(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        AlphaLifParams::flywire(),
        &SUGAR_GRN,
        200.0,
        68.75,
    )
    .expect("GPU worker init (NVRTC + device alloc + CSC build)");
    println!("[gpu] init={:.3}s (NVRTC compile + CSR upload)", t1.elapsed().as_secs_f64());

    // ── CPU воркер (эталон) ──
    let n_neurons = conn.n_neurons;
    let mut cpu = AlphaLifWorker::new(conn, ShiuParams::flywire()).expect("CPU worker init");
    cpu.set_sugar_grn(&SUGAR_GRN);

    // ── Прогон ──
    let tg = Instant::now();
    let gout = gpu.run_seconds(t_run_s).expect("GPU run");
    let gpu_s = tg.elapsed().as_secs_f64();

    let tc = Instant::now();
    let cout = cpu.run_seconds(t_run_s);
    let cpu_s = tc.elapsed().as_secs_f64();

    println!(
        "[gpu] t={t_run_s}s spikes={} active={} wall={:.3}s (ratio {:.3}x)",
        gout.total_spikes, gout.active_neurons, gpu_s, t_run_s as f64 / gpu_s
    );
    println!(
        "[cpu] t={t_run_s}s spikes={} active={} wall={:.3}s (ratio {:.3}x)",
        cout.total_spikes, cout.active_neurons, cpu_s, t_run_s as f64 / cpu_s
    );
    println!("[speedup] {:.2}x", cpu_s / gpu_s);

    // Близость финального v (нейрон bit-exact; scatter атомарный → ~1 ulp).
    let cpu_v = cpu.v().to_vec();
    let gpu_v = gpu.download_v().expect("GPU v download");
    let mut max_abs = 0.0f32;
    let mut diverged = 0usize;
    for i in 0..n_neurons {
        let d = (cpu_v[i] - gpu_v[i]).abs();
        max_abs = max_abs.max(d);
        if cpu_v[i].to_bits() != gpu_v[i].to_bits() {
            diverged += 1;
        }
    }
    println!("[parity] max|Δv|={max_abs:.2e} non_bit_exact={diverged}/{n_neurons}");
}
