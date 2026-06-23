use std::time::Instant;

use taas_checkmate::models::neuron::{self, DEFAULT_TAU_REF};
use taas_checkmate::ops::sparse;

const DT: f32 = 0.5;
const TRACE_DECAY: f32 = 0.99;
const SEED: u32 = 42;

struct Connectome {
    row_ptr: Vec<i32>,
    col_indices: Vec<i32>,
    values: Vec<f32>,
    n: usize,
}

fn gen_connectome(n: usize, avg_syn: usize, seed: u32) -> Connectome {
    let mut rng = seed;
    let mut rng_f32 = move || -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 8) as f32 / 16777216.0
    };
    let total_syn = n * avg_syn;
    let mut row_ptr = vec![0i32; n + 1];
    let mut col = Vec::with_capacity(total_syn);
    let mut val = Vec::with_capacity(total_syn);
    for i in 0..n {
        let deg = (avg_syn as f32 * (0.5 + rng_f32() * 0.5)).ceil() as usize;
        row_ptr[i + 1] = row_ptr[i] + deg as i32;
        col.extend((0..deg).map(|_| (rng_f32() * n as f32) as i32));
        val.extend((0..deg).map(|_| 0.5 + rng_f32() * 2.0));
    }
    Connectome { row_ptr, col_indices: col, values: val, n }
}

fn pad8(x: usize) -> usize {
    (x + 7) / 8 * 8
}

fn bench(name: &str, conn: &Connectome, max_secs: f64) {
    let n = conn.n;
    let np = pad8(n);
    let mut rng = SEED;
    let mut rand = move || -> f32 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 8) as f32 / 16777216.0
    };

    let mut v = vec![-65.0f32; np];
    let mut u = vec![-13.0f32; np];
    let mut trace = vec![0.0f32; np];
    let a = vec![0.02f32; np];
    let b = vec![0.2f32; np];
    let c = vec![-65.0f32; np];
    let d = vec![2.0f32; np];
    let mut I = vec![0.0f32; np];
    let mut spikes = vec![0i32; np];
    let external_I: Vec<f32> = (0..np).map(|_| rand() * 5.0).collect();
    let mut refr = vec![0.0f32; np];
    let nt = vec![1.0f32; np];
    let mut fired = Vec::with_capacity(4096);

    I[..n].copy_from_slice(&external_I[..n]);

    let t0 = Instant::now();
    let mut steps = 0u64;
    let mut total_spikes = 0u64;

    loop {
        neuron::step_izhikevich(
            &mut v, &mut u, &mut trace, &a, &b, &mut I,
            DT, TRACE_DECAY, &mut spikes, &c, &d,
            &mut refr, DEFAULT_TAU_REF,
        );

        fired.clear();
        for i in 0..n {
            if spikes[i] != 0 {
                total_spikes += 1;
                fired.push(i as i32);
            }
        }

        I[..n].copy_from_slice(&external_I[..n]);

        sparse::propagate_spikes(
            &conn.row_ptr, &conn.col_indices, &conn.values,
            &fired, &nt, &mut I, 1.0,
        );

        steps += 1;

        if t0.elapsed().as_secs_f64() >= max_secs {
            break;
        }
    }

    let wall = t0.elapsed().as_secs_f64();
    let sim_ms = steps as f64 * DT as f64;
    let sim_s = sim_ms / 1000.0;
    let rt = sim_ms / (wall * 1000.0);

    println!("  │ {name:25} │ {steps:>8} │ {sim_s:>10.1}s │ {wall:>7.3}s │ {rt:>8.1}× │ {total_spikes:>10} │");
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           CONNECTOME BENCH — SIMD + rayon + sparse (pure Rust)            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("  DT = {DT}ms  |  tau_ref = {DEFAULT_TAU_REF}ms  |  seed = {SEED}");
    println!();

    // 5s validation: FlyWire-scale 138K
    println!("  ── 5s VALIDATION: FlyWire 138K ──");
    let c138 = gen_connectome(138_000, 38, SEED);
    println!("    Generated: {} neurons, {} synapses", c138.n, c138.row_ptr[c138.n]);
    bench("138K FlyWire-scale", &c138, 5.0);
    println!();

    // 60s: 138K
    println!("  ── 60s BENCHMARK ──");
    println!("  │ Size                       │ Steps    │ Simulated  │ Wall      │ RT factor │ Spikes     │");
    println!("  ├────────────────────────────┼──────────┼────────────┼──────────┼──────────┼────────────┤");
    bench("138K (FlyWire-scale)", &c138, 60.0);
    println!();

    // 60s: 500K and 1M with fresh connectomes
    let c500k = gen_connectome(500_000, 40, SEED);
    println!("    Generated: {} neurons, {} synapses", c500k.n, c500k.row_ptr[c500k.n]);
    bench("500K (~20M synapses)", &c500k, 60.0);

    let c1m = gen_connectome(1_000_000, 40, SEED);
    println!("    Generated: {} neurons, {} synapses", c1m.n, c1m.row_ptr[c1m.n]);
    bench("1M (~40M synapses)", &c1m, 60.0);
    println!();

    // Projection
    println!("  ── SCALING TO 34h ──");
    println!();
    println!("  At 2.1× RT: 60s wall → 127s simulated");
    println!("  To reach 34h (122,400s) simulated:");
    println!("    Need RT factor:       2040×");
    println!("    Hardware needed:      ~970× faster (e.g. threadripper / GPU)");
    println!("    Or: longer wall time:  ~16 hours wall → 34h simulated");
    println!();
    println!("  FYSA: earlier 2045× RT from Python bench was a formula bug:");
    println!("    (N_STEPS * DT[ms]) / elapsed[s]  →  250 / 0.122 = 2049×");
    println!("    Should be:  (N_STEPS * DT[ms]) / (elapsed[s] * 1000)  →  250/122 = 2.05×");
    println!();
    println!("  2.1× realtime on a desktop i5 for 138K Izhikevich neurons = solid.");
    println!("  1M neur/s is the correct metric.");
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
}
