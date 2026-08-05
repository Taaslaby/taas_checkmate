//! 🚀 LIF 138k — CPU SIMD vs pillars-cuda (naive per-step vs persistent).
//!
//! The whole point of the persistent path: state/params live in VRAM, the
//! kernel runs in place, and only the *spikes* cross PCIe each step.
//!
//! Run: `cargo run -p pillars-cuda --example persistent_bench --release`
//!
//! Reference numbers from the legacy crate (crates/cuda, `lif_cpu_vs_cuda`
//! example): its per-step `step_lif` loses to CPU SIMD by ~100× at 138k —
//! exactly the host↔device round-trip trap. (Note: the legacy crate does
//! not currently build on this machine — nvcc fails on xor_attention.cu —
//! so the comparison is CPU vs the two pillars-cuda paths.)

use std::time::Instant;

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, LIFModel, LIF_CUDA_SOURCE};

const N: usize = 138_000; // fly-sized population
const DT: f32 = 0.1;
const STEPS: usize = 1000;

fn setup() -> (NeuralPopulation<LIFModel>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pop = NeuralPopulation::<LIFModel>::new(N);
    pop.fill_state(0, -70.0); // v
    pop.fill_state(1, 0.0); // refr
    pop.fill_param(0, DT);
    pop.fill_param(1, 0.1); // inv_tau_m (tau_m = 10)
    pop.fill_param(2, -70.0);
    pop.fill_param(3, -54.0);
    pop.fill_param(4, -70.0);
    pop.fill_param(5, 2.0);

    let ns = LIFModel::N_STATE;
    let np = LIFModel::N_PARAMS;
    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 30.0 } else { 0.0 })
        .collect();
    (pop, g_states, g_params, i_in)
}

fn main() {
    println!("═══ LIF {N}: CPU SIMD vs pillars-cuda ═══");
    println!("  steps = {STEPS}, dt = {DT} ms\n");

    // ── CPU SIMD ─────────────────────────────────────────────────────
    let (mut pop, mut g_states, g_params, i_in) = setup();
    let init_states = g_states.clone(); // pristine start for both GPU paths
    let mut spikes = vec![0i32; N];
    let mut cpu_total = 0i64;
    let t0 = Instant::now();
    for _ in 0..STEPS {
        pop.step(&i_in, &mut spikes);
        cpu_total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── pillars-cuda: naive per-step (full round-trip every step) ────
    let model =
        CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N).expect("CUDA compile failed");
    let mut g_spikes = vec![0i32; N];
    let t1 = Instant::now();
    for _ in 0..STEPS {
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("per-step gpu");
    }
    let naive_ms = t1.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── pillars-cuda: persistent (upload once, only spikes cross) ────
    let mut g_states2 = init_states.clone();
    let model2 =
        CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N).expect("CUDA compile failed");
    model2.upload(&g_states2, &g_params, &i_in).expect("upload");
    let mut gpu_total = 0i64;
    let mut g_spikes2 = vec![0i32; N];
    let t2 = Instant::now();
    for _ in 0..STEPS {
        model2.step_device(DT).expect("persistent step");
        model2
            .download_spikes(&mut g_spikes2)
            .expect("download spikes");
        gpu_total += g_spikes2.iter().map(|&s| s as i64).sum::<i64>();
    }
    let persistent_ms = t2.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── pillars-cuda: persistent, spikes downloaded every 100 steps ──
    // (realistic usage: you read spike trains at observation cadence, not
    //  every integration step)
    let g_states3 = init_states.clone();
    let model3 =
        CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N).expect("CUDA compile failed");
    model3.upload(&g_states3, &g_params, &i_in).expect("upload");
    let mut g_spikes3 = vec![0i32; N];
    let t3 = Instant::now();
    for s in 0..STEPS {
        model3.step_device(DT).expect("persistent step");
        if s % 100 == 99 {
            model3
                .download_spikes(&mut g_spikes3)
                .expect("download spikes");
        }
    }
    let batched_ms = t3.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── pillars-cuda: CUDA Graph (kernel captured once, replayed per step)
    //  — same observation cadence as batched for a fair comparison. ─────
    let g_states4 = init_states.clone();
    let model4 =
        CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N).expect("CUDA compile failed");
    model4.upload(&g_states4, &g_params, &i_in).expect("upload");
    let mut g_spikes4 = vec![0i32; N];
    let mut graph_total = 0i64;
    let t4 = Instant::now();
    for _ in 0..STEPS {
        model4.step_graph(DT).expect("graph step");
        model4.sync().expect("graph sync");
        model4
            .download_spikes(&mut g_spikes4)
            .expect("download spikes");
        graph_total += g_spikes4.iter().map(|&s| s as i64).sum::<i64>();
    }
    let graph_ms = t4.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── pillars-cuda: graph + on-device spike compaction (download only the
    //  active indices, not the whole n-array) — the full production path. ──
    let g_states5 = init_states.clone();
    let model5 =
        CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", 2, 6, N, N).expect("CUDA compile failed");
    model5.upload(&g_states5, &g_params, &i_in).expect("upload");
    let mut compact_buf = vec![0i32; N];
    let mut compact_total = 0i64;
    let t5 = Instant::now();
    for _ in 0..STEPS {
        model5.step_graph(DT).expect("graph step");
        let count = model5.compact_spikes().expect("compact");
        compact_total += count as i64;
        model5
            .download_compact(&mut compact_buf, count)
            .expect("download compact");
    }
    let compact_ms = t5.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    // ── Report ───────────────────────────────────────────────────────
    println!("  ms/step @ {N} neurons:");
    println!(
        "    CPU SIMD              : {cpu_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / cpu_ms / 1e3
    );
    println!(
        "    CUDA per-step          : {naive_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / naive_ms / 1e3
    );
    println!(
        "    CUDA persistent        : {persistent_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / persistent_ms / 1e3
    );
    println!(
        "    CUDA persistent+spikes : {batched_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / batched_ms / 1e3
    );
    println!(
        "    CUDA graph             : {graph_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / graph_ms / 1e3
    );
    println!(
        "    CUDA graph+compact     : {compact_ms:8.4}  ({:>6.1} M neuron-step/s)",
        N as f64 / compact_ms / 1e3
    );
    println!();
    println!("  speedup vs CPU:");
    println!("    per-step          : {:.2}×", cpu_ms / naive_ms);
    println!("    persistent        : {:.2}×", cpu_ms / persistent_ms);
    println!("    persistent+spikes : {:.2}×", cpu_ms / batched_ms);
    println!("    graph              : {:.2}×", cpu_ms / graph_ms);
    println!("    graph+compact      : {:.2}×", cpu_ms / compact_ms);
    println!();
    // Graph mode must produce identical spikes to the persistent mode.
    let mut g_states4 = init_states.clone();
    model4
        .download_states(&mut g_states4)
        .expect("download states");
    let mut graph_bits = 0u32;
    for c in 0..LIFModel::N_STATE {
        for i in 0..N {
            graph_bits = graph_bits.max(
                (pop.state_slice(c)[i].to_bits() ^ g_states4[c * N + i].to_bits()).count_ones(),
            );
        }
    }
    println!(
        "  graph-mode spikes (cumulative): {graph_total} — {}",
        if graph_total == gpu_total {
            "identical to persistent ✅"
        } else {
            "DIVERGED ❌"
        }
    );
    println!("  max differing bits (CPU vs graph GPU): {graph_bits} (0 = bitwise identical)");
    println!(
        "  graph+compact spike total : {compact_total} — {}",
        if compact_total == gpu_total {
            "identical to persistent ✅"
        } else {
            "DIVERGED ❌"
        }
    );
    println!(
        "  spikes (CPU / persistent GPU): {cpu_total} / {gpu_total} — {}",
        if cpu_total == gpu_total {
            "identical ✅"
        } else {
            "DIVERGED ❌"
        }
    );

    // LIF is exp-free → bit-for-bit parity is expected; prove the persistent
    // path didn't drift.
    model2
        .download_states(&mut g_states2)
        .expect("download states");
    let mut max_bits = 0u32;
    for c in 0..LIFModel::N_STATE {
        for i in 0..N {
            max_bits = max_bits.max(
                (pop.state_slice(c)[i].to_bits() ^ g_states2[c * N + i].to_bits()).count_ones(),
            );
        }
    }
    println!("  max differing bits in final state (CPU vs persistent GPU): {max_bits} (0 = bitwise identical)");
}
