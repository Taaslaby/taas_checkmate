/// FAFB — полный мозг дрозофилы на GPU. Загружает из .taas.
///
/// Запуск: cargo run --features cuda --example bench_fly_brain --release
use std::fs::File;
use std::time::Instant;
use cuda_kernels::GpuContext;
use taas_connectome::{BrainState, ConnectomeData, LifParams, NeuronModel, sim_lif};

const DT_MS: f32 = 0.1;
const DT_S: f32 = 0.1e-3;
const TAU_M: f32 = 10e-3;
const V_REST: f32 = -65e-3;
const V_THRESH: f32 = -50e-3;
const V_RESET: f32 = -65e-3;
const TAU_REF: f32 = 2e-3;
const N_STEPS: usize = 100;
const N_SAMPLE: usize = 3;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  FAFB — полный мозг дрозофилы на GPU (из .taas)        ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── Load baked connectome ───────────────────────────
    let t_load = Instant::now();
    let mut f = File::open("fafb.taas").expect("fafb.taas not found — run `cargo run --example bake_fafb --release` first");
    let conn = ConnectomeData::load_baked(&mut f).unwrap();
    println!("  Loaded .taas in {:.3}s", t_load.elapsed().as_secs_f64());

    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    let avg_syn = n_syn as f64 / n as f64;
    println!("  {n} neurons, {n_syn} synapses ({avg_syn:.0} syn/neu)");
    let mem_lif = n as f64 * 44.0;
    let mem_csr = n_syn as f64 * 12.0 + (n + 1) as f64 * 4.0;
    let mem_total = mem_lif + mem_csr;
    println!("  VRAM: LIF {:.1}MB + CSR {:.1}MB = {:.1}MB ({:.1}% of 8GB)",
             mem_lif / 1_048_576.0, mem_csr / 1_048_576.0,
             mem_total / 1_048_576.0, mem_total / (8.0 * 1_073_741_824.0) * 100.0);
    println!();

    let external_I = vec![2e-9; n];

    // ─── CPU SIMD LIF + CSR ─────────────────────────────
    println!("─── 1. CPU SIMD LIF + CSR ───");
    let lif_params = LifParams::new(
        &vec![TAU_M; n], &vec![V_REST; n], &vec![V_THRESH; n], &vec![V_RESET; n],
        DT_MS, TAU_REF,
    );

    let mut cpu_t_avg = 0.0;
    for _ in 0..N_SAMPLE {
        let mut s = BrainState::new(n, NeuronModel::LIF);
        let t0 = Instant::now();
        let _ = sim_lif(&conn, &mut s, &lif_params, &external_I, N_STEPS);
        cpu_t_avg += t0.elapsed().as_secs_f64();
    }
    cpu_t_avg /= N_SAMPLE as f64;
    let cpu_thr = n as f64 * N_STEPS as f64 / cpu_t_avg;
    let cpu_us = cpu_t_avg / N_STEPS as f64 * 1e6;
    println!("  {:.3}s / {N_STEPS} steps, {cpu_us:.1} us/step, {cpu_thr:.2e} n·s/s",
             cpu_t_avg);
    println!();

    // ─── GPU Izhikevich + CSR ────────────────────────────
    println!("─── 2. GPU Izhikevich + CSR ───");
    let ctx = GpuContext::new(n, n_syn).expect("GpuContext::new");
    ctx.upload_lif_params(DT_S, &vec![TAU_M; n], &vec![V_REST; n],
                          &vec![V_THRESH; n], &vec![V_RESET; n]).unwrap();
    ctx.upload_I(&external_I).unwrap();
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &external_I).unwrap();
    let d_spike_counter = ctx.alloc_spike_counter().unwrap();
    ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
    ctx.sync().unwrap();

    let mut gpu_t_avg = 0.0;
    for _ in 0..N_SAMPLE {
        ctx.upload_I(&external_I).unwrap();
        let t0 = Instant::now();
        for _ in 0..N_STEPS {
            ctx.upload_I(&external_I).unwrap();
            ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
        }
        ctx.sync().unwrap();
        gpu_t_avg += t0.elapsed().as_secs_f64();
    }
    gpu_t_avg /= N_SAMPLE as f64;
    let gpu_thr = n as f64 * N_STEPS as f64 / gpu_t_avg;

    let mut gpu_t_opt = 0.0;
    for _ in 0..N_SAMPLE {
        ctx.upload_I(&external_I).unwrap();
        let t0 = Instant::now();
        for _ in 0..N_STEPS {
            ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
        }
        ctx.sync().unwrap();
        gpu_t_opt += t0.elapsed().as_secs_f64();
    }
    gpu_t_opt /= N_SAMPLE as f64;
    let gpu_thr_opt = n as f64 * N_STEPS as f64 / gpu_t_opt;

    println!("  with re-upload I: {gpu_t_avg:.4}s, {:.2e} n·s/s",
             gpu_thr);
    println!("  без re-upload I:  {gpu_t_opt:.4}s, {:.2e} n·s/s",
             gpu_thr_opt);
    println!();

    // ─── Comparison ──────────────────────────────────────
    println!("─── 3. Сравнение ───");
    let speedup_opt = cpu_t_avg / gpu_t_opt;
    let speedup_real = cpu_t_avg / gpu_t_avg;
    println!("  CPU: {:.2e}s/step, {:.2e} n·s/s",
             cpu_t_avg / N_STEPS as f64, cpu_thr);
    println!("  GPU: {:.2e}s/step, {:.2e} n·s/s (+upload I)",
             gpu_t_avg / N_STEPS as f64, gpu_thr);
    println!("  GPU: {:.2e}s/step, {:.2e} n·s/s (optim.)",
             gpu_t_opt / N_STEPS as f64, gpu_thr_opt);
    println!("  Speedup: {speedup_real:.0}× (real) / {speedup_opt:.0}× (ideal)");
    println!("  Полный мозг мухи на RTX 4060 Ti — {:.0}% VRAM",
             mem_total / (8.0 * 1_073_741_824.0) * 100.0);
}
