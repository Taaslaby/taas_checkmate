/// GPU Kuramoto CSR coupling benchmark.
///
/// Загружает .taas коннектом, запускает N_STEPS Kuramoto на GPU (prepare →
/// CSR coupling → update), замеряет скорость.
///
/// ## Запуск
/// ```bash
/// cargo run --features cuda --example gpu_kuramoto_bench --release -- <input.taas>
/// ```
/// Env: N_STEPS=80 (default), DT=0.05 (default)
use rand::SeedableRng;
use std::env;
use std::fs::File;
use std::time::Instant;

const DEFAULT_N_STEPS: usize = 80;
const DEFAULT_DT: f32 = 0.05;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Kuramoto CSR Coupling Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── GPU check ─────────────────────────────────────
    let n_gpus = cuda_kernels::get_device_count().unwrap_or(0);
    if n_gpus == 0 { eprintln!("❌ No CUDA devices"); std::process::exit(1); }
    let _ = cuda_kernels::set_device(0);
    println!("  GPU: {n_gpus} device(s)\n");

    // ─── CLI ───────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.taas>", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let n_steps: usize = env::var("N_STEPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_N_STEPS);
    let dt: f32 = env::var("DT").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_DT);

    println!("  Config: N_STEPS={n_steps}, DT={dt}\n");

    // ─── Load .taas ─────────────────────────────────────
    let t = Instant::now();
    let mut f = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f)
        .unwrap_or_else(|e| panic!("Cannot load .taas: {e}"));
    let n = conn.n_neurons;
    let s = conn.n_synapses;
    println!("  Loaded: {n} neurons, {s} synapses ({:.0} syn/neu, {:.3}s)",
             s as f64 / n.max(1) as f64, t.elapsed().as_secs_f64());

    // ─── Upload to GPU ─────────────────────────────────
    let ctx = cuda_kernels::GpuContext::new(n, s)
        .unwrap_or_else(|e| panic!("GpuContext::new: {e}"));
    let ext_I = vec![0.0f32; n];
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &ext_I).unwrap();
    println!("  Uploaded to GPU");

    // ─── Pre-allocate ──────────────────────────────────
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
    let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
    let coupling = vec![0.0f32; n];

    ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();
    println!("  Kuramoto state uploaded\n");

    // ─── Warmup ─────────────────────────────────────────
    println!("  Warmup...");
    for _ in 0..10 {
        ctx.kuramoto_prepare().unwrap();
        ctx.kuramoto_csr_coupling().unwrap();
        ctx.kuramoto_update(dt).unwrap();
    }
    ctx.sync().unwrap();
    println!("  Warmup done\n");

    // ─── Benchmark ──────────────────────────────────────
    println!("─── GPU Kuramoto CSR benchmark ({n_steps} steps) ───");
    let bench_start = Instant::now();

    for step in 0..n_steps {
        ctx.kuramoto_prepare().unwrap();
        ctx.kuramoto_csr_coupling().unwrap();
        ctx.kuramoto_update(dt).unwrap();
    }
    ctx.sync().unwrap();

    let elapsed = bench_start.elapsed();
    let us_per_step = elapsed.as_secs_f64() / n_steps as f64 * 1_000_000.0;
    let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU Kuramoto CSR: {:.3}s for {n_steps} steps         ║",
             elapsed.as_secs_f64());
    println!("║  Per step: {us_per_step:.1} µs                          ║");
    println!("║  Throughput: {steps_per_sec:.0} steps/s                    ║");
    println!("║  Synapses/step: {s}                                      ║");
    println!("║  Syn·step/s: {:.2e}                                    ║",
             s as f64 * steps_per_sec);
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── Download phases & coherence ────────────────────
    let mut final_phases = vec![0.0f32; n];
    ctx.download_phases(&mut final_phases).unwrap();

    let (sum_s, sum_c): (f64, f64) = final_phases.iter().fold((0.0, 0.0), |(s, c), &th| {
        (s + th.sin() as f64, c + th.cos() as f64)
    });
    let coherence = ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32;
    println!("  Final coherence R = {:.4}\n", coherence);

    // ─── CPU reference benchmark ────────────────────────
    println!("─── CPU Kuramoto reference ({n_steps} steps) ───");
    let mut phases_cpu = phases.clone();
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];

    let cpu_start = Instant::now();
    for _ in 0..n_steps {
        for i in 0..n { sin_buf[i] = phases_cpu[i].sin(); cos_buf[i] = phases_cpu[i].cos(); }
        for i in 0..n {
            let start = conn.row_ptr[i].max(0) as usize;
            let end = (conn.row_ptr[i + 1] as usize).min(s);
            let mut cpl = 0.0f32;
            for j in start..end {
                let target = conn.col_indices[j] as usize;
                if target >= n { continue; }
                cpl += conn.values[j] * (sin_buf[target] * cos_buf[i] - cos_buf[target] * sin_buf[i]);
            }
            phases_cpu[i] += dt * (omegas[i] + cpl);
        }
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_us = cpu_elapsed.as_secs_f64() / n_steps as f64 * 1_000_000.0;
    let speedup = cpu_elapsed.as_secs_f64() / elapsed.as_secs_f64();

    println!("  CPU: {cpu_us:.0} µs/step ({:.3}s total)", cpu_elapsed.as_secs_f64());
    println!("  GPU: {us_per_step:.1} µs/step ({:.3}s total)", elapsed.as_secs_f64());
    println!("  Speedup: {speedup:.0}× 🚀\n");

    // ─── Verify correctness ─────────────────────────────
    let diff: f32 = phases_cpu.iter().zip(final_phases.iter())
        .map(|(c, g)| (c - g).abs()).sum::<f32>() / n as f32;
    println!("  Mean phase diff (CPU vs GPU): {:.6e}", diff);
    if diff < 0.01 {
        println!("  ✅ GPU Kuramoto CSR coupling — correct!");
    } else {
        println!("  ⚠️  Large phase diff — possible kernel bug");
    }
}
