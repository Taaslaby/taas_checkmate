//! Comprehensive benchmark: CPU (ASM) vs CUDA на всех масштабах.
//!
//! ## Что меряем
//!
//! 1. **XOR attention** — CPU (koit ASM) vs CUDA (persistent XorAttention)
//! 2. **Simulation step** — GpuContext: safety_clamp → fused_izhikevich+stdp → kuramoto_prepare → kuramoto_update
//! 3. **CUDA Graph replay** — один захват, N реплеев vs individual kernel launches
//! 4. **Fused vs separate** — izhikevich + stdp в одном launch vs два отдельных
//!
//! Масштабы: SEZ 12K, FlyWire 138K, 500K, 1M нейронов.

use std::time::Instant;
use neuron_engine::koit;
use cuda_kernels;

// ═══════════════════════════════════════════════════════════
//  Простой inline RNG (без зависимостей)
// ═══════════════════════════════════════════════════════════

fn rng_u8() -> u8 {
    static mut S: u64 = 42;
    unsafe { S = S.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (S >> 32) as u8 }
}

fn rng_f32() -> f32 {
    static mut S: u64 = 12345;
    unsafe { S = S.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((S >> 32) as f32) / 4294967296.0 }
}

fn rng_vec_f32(n: usize) -> Vec<f32> {
    (0..n).map(|_| rng_f32()).collect()
}

fn rng_vec_u8(n: usize) -> Vec<u8> {
    (0..n).map(|_| rng_u8()).collect()
}

// ═══════════════════════════════════════════════════════════
//  1. XOR Attention: CPU (ASM) vs CUDA
// ═══════════════════════════════════════════════════════════

fn bench_xor_attention(label: &str, n_keys: usize, n_bytes: usize, n_iter: usize, gpus: i32) {
    let query = rng_vec_u8(n_bytes);
    let keys = rng_vec_u8(n_keys * n_bytes);
    let values = rng_vec_f32(n_keys);

    // ─── CPU ASM ───────────────────────────────────────
    let mut cpu_out = vec![0.0f32];
    for _ in 0..3 { koit::xor_attention(&query, &keys, &values, &mut cpu_out, n_keys, n_bytes); }
    let t0 = Instant::now();
    for _ in 0..n_iter { koit::xor_attention(&query, &keys, &values, &mut cpu_out, n_keys, n_bytes); }
    let cpu_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    // ─── CUDA ─────────────────────────────────────────
    let mut gpu_ms = f64::NAN;
    let mut gpu_out = vec![0.0f32];
    if gpus > 0 {
        let ctx = cuda_kernels::XorAttention::new(n_keys, n_bytes).unwrap();
        for _ in 0..3 { let _ = ctx.compute(&query, &keys, &values, &mut gpu_out); }
        let t0 = Instant::now();
        for _ in 0..n_iter { let _ = ctx.compute(&query, &keys, &values, &mut gpu_out); }
        gpu_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

        // Информационное сравнение (float rounding накапливается для больших N)
        let diff = (cpu_out[0] - gpu_out[0]).abs();
        if diff > 0.1 {
            println!("       ⚠ CPU/GPU diff: {:.6} (float rounding over {} keys)", diff, n_keys);
        }
    }

    let speedup = if gpu_ms.is_finite() && gpu_ms > 0.0 { cpu_ms / gpu_ms } else { f64::NAN };
    let sp_str = if speedup.is_finite() { format!("{:>5.1}×", speedup) } else { "  N/A".into() };
    println!("  {:<22} CPU:{:.4}ms  GPU:{:.4}ms  {}", label, cpu_ms, gpu_ms, sp_str);
}

// ═══════════════════════════════════════════════════════════
//  2. Simulation step: GpuContext
// ═══════════════════════════════════════════════════════════

fn bench_simulation_step(label: &str, n: usize, n_iter: usize) {
    let ctx = cuda_kernels::GpuContext::new(n, n).unwrap();

    // Init neuron state
    let v = vec![-65.0f32; n];
    let u = vec![-13.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];
    ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    let phases = vec![0.0f32; n];
    let omegas = vec![1.0f32; n];
    let coupling = vec![0.0f32; n];
    ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

    // ─── Warmup ───────────────────────────────────────
    for _ in 0..3 { ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0).unwrap(); }

    // ─── Benchmark: CUDA Graph replay ─────────────────
    let t0 = Instant::now();
    for _ in 0..n_iter { ctx.graph_replay().unwrap(); }
    ctx.sync().unwrap();
    let graph_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    // ─── Verify it actually ran ───────────────────────
    let mut spikes = vec![0i32; n];
    ctx.download_spikes(&mut spikes).unwrap();
    let n_spikes = spikes.iter().filter(|&&s| s > 0).count();

    println!("  {:<22} CUDA Graph replay: {:.4}ms/step  (n={}, {} spikes)",
        label, graph_ms, n, n_spikes);
}

// ═══════════════════════════════════════════════════════════
//  3. Fused vs Separate kernels
// ═══════════════════════════════════════════════════════════

fn bench_fused_vs_separate(label: &str, n: usize, n_iter: usize) {
    let ctx = cuda_kernels::GpuContext::new(n, n).unwrap();

    let v = vec![-65.0f32; n];
    let u = vec![-13.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];
    ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

    // ─── Separate: izhikevich + stdp (2 launches) ─────
    ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    for _ in 0..3 {
        ctx.izhikevich_step(0.001, 0.99).unwrap();
        ctx.stdp_update(0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iter {
        ctx.izhikevich_step(0.001, 0.99).unwrap();
        ctx.stdp_update(0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
    }
    let separate_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    // ─── Fused: 1 launch ──────────────────────────────
    ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    for _ in 0..3 {
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iter {
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
    }
    let fused_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    let speedup = if fused_ms > 0.0 { separate_ms / fused_ms } else { f64::NAN };
    println!("  {:<22} Separate: {:.4}ms  Fused: {:.4}ms  {:.1}× faster",
        label, separate_ms, fused_ms, speedup);
}

// ═══════════════════════════════════════════════════════════
//  4. CUDA Graph vs Individual launches
// ═══════════════════════════════════════════════════════════

fn bench_graph_vs_individual(label: &str, n: usize, n_iter: usize) {
    let ctx = cuda_kernels::GpuContext::new(n, n).unwrap();

    let v = vec![-65.0f32; n];
    let u = vec![-13.0f32; n];
    let a = vec![0.02f32; n];
    let b = vec![0.2f32; n];
    let c = vec![-65.0f32; n];
    let d = vec![8.0f32; n];
    ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();
    let phases = vec![0.0f32; n];
    let omegas = vec![1.0f32; n];
    let coupling = vec![0.0f32; n];
    ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

    // ─── Individual kernel launches (4 per step) ──────
    for _ in 0..3 {
        ctx.safety_clamp(-100.0, 100.0).unwrap();
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.kuramoto_prepare().unwrap();
        ctx.kuramoto_update(0.001).unwrap();
        ctx.sync().unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iter {
        ctx.safety_clamp(-100.0, 100.0).unwrap();
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.kuramoto_prepare().unwrap();
        ctx.kuramoto_update(0.001).unwrap();
        ctx.sync().unwrap();
    }
    let individual_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    // ─── CUDA Graph replay ────────────────────────────
    // First call captures the graph via simulation_step
    ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0).unwrap();

    let t0 = Instant::now();
    for _ in 0..n_iter {
        ctx.graph_replay().unwrap();
    }
    ctx.sync().unwrap();
    let graph_ms = t0.elapsed().as_secs_f64() / n_iter as f64 * 1000.0;

    let speedup = if graph_ms > 0.0 { individual_ms / graph_ms } else { f64::NAN };
    println!("  {:<22} 4 launches: {:.4}ms  Graph: {:.4}ms  {:.1}× faster",
        label, individual_ms, graph_ms, speedup);
}

// ═══════════════════════════════════════════════════════════
//  Main benchmark
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_all() {
    let gpus = cuda_kernels::get_device_count().unwrap_or(0);
    let _ = cuda_kernels::set_device(0);
    println!("\n═══ 🚀 CPU (ASM) vs CUDA — Comprehensive Benchmark ═══");
    println!("GPU: {} device(s)", gpus);
    println!("Arch: sm_89 (Ada Lovelace) + --use_fast_math");
    if gpus == 0 { println!("⚠ No GPU — skipping CUDA tests"); return; }

    // Конфигурации: (label, n_neurons, n_bytes, n_iter)
    let xor_cfgs = [
        ("SEZ   64b",    11797,   8, 100),
        ("SEZ  128b",    11797,  16,  50),
        ("SEZ  256b",    11797,  32,  30),
        ("FW   64b",    138584,   8,  30),
        ("FW  128b",    138584,  16,  15),
        ("FW  256b",    138584,  32,  10),
        ("500K 64b",    500000,   8,   5),
        ("1M   64b",   1000000,   8,   3),
    ];

    // ─── 1. XOR Attention ─────────────────────────────
    println!("\n─── 1. XOR Attention: CPU (ASM) vs CUDA ───");
    for (label, nk, nb, ni) in &xor_cfgs {
        bench_xor_attention(label, *nk, *nb, *ni, gpus);
    }

    // ─── 2. Simulation Step (GpuContext) ──────────────
    println!("\n─── 2. Simulation Step: GpuContext (CUDA Graph) ───");
    let step_cfgs = [
        ("SEZ 12K",     11797,  100),
        ("FW  138K",   138584,   30),
        ("500K",       500000,   10),
        ("1M",        1000000,    5),
    ];
    for (label, n, ni) in &step_cfgs {
        bench_simulation_step(label, *n, *ni);
    }

    // ─── 3. Fused vs Separate ─────────────────────────
    println!("\n─── 3. Fused Izhikevich+STDP vs Separate ───");
    let fuse_cfgs = [
        ("SEZ 12K", 11797,  30),
        ("FW  138K", 138584, 10),
    ];
    for (label, n, ni) in &fuse_cfgs {
        bench_fused_vs_separate(label, *n, *ni);
    }

    // ─── 4. CUDA Graph vs Individual ──────────────────
    println!("\n─── 4. CUDA Graph vs 4 Individual Launches ───");
    let graph_cfgs = [
        ("SEZ 12K", 11797,  100),
        ("FW  138K", 138584, 30),
        ("500K",    500000,  10),
    ];
    for (label, n, ni) in &graph_cfgs {
        bench_graph_vs_individual(label, *n, *ni);
    }

    println!("\n═══ BENCHMARK COMPLETE ═══");
}
