/// FAFB (полный мозг дрозофилы) + HH на GPU — benchmark.
///
/// Запуск: cargo run --features cuda --example bench_fly_hh --release
///
/// Требует: fafb.taas (запусти `cargo run --example bake_fafb --release` сначала)
use std::fs::File;
use std::time::Instant;
use cuda_kernels::GpuContext;

const N_STEPS: usize = 200;
const N_SAMPLE: usize = 3;
const DT: f32 = 0.01; // ms (HH требует мелкого шага)
const TAU_REF: f32 = 2.0; // ms refractory
const TAU_SYN: f32 = 5.0; // ms synaptic decay
const WEIGHT_SCALE: f32 = 0.005;

const N_GATES: usize = 4;
const N_CHANNELS: usize = 3;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  FAFB + HH на GPU (Hodgkin-Huxley, полный мозг мухи)   ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── Load baked connectome ───────────────────────────
    let t = Instant::now();
    let mut f = File::open("fafb.taas")
        .expect("fafb.taas not found — run `cargo run --example bake_fafb --release`");
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    let avg_syn = n_syn as f64 / n as f64;
    println!("  Loaded .taas in {:.3}s — {n} neurons, {n_syn} synapses ({avg_syn:.0} syn/neu)",
             t.elapsed().as_secs_f64());

    let mem_lif = n as f64 * 44.0;
    let mem_csr = n_syn as f64 * 12.0 + (n + 1) as f64 * 4.0;
    let mem_hh = n as f64 * 36.0; // v,I,spikes,refr,gates(4),g_leak,E_leak,C_m,I_syn
    let mem_total = mem_lif + mem_csr + mem_hh;
    println!("  VRAM: {:.0}MB / 8192MB ({:.1}%)",
             mem_total / 1_048_576.0, mem_total / (8.0 * 1_073_741_824.0) * 100.0);
    println!();

    // ─── GPU context + connectome upload ─────────────────
    let ctx = GpuContext::new(n, n_syn).expect("GpuContext::new");
    let mut external_I = vec![0.0f32; n];
    let n_stim = n / 10;
    for i in 0..n_stim { external_I[i] = 5.0; } // 5nA to 10%
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &external_I).unwrap();
    println!("  GpuContext + connectome uploaded ({n_stim}/{n} stimulated with 5nA)");

    // ─── HH: NaV (m³h) + KV (n⁴) + CaT (c¹) ─────────────
    let v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * N_GATES];
    for i in 0..n {
        gates[i * N_GATES + 0] = 0.05; // m
        gates[i * N_GATES + 1] = 0.6;  // h
        gates[i * N_GATES + 2] = 0.3;  // n
        gates[i * N_GATES + 3] = 0.0;  // c
    }
    let Ii_init = vec![0.0f32; n];
    let g_leak = vec![0.3f32; n];
    let E_leak = vec![-65.0f32; n];
    let C_m = vec![1.0f32; n];

    ctx.hh_upload_state(&v_init, &gates, &Ii_init,
                        &g_leak, &E_leak, &C_m, N_GATES).unwrap();

    let E_rev_arr = vec![50.0, -77.0, 120.0];
    let g_max_arr = vec![120.0, 36.0, 0.5];
    let powers = vec![3u32, 1, 4, 1]; // m³, h¹, n⁴, c¹
    let n_gates_per_ch = vec![2i32, 1, 1]; // NaV:2, KV:1, CaT:1

    ctx.hh_upload_channels(&E_rev_arr, &g_max_arr, &powers, &n_gates_per_ch,
                           N_GATES, N_CHANNELS).unwrap();
    println!("  HH (NaV+KV+CaT) uploaded, {N_GATES} gates, {N_CHANNELS} channels");

    // ─── Warmup ──────────────────────────────────────────
    let max_steps = N_STEPS.max(200);
    let mut host_spike = vec![0i32; max_steps * n];
    let mut host_v = vec![0.0f32; n];
    let mut host_I_syn = vec![0.0f32; n];

    ctx.batch_hh_steps(&external_I, &mut host_spike, &mut host_v, &mut host_I_syn,
                       200, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();

    let ws: i32 = host_spike[..200 * n].iter().sum();
    let wa = host_spike[..200 * n].iter().filter(|&&s| s > 0).count();
    let vm = host_v.iter().sum::<f32>() / n as f32;
    println!("  Warmup 200 steps: {ws} spikes, {wa} active, V_mean={vm:.1}mV");
    println!();

    // ─── Benchmark ───────────────────────────────────────
    println!("─── Benchmark: batch_hh_steps {N_STEPS} steps ───");
    let mut t_avg = 0.0;
    for si in 0..N_SAMPLE {
        let mut sb = vec![0i32; N_STEPS * n];
        let t0 = Instant::now();
        ctx.batch_hh_steps(&external_I, &mut sb, &mut host_v, &mut host_I_syn,
                           N_STEPS, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();
        t_avg += t0.elapsed().as_secs_f64();
        if si == 0 {
            let s: i32 = sb.iter().sum();
            let a = sb.iter().filter(|&&s| s > 0).count();
            let rate = s as f64 / (N_STEPS as f64 * n as f64 / DT as f64 * 1000.0);
            println!("  Spikes: {s}, active: {a}/{n}, rate ≈ {rate:.0}Hz");
        }
    }
    t_avg /= N_SAMPLE as f64;
    let thr = n as f64 * N_STEPS as f64 / t_avg;
    let us = t_avg / N_STEPS as f64 * 1e6;
    println!("  {:.3}s total, {us:.0} us/step, {thr:.2e} n·s/s", t_avg);

    let bio_ms = N_STEPS as f32 * DT;
    println!("  dt={DT}ms × {N_STEPS} = {bio_ms}ms bio in {:.3}s wall = {:.0}× realtime",
             t_avg, t_avg / (bio_ms as f64 / 1000.0));
    println!();
    println!("─── VRAM: {:.0}MB / 8192MB ({:.1}%)",
             mem_total / 1_048_576.0, mem_total / (8.0 * 1_073_741_824.0) * 100.0);
}
