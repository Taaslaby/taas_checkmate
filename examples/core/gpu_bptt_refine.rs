/// GPU-accelerated ConnectomeRefiner — full GPU pipeline.
///
/// ## Pipeline (all on GPU, no per-step CPU transfers)
///
/// Phase 1 — Forward: prepare → CSR coupling → **D2D copy coupling** → update → **D2D copy phases**
/// Phase 2 — Loss grad: **GPU** bptt_loss_grad (coherence + temporal)
/// Phase 3 — BPTT backward: **GPU** bptt_step loop (all pointers into device trajectory)
/// Phase 4 — Weight update: **GPU** weight_update (SGD + sparsity + homeostasis + clamp)
/// Phase 5 — Stats: single D2H download of final phases + weights (just for reporting)
///
/// All trajectory data (phases, coupling, dtheta, grad_weights) lives on GPU via DevicePtr.
/// Only 2 D2H transfers per epoch: final phases (coherence) + weights (mean|w|).
///
/// ## Usage
/// ```bash
/// cargo run --features cuda --release --example gpu_bptt_refine -- <input.taas>
/// ```
use rand::Rng;
use rand::SeedableRng;
use std::env;
use std::fs::File;
use std::time::Instant;

// ─── Constants ───────────────────────────────────────────
const DEFAULT_N_STEPS: usize = 80;
const DEFAULT_DT: f32 = 0.05;
const DEFAULT_LR: f32 = 0.08;
const DEFAULT_EPOCHS: usize = 500;
const DEFAULT_COHERENCE: f32 = 1.0;
const DEFAULT_SPARSITY: f32 = 0.002;
const DEFAULT_TEMPORAL: f32 = 0.01;
const DEFAULT_HOMEOSTASIS: f32 = 0.01;
const DEFAULT_TARGET_MEAN_W: f32 = 0.5;
const DEFAULT_WEIGHT_CLAMP: f32 = 5.0;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GPU ConnectomeRefiner — FULL GPU pipeline              ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // ─── GPU check ─────────────────────────────────────
    let n_gpus = cuda_kernels::get_device_count().unwrap_or(0);
    if n_gpus == 0 { eprintln!("❌ No CUDA devices"); std::process::exit(1); }
    let _ = cuda_kernels::set_device(0);
    println!("  GPU: {n_gpus} device(s)");

    // ─── CLI + config ─────────────────────────────────
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.taas> [output.taas]", args[0]);
        std::process::exit(1);
    }
    let path = &args[1];
    let out_path = args.get(2).cloned().unwrap_or_else(|| {
        let p = path.strip_suffix(".taas").unwrap_or(path);
        format!("{}_refined.taas", p)
    });

    let n_steps = env_or("N_STEPS", DEFAULT_N_STEPS);
    let dt = env_or("DT", DEFAULT_DT);
    let lr = env_or("LR", DEFAULT_LR);
    let n_epochs = env_or("EPOCHS", DEFAULT_EPOCHS);
    let coherence_w = env_or("COHERENCE", DEFAULT_COHERENCE);
    let sparsity_w = env_or("SPARSITY", DEFAULT_SPARSITY);
    let temporal_w = env_or("TEMPORAL", DEFAULT_TEMPORAL);
    let homeo_w = env_or("HOMEOSTASIS", DEFAULT_HOMEOSTASIS);
    let target_mean_w = env_or("TARGET_MEAN_W", DEFAULT_TARGET_MEAN_W);
    let weight_clamp = env_or("WEIGHT_CLAMP", DEFAULT_WEIGHT_CLAMP);
    let grad_clip_norm = env_or("GRAD_CLIP_NORM", 1.0);  // 0 = disabled
    let coherence_start = n_steps / 2;

    println!("  Input:  {path}");
    println!("  Output: {out_path}");
    println!("  Config: N_STEPS={n_steps}, DT={dt}, LR={lr}, EPOCHS={n_epochs}");
    println!("          COHERENCE={coherence_w}, SPARSITY={sparsity_w}, TEMPORAL={temporal_w}");
    println!("          HOMEOSTASIS={homeo_w}, TARGET_MEAN_W={target_mean_w}\n");

    // ─── Load .taas ─────────────────────────────────────
    let t = Instant::now();
    let mut f = File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut conn = taas_connectome::ConnectomeData::load_baked(&mut f)
        .unwrap_or_else(|e| panic!("Cannot load .taas: {e}"));
    let n = conn.n_neurons;
    let s = conn.n_synapses;
    println!("  Loaded: {n} neurons, {s} synapses ({:.1} syn/neu, {:.3}s)",
             s as f64 / n.max(1) as f64, t.elapsed().as_secs_f64());

    if n == 0 || s == 0 { eprintln!("Empty connectome"); std::process::exit(0); }

    // ─── Upload to GPU ─────────────────────────────────
    let ctx = cuda_kernels::GpuContext::new(n, s)
        .unwrap_or_else(|e| panic!("GpuContext::new: {e}"));
    let ext_I = vec![0.0f32; n];
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &ext_I).unwrap();
    println!("  Uploaded to GPU");

    // ─── Init Kuramoto ──────────────────────────────────
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let init_phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
    let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();

    // ─── Allocate GPU trajectory buffers ────────────────
    // All data for the entire pipeline lives on GPU — no H2D/D2H per step.
    let mut d_phases_traj = cuda_kernels::DevicePtr::<f32>::new(n_steps * n).unwrap();
    let mut d_coupling_traj = cuda_kernels::DevicePtr::<f32>::new(n_steps * n).unwrap();
    let mut d_dtheta_traj = cuda_kernels::DevicePtr::<f32>::new(n_steps * n).unwrap();
    let mut d_grad_weights = cuda_kernels::DevicePtr::<f32>::new(s).unwrap();
    // Gradient norm buffers (fully on GPU — no D2H transfer)
    let n_partial_blocks = (s + 255) / 256;
    let mut d_grad_norm = cuda_kernels::DevicePtr::<f32>::new(1).unwrap();
    let mut d_grad_norm_partial = cuda_kernels::DevicePtr::<f32>::new(n_partial_blocks).unwrap();

    // Upload omegas + init phases
    ctx.upload_kuramoto(&init_phases, &omegas, &vec![0.0f32; n]).unwrap();

    // ─── Warmup ─────────────────────────────────────────
    println!("  Warmup...");
    for _ in 0..5 {
        ctx.kuramoto_prepare().unwrap();
        ctx.kuramoto_csr_coupling().unwrap();
        ctx.kuramoto_update(dt).unwrap();
    }
    ctx.sync().unwrap();
    println!("  Warmup done\n");

    // ─── Host staging buffers (for per-epoch stats only) ─
    let mut h_phases_full = vec![0.0f32; n_steps * n];
    let mut h_weights = vec![0.0f32; s];
    let mut mean_abs_w = conn.values.iter().map(|w| w.abs()).sum::<f32>() / s as f32;

    // ─── Main refine loop — ALL on GPU ─────────────────
    println!("─── Refining: {n_epochs} epochs (all GPU) ───");
    let total_t0 = Instant::now();

    // ─── Main refine loop — each epoch is a CUDA Graph replay ─
    let total_t0 = Instant::now();

    for epoch in 0..n_epochs {
        let epoch_t0 = Instant::now();

        // ═══════════════════════════════════════════════
        //  Phase 1-4: CUDA Graph — forward + loss + BPTT + clip
        //  First call captures, subsequent calls replay
        //  (~480 kernel launches → 1 graph launch)
        // ═══════════════════════════════════════════════
        ctx.upload_kuramoto(&init_phases, &omegas, &vec![0.0f32; n]).unwrap();

        ctx.bptt_epoch_graph_replay(
            d_phases_traj.as_mut_ptr(),
            d_coupling_traj.as_mut_ptr(),
            d_dtheta_traj.as_mut_ptr(),
            d_grad_weights.as_mut_ptr(),
            d_grad_norm.as_mut_ptr(),
            d_grad_norm_partial.as_mut_ptr(),
            n_steps, dt, coherence_start, coherence_w, temporal_w, grad_clip_norm,
        ).unwrap();

        // ═══════════════════════════════════════════════
        //  Phase 5: Weight Update (outside graph — per-epoch scalars)
        // ═══════════════════════════════════════════════
        ctx.sync().unwrap();
        ctx.download_csr_values(&mut h_weights).unwrap();
        mean_abs_w = h_weights.iter().map(|w| w.abs()).sum::<f32>() / s as f32;

        let sparsity_scale = sparsity_w / s as f32;
        let homeo_diff = mean_abs_w - target_mean_w;
        let homeostasis_scale = 2.0 * homeo_w * homeo_diff / s as f32;

        let sparsity_scale = if sparsity_scale.is_finite() { sparsity_scale } else { 0.0 };
        let homeostasis_scale = if homeostasis_scale.is_finite() { homeostasis_scale } else { 0.0 };

        ctx.weight_update_async(
            d_grad_weights.as_ptr() as *const f32,
            lr, sparsity_scale, homeostasis_scale, weight_clamp,
        ).unwrap();
        ctx.sync().unwrap();

        // ═══════════════════════════════════════════════
        //  Stats
        // ═══════════════════════════════════════════════
        if epoch % 100 == 0 || epoch == n_epochs - 1 {
            d_phases_traj.copy_to_host(&mut h_phases_full).unwrap();

            let epoch_ms = epoch_t0.elapsed().as_secs_f64() * 1000.0;
            let final_off = (n_steps - 1) * n;
            let (sum_s, sum_c): (f64, f64) = h_phases_full[final_off..final_off + n]
                .iter().fold((0.0, 0.0), |(s, c), &th| (s + th.sin() as f64, c + th.cos() as f64));
            let coherence = ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32;

            println!("  epoch {epoch:>4}/{n_epochs}: R={coherence:.4} mean|w|={mean_abs_w:.4} ({epoch_ms:.0}ms)");
        }
    }

    let total_sec = total_t0.elapsed().as_secs_f64();
    println!("\n  Total refine time: {total_sec:.2}s ({:.2}s/epoch)",
             total_sec / n_epochs as f64);

    // ─── Download refined weights & save ──────────────
    ctx.download_csr_values(&mut conn.values).unwrap();
    println!("\n  Saving to {out_path}...");

    let mut out_f = File::create(&out_path)
        .unwrap_or_else(|e| panic!("Cannot create {out_path}: {e}"));
    conn.save_baked(&mut out_f)
        .unwrap_or_else(|e| panic!("Cannot save .taas: {e}"));
    println!("  ✅ Saved!\n");
}
