//! # refine_connectome — Kuramoto BPTT denoising for real connectomes
//!
//! Загружает baked .taas коннектом, прогоняет ConnectomeRefiner (differentiable
//! Kuramoto + BPTT) для очистки синаптических весов, и сохраняет результат.
//!
//! ## Запуск
//! ```bash
//! cargo run --example refine_connectome -p connectome-gpt --release -- <input.taas> [output.taas]
//! ```
//!
//! По умолчанию output = `<input>_refined.taas`.
//!
//! ## Параметры refiner (через переменные окружения):
//! - `N_STEPS=80` — шаги Kuramoto
//! - `LR=0.08` — learning rate
//! - `EPOCHS=500` — число эпох
//! - `COHERENCE=1.0` — weight for coherence loss
//! - `SPARSITY=0.002` — weight for L1 sparsity
//! - `HOMEOSTASIS=0.01` — weight для homeostasis

use connectome_gpt::{
    ConnectomeSample, ConnectomeRefiner, RefinerConfig,
};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Resolve .taas file path relative to project root.
/// When running via `cargo run -p connectome-gpt`, CWD is `crates/connectome-gpt/`.
fn resolve_taas_path(path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.exists() { return p.to_path_buf(); }
    // Try relative to project root (../../ from crates/connectome-gpt/)
    let alt = Path::new("../../").join(path);
    if alt.exists() { return alt; }
    // Try absolute or relative to CWD
    p.to_path_buf()
}

fn from_env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Convert ConnectomeData → ConnectomeSample (for the refiner).
fn conn_to_sample(conn: &taas_connectome::ConnectomeData, source: &str) -> ConnectomeSample {
    let groups: HashMap<String, Vec<usize>> = conn.groups.iter()
        .map(|g| (g.name.clone(), g.indices.clone()))
        .collect();

    ConnectomeSample {
        source: source.to_string(),
        n_neurons: conn.n_neurons,
        n_synapses: conn.n_synapses,
        row_ptr: conn.row_ptr.clone(),
        col_indices: conn.col_indices.clone(),
        values: conn.values.clone(),
        nt_types: conn.nt_types.clone(),
        groups,
        features: vec![],
        depth: None,
    }
}

/// Convert refined ConnectomeSample → ConnectomeData (preserve original metadata).
fn sample_to_conn(sample: &ConnectomeSample, orig: &taas_connectome::ConnectomeData) -> taas_connectome::ConnectomeData {
    let groups: Vec<taas_connectome::NeuronGroup> = sample.groups.iter()
        .map(|(name, indices)| taas_connectome::NeuronGroup {
            name: name.clone(),
            indices: indices.clone(),
        })
        .collect();

    taas_connectome::ConnectomeData {
        n_neurons: sample.n_neurons,
        n_synapses: sample.n_synapses,
        row_ptr: sample.row_ptr.clone(),
        col_indices: sample.col_indices.clone(),
        values: sample.values.clone(),
        nt_types: sample.nt_types.clone(),
        receptor_types: orig.receptor_types.clone(), // preserve original receptor mapping
        groups,
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════╗");
    println!("║  ConnectomeRefiner — Kuramoto BPTT denoising for .taas ║");
    println!("╚═════════════════════════════════════════════════════════╝");

    // ─── CLI args ───────────────────────────────────
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.taas> [output.taas]", args[0]);
        eprintln!();
        eprintln!("Environment variables:");
        eprintln!("  N_STEPS=80      — Kuramoto steps per forward");
        eprintln!("  LR=0.08         — learning rate");
        eprintln!("  EPOCHS=500      — number of refinement epochs");
        eprintln!("  COHERENCE=1.0   — coherence loss weight");
        eprintln!("  SPARSITY=0.002  — L1 sparsity weight");
        eprintln!("  HOMEOSTASIS=0.01 — homeostasis weight");
        std::process::exit(1);
    }

    let input_arg = &args[1];
    let input_path = resolve_taas_path(input_arg);
    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        let stem = input_path.to_str().unwrap_or(input_arg)
            .strip_suffix(".taas").unwrap_or(input_arg);
        format!("{}_refined.taas", stem)
    };

    // ─── Config from env ────────────────────────────
    let n_steps = from_env_or("N_STEPS", 80usize);
    let lr = from_env_or("LR", 0.08f32);
    let n_epochs = from_env_or("EPOCHS", 500usize);
    let coherence_w = from_env_or("COHERENCE", 1.0f32);
    let sparsity_w = from_env_or("SPARSITY", 0.002f32);
    let homeostasis_w = from_env_or("HOMEOSTASIS", 0.01f32);
    let dt = from_env_or("DT", 0.05f32);

    println!();
    println!("  Input:  {}", input_path.display());
    println!("  Output: {output_path}");
    println!("  Config: N_STEPS={n_steps}, LR={lr}, EPOCHS={n_epochs}");
    println!("          COHERENCE={coherence_w}, SPARSITY={sparsity_w}, HOMEOSTASIS={homeostasis_w}");
    println!();

    // ─── Load .taas ─────────────────────────────────
    let t = Instant::now();
    let mut f = File::open(&input_path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {e}", input_path.display()));
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f)
        .unwrap_or_else(|e| panic!("Cannot load .taas: {e}"));
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;

    let n = conn.n_neurons;
    let s = conn.n_synapses;
    let avg_syn = if n > 0 { s as f64 / n as f64 } else { 0.0 };
    let n_exc = conn.nt_types.iter().filter(|&&t| t > 0.0).count();
    let n_inh = n - n_exc;
    println!("  Loaded: {n} neurons, {s} synapses ({avg_syn:.0} syn/neu) in {load_ms:.1}ms");
    println!("  E/I: {n_exc}/{n_inh} ({:.1}%)", n_exc as f64 / n.max(1) as f64 * 100.0);
    println!("  Groups: {}", conn.groups.len());
    for g in &conn.groups {
        println!("    {}: {} neurons", g.name, g.indices.len());
    }
    println!("  Has receptor_types: {}", !conn.receptor_types.is_empty());
    println!();

    // ─── Validate ───────────────────────────────────
    if n == 0 || s == 0 {
        eprintln!("Error: empty connectome (n={n}, syn={s})");
        std::process::exit(1);
    }

    // ─── Convert to ConnectomeSample ────────────────
    let sample = conn_to_sample(&conn, input_path.to_str().unwrap_or(input_arg));
    let mean_w: f32 = sample.values.iter().map(|w| w.abs()).sum::<f32>() / s.max(1) as f32;
    let w_min = sample.values.iter().cloned().fold(f32::MAX, f32::min);
    let w_max = sample.values.iter().cloned().fold(f32::MIN, f32::max);
    println!("  Weights: min={:.6e}, max={:.6e}, mean|w|={:.6e}", w_min, w_max, mean_w);

    // ─── Compute baseline coherence ─────────────────
    // Quick Kuramoto sim to measure pre-refinement coherence
    println!();
    println!("  Computing baseline coherence...");
    let t0 = Instant::now();
    let baseline_coherence = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
        let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
        let mut sin_buf = vec![0.0f32; n];
        let mut cos_buf = vec![0.0f32; n];

        for _ in 0..n_steps {
            for i in 0..n { sin_buf[i] = phases[i].sin(); cos_buf[i] = phases[i].cos(); }
            for i in 0..n {
                let start = sample.row_ptr[i].max(0) as usize;
                let end = (sample.row_ptr[i + 1] as usize).min(s);
                let mut cpl = 0.0f32;
                for idx in start..end {
                    let j = sample.col_indices[idx] as usize;
                    if j >= n { continue; }
                    cpl += sample.values[idx] * (sin_buf[j] * cos_buf[i] - cos_buf[j] * sin_buf[i]);
                }
                phases[i] += dt * (omegas[i] + cpl);
            }
        }
        let (sum_s, sum_c): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &th| {
            (s + th.sin() as f64, c + th.cos() as f64)
        });
        ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32
    };
    println!("  Baseline coherence R = {:.4} ({:.1}ms)", baseline_coherence,
        t0.elapsed().as_secs_f64() * 1000.0);

    // ─── Run refiner ────────────────────────────────
    println!();
    println!("▶ Running ConnectomeRefiner ({n_epochs} epochs)...");
    println!();

    let config = RefinerConfig {
        n_steps,
        dt,
        lr,
        coherence_weight: coherence_w,
        sparsity_weight: sparsity_w,
        temporal_weight: from_env_or("TEMPORAL", 0.05f32),
        homeostasis_weight: homeostasis_w,
        target_mean_weight: mean_w, // keep mean weight close to original
        omegas: vec![],
        seed: from_env_or("SEED", 42u64),
        max_grad_norm: from_env_or("MAX_GRAD", 5.0f32),
        weight_clamp: from_env_or("WEIGHT_CLAMP", 0.0f32), // 0 = no clamp
        ..Default::default()
    };
    let mut refiner = ConnectomeRefiner::new(config);
    let refine_start = Instant::now();
    let (refined, history) = refiner.refine(&sample, n_epochs);
    let refine_elapsed = refine_start.elapsed();

    let steps_per_sec = n_epochs as f64 / refine_elapsed.as_secs_f64();
    let us_per_step = refine_elapsed.as_secs_f64() / n_epochs as f64 * 1_000_000.0;

    // ─── Results ────────────────────────────────────
    let first = history.first().unwrap();
    let last = history.last().unwrap();

    println!();
    println!("╔══════════════════════════════╦═══════════════╦═══════════════╗");
    println!("║  Metric                     ║   Before      ║   After       ║");
    println!("╠══════════════════════════════╬═══════════════╬═══════════════╣");
    println!("║  Loss                        ║ {:>12.6} ║ {:>12.6} ║", first.loss, last.loss);
    println!("║  Coherence R                 ║ {:>12.4} ║ {:>12.4} ║", first.coherence, last.coherence);
    println!("║  Sparsity (L1 mean)          ║ {:>12.6} ║ {:>12.6} ║", first.sparsity_penalty, last.sparsity_penalty);
    println!("║  Temporal penalty            ║ {:>12.6} ║ {:>12.6} ║", first.temporal_penalty, last.temporal_penalty);
    println!("║  Homeostasis penalty         ║ {:>12.6} ║ {:>12.6} ║", first.homeostasis_penalty, last.homeostasis_penalty);
    println!("║  Mean |weight|               ║ {:>12.6e} ║ {:>12.6e} ║", first.mean_weight, last.mean_weight);
    println!("║  PLV (phase locking)         ║ {:>12.6} ║ {:>12.6} ║", first.plv, last.plv);
    println!("╠══════════════════════════════╩═══════════════╩═══════════════╣");
    println!("║  Time: {:.3}s ({:.0} steps/s, {:.0} µs/step)           ║",
        refine_elapsed.as_secs_f64(), steps_per_sec, us_per_step);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // ── Weight change stats ──
    let mut total_change = 0.0f32;
    let mut max_change = 0.0f32;
    let mut n_increased = 0usize;
    let mut n_decreased = 0usize;
    for (orig, new) in sample.values.iter().zip(refined.values.iter()) {
        let delta = (new - orig).abs();
        total_change += delta;
        max_change = max_change.max(delta);
        if *new > *orig { n_increased += 1; }
        else if *new < *orig { n_decreased += 1; }
    }
    let mean_change = total_change / s.max(1) as f32;
    let mean_w_after: f32 = refined.values.iter().map(|w| w.abs()).sum::<f32>() / s.max(1) as f32;

    println!();
    println!("  Weight changes:");
    println!("    Mean |Δw|:   {:.6e}  ({:.1}% of mean|w|)",
        mean_change, mean_change / mean_w_after.max(1e-20) * 100.0);
    println!("    Max |Δw|:    {:.6e}", max_change);
    println!("    Increased:   {n_increased} synapses");
    println!("    Decreased:   {n_decreased} synapses");
    println!("    Mean |w|:    {:.6e} → {:.6e}", mean_w, mean_w_after);

    // ── Compute post-refinement coherence ──
    let _t0 = Instant::now();
    let refined_coherence = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
        let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
        let mut sin_buf = vec![0.0f32; n];
        let mut cos_buf = vec![0.0f32; n];

        for _ in 0..n_steps {
            for i in 0..n { sin_buf[i] = phases[i].sin(); cos_buf[i] = phases[i].cos(); }
            for i in 0..n {
                let start = refined.row_ptr[i].max(0) as usize;
                let end = (refined.row_ptr[i + 1] as usize).min(s);
                let mut cpl = 0.0f32;
                for idx in start..end {
                    let j = refined.col_indices[idx] as usize;
                    if j >= n { continue; }
                    cpl += refined.values[idx] * (sin_buf[j] * cos_buf[i] - cos_buf[j] * sin_buf[i]);
                }
                phases[i] += dt * (omegas[i] + cpl);
            }
        }
        let (sum_s, sum_c): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &th| {
            (s + th.sin() as f64, c + th.cos() as f64)
        });
        ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32
    };
    println!("  Coherence R:  {:.4} → {:.4}  ({:+.1}%)",
        baseline_coherence, refined_coherence,
        (refined_coherence / baseline_coherence.max(1e-10) - 1.0) * 100.0);

    // ─── Save refined .taas ────────────────────────
    println!();
    println!("▶ Saving refined connectome → {output_path}");

    let conn_refined = sample_to_conn(&refined, &conn);
    let mut out_f = File::create(&output_path)
        .unwrap_or_else(|e| panic!("Cannot create {output_path}: {e}"));
    conn_refined.save_baked(&mut out_f)
        .unwrap_or_else(|e| panic!("Cannot save .taas: {e}"));

    let file_size = out_f.metadata().map(|m| m.len()).unwrap_or(0);
    println!("  Saved {:.1}MB", file_size as f64 / 1_048_576.0);
    println!();

    // ─── Verdict ────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════╗");
    if last.loss < first.loss {
        println!("║  ✅ Refinement successful — loss decreased by {:.2}%      ║",
            (1.0 - last.loss / first.loss.abs().max(1e-10)) * 100.0);
    } else {
        println!("║  ⚠️  Loss did not decrease — try different hyperparams   ║");
    }
    if refined_coherence > baseline_coherence {
        println!("║  ✅ Coherence improved: {:.4} → {:.4}                     ║",
            baseline_coherence, refined_coherence);
    } else {
        println!("║  → Coherence stable: {:.4} → {:.4}                        ║",
            baseline_coherence, refined_coherence);
    }
    println!("║                                                          ║");
    println!("║  Next: run HH stimulation on refined connectome:         ║");
    println!("║    cargo run --features cuda --example t2_vision         ║");
    println!("║    --release -- {output_path}              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}
