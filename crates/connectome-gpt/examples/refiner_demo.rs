//! # ConnectomeRefiner Demo
//!
//! Запуск:
//! ```bash
//! cargo run --example refiner_demo -p connectome-gpt
//! ```
//!
//! 1. Создаём коннектом с умеренной синхронизацией (sparse E/I смесь)
//! 2. Добавляем шум — coherence падает
//! 3. Refiner восстанавливает coherence через SpikeGrad BPTT
//! 4. Отчёт

use connectome_gpt::{
    ConnectomeSample, ConnectomeRefiner, RefinerConfig,
};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Создать коннектом с умеренной когерентностью.
/// - Mixed excitatory/inhibitory (80/20)
/// - Sparse (density 0.08)
/// - Умеренные веса — Kuramoto не полностью синхронизуется, но и не хаос
fn make_biorealistic_connectome(n: usize, seed: u64) -> ConnectomeSample {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut nt_types = vec![1.0f32; n];
    let density = 0.08;

    for i in 0..n {
        // 20% inhibitory
        let is_inh = rng.gen::<f32>() < 0.2;
        nt_types[i] = if is_inh { -1.0 } else { 1.0 };

        for j in 0..n {
            if i != j && rng.gen::<f32>() < density {
                let w = if is_inh {
                    -(0.3 + rng.gen::<f32>() * 0.4) // [-0.7, -0.3]
                } else {
                    0.2 + rng.gen::<f32>() * 0.5   // [0.2, 0.7]
                };
                col_indices.push(j as i32);
                values.push(w);
            }
        }
        row_ptr[i + 1] = col_indices.len() as i32;
    }

    let mut groups = HashMap::new();
    groups.insert("dn".to_string(), (0..n/10).collect());
    groups.insert("mn".to_string(), (n/5..n/5+n/20).collect());

    ConnectomeSample {
        source: "biorealistic".into(),
        n_neurons: n,
        n_synapses: values.len(),
        row_ptr, col_indices, values, nt_types, groups,
        features: vec![],
        depth: None,
    }
}

/// Быстрая Kuramoto симуляция + coherence R.
fn compute_coherence(sample: &ConnectomeSample, n_steps: usize, dt: f32, seed: u64) -> f32 {
    let n = sample.n_neurons;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
    let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];

    for _ in 0..n_steps {
        for i in 0..n { sin_buf[i] = phases[i].sin(); cos_buf[i] = phases[i].cos(); }
        for i in 0..n {
            let start = sample.row_ptr[i].max(0) as usize;
            let end = (sample.row_ptr[i + 1] as usize).min(sample.col_indices.len());
            let mut cpl = 0.0f32;
            for idx in start..end {
                let j = sample.col_indices[idx] as usize;
                if j >= n { continue; }
                let w = sample.values[idx];
                cpl += w * (sin_buf[j] * cos_buf[i] - cos_buf[j] * sin_buf[i]);
            }
            phases[i] += dt * (omegas[i] + cpl);
        }
    }

    let (sum_s, sum_c): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &th| {
        (s + th.sin() as f64, c + th.cos() as f64)
    });
    ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len().max(1) as f32
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  ConnectomeRefiner — SpikeGrad Synaptic Denoise ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let n = 32;
    let n_steps = 80;
    let dt = 0.05;

    // ── 1. Create connectome ──
    println!("▶ Creating biorealistic connectome (n={}, density=0.08, E/I=80/20)...", n);
    let clean = make_biorealistic_connectome(n, 42);
    println!("   Synapses: {}", clean.values.len());
    let baseline_r = compute_coherence(&clean, n_steps, dt, 0);
    println!("   Baseline coherence: {:.4}", baseline_r);
    println!();

    // ── 2. Add noise ──
    println!("▶ Adding noise (σ=0.5 additive)...");
    let mut noisy = clean.clone();
    let mut rng = StdRng::seed_from_u64(999);
    for w in &mut noisy.values {
        *w += rng.gen::<f32>() * 0.5;
    }
    let noisy_r = compute_coherence(&noisy, n_steps, dt, 0);
    let noisy_mse = mse(&noisy.values, &clean.values);
    println!("   Noisy coherence: {:.4}  (⬇ {:.1}%)",
        noisy_r, (1.0 - noisy_r / baseline_r) * 100.0);
    println!("   MSE: {:.6}", noisy_mse);
    println!();

    // ── 3. Run refiner ──
    println!("▶ Running ConnectomeRefiner (500 epochs)...");
    println!();

    let config = RefinerConfig {
        n_steps,
        dt,
        lr: 0.08,
        coherence_weight: 1.0,
        sparsity_weight: 0.002,
        temporal_weight: 0.05,
        homeostasis_weight: 0.01,
        target_mean_weight: 0.4,
        omegas: vec![],
        seed: 42,
        ..Default::default()
    };
    let mut refiner = ConnectomeRefiner::new(config);
    let start = std::time::Instant::now();
    let (refined, history) = refiner.refine(&noisy, 500);
    let elapsed = start.elapsed();

    // ── 4. Final metrics ──
    let refined_r = compute_coherence(&refined, n_steps, dt, 0);
    let refined_mse = mse(&refined.values, &clean.values);

    let first = history.first().unwrap();
    let last = history.last().unwrap();

    // ── 5. Report ──
    println!();
    println!("╔══════════════════════════════╦══════════╦══════════╦══════════╗");
    println!("║  Metric                     ║ Baseline ║  Noisy   ║ Refined  ║");
    println!("╠══════════════════════════════╬══════════╬══════════╬══════════╣");
    println!("║  Coherence R (order param)  ║ {:>7.4} ║ {:>7.4} ║ {:>7.4} ║",
        baseline_r, noisy_r, refined_r);
    println!("║  Loss (coherence + regularize)║         ║ {:>+8.4} ║ {:>+8.4}║", first.loss, last.loss);
    println!("║  Sparsity penalty (L1 mean)  ║         ║ {:>8.4} ║ {:>8.4}║",
        first.sparsity_penalty, last.sparsity_penalty);
    println!("║  Temporal smoothness penalty ║         ║ {:>8.6} ║ {:>8.6}║",
        first.temporal_penalty, last.temporal_penalty);
    println!("║  Homeostasis penalty         ║         ║ {:>8.6} ║ {:>8.6}║",
        first.homeostasis_penalty, last.homeostasis_penalty);
    println!("║  Mean |weight|               ║         ║ {:>8.4} ║ {:>8.4}║",
        first.mean_weight, last.mean_weight);
    println!("║  MSE vs clean connectome     ║         ║ {:>8.6} ║ {:>8.6}║", noisy_mse, refined_mse);
    println!("╠══════════════════════════════╩══════════╩══════════╩══════════╣");
    println!("║  Time: {:.3}s                                          ║", elapsed.as_secs_f32());
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 6. Summary ──
    println!("📊 **Summary**");
    println!();

    let coherence_loss = baseline_r - noisy_r;
    let coherence_gain = refined_r - noisy_r;
    let recovery_pct = if coherence_loss > 0.0 {
        coherence_gain / coherence_loss * 100.0
    } else {
        0.0
    };
    let mse_change_pct = (1.0 - refined_mse / noisy_mse) * 100.0;

    if coherence_gain > 0.01 {
        println!("✅ Coherence recovered: {:.3} → {:.3} → {:.3} (+{:.1}% of lost coherence)",
            baseline_r, noisy_r, refined_r, recovery_pct);
    } else {
        println!("⚠️  Coherence: {:.3} → {:.3} → {:.3} (minimal change)",
            baseline_r, noisy_r, refined_r);
    }

    if mse_change_pct > 1.0 {
        println!("✅ MSE vs clean: {:.6} → {:.6}  (⬇ {:.1}%)",
            noisy_mse, refined_mse, mse_change_pct);
    } else if mse_change_pct < -1.0 {
        println!("⚠️  MSE vs clean: {:.6} → {:.6}  (⬆ {:.1}%) — weights diverged",
            noisy_mse, refined_mse, -mse_change_pct);
    } else {
        println!("→ MSE vs clean: {:.6} → {:.6} (stable)",
            noisy_mse, refined_mse);
    }

    if history.last().unwrap().loss < history.first().unwrap().loss {
        println!("✅ Loss: {:.4} → {:.4}  (⬇ via BPTT gradient descent)",
            first.loss, last.loss);
    }

    println!();
    println!("🔬 **Interpretation**");
    println!("   ConnectomeRefiner использует SpikeGrad (BPTT через Kuramoto)");
    println!("   чтобы найти градиенты dL/dW_ij и обновить синаптические веса.");
    println!("   Loss = -R + λ₁·mean|W| + λ₂·temporal + λ₃·|mean|W|-target|²");
    println!();
    if coherence_gain > 0.01 {
        println!("   🎯 Denoising successful — synaptic cleanup improved phase synchrony!");
    } else {
        println!("   💡 Tip: На реальном FlyWire коннектоме (>100K нейронов) эффект сильнее,");
        println!("       т.к. градиенты усредняются по большему числу синапсов.");
    }
}
