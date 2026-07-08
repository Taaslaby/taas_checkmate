//! # Synthetic Sensorimotor Test — SpikeGrad PLV Validation
//!
//! Игрушечный connectome с гарантированным сенсор→интегратор→мотор паттерном.
//! Проверяет что SpikeGrad PLV действительно увеличивает Phase Locking Value.
//!
//! Архитектура:
//!   5 sensory → 10 central → 5 motor
//!   (dense feedforward: каждый сенсор → каждый централ → каждый мотор)
//!
//! Запуск:
//! ```bash
//! cargo run --example fly_synth_sensorimotor -p connectome-gpt
//! ```
//!
//! Ожидается:
//!   PLV = 0.0x (before) → 0.y (after) — должен расти!

use connectome_gpt::{ConnectomeSample, ConnectomeRefiner, RefinerConfig};
use std::collections::HashMap;

fn make_synth_sensorimotor() -> ConnectomeSample {
    let n_sensory = 5;
    let n_central = 10;
    let n_motor = 5;
    let n = n_sensory + n_central + n_motor; // 20

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut nt_types = vec![1.0f32; n]; // all excitatory

    // ===== CSR convention: row_ptr[i] → j means edge FROM j TO i (incoming).
    // coupling[i] += W_ij * sin(θ_j - θ_i) — i is influenced by j.
    // For forward propagation: sensory→central→motor
    // =====

    // Sensory neurons (0..4): no incoming edges (driven only by stimulus)
    for s in 0..n_sensory {
        row_ptr[s + 1] = col_indices.len() as i32; // empty row
    }

    // Central (5..14) receives from Sensory (0..4): all-to-all, weight = 0.5
    // coupling[central] += W * sin(θ_s - θ_c) — central follows sensory
    for c in n_sensory..n_sensory + n_central {
        for s in 0..n_sensory {
            col_indices.push(s as i32);
            values.push(0.5);
        }
        row_ptr[c + 1] = col_indices.len() as i32;
    }

    // Motor (15..19) receives from Central (5..14): all-to-all, weight = 0.5
    // coupling[motor] += W * sin(θ_c - θ_m) — motor follows central
    for m in n_sensory + n_central..n {
        for c in n_sensory..n_sensory + n_central {
            col_indices.push(c as i32);
            values.push(0.5);
        }
        row_ptr[m + 1] = col_indices.len() as i32;
    }

    // Groups
    let mut groups = HashMap::new();
    groups.insert("sensory".into(), (0..n_sensory).collect());
    groups.insert("central".into(), (n_sensory..n_sensory + n_central).collect());
    groups.insert("motor".into(), (n_sensory + n_central..n).collect());

    ConnectomeSample {
        source: "synth_sensorimotor".into(),
        n_neurons: n,
        n_synapses: col_indices.len(),
        row_ptr, col_indices, values, nt_types, groups,
        features: vec![],
        depth: None,
    }
}

fn compute_plv_external(
    values: &[f32], row_ptr: &[i32], col_indices: &[i32],
    n: usize, sensory_idx: &[usize], motor_idx: &[usize],
    stim_freq: f32, stim_amp: f32, n_steps: usize, dt: f32, seed: u64,
) -> f32 {
    use rand::Rng;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
    let omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];

    for step in 0..n_steps {
        let t = step as f32 * dt;
        let stimulus = (stim_freq * t * std::f32::consts::TAU).sin() * stim_amp;

        for i in 0..n {
            sin_buf[i] = phases[i].sin();
            cos_buf[i] = phases[i].cos();
        }

        for i in 0..n {
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1] as usize).min(col_indices.len());
            let mut cpl = 0.0f32;
            for idx in start..end {
                let j = col_indices[idx] as usize;
                if j >= n { continue; }
                cpl += values[idx] * (sin_buf[j] * cos_buf[i] - cos_buf[j] * sin_buf[i]);
            }
            let drive = if sensory_idx.contains(&i) { stimulus } else { 0.0 };
            phases[i] += dt * (omegas[i] + cpl + drive);
        }
    }

    // PLV over last 50%
    let plv_start = n_steps / 2;
    let mut complex_sum = (0.0f64, 0.0f64);
    let mut count = 0usize;

    for step in plv_start..n_steps {
        let t = step as f32 * dt;
        let stim_phase = (stim_freq * t * std::f32::consts::TAU).fract() * std::f32::consts::TAU;
        let mut m_sin = 0.0f64;
        let mut m_cos = 0.0f64;
        for &mi in motor_idx {
            let diff = phases[mi] as f64 - stim_phase as f64;
            m_sin += diff.sin();
            m_cos += diff.cos();
        }
        if !motor_idx.is_empty() {
            complex_sum.0 += m_sin / motor_idx.len() as f64;
            complex_sum.1 += m_cos / motor_idx.len() as f64;
            count += 1;
        }
    }

    if count > 0 {
        let (s, c) = complex_sum;
        ((s * s + c * c).sqrt() / count as f64) as f32
    } else {
        0.0
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Synthetic Sensorimotor — SpikeGrad PLV Test    ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let sample = make_synth_sensorimotor();
    let n = sample.n_neurons;
    let sensory_idx: Vec<usize> = (0..5).collect();
    let motor_idx: Vec<usize> = (15..20).collect();

    println!("📊 Connectome: {} neurons, {} synapses", n, sample.n_synapses);
    println!("   Sensory (0..4) → Central (5..14) → Motor (15..19)");
    println!();

    // ── Before refinement ──
    println!("▶ Before refinement...");
    let plv_before = compute_plv_external(
        &sample.values, &sample.row_ptr, &sample.col_indices, n,
        &sensory_idx, &motor_idx, 0.5, 3.0, 100, 0.05, 42,
    );
    println!("   Motor PLV: {:.6}", plv_before);
    println!();

    // ── Run PLV-aware refiner ──
    println!("▶ Running SpikeGrad PLV refiner (500 epochs, lr=0.01, decay=0.995)...");
    let config = RefinerConfig {
        n_steps: 50, dt: 0.05, lr: 0.01,
        lr_decay: 0.995,
        coherence_weight: 0.0,
        sparsity_weight: 0.001,
        temporal_weight: 0.0,
        homeostasis_weight: 0.0,
        target_mean_weight: 0.5,
        omegas: vec![],
        seed: 42,
        sensory_idx: sensory_idx.clone(),
        motor_idx: motor_idx.clone(),
        stim_freq: 0.5,
        stim_amp: 3.0,
        plv_weight: 10.0,
        max_grad_norm: 5.0,
        weight_clamp: 5.0,
        n_bptt_steps: 15,
    };

    // Debug: inspect phase gradient directly
    println!("\n🔬 **Debug: PLV gradient analysis**");
    {
        let mut dbg_refiner = ConnectomeRefiner::new(config.clone());
        let traj = dbg_refiner.forward(&sample, None);
        let (_loss, _coh, plv, _sp, _tp, _hp) = dbg_refiner.loss(&traj);
        let grad = dbg_refiner.backward(&traj);
        
        // Debug gradient norms
        // NOTE: central's row contains INCOMING edges from sensory (sensory→central)
        //       motor's row contains INCOMING edges from central (central→motor)
        let mut grad_sens_cent = 0.0f32;
        let mut grad_cent_mot = 0.0f32;
        let mut total_grad_norm = 0.0f32;
        // Central (5..14) incoming from sensory → sensor→central weights
        for i in 5..15 {
            let start = sample.row_ptr[i] as usize;
            let end = sample.row_ptr[i + 1] as usize;
            for idx in start..end {
                let g = grad[idx];
                grad_sens_cent += g * g;
                total_grad_norm += g * g;
            }
        }
        // Motor (15..19) incoming from central → central→motor weights
        for i in 15..20 {
            let start = sample.row_ptr[i] as usize;
            let end = sample.row_ptr[i + 1] as usize;
            for idx in start..end {
                let g = grad[idx];
                grad_cent_mot += g * g;
                total_grad_norm += g * g;
            }
        }
        println!("   Phase err at epoch 0: {:.6}", plv);
        println!("   ‖grad‖ sensory→central: {:.6}", grad_sens_cent.sqrt());
        println!("   ‖grad‖ central→motor:   {:.6}", grad_cent_mot.sqrt());
        println!("   ‖grad‖ total:           {:.6}", total_grad_norm.sqrt());
    }

    let mut refiner = ConnectomeRefiner::new(config);
    let refine_start = std::time::Instant::now();
    let (refined, history) = refiner.refine(&sample, 500);
    let refine_elapsed = refine_start.elapsed();

    // ── Trajectory analysis: PLV every 20 epochs ──        println!("\n📈 **Phase loss trajectory (every 50 epochs)**");
    for (i, r) in history.iter().enumerate() {
        if i % 50 == 0 {
            println!("   epoch {:>3}: loss={:>+8.4}  R={:.4}  phase_err={:.4}  |Δw|={:.6}",
                i, r.loss, r.coherence, r.plv, r.weight_change_norm);
        }
    }

    // ── After refinement ──
    println!("\n▶ After refinement...");
    let plv_after = compute_plv_external(
        &refined.values, &refined.row_ptr, &refined.col_indices, n,
        &sensory_idx, &motor_idx, 0.5, 3.0, 100, 0.05, 42,
    );
    println!("   Motor PLV: {:.6}", plv_after);

    let first = history.first().unwrap();
    let last = history.last().unwrap();

    // ── Weight change analysis ──
    // NOTE: with forward-propagation edge direction:
    //   - sensory→central weights live in central's rows (central receives from sensory)
    //   - central→motor weights live in motor's rows (motor receives from central)
    println!("\n🔬 **Weight change analysis**");
    let mut mean_w_sens_cent = 0.0f32;
    let mut cnt_sens_cent = 0;
    let mut mean_w_cent_mot = 0.0f32;
    let mut cnt_cent_mot = 0;
    // Central (5..14) incoming from sensory → sensory→central weights
    for i in 5..15 {
        let start = refined.row_ptr[i] as usize;
        let end = refined.row_ptr[i + 1] as usize;
        for idx in start..end {
            mean_w_sens_cent += refined.values[idx];
            cnt_sens_cent += 1;
        }
    }
    // Motor (15..19) incoming from central → central→motor weights
    for i in 15..20 {
        let start = refined.row_ptr[i] as usize;
        let end = refined.row_ptr[i + 1] as usize;
        for idx in start..end {
            mean_w_cent_mot += refined.values[idx];
            cnt_cent_mot += 1;
        }
    }
    println!("   Mean weight sensory→central: {:.4} (was 0.5000)", mean_w_sens_cent / cnt_sens_cent as f32);
    println!("   Mean weight central→motor:   {:.4} (was 0.5000)", mean_w_cent_mot / cnt_cent_mot as f32);

    println!();
    println!("╔══════════════════════════════╦═══════════╦═══════════╗");
    println!("║  Metric                     ║ Before    ║ After     ║");
    println!("╠══════════════════════════════╬═══════════╬═══════════╣");
    println!("║  Loss                       ║ {:>+8.4} ║ {:>+8.4} ║", first.loss, last.loss);
    println!("║  Coherence R                ║ {:>8.4} ║ {:>8.4} ║", first.coherence, last.coherence);
    println!("║  PLV (refiner)              ║ {:>8.4} ║ {:>8.4} ║", first.plv, last.plv);
    println!("║  External PLV               ║ {:>8.4} ║ {:>8.4} ║", plv_before, plv_after);
    println!("║  Weight Δ sens→cent         ║ {:>8.4} ║ {:>8.4} ║", 0.5, mean_w_sens_cent / cnt_sens_cent as f32);
    println!("║  Weight Δ cent→motor        ║ {:>8.4} ║ {:>8.4} ║", 0.5, mean_w_cent_mot / cnt_cent_mot as f32);
    println!("╠══════════════════════════════╩═══════════╩═══════════╣");
    println!("║  N={}, E={}, 200 epochs in {:.2}s        ║",
        n, sample.n_synapses, refine_elapsed.as_secs_f32());
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    // ── Verdict: phase_err lower = better (0 = perfect phase lock) ──
    if last.plv < first.plv {
        let drop = (1.0 - last.plv / first.plv) * 100.0;
        println!("✅✅ Phase error: {:.4} → {:.4} (↓{:.1}%)", first.plv, last.plv, drop);
    } else {
        println!("→ Phase error: {:.4} → {:.4} (increased)", first.plv, last.plv);
    }
    if last.coherence > first.coherence {
        let gain = (last.coherence / first.coherence - 1.0) * 100.0;
        println!("✅ Coherence: {:.4} → {:.4} (+{:.1}%)", first.coherence, last.coherence, gain);
    }
    if last.loss < first.loss {
        println!("✅ Loss: {:.4} → {:.4}", first.loss, last.loss);
    }

    // ── Diagnosis ──
    println!();
    println!("┌────────────────────────────────────────────────────┐");
    println!("│ 🔍 **Diagnosis**                                   │");
    if last.plv < first.plv {
        println!("│   ✅ Phase error decreased — motor is locking to    │");
        println!("│      stimulus.                                     │");
        if last.plv < 0.5 {
            println!("│   🎯 Strong lock: phase_err < 0.5!                 │");
        }
    } else {
        println!("│   ❌ Phase error did not improve.                   │");
        println!("│   Gradient may not be reaching correct synapses.    │");
    }
    println!("└────────────────────────────────────────────────────┘");
    println!();
    println!("✅ Debug synthetic test complete!");
}
