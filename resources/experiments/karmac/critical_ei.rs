/// Critical E/I — фазовый переход workspace selectivity при балансе возбуждения/торможения.
///
/// ## Гипотеза
///
/// Существует критическое E/I ratio, при котором workspace selectivity
/// максимальна, а variance (чувствительность к random seed) пиковая.
/// Это signature фазового перехода: система на границе "local processing"
/// и "global broadcast".
///
/// ## Почему это важно
///
/// 1. Биологический мозг имеет ~80% E / ~20% I — мы проверим, это ли
///    оптимальный баланс для information transmission через GNW.
/// 2. Аномалии E/I balance связаны с аутизмом, шизофренией, эпилепсией.
///    Если мы найдём точное критическое E/I — это модельный предиктор.
/// 3. Наш предыдущий experiment показал: E/I=0.8 → единственный
///    положительный scaling exponent. Теперь проверим детально.
///
/// ## Метод
///
/// 1. Фиксируем N=2048, degree=12, n_blocks=8
/// 2. Сканируем E/I от 0.0 до 1.0 с шагом 0.05 (21 точка)
/// 3. Для каждого E/I:
///    a. Генерируем 5 случайных connectome (разные seed)
///    b. Для каждого — 8 GNW seeds
///    c. Измеряем selectivity + variance
/// 4. Анализ:
///    - Пик selectivity = оптимальный E/I
///    - Пик variance = критическая точка (фазовый переход)
///    - Отношение signal/noise в workspace
///
/// Запуск: cargo run --package karmac --example critical_ei --release

use std::time::Instant;
use rand::Rng;

// ─── Params ─────────────────────────────────────────────────────────────────

/// Number of neurons (fixed)
const N_NEURONS: usize = 2048;
/// Number of SBM blocks
const N_BLOCKS: usize = 8;
/// Target average degree (fixed)
const TARGET_DEGREE: f32 = 12.0;
/// Fraction of within-block connections
const WITHIN_FRAC: f32 = 0.75;
/// Workspace size
const WORKSPACE_SIZE: usize = 32;
/// Number of GNW seeds per measurement
const N_GNW_SEEDS: usize = 8;
/// Number of random connectome instances per E/I
const N_CONN_SEEDS: usize = 5;

// ─── SBM probabilities ──────────────────────────────────────────────────────

fn sbm_probs(n: usize, n_blocks: usize, target_degree: f32, within_frac: f32) -> (f32, f32) {
    let block_size = n / n_blocks;
    let d_within = target_degree * within_frac;
    let d_between = target_degree * (1.0 - within_frac);
    (
        d_within / (block_size as f32 - 1.0),
        d_between / ((n - block_size) as f32 + 1.0),
    )
}

// ─── Connectome generation ──────────────────────────────────────────────────

fn generate_connectome(n: usize, n_blocks: usize, ei_ratio: f32,
    target_degree: f32, within_frac: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>)
{
    let mut rng = rand::thread_rng();
    let (within_prob, between_prob) = sbm_probs(n, n_blocks, target_degree, within_frac);
    let block_size = n / n_blocks;

    let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        let bi = i / block_size;
        for j in 0..n {
            if i == j { continue; }
            let bj = j / block_size;
            let p = if bi == bj { within_prob } else { between_prob };
            if rng.gen::<f32>() < p {
                rows[i].push((j, rng.gen::<f32>() * 0.5 + 0.5));
            }
        }
    }

    let n_syn: usize = rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = vec![0i32; n_syn];
    let mut values = vec![0.0f32; n_syn];
    let mut pos = 0;
    for i in 0..n {
        row_ptr[i] = pos as i32;
        for &(j, w) in &rows[i] {
            col_indices[pos] = j as i32;
            values[pos] = w;
            pos += 1;
        }
    }
    row_ptr[n] = pos as i32;

    let mut nt_types = vec![0.0f32; n];
    for i in 0..n {
        nt_types[i] = if rng.gen::<f32>() < ei_ratio { 1.0 } else { -1.0 };
    }

    (row_ptr, col_indices, values, nt_types)
}

// ─── State generation ───────────────────────────────────────────────────────

fn generate_state(row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    nt_types: &[f32], n: usize, n_blocks: usize) -> Vec<f32>
{
    let mut rng = rand::thread_rng();
    let block_size = n / n_blocks.max(1);

    let block_baselines: Vec<f32> = (0..n_blocks)
        .map(|b| match b {
            0 => 0.5 + rng.gen::<f32>() * 0.3,
            _ if b == n_blocks - 1 => 0.3 + rng.gen::<f32>() * 0.2,
            _ => rng.gen::<f32>() * 0.3,
        })
        .collect();

    let mut rates = vec![0.0f32; n];
    for i in 0..n {
        let bi = (i / block_size).min(n_blocks - 1);
        rates[i] = block_baselines[bi] + rng.gen::<f32>() * 0.1;
    }

    for _ in 0..5 {
        let mut new_rates = rates.clone();
        for i in 0..n {
            let start = row_ptr[i].max(0) as usize;
            let end = row_ptr[i + 1].max(0) as usize;
            let mut drive = 0.0f32;
            for j in start..end {
                if j < col_indices.len() && j < values.len() {
                    let pre = col_indices[j].max(0) as usize;
                    if pre < n {
                        drive += values[j] * rates[pre];
                    }
                }
            }
            let sig = 1.0 / (1.0 + (-drive * 0.1).exp());
            new_rates[i] = new_rates[i] * 0.7 + sig * nt_types[i].abs() * 0.3;
        }
        rates = new_rates;
    }

    let max_r = rates.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    for r in rates.iter_mut() { *r /= max_r; }
    rates
}

// ─── Selectivity ────────────────────────────────────────────────────────────

fn compute_selectivity(gate_weights: &[f32], n: usize, ws: usize, r: &[f32]) -> f32 {
    let mut gates = vec![0.0f32; ws];
    for i in 0..ws {
        let row_start = i * n;
        gates[i] = (0..n)
            .map(|j| gate_weights[row_start + j] * r[j])
            .sum();
    }

    let max_val = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for g in gates.iter_mut() {
        *g = (*g - max_val).exp();
        sum += *g;
    }
    let inv_sum = 1.0 / sum;
    for g in gates.iter_mut() {
        *g *= inv_sum;
    }

    let eps = 1e-12;
    let mut entropy = 0.0f32;
    for &g in gates.iter() {
        if g > eps { entropy -= g * g.log2(); }
    }

    let max_entropy = (ws as f32).log2();
    if max_entropy > 0.0 { 1.0 - (entropy / max_entropy) } else { 0.0 }
}

fn mean_selectivity(r: &[f32], n: usize, ws: usize, n_seeds: usize) -> (f32, f32) {
    let mut vals = Vec::with_capacity(n_seeds);
    for _ in 0..n_seeds {
        let gnw = karmac::gnw::GlobalWorkspace::new(n, ws);
        let gws = gnw.gate_weights().to_vec();
        drop(gnw);
        let s = compute_selectivity(&gws, n, ws, r);
        vals.push(s);
    }
    let mean = vals.iter().sum::<f32>() / n_seeds as f32;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n_seeds as f32;
    (mean, var.sqrt())
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    println!("═══ Critical E/I Experiment ═══");
    println!();
    println!("Гипотеза: workspace selectivity максимальна при E/I ≈ 0.8,");
    println!("а variance пиковая — signature фазового перехода.");
    println!();
    println!("Параметры:");
    println!("  N = {N_NEURONS}, degree = {TARGET_DEGREE}, blocks = {N_BLOCKS}");
    println!("  Workspace = {WORKSPACE_SIZE}, GNW seeds = {N_GNW_SEEDS}");
    println!("  Connectome seeds = {N_CONN_SEEDS}");
    println!();

    // Generate E/I ratios from 0.0 to 1.0 in steps of 0.05
    let ei_values: Vec<f32> = (0..=20).map(|i| i as f32 * 0.05).collect();

    println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "E/I", "S_mean", "S_std", "S_rand", "Δ", "Δ_mean", "Δ_std");
    println!("{}", "─".repeat(72));

    let mut results: Vec<(f32, f32, f32, f32, f32, f32, f32)> = Vec::new();

    for &ei in &ei_values {
        let mut s_all = Vec::with_capacity(N_CONN_SEEDS);
        let mut delta_all = Vec::with_capacity(N_CONN_SEEDS);

        for _conn_seed in 0..N_CONN_SEEDS {
            let (rp, ci, v, nt) = generate_connectome(
                N_NEURONS, N_BLOCKS, ei, TARGET_DEGREE, WITHIN_FRAC);
            let r_state = generate_state(&rp, &ci, &v, &nt, N_NEURONS, N_BLOCKS);

            let mut rng = rand::thread_rng();
            let r_rand: Vec<f32> = (0..N_NEURONS).map(|_| rng.gen::<f32>()).collect();

            let (s, _) = mean_selectivity(&r_state, N_NEURONS, WORKSPACE_SIZE, N_GNW_SEEDS);
            let (s_rand, _) = mean_selectivity(&r_rand, N_NEURONS, WORKSPACE_SIZE, N_GNW_SEEDS);

            s_all.push(s);
            delta_all.push(s - s_rand);
        }

        // Compute stats over connectome seeds
        let s_mean = s_all.iter().sum::<f32>() / N_CONN_SEEDS as f32;
        let s_var = s_all.iter().map(|&x| (x - s_mean).powi(2)).sum::<f32>() / N_CONN_SEEDS as f32;
        let s_std = s_var.sqrt();        // Mean random selectivity (averaged)
            let mut rng = rand::thread_rng();
            let r_rand: Vec<f32> = (0..N_NEURONS).map(|_| rng.gen::<f32>()).collect();
        let (s_rand_ref, _) = mean_selectivity(&r_rand, N_NEURONS, WORKSPACE_SIZE, N_GNW_SEEDS);

        let delta_mean = delta_all.iter().sum::<f32>() / N_CONN_SEEDS as f32;
        let delta_var = delta_all.iter().map(|&x| (x - delta_mean).powi(2)).sum::<f32>() / N_CONN_SEEDS as f32;
        let delta_std = delta_var.sqrt();
        let delta_std_pct = if delta_mean.abs() > 1e-10 { delta_std / delta_mean.abs() * 100.0 } else { 0.0 };

        results.push((ei, s_mean, s_std, s_rand_ref, delta_mean, delta_std, delta_std_pct));

        let marker = if s_mean > 0.14 { "⬆" } else if s_mean > 0.10 { "—" } else { "⬇" };
        println!("{marker} {:.2}   {:.4}   {:.4}   {:.4}   {:.4}   {:.4}   {:.1}%",
                 ei, s_mean, s_std, s_rand_ref, delta_mean, delta_std, delta_std_pct);
    }

    // ─── Analysis ──────────────────────────────────────────

    println!();
    println!("═══ Analysis ═══");
    println!();

    // Find max selectivity
    let max_s = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("Peak selectivity:  E/I = {:.2}, S = {:.4} ± {:.4}",
             max_s.0, max_s.1, max_s.2);

    // Find max variance (std) — signature of criticality
    let max_var = results.iter().max_by(|a, b| a.5.partial_cmp(&b.5).unwrap()).unwrap();
    println!("Peak Δ variance:   E/I = {:.2}, Δ_std = {:.4} ({:.1}%)",
             max_var.0, max_var.5, max_var.6);

    // Find min selectivity
    let min_s = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("Min selectivity:   E/I = {:.2}, S = {:.4} ± {:.4}",
             min_s.0, min_s.1, min_s.2);

    // Compute dynamic range
    let dyn_range = max_s.1 - min_s.1;
    println!("Dynamic range:     ΔS = {:.4}", dyn_range);

    // Check if biological optimum (0.75-0.85) is in the top region
    println!();
    println!("Biological E/I range (0.75-0.85):");
    for r in results.iter().filter(|r| r.0 >= 0.70 && r.0 <= 0.90) {
        let is_peak = (r.1 - max_s.1).abs() < 0.01;
        let tag = if is_peak { " ← CRITICAL" } else { "" };
        println!("  E/I = {:.2}: S = {:.4} ± {:.4} (Δ_std = {:.3}){}",
                 r.0, r.1, r.2, r.5, tag);
    }

    // Phase transition detection
    println!();
    println!("Phase transition analysis:");
    println!("  (Variance peak near max selectivity → critical point)");
    let critical_ei = max_s.0;
    let var_at_critical = results.iter().find(|r| (r.0 - critical_ei).abs() < 0.01)
        .map(|r| r.5).unwrap_or(0.0);
    let var_at_min = results.iter().find(|r| (r.0 - min_s.0).abs() < 0.01)
        .map(|r| r.5).unwrap_or(0.0);

    if var_at_critical > var_at_min * 1.5 {
        println!("  ✅ Phase transition detected at E/I = {:.2}:", critical_ei);
        println!("     Variance at critical: {:.4} vs min: {:.4} (ratio: {:.1}×)",
                 var_at_critical, var_at_min, var_at_critical / var_at_min.max(1e-10));
    } else {
        println!("  ⚠  No clear phase transition: variance at critical ({:.4})", var_at_critical);
        println!("     vs min ({:.4}) — ratio {:.1}× below 1.5× threshold",
                 var_at_min, var_at_critical / var_at_min.max(1e-10));
    }

    // Signal-to-noise ratio
    println!();
    println!("Signal-to-noise (Δ / Δ_std):");
    for r in results.iter().filter(|r| r.0 >= 0.70 && r.0 <= 0.90) {
        let snr = if r.5 > 1e-10 { r.4 / r.5 } else { 0.0 };
        let tag = if snr > 2.0 { " ✅" } else if snr > 1.0 { " —" } else { " ⚠" };
        println!("  E/I = {:.2}: SNR = {:.1}{}", r.0, snr, tag);
    }

    println!();
    let elapsed = t0.elapsed();
    println!("═══ Done in {:.2}s ═══", elapsed.as_secs_f64());
    println!();
    println!("Interpretation:");
    println!("  If S peaks at E/I ≈ 0.8 → biological balance is optimal for workspace.");
    println!("  If variance peaks at same E/I → phase transition (critical point).");
    println!("  If both → strong evidence for self-organized criticality (SOC).");
    println!("  Clinical: E/I imbalances linked to autism (↑E/I), schizophrenia (↓E/I).");
}
