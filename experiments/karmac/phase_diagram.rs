/// Phase Diagram: E/I × Coupling λ — joint scan for GNW ignition.
///
/// ## Гипотеза
///
/// GNW ignition может требовать ОДНОВРЕМЕННО:
///   - E/I ≈ 0.8 (биологический баланс)
///   - λ > λ_crit (достаточная сила обратной связи)
///
/// Если ignition это фазовый переход второго порядка, он должен
/// проявляться только на joint (E/I, λ) диаграмме, а не в 1D scan.
///
/// ## Метод
///
/// 1. Фиксируем N=2048, degree=12
/// 2. Сканируем E/I: 0.0, 0.1, ... 0.9, 1.0 (11 точек)
/// 3. Для каждого E/I сканируем λ: logspace от 0.01 до 10.0 (15 точек)
/// 4. На каждой точке (E/I, λ):
///    a. Генерируем connectome + state
///    b. Запускаем GNW с broadcast = λ × W_bc
///    c. Измеряем: selectivity, ignition_score, entropy_stability
/// 5. Строим phase diagram
///
/// Запуск: cargo run --package karmac --example phase_diagram --release

use std::time::Instant;
use rand::Rng;

// ─── Params ─────────────────────────────────────────────────────────────────

const N_NEURONS: usize = 2048;
const N_BLOCKS: usize = 8;
const TARGET_DEGREE: f32 = 12.0;
const WITHIN_FRAC: f32 = 0.75;
const WORKSPACE_SIZE: usize = 32;
const N_GNW_SEEDS: usize = 4;
const N_SWEEP_STEPS: usize = 20;      // GNW iterations per (E/I, λ) point
const N_EI_STEPS: usize = 11;          // 0.0 to 1.0 in 0.1 steps
const N_LAMBDA_STEPS: usize = 15;     // logspace 0.01 to 10

// ─── Output structure ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PhasePoint {
    ei: f32,
    lambda: f32,
    /// Static selectivity (before GNW recurrence)
    selectivity_static: f32,
    /// Ignition score after N_SWEEP_STEPS iterations
    ignition_score: f32,
    /// Selectivity AFTER recurrence (how it evolved)
    selectivity_final: f32,
    /// Entropy stability: std(entropy) over last 10 steps
    entropy_stability: f32,
    /// Whether gate converged (∆entropy < 0.01 over last 5 steps)
    converged: bool,
}

// ─── SBM ────────────────────────────────────────────────────────────────────

fn sbm_probs(n: usize, n_blocks: usize, target_degree: f32, within_frac: f32) -> (f32, f32) {
    let block_size = n / n_blocks;
    let d_within = target_degree * within_frac;
    let d_between = target_degree * (1.0 - within_frac);
    (d_within / (block_size as f32 - 1.0), d_between / ((n - block_size) as f32 + 1.0))
}

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
            values[pos] = w; pos += 1;
        }
    }
    row_ptr[n] = pos as i32;
    let mut nt_types = vec![0.0f32; n];
    for i in 0..n { nt_types[i] = if rng.gen::<f32>() < ei_ratio { 1.0 } else { -1.0 }; }
    (row_ptr, col_indices, values, nt_types)
}

// ─── State ──────────────────────────────────────────────────────────────────

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
        }).collect();
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
                    if pre < n { drive += values[j] * rates[pre]; }
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

// ─── GNW operations ─────────────────────────────────────────────────────────

/// Compute gate softmax + entropy + selectivity from raw dot products.
fn gates_from_weights(gate_weights: &[f32], n: usize, ws: usize, r: &[f32])
    -> (Vec<f32>, f32)
{
    let mut gates = vec![0.0f32; ws];
    for i in 0..ws {
        let row_start = i * n;
        gates[i] = (0..n).map(|j| gate_weights[row_start + j] * r[j]).sum();
    }
    let max_val = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for g in gates.iter_mut() { *g = (*g - max_val).exp(); sum += *g; }
    let inv_sum = 1.0 / sum;
    for g in gates.iter_mut() { *g *= inv_sum; }

    let eps = 1e-12;
    let mut entropy = 0.0f32;
    for &g in gates.iter() { if g > eps { entropy -= g * g.log2(); } }
    let max_entropy = (ws as f32).log2();
    let selectivity = if max_entropy > 0.0 { 1.0 - entropy / max_entropy } else { 0.0 };
    (gates, selectivity)
}

/// GNW recurrence: r += broadcast(softmax(gate(r))) × λ
fn gnw_step(r: &mut [f32], gate_w: &[f32], broadcast_w: &[f32],
    n: usize, ws: usize, lambda: f32)
{
    // gate
    let (gates, _) = gates_from_weights(gate_w, n, ws, r);
    // broadcast: r += λ · broadcast_w @ gates
    for i in 0..n {
        let row_start = i * ws;
        let delta: f32 = (0..ws)
            .map(|k| broadcast_w[row_start + k] * gates[k])
            .sum::<f32>() * lambda;
        r[i] += delta;
    }
}

/// Full sweep: run N_SWEEP_STEPS GNW iterations, measure metrics.
fn sweep_point(
    gate_w: &[f32], broadcast_w: &[f32],
    r_init: &[f32], n: usize, ws: usize, lambda: f32,
) -> (f32, f32, f32, f32, bool) {
    // Static selectivity (before recurrence)
    let (_, sel_static) = gates_from_weights(gate_w, n, ws, r_init);

    // Recurrence
    let mut r = r_init.to_vec();
    let mut entropy_log = Vec::with_capacity(N_SWEEP_STEPS);

    for _ in 0..N_SWEEP_STEPS {
        gnw_step(&mut r, gate_w, broadcast_w, n, ws, lambda);
        let (_, sel) = gates_from_weights(gate_w, n, ws, &r);
        let max_entropy = (ws as f32).log2();
        let entropy = if max_entropy > 0.0 { (1.0 - sel) * max_entropy } else { 0.0 };
        entropy_log.push(entropy);
    }

    let (_, sel_final) = gates_from_weights(gate_w, n, ws, &r);

    // Ignition score: low final entropy + high selectivity = ignited
    let max_entropy = (ws as f32).log2();
    let entropy_final = if max_entropy > 0.0 { (1.0 - sel_final) * max_entropy } else { 0.0 };
    let ignition_score = if max_entropy > 0.0 {
        1.0 - (entropy_final / max_entropy).powf(0.5) // sqrt to boost low-entropy states
    } else { 0.0 };

    // Stability: std of entropy over last 10 steps
    let window = entropy_log.len().min(10);
    let recent = &entropy_log[entropy_log.len() - window..];
    let mean_e = recent.iter().sum::<f32>() / window as f32;
    let var_e = recent.iter().map(|&e| (e - mean_e).powi(2)).sum::<f32>() / window as f32;
    let entropy_stability = var_e.sqrt();

    // Convergence: entropy change < 0.01 over last 5 steps
    let converged = if entropy_log.len() >= 5 {
        let last5 = &entropy_log[entropy_log.len() - 5..];
        let max_d = (0..4).map(|i| (last5[i+1] - last5[i]).abs()).fold(0.0f32, f32::max);
        max_d < 0.01
    } else { false };

    (sel_static, sel_final, ignition_score, entropy_stability, converged)
}

// ═══ Main ══════════════════════════════════════════════════════════════════

fn main() {
    let t0 = Instant::now();
    println!("═══ Phase Diagram: E/I × Coupling λ ═══");
    println!();
    println!("Параметры:");
    println!("  N={N_NEURONS}, degree={TARGET_DEGREE}, blocks={N_BLOCKS}");
    println!("  Workspace={WORKSPACE_SIZE}, GNW seeds={N_GNW_SEEDS}");
    println!("  E/I steps={N_EI_STEPS}, λ steps={N_LAMBDA_STEPS}");
    println!("  GNW iterations/point={N_SWEEP_STEPS}");
    println!();

    // Lambda range: logspace 0.01 to 10.0
    let lambdas: Vec<f32> = (0..N_LAMBDA_STEPS)
        .map(|i| 10.0_f32.powf(-2.0 + i as f32 / (N_LAMBDA_STEPS - 1) as f32 * 3.0))
        .collect();
    // E/I range: 0.0 to 1.0 in N_EI_STEPS
    let eis: Vec<f32> = (0..N_EI_STEPS)
        .map(|i| i as f32 / (N_EI_STEPS - 1) as f32)
        .collect();

    // ─── Collect all results ───
    // Store per (ei, lambda) averaged over GNW seeds
    let mut all_results: Vec<PhasePoint> = Vec::new();
    let total = eis.len() * lambdas.len();
    let mut done = 0usize;

    for &ei in &eis {
        // Generate one connectome + state per E/I (reused across λ)
        let (rp, ci, v, nt) = generate_connectome(
            N_NEURONS, N_BLOCKS, ei, TARGET_DEGREE, WITHIN_FRAC);
        let r_state = generate_state(&rp, &ci, &v, &nt, N_NEURONS, N_BLOCKS);

        for &lambda in &lambdas {
            let mut acc_sel_static = 0.0f32;
            let mut acc_sel_final = 0.0f32;
            let mut acc_ignition = 0.0f32;
            let mut acc_stability = 0.0f32;
            let mut acc_converged = 0usize;

            for _ in 0..N_GNW_SEEDS {
                let gnw = karmac::gnw::GlobalWorkspace::new(N_NEURONS, WORKSPACE_SIZE);
                let gw = gnw.gate_weights().to_vec();
                let bw = gnw.broadcast_weights().to_vec();
                drop(gnw);

                let (sel_s, sel_f, ign, stab, conv) = sweep_point(
                    &gw, &bw, &r_state, N_NEURONS, WORKSPACE_SIZE, lambda);

                acc_sel_static += sel_s;
                acc_sel_final += sel_f;
                acc_ignition += ign;
                acc_stability += stab;
                if conv { acc_converged += 1; }
            }

            let n = N_GNW_SEEDS as f32;
            all_results.push(PhasePoint {
                ei, lambda,
                selectivity_static: acc_sel_static / n,
                ignition_score: acc_ignition / n,
                selectivity_final: acc_sel_final / n,
                entropy_stability: acc_stability / n,
                converged: acc_converged as f32 / n > 0.5,
            });

            done += 1;
            if done % 20 == 0 || done == total {
                print!("\r  {}/{} ({:.0}%)", done, total, done as f32 / total as f32 * 100.0);
                use std::io::{Write, stdout};
                stdout().flush().unwrap();
            }
        }
    }
    println!();

    let elapsed = t0.elapsed();
    println!("\n  Done in {:.1}s", elapsed.as_secs_f64());

    // ─── Phase diagram table ───
    println!();
    println!("═══ Phase Diagram ═══");
    println!();

    // Header
    print!("{:>5}", "E/I");
    for &l in &lambdas { print!(" {:>7.2}", l); }
    println!();

    // Data rows: ignition_score
    println!("─── ignition_score ───");
    for &ei in &eis {
        print!("{:>5.1}", ei);
        for &l in &lambdas {
            if let Some(pt) = all_results.iter().find(|r| (r.ei - ei).abs() < 0.01 && (r.lambda - l).abs() / l.max(0.01) < 0.1) {
                let ign = pt.ignition_score;
                let marker = if ign > 0.6 { "🔥" } else if ign > 0.4 { "⬆" } else if ign > 0.2 { "—" } else if ign > 0.1 { "⬇" } else { "·" };
                print!(" {:>7}", marker);
            } else { print!(" {:>7}", "?"); }
        }
        println!();
    }

    // Data rows: selectivity_final
    println!();
    println!("─── selectivity_final ───");
    for &ei in &eis {
        print!("{:>5.1}", ei);
        for &l in &lambdas {
            if let Some(pt) = all_results.iter().find(|r| (r.ei - ei).abs() < 0.01 && (r.lambda - l).abs() / l.max(0.01) < 0.1) {
                let s = pt.selectivity_final;
                let marker = if s > 0.20 { "⬆" } else if s > 0.15 { "—" } else if s > 0.10 { "⬇" } else { "··" };
                print!(" {:>7}", marker);
            } else { print!(" {:>7}", "?"); }
        }
        println!();
    }

    // Data rows: converged
    println!();
    println!("─── converged ───");
    for &ei in &eis {
        print!("{:>5.1}", ei);
        for &l in &lambdas {
            if let Some(pt) = all_results.iter().find(|r| (r.ei - ei).abs() < 0.01 && (r.lambda - l).abs() / l.max(0.01) < 0.1) {
                let marker = if pt.converged { "✓" } else { "·" };
                print!(" {:>7}", marker);
            } else { print!(" {:>7}", "?"); }
        }
        println!();
    }

    // ─── Analysis ───
    println!();
    println!("═══ Analysis ═══");
    println!();

    // Find max ignition
    let max_ign = all_results.iter()
        .max_by(|a, b| a.ignition_score.partial_cmp(&b.ignition_score).unwrap())
        .unwrap();
    println!("Max ignition:     E/I={:.1}, λ={:.3}, score={:.3}",
             max_ign.ei, max_ign.lambda, max_ign.ignition_score);

    // Find region with both high ignition AND convergence
    let stable_ignition: Vec<&PhasePoint> = all_results.iter()
        .filter(|r| r.ignition_score > 0.3 && r.converged)
        .collect();
    println!("Stable ignition (score>0.3 + converged): {} points", stable_ignition.len());
    for pt in stable_ignition.iter().take(5) {
        println!("  E/I={:.1}, λ={:.3}, ign={:.3}, sel={:.3}",
                 pt.ei, pt.lambda, pt.ignition_score, pt.selectivity_final);
    }

    // Find: does ignition require BOTH E/I ≈ 0.8 AND λ > λ_crit?
    let high_ei: Vec<&PhasePoint> = all_results.iter()
        .filter(|r| r.ei >= 0.7 && r.lambda > 1.0)
        .collect();
    let low_ei: Vec<&PhasePoint> = all_results.iter()
        .filter(|r| r.ei < 0.7 && r.lambda > 1.0)
        .collect();

    let mean_high = high_ei.iter().map(|r| r.ignition_score).sum::<f32>() / high_ei.len().max(1) as f32;
    let mean_low = low_ei.iter().map(|r| r.ignition_score).sum::<f32>() / low_ei.len().max(1) as f32;

    println!();
    println!("Joint phase analysis:");
    println!("  Mean ignition at E/I≥0.7, λ>1: {:.3} ({} pts)", mean_high, high_ei.len());
    println!("  Mean ignition at E/I<0.7, λ>1: {:.3} ({} pts)", mean_low, low_ei.len());

    let ratio = if mean_low > 0.0 { mean_high / mean_low } else { 0.0 };
    if ratio > 1.5 {
        println!("  ✅ JOINT PHASE TRANSITION: high E/I + high λ → {}× ignition boost", ratio);
    } else if ratio > 1.1 {
        println!("  — Weak joint effect: {}× ignition at E/I≥0.7 + λ>1", ratio);
    } else {
        println!("  ⚠  No joint phase transition: E/I and λ are independent factors.");
        println!("     Ignition depends on λ only, not on E/I.");
    }

    // Stability analysis
    println!();
    let stable_pts = all_results.iter().filter(|r| r.entropy_stability < 0.5).count();
    println!("Entropy stability < 0.5: {}/{} points ({:.0}%)",
             stable_pts, all_results.len(),
             stable_pts as f32 / all_results.len().max(1) as f32 * 100.0);

    println!();
    println!("═══ Done ({:.1}s) ═══", elapsed.as_secs_f64());
    println!();
    println!("Interpretation:");
    println!("  If ignition peaks at BOTH high E/I AND high λ → joint phase transition.");
    println!("  If ignition depends on λ only → no joint transition, λ is the sole driver.");
    println!("  If never ignites → GNW with random weights cannot ignite at any (E/I, λ).");
}
