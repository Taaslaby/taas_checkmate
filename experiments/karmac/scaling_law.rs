/// Connectome Scaling Law — информационная ёмкость коннектома vs N.
///
/// ## Гипотеза
///
/// Информационная ёмкость коннектома (измеренная как selectivity GNW workspace)
/// следует степенному закону: S(N) = c · N^α, где α — универсальный показатель.
///
/// ## Мотивация
///
/// ML scaling laws (Kaplan et al. 2020, Chinchilla 2022) показывают, что
/// loss ~ N^(-α) для языковых моделей. НО: для биологических нейросетей
/// такой закон никто не измерял — не было инструментов.
///
/// Мы можем это сделать:
///   1. Сгенерировать коннектомы разного размера (SBM / CookieCutter)
///   2. Пропустить активность через GNW workspace
///   3. Измерить selectivity S = 1 - H(g) / log₂(ws)
///   4. Сравнить с случайным контролем (shuffled state)
///   5. Найти α
///
/// ## Контроль
///
/// Сравниваем selectivity на:
///   - connectome-driven state (S_c)
///   - random state (S_r) — контрольный baseline
///
/// ΔS = S_c - S_r чистая мера вклада структуры коннектома.
///
/// ## E/I ratio scan
///
/// Для каждого N пробуем разные excitation/inhibition ratios:
///   0.5, 0.7, 0.8, 0.9, 0.95, 1.0 (all excitatory)
///
/// Это позволяет найти оптимальный E/I для передачи информации.
///
/// Запуск: cargo run --package karmac --example scaling_law --release
///
/// Результат: α, ΔS(N), оптимальный E/I ratio.

use std::time::Instant;
use rand::Rng;

// ─── Params ─────────────────────────────────────────────────────────────────

/// Scales (neurons) to test
const SCALES: &[usize] = &[256, 512, 1024, 2048, 4096];
/// Number of GNW random seeds per scale
const N_GNW_SEEDS: usize = 8;
/// Workspace size (fixed across scales)
const WORKSPACE_SIZE: usize = 32;
/// E/I ratios to scan
const EI_RATIOS: &[f32] = &[0.5, 0.7, 0.8, 0.9, 0.95, 1.0];
/// SBM blocks (scales with N)
const N_BLOCKS: usize = 8;
/// Target average degree (synapses per neuron) — kept constant across scales
const TARGET_DEGREE: f32 = 12.0;
/// Fraction of connections that are within-block (vs between-block)
const WITHIN_FRAC: f32 = 0.75;

// ─── Metrics ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ScaleResult {
    n_neurons: usize,
    ei_ratio: f32,
    /// Mean selectivity over GNW seeds (connectome-driven)
    selectivity_mean: f32,
    /// Std selectivity over GNW seeds
    selectivity_std: f32,
    /// Mean selectivity for random state (control)
    selectivity_random: f32,
    /// Δ = connectome - random (pure structural contribution)
    selectivity_delta: f32,
    /// Mean workspace entropy
    entropy_mean: f32,
}

// ─── Connectome generation ──────────────────────────────────────────────────

/// Generate SBM connectome. Returns (row_ptr, col_indices, values, nt_types).
/// Compute SBM probabilities that keep average degree constant across scales.
fn sbm_probs(n: usize, n_blocks: usize, target_degree: f32, within_frac: f32) -> (f32, f32) {
    let block_size = n / n_blocks;
    let d_within = target_degree * within_frac;
    let d_between = target_degree * (1.0 - within_frac);
    (d_within / (block_size as f32 - 1.0), d_between / ((n - block_size) as f32))
}

/// Generate SBM connectome with constant average degree.
fn generate_connectome(n: usize, n_blocks: usize, ei_ratio: f32,
    target_degree: f32, within_frac: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>)
{
    let (within_prob, between_prob) = sbm_probs(n, n_blocks, target_degree, within_frac);
    let mut rng = rand::thread_rng();
    let block_size = n / n_blocks;

    // Single pass: per-row edge lists
    let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        let bi = i / block_size;
        for j in 0..n {
            if i == j { continue; }
            let bj = j / block_size;
            let p = if bi == bj { within_prob } else { between_prob };
            if rng.gen::<f32>() < p {
                rows[i].push((j, rng.gen::<f32>() * 0.5 + 0.5)); // weight in [0.5, 1.0]
            }
        }
    }

    // Flatten to CSR
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

    // nt_types: ei_ratio fraction = excitatory (1.0), rest = inhibitory (-1.0)
    let mut nt_types = vec![0.0f32; n];
    for i in 0..n {
        nt_types[i] = if rng.gen::<f32>() < ei_ratio { 1.0 } else { -1.0 };
    }

    (row_ptr, col_indices, values, nt_types)
}

/// Generate neural state from connectome-driven activity.
fn generate_state(row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    nt_types: &[f32], n: usize, n_blocks: usize) -> Vec<f32>
{
    let mut rng = rand::thread_rng();
    let block_size = n / n_blocks.max(1);

    // Block baselines — sensory block is more active
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

    // Iterative propagation through connectome (rate-based)
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

    // Normalize
    let max_r = rates.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    for r in rates.iter_mut() { *r /= max_r; }
    rates
}

// ─── GNW selectivity ────────────────────────────────────────────────────────

/// Compute workspace selectivity from state r through a GNW.
///
/// selectivity = 1 - H(g) / log₂(ws)
///   = 0: uniform gates (no structure)
///   = 1: single gate dominates (max structure)
fn compute_selectivity(gnw: &mut karmac::gnw::GlobalWorkspace, r: &[f32]) -> f32 {
    let n = gnw.n_neurons();
    let ws = gnw.workspace_size();
    let gate_weights = gnw.gate_weights();

    // Compute gate = W_gate @ r
    let mut gates = vec![0.0f32; ws];
    for i in 0..ws {
        let row_start = i * n;
        gates[i] = (0..n)
            .map(|j| gate_weights[row_start + j] * r[j])
            .sum();
    }

    // Softmax
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

    // Entropy and selectivity
    let eps = 1e-12;
    let mut entropy = 0.0f32;
    for &g in gates.iter() {
        if g > eps {
            entropy -= g * g.log2();
        }
    }

    let max_entropy = (ws as f32).log2();
    if max_entropy > 0.0 {
        1.0 - (entropy / max_entropy)
    } else {
        0.0
    }
}

/// Compute selectivity over N random GNW instances (using thread_rng).
fn mean_selectivity(r: &[f32], n: usize, ws: usize, n_seeds: usize) -> (f32, f32) {
    let mut selectivities = Vec::with_capacity(n_seeds);
    for _ in 0..n_seeds {
        let mut gnw = karmac::gnw::GlobalWorkspace::new(n, ws);
        let s = compute_selectivity(&mut gnw, r);
        selectivities.push(s);
    }

    let mean = selectivities.iter().sum::<f32>() / n_seeds as f32;
    let variance = selectivities.iter()
        .map(|&s| (s - mean).powi(2))
        .sum::<f32>() / n_seeds as f32;
    (mean, variance.sqrt())
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    println!("═══ Connectome Scaling Law ═══");
    println!();
    println!("Гипотеза: S(N) = c · N^α — информационная ёмкость коннектома");
    println!("следует степенному закону, как Chinchilla scaling law для LLM.");
    println!();
    println!("Параметры:");
    println!("  Scales N: {:?}", SCALES);
    println!("  Workspace: {}", WORKSPACE_SIZE);
    println!("  GNW seeds: {}", N_GNW_SEEDS);
    println!("  E/I ratios: {:?}", EI_RATIOS);
    println!("  SBM: {} blocks, target_degree={}, within_frac={}", N_BLOCKS, TARGET_DEGREE, WITHIN_FRAC);
    println!();

    // Store results for power-law fit
    let mut all_results: Vec<ScaleResult> = Vec::new();

    for &ei_ratio in EI_RATIOS {
        println!("─── E/I = {ei_ratio} ───");

        for &n in SCALES {
            // 1. Generate connectome (keeps average degree ~ TARGET_DEGREE across all scales)
            let (row_ptr, col_indices, values, nt_types) = generate_connectome(
                n, N_BLOCKS, ei_ratio, TARGET_DEGREE, WITHIN_FRAC);
            let n_syn = col_indices.len();
            let avg_deg = n_syn as f64 / n as f64;

            // 2. Generate connectome-driven state
            let r_state = generate_state(
                &row_ptr, &col_indices, &values, &nt_types, n, N_BLOCKS);

            // 3. Generate random state (control)
            let mut rng = rand::thread_rng();
            let r_random: Vec<f32> = (0..n).map(|_| rng.gen::<f32>()).collect();

            // 4. Shuffled connectome control: randomize row assignments
            //    Preserves degree distribution, destroys modular structure
            let mut col_shuffled = col_indices.clone();
            let mut rng_shuf = rand::thread_rng();
            // Fisher-Yates shuffle of col_indices within each row
            for i in 0..n {
                let start = row_ptr[i] as usize;
                let end = row_ptr[i + 1] as usize;
                let row_len = end - start;
                if row_len > 1 {
                    for a in start..end-1 {
                        let b = rng_shuf.gen_range(a..end);
                        col_shuffled.swap(a, b);
                    }
                }
            }
            // After in-row shuffle, randomly reassign targets to destroy structure
            // but preserve degree: shuffle ALL col_indices globally
            let mut rng_shuf2 = rand::thread_rng();
            for i in 0..n_syn {
                let j = rng_shuf2.gen_range(0..n_syn);
                col_shuffled.swap(i, j);
            }
            // Generate state from shuffled connectome
            let r_shuffled = generate_state(
                &row_ptr, &col_shuffled, &values, &nt_types, n, N_BLOCKS);
            let (sel_shuf_mean, _) = mean_selectivity(&r_shuffled, n, WORKSPACE_SIZE, N_GNW_SEEDS);

            // 5. Measure selectivity
            let (sel_mean, sel_std) = mean_selectivity(&r_state, n, WORKSPACE_SIZE, N_GNW_SEEDS);
            let (sel_rand_mean, _) = mean_selectivity(&r_random, n, WORKSPACE_SIZE, N_GNW_SEEDS);

            let delta = sel_mean - sel_rand_mean;
            let delta_shuf = sel_shuf_mean - sel_rand_mean;

            // 6. Compute entropy
            let max_entropy = (WORKSPACE_SIZE as f32).log2();
            let entropy = max_entropy * (1.0 - sel_mean);

            all_results.push(ScaleResult {
                n_neurons: n,
                ei_ratio,
                selectivity_mean: sel_mean,
                selectivity_std: sel_std,
                selectivity_random: sel_rand_mean,
                selectivity_delta: delta,
                entropy_mean: entropy,
            });

            println!("  N={0:5} deg={1:.1} S={2:.4}±{3:.4} S_rand={4:.4} Δ={5:.4} Δ_shuf={7:.4} H={6:.2}",
                     n, avg_deg, sel_mean, sel_std, sel_rand_mean, delta, entropy, delta_shuf);
        }
        println!();
    }

    // ─── Power law fit ──────────────────────────────────────────
    //
    // Fit: S(N) = c · N^α  →  ln(S) = ln(c) + α · ln(N)
    // Only fit the Δ curves (connectome - random) since that's the
    // pure structural contribution.

    println!("═══ Power Law Fit: S = c · N^α ═══");
    println!();

    for &ei_ratio in EI_RATIOS {
        let scale_results: Vec<&ScaleResult> = all_results.iter()
            .filter(|r| (r.ei_ratio - ei_ratio).abs() < 0.01)
            .filter(|r| r.selectivity_delta > 0.0) // skip negative Δ
            .collect();

        if scale_results.len() < 3 { continue; }

        // Collect ln(N), ln(S) for Δ values
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_xx = 0.0f32;
        let mut sum_xy = 0.0f32;
        let mut count = 0usize;

        for r in &scale_results {
            let ln_n = (r.n_neurons as f32).ln();
            // Use Δ selectivity (positive only) for fitting
            let s_val = r.selectivity_delta.max(1e-10);
            let ln_s = s_val.ln();
            sum_x += ln_n;
            sum_y += ln_s;
            sum_xx += ln_n * ln_n;
            sum_xy += ln_n * ln_s;
            count += 1;
        }

        if count > 1 {
            let n_f = count as f32;
            let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n_f;
            let c_val = intercept.exp();

            // R²
            let y_mean = sum_y / n_f;
            let ss_tot: f32 = scale_results.iter()
                .map(|r| {
                    let s_val = r.selectivity_delta.max(1e-10);
                    (s_val.ln() - y_mean).powi(2)
                })
                .sum();
            let ss_res: f32 = scale_results.iter()
                .map(|r| {
                    let s_val = r.selectivity_delta.max(1e-10);
                    let ln_n = (r.n_neurons as f32).ln();
                    let pred = slope * ln_n + intercept;
                    (s_val.ln() - pred).powi(2)
                })
                .sum();
            let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

            let marker = if r2 > 0.8 { "✅" } else { "⚠" };
            println!("{0} E/I={1}:  S = {2:.4} · N^{3:.3}  (R²={4:.3}, N={5})",
                     marker, ei_ratio, c_val, slope, r2, count);
        }
    }

    println!();
    println!("═══ Results (all data) ═══");
    println!();
    println!("{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "N", "E/I", "S_conn", "S_rand", "Δ", "α(N)", "H(g)");
    println!("{}", "─".repeat(72));

    for r in &all_results {
        // Approximate local α: log(S(N2)/S(N1)) / log(N2/N1)
        let local_alpha = "—".to_string();
        println!("{:>6} {:>6} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10.2}",
                 r.n_neurons, r.ei_ratio, r.selectivity_mean,
                 r.selectivity_random, r.selectivity_delta,
                 local_alpha, r.entropy_mean);
    }

    println!();
    let elapsed = t0.elapsed();
    println!("═══ Done in {:.2}s ═══", elapsed.as_secs_f64());
    println!();
    println!("Interpretation:");
    println!("  α > 0:  information capacity grows with N (expected)");
    println!("  α ≈ 0:  no scaling — workspace saturates immediately");
    println!("  α > 1:  super-linear — synergy, whole > sum of parts"); 
    println!("  α < 0.3: sub-linear — redundancy, diminishing returns");
    println!();
    println!("  The EI ratio that maximizes α is the optimal balance for");
    println!("  information transmission in a modular connectome.");
    println!("  This corresponds to the 'critical E/I' hypothesis.");
}
