/// GNW Ignition Experiment — поиск фазового перехода в Global Workspace.
///
/// Проверяет гипотезу Деана (Dehaene 1998, 2011): при критической силе
/// обратной связи workspace "загорается" — переходит от локальной активности
/// к глобальному вещанию (global ignition).
///
/// ## Эксперимент
///
/// 1. Загружаем FlyWire connectome (или генерируем modular synthetic)
/// 2. Создаём GNW с n_neurons = n, workspace_size = 64
/// 3. Сканируем coupling multiplier λ от 0 до 10
/// 4. На каждом λ измеряем:
///    - workspace entropy H = -Σ p·log₂(p)
///    - max softmax probability
///    - broadcast strength = mean(abs(delta))
/// 5. Ищем критический λ, где workspace_ignition > 0.5
///
/// ## Теория
///
/// GNW ignition = phase transition:
///   λ < λ_crit: gates равномерны (нейроны работают локально)
///   λ ≥ λ_crit: один gate доминирует (global broadcast)
///
/// Запуск: cargo run --package karmac --example gnw_ignition --release
///
/// Без fafb.taas: сгенерирует synthetic modular connectome.
/// С fafb.taas: загрузит реальный мозг мухи.

use std::time::Instant;
use rand::Rng;

// ─── Параметры эксперимента ─────────────────────────────────────────────────
const WORKSPACE_SIZE: usize = 64;
const N_SCAN_STEPS: usize = 10;             // шагов GNW итерации на λ
const N_LAMBDA_STEPS: usize = 40;           // сколько λ пробуем

// Для synthetic connectome
const SYNTH_N_NEURONS: usize = 2048;
const SYNTH_N_BLOCKS: usize = 8;
const SYNTH_WITHIN_PROB: f32 = 0.15;        // внутри блока
const SYNTH_BETWEEN_PROB: f32 = 0.01;       // между блоками

// ─── Metrics ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct IgnitionMetrics {
    lambda: f32,
    /// Entropy H = -Σ p_i · log₂(p_i), max = log₂(workspace_size)
    workspace_entropy: f32,
    /// Max softmax probability (measure of WTA sharpness)
    max_gate: f32,
    /// Mean absolute broadcast delta across all neurons
    broadcast_strength: f32,
    /// Fraction of workspace units with gate > 0.1
    active_units: f32,
    /// Ignition score: 0 = no ignition, 1 = fully ignited
    /// Computed as: 1 - entropy_normalized + max_gate * 0.5
    ignition_score: f32,
}

// ─── GNW Ignition scan ─────────────────────────────────────────────────────

/// Запустить GNW на state `r`, вычислить метрики.
fn compute_ignition_metrics(gnw: &mut karmac::gnw::GlobalWorkspace, r: &mut [f32]) -> IgnitionMetrics {
    // Сохраняем state до GNW
    let before = r.to_vec();

    // GNW forward: r += broadcast(softmax(gate(r)))
    gnw.forward(r);

    // Compute delta
    let broadcast_strength: f32 = r.iter()
        .zip(before.iter())
        .map(|(new, old)| (new - old).abs())
        .sum::<f32>() / r.len() as f32;

    // Compute gate activities (re-compute gate and softmax)
    let n = gnw.n_neurons();
    let ws = gnw.workspace_size();
    let gate_weights = gnw.gate_weights();

    let mut gates = vec![0.0f32; ws];
    for i in 0..ws {
        let row_start = i * n;
        gates[i] = (0..n)
            .map(|j| gate_weights[row_start + j] * before[j])
            .sum();
    }

    // Softmax
    let max_gate_val = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for g in gates.iter_mut() {
        *g = (*g - max_gate_val).exp();
        sum += *g;
    }
    let inv_sum = 1.0 / sum;
    for g in gates.iter_mut() {
        *g *= inv_sum;
    }

    // Entropy
    let eps = 1e-12;
    let mut entropy = 0.0f32;
    let mut max_gate = 0.0f32;
    let mut active = 0usize;
    for &g in gates.iter() {
        if g > 0.1 { active += 1; }
        if g > max_gate { max_gate = g; }
        if g > eps {
            entropy -= g * g.log2();
        }
    }

    let max_entropy = (ws as f32).log2();
    let entropy_norm = if max_entropy > 0.0 { entropy / max_entropy } else { 1.0 };
    let active_frac = active as f32 / ws as f32;

    // Ignition score: 0..1, higher = more ignited
    // Комбинируем: низкая entropy + высокий max_gate + высокая broadcast_strength
    let score = (1.0 - entropy_norm) * 0.4 + max_gate * 0.4 + active_frac.min(1.0) * 0.2;

    IgnitionMetrics {
        lambda: 0.0, // will be filled by caller
        workspace_entropy: entropy,
        max_gate,
        broadcast_strength,
        active_units: active_frac,
        ignition_score: score,
    }
}

/// Сканировать coupling strength λ.
///
/// На каждом λ:
///   1. Копируем state `r_base`
///   2. Масштабируем broadcast weights: W_bc_scaled = W_bc * λ
///   3. Запускаем N_SCAN_STEPS GNW forward pass
///   4. Усредняем метрики по последним N_SCAN_STEPS/2 шагам
///   5. Восстанавливаем weights
///
/// Важно: λ умножается на broadcast weights, не gate weights.
/// Gate определяет КТО в workspace, broadcast определяет
/// НАСКОЛЬКО workspace влияет на нейроны.
fn scan_coupling(
    gnw_base: &karmac::gnw::GlobalWorkspace,
    r_base: &[f32],
    lambda_range: &[f32],
) -> Vec<IgnitionMetrics> {
    let n = gnw_base.n_neurons();
    let ws = gnw_base.workspace_size();
    let n_probe = lambda_range.len();

    // Pre-allocate buffers
    let mut results = Vec::with_capacity(n_probe);
    let mut state = r_base.to_vec();
    let broadcast_orig = gnw_base.broadcast_weights().to_vec();

    let mut gnw = karmac::gnw::GlobalWorkspace::from_weights(
        gnw_base.gate_weights().to_vec(),
        broadcast_orig.clone(),
        n, ws,
    );

    for &lambda in lambda_range {
        // Scale broadcast weights
        let broadcast_scaled: Vec<f32> = broadcast_orig.iter()
            .map(|w| w * lambda)
            .collect();
        gnw = karmac::gnw::GlobalWorkspace::from_weights(
            gnw.gate_weights().to_vec(),
            broadcast_scaled,
            n, ws,
        );

        // Reset state to baseline
        state.copy_from_slice(r_base);

        // Run N_SCAN_STEPS GNW passes
        let half = N_SCAN_STEPS / 2;
        let mut metrics_sum = IgnitionMetrics {
            lambda: 0.0, workspace_entropy: 0.0, max_gate: 0.0,
            broadcast_strength: 0.0, active_units: 0.0, ignition_score: 0.0,
        };

        for step in 0..N_SCAN_STEPS {
            let m = compute_ignition_metrics(&mut gnw, &mut state);
            if step >= half {
                metrics_sum.workspace_entropy += m.workspace_entropy;
                metrics_sum.max_gate += m.max_gate;
                metrics_sum.broadcast_strength += m.broadcast_strength;
                metrics_sum.active_units += m.active_units;
                metrics_sum.ignition_score += m.ignition_score;
            }
        }

        let n_avg = (N_SCAN_STEPS - half) as f32;
        results.push(IgnitionMetrics {
            lambda,
            workspace_entropy: metrics_sum.workspace_entropy / n_avg,
            max_gate: metrics_sum.max_gate / n_avg,
            broadcast_strength: metrics_sum.broadcast_strength / n_avg,
            active_units: metrics_sum.active_units / n_avg,
            ignition_score: metrics_sum.ignition_score / n_avg,
        });
    }

    results
}

// ─── Connectome generation ──────────────────────────────────────────────────

/// Generate a modular synthetic connectome using SBM (Stochastic Block Model).
///
/// Single-pass approach: строим per-row edge lists, потом flatten в CSR.
/// Гарантирует consistency: row_ptr[i+1] - row_ptr[i] == rows[i].len().
///
/// Returns (row_ptr, col_indices, values, nt_types).
fn generate_sbm_connectome(
    n: usize,
    n_blocks: usize,
    within_prob: f32,
    between_prob: f32,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>) {
    let block_size = n / n_blocks;
    let mut rng = rand::thread_rng();

    // Single pass: per-row edge lists (не CSR ещё)
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

    // nt_types: 1 = excitatory (80%), -1 = inhibitory (20%)
    let mut nt_types = vec![0.0f32; n];
    for i in 0..n {
        nt_types[i] = if rng.gen::<f32>() < 0.8 { 1.0 } else { -1.0 };
    }

    (row_ptr, col_indices, values, nt_types)
}

/// Load FlyWire connectome from fafb.taas.
fn load_flywire() -> Option<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, usize)> {
    let path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); p.pop(); // karmac → crates → project root
        p.push("examples/data/fafb.taas");
        p
    };

    if !path.exists() {
        println!("  ⚠  fafb.taas не найден: {path:?}");
        return None;
    }

    println!("  Загружаю FlyWire connectome...");
    let mut f = std::fs::File::open(&path).unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    println!("    нейронов: {}, синапсов: {}", conn.n_neurons, conn.col_indices.len());

    Some((
        conn.row_ptr.clone(),
        conn.col_indices.clone(),
        conn.values.clone(),
        conn.nt_types.clone(),
        conn.n_neurons,
    ))
}

// ─── State generation ───────────────────────────────────────────────────────

/// Generate initial neural state r from connectome-driven activity.
///
/// Для synthetic: случайная активность + небольшая модуляция по блокам.
/// Для FlyWire: запускаем brief симуляцию и берём rates.
fn generate_state(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    n: usize,
    n_blocks: usize,
) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let block_size = n / n_blocks.max(1);

    // Generate baseline rates: каждый блок имеет свою baseline активность
    // (имитация разных функциональных зон мозга)
    let mut rates = vec![0.0f32; n];

    // Блок 0 = "sensory" (высокая активность), блок N-1 = "motor" (средняя)
    // Остальные = random baseline
    let block_baselines: Vec<f32> = (0..n_blocks)
        .map(|b| {
            match b {
                0 => 0.5 + rng.gen::<f32>() * 0.3,
                _ if b == n_blocks - 1 => 0.3 + rng.gen::<f32>() * 0.2,
                _ => rng.gen::<f32>() * 0.3,
            }
        })
        .collect();

    for i in 0..n {
        let bi = (i / block_size).min(n_blocks - 1);
        rates[i] = block_baselines[bi] + rng.gen::<f32>() * 0.1;
    }

    // Добавляем network-driven activity: weighted sum of neighbors' activity
    // Несколько итераций распространения (как rate-based симуляция)
    let n_iter = 5;
    for _iter in 0..n_iter {
        let mut new_rates = rates.clone();
        for i in 0..n {
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1]).max(0) as usize;
            let mut drive = 0.0f32;
            for j in start..end {
                if j < col_indices.len() && j < values.len() {
                    let pre = col_indices[j].max(0) as usize;
                    if pre < n {
                        drive += values[j] * rates[pre];
                    }
                }
            }
            let nt = if i < nt_types.len() { nt_types[i] } else { 1.0 };
            // Sigmoid transfer: r = σ(drive), nt определяет sign
            let sig = 1.0 / (1.0 + (-drive * 0.1).exp());
            new_rates[i] = new_rates[i] * 0.7 + sig * nt.abs() * 0.3;
        }
        rates = new_rates;
    }

    // Normalize to [0, 1]
    let max_r = rates.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    for r in rates.iter_mut() { *r /= max_r; }

    rates
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    println!("═══ GNW Ignition Experiment ═══");
    println!();
    println!("Гипотеза: при критической силе обратной связи (λ_crit)");
    println!("workspace переходит от локальной активности к глобальному вещанию.");
    println!("Это phase transition — аналог Dehaene's global ignition.");
    println!();

    println!("─── 1. Connectome ───");
    let (row_ptr, col_indices, values, nt_types, n, n_blocks) =
        if let Some((rp, ci, v, nt, nn)) = load_flywire() {
            (rp, ci, v, nt, nn, 50usize) // FlyWire: ~50 functional blocks
        } else {
            println!("  → Генерирую synthetic modular connectome");
            println!("    {SYNTH_N_NEURONS} нейронов, {SYNTH_N_BLOCKS} блоков");
            let (rp, ci, v, nt) = generate_sbm_connectome(
                SYNTH_N_NEURONS, SYNTH_N_BLOCKS,
                SYNTH_WITHIN_PROB, SYNTH_BETWEEN_PROB,
            );
            let n_syn = ci.len();
            println!("    синапсов: {n_syn} ({:.1} syn/neur)", n_syn as f64 / SYNTH_N_NEURONS as f64);
            (rp, ci, v, nt, SYNTH_N_NEURONS, SYNTH_N_BLOCKS)
        };
    println!();

    // ─── 2. GNW ───
    println!("─── 2. Global Workspace ───");
    println!("    нейронов: {n}, workspace: {WORKSPACE_SIZE}");
    let gnw = karmac::gnw::GlobalWorkspace::new(n, WORKSPACE_SIZE);
    println!("    gate weights: {} × {} = {}",
             WORKSPACE_SIZE, n, WORKSPACE_SIZE * n);
    println!("    broadcast weights: {} × {} = {}",
             n, WORKSPACE_SIZE, n * WORKSPACE_SIZE);
    println!();

    // ─── 3. State ───
    println!("─── 3. Initial state ───");
    let r_base = generate_state(&row_ptr, &col_indices, &values, &nt_types, n, n_blocks);
    let mean_r = r_base.iter().sum::<f32>() / n as f32;
    let max_r = r_base.iter().cloned().fold(0.0f32, f32::max);
    let nonzero = r_base.iter().filter(|&&r| r > 0.01).count();
    println!("    mean rate: {mean_r:.3}, max rate: {max_r:.3}, active: {nonzero}/{n}");
    println!();

    // ─── 4. Scan coupling ───
    println!("─── 4. Сканирование coupling λ ───");
    let lambda_range: Vec<f32> = (0..N_LAMBDA_STEPS)
        .map(|i| {
            let frac = i as f32 / (N_LAMBDA_STEPS - 1).max(1) as f32;
            // log-scale от 0.01 до 10 (начинаем с маленького λ для resolution)
            10.0_f32.powf(-2.0 + frac * 3.0)
        })
        .collect();

    let results = scan_coupling(&gnw, &r_base, &lambda_range);
    let elapsed = t0.elapsed();
    println!("    λ от {:.4} до {:.1}, {N_LAMBDA_STEPS} шагов, {N_SCAN_STEPS} GNW итераций на λ",
             lambda_range[0], lambda_range[lambda_range.len()-1]);
    println!("    Сканирование завершено за {:.2}s", elapsed.as_secs_f64());
    println!();

    // ─── 5. Results ───
    println!("─── 5. Результаты ───");
    println!();
    println!("   {:>8} │ {:>12} {:>10} {:>12} {:>12} {:>14}",
             "λ", "entropy", "max_gate", "broadcast", "active_frac", "ignition_score");
    println!("   {:>8}─┼{:─^61}", "", "");

    for m in &results {
        let marker = if m.ignition_score > 0.5 { "🔥" } else { "  " };
        println!("{marker} {:.4} │ {:.4}     {:.4}   {:.4}     {:.4}      {:.4}",
                 m.lambda,
                 m.workspace_entropy,
                 m.max_gate,
                 m.broadcast_strength,
                 m.active_units,
                 m.ignition_score);
    }

    // ─── 6. Find critical λ_crit ───
    println!();
    let ignition_threshold = 0.5;
    let crit = results.iter().find(|m| m.ignition_score > ignition_threshold);

    match crit {
        Some(crit_m) => {
            let idx = results.iter().position(|m| m.ignition_score > ignition_threshold).unwrap_or(0);
            let prev_idx = idx.saturating_sub(1);
            let prev_m = &results[prev_idx];

            println!("─── 6. Critical coupling λ_crit ───");
            println!("    🔥 IGNITION ОБНАРУЖЕН при λ ≈ {:.2}", crit_m.lambda);
            println!("    До перехода (λ={:.2}): entropy={:.3}, max_gate={:.3}, score={:.3}",
                     prev_m.lambda, prev_m.workspace_entropy, prev_m.max_gate, prev_m.ignition_score);
            println!("    После перехода (λ={:.2}): entropy={:.3}, max_gate={:.3}, score={:.3}",
                     crit_m.lambda, crit_m.workspace_entropy, crit_m.max_gate, crit_m.ignition_score);
            println!();
            println!("    → Global Workspace ignition подтверждён!");
            println!("    → λ_crit ≈ {:.2} — фазовая граница", crit_m.lambda);
        }
        None => {
            let max_score = results.iter().map(|m| m.ignition_score).fold(0.0f32, f32::max);
            let max_m = results.iter().find(|m| (m.ignition_score - max_score).abs() < 1e-6).unwrap();
            println!("─── 6. Critical coupling λ_crit ───");
            println!("    ⚠  IGNITION НЕ ОБНАРУЖЕН (max_score={:.3} при λ={:.2})",
                     max_score, max_m.lambda);
            println!();
            if max_score > 0.3 {
                println!("    → Частичная модуляция workspace есть, но без phase transition.");
                println!("    → Возможно нужен больший λ или более сильная начальная активность.");
            } else {
                println!("    → Workspace остаётся равномерным при всех λ.");
                println!("    → GNW не может сфокусироваться на данном connectome.");
                println!("    → Попробуйте увеличить WORKSPACE_SIZE или изменить gate weights.");
            }
        }
    }

    println!();
    println!("═══ Experiment complete ({:.2}s) ═══", elapsed.as_secs_f64());
    println!();
    println!("Interpretation:");
    println!("  Dehaene's GNW theory: ignition = phase transition in workspace.");
    println!("  λ_crit > 0 → connectome поддерживает global broadcast.");
    println!("  λ_crit → ∞ → connectome не поддерживает workspace (no ignition).");
    println!("  λ_crit ≈ 0 → trivial ignition (workspace всегда доминирует).");
}
