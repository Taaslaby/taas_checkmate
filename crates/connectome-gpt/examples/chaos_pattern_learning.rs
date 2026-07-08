//! # Chaos vs Pattern — dopamine-STDP обучение на Kuramoto коннектоме
//!
//! ## Теория
//!
//! ```text
//! δ(t) = R(t) − α · |H(t) − H_target|
//! Δw_ij = η · δ · e_ij
//! ```
//!
//! - **R(t)** — Kuramoto order parameter (фазовая когерентность) — награда за паттерн
//! - **H(t)** — спайковая энтропия (хаос-штраф), H_penalty = |H − H_target|
//! - **δ(t)** — dopamine reward signal для 3-factor STDP (Izhikevich 2007)
//! - **e_ij** — STDP eligibility trace (pre-post spike timing, а не фазовая разница!)
//!
//! ## Как это работает
//!
//! 1. Kuramoto осцилляторы на CSR коннектоме
//! 2. Каждый осциллятор генерирует "спайк" при пересечении фазой порога π·0.8
//! 3. Для каждого синапса i→j: pre_spike = spike[i], post_spike = spike[j]
//! 4. `DopamineSTDP::step()` обновляет eligibility через STDP:
//!    e += A_ltp · tr_pre · post_spike − A_ltd · tr_post · pre_spike
//!    e *= (1 − dt/τ_e)
//! 5. В конце эпизоды: δ = R − α · |H − H_target|
//! 6. `apply_dopamine(weights, δ, η)` — Δw = η · δ · e
//!
//! ## Ожидаемый результат
//!
//! - STDP eligibility накапливается от реальных spike пар → сильный сигнал
//! - Синапсы, где pre опережает post → LTP → вес растёт → синхронизация
//! - Синапсы, где post опережает pre → LTD → вес падает → шум подавляется
//! - R растёт, H → H_target

use std::time::Instant;

use taas_core::kuramoto::{kuramoto_step_sparse, phase_coherence};
use taas_core::neuron::DopamineSTDP;
use taas_kernels::{spike_entropy};

// ═══════════════════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════════════════

const N_OSC: usize = 50;           // осцилляторов
const N_STEPS: usize = 200;        // шагов Kuramoto за эпизоду
const DT: f32 = 0.01;              // временной шаг (s)
const N_EPISODES: usize = 300;     // эпизод обучения
const ALPHA: f32 = 0.3;            // вес штрафа энтропии
const H_TARGET: f32 = 2.5;         // целевая энтропия
const LR: f32 = 0.2;               // learning rate для dopamine-STDP
const W_MIN: f32 = 0.0;            // мин вес синапса
const W_MAX: f32 = 3.0;            // макс вес синапса
const OMEGA_BASE: f32 = 5.0;       // базовая частота (rad/s)
const OMEGA_SPREAD: f32 = 1.5;     // разброс частот
const SPIKE_THRESHOLD: f32 = 2.5;  // порог спайка (фаза, радианы)

// ═══════════════════════════════════════════════════════════
//  LCG RNG (zero-dep)
// ═══════════════════════════════════════════════════════════

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ═══════════════════════════════════════════════════════════
//  Синтетический коннектом: ring + random shortcuts
// ═══════════════════════════════════════════════════════════

fn make_connectome(n: usize, shortcut_prob: f32, seed: u32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let mut edge_list: Vec<(usize, usize, f32)> = Vec::new();

    // Ring: каждый i → i+1 (с замыканием)
    for i in 0..n {
        let j = (i + 1) % n;
        let w = rng.next_f32_range(0.5, 1.5);
        edge_list.push((i, j, w));
    }

    // Random shortcuts — сильные, чтобы создать конкуренцию путей
    for i in 0..n {
        for _ in 0..5 {
            if rng.next_f32() < shortcut_prob {
                let j = (rng.next_f32() * n as f32) as usize % n;
                if i != j && !edge_list.iter().any(|(a, b, _)| *a == i && *b == j) {
                    let w = rng.next_f32_range(0.2, 0.8);
                    edge_list.push((i, j, w));
                }
            }
        }
    }

    edge_list.sort_by_key(|(i, _, _)| *i);

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    let mut current = 0usize;
    for (i, j, w) in &edge_list {
        while current <= *i {
            row_ptr[current] = col_indices.len() as i32;
            current += 1;
        }
        col_indices.push(*j as i32);
        values.push(*w);
    }
    while current <= n {
        row_ptr[current] = col_indices.len() as i32;
        current += 1;
    }

    (row_ptr, col_indices, values)
}

// ═══════════════════════════════════════════════════════════
//  Spike detection для Kuramoto
// ═══════════════════════════════════════════════════════════

fn detect_spikes(phases: &[f32], phases_prev: &[f32], spikes: &mut [i32]) {
    for i in 0..phases.len().min(spikes.len()) {
        spikes[i] = if phases[i] > SPIKE_THRESHOLD && phases_prev[i] <= SPIKE_THRESHOLD {
            1
        } else {
            0
        };
    }
}

fn count_spikes(spike_buffer: &[i32], n_osc: usize, n_steps: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; n_osc];
    for step in 0..n_steps {
        let base = step * n_osc;
        for i in 0..n_osc {
            if spike_buffer[base + i] != 0 {
                counts[i] += 1.0;
            }
        }
    }
    counts
}

/// Заполняет per-synapse pre/post spike массивы для STDP.
/// Для каждого синапса idx (i→j): pre_spikes[idx] = spikes[i], post_spikes[idx] = spikes[j].
fn build_synapse_spikes(
    syn_pre_spikes: &mut [i32],
    syn_post_spikes: &mut [i32],
    spikes: &[i32],
    row_ptr: &[i32],
    col_indices: &[i32],
    n_osc: usize,
) {
    let max_syn = col_indices.len().min(syn_pre_spikes.len()).min(syn_post_spikes.len());
    for i in 0..n_osc {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let pre_val = if i < spikes.len() { spikes[i] } else { 0 };
        for idx in start..end {
            let j = col_indices[idx] as usize;
            syn_pre_spikes[idx] = pre_val;
            syn_post_spikes[idx] = if j < spikes.len() { spikes[j] } else { 0 };
        }
    }
}

/// Вычисляет среднюю абсолютную связность (для оценки изменений).
fn mean_abs_weight(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 0 { return 0.0; }
    values.iter().map(|v| v.abs()).sum::<f32>() / n as f32
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════

fn main() {
    println!("{}", "═".repeat(64));
    println!("  🧠 Chaos vs Pattern — dopamine-STDP на Kuramoto коннектоме");
    println!("  δ = R − α·|H − H_target|,  Δw = η·δ·e  (Izhikevich 2007)");
    println!("{}", "═".repeat(64));

    let t_start = Instant::now();

    // ── 1. Генерация коннектома ──
    println!("\n─── Connectome ───");
    let (row_ptr, col_indices, mut values) = make_connectome(N_OSC, 0.4, 42);
    let n_syn = values.len();
    let avg_degree = n_syn as f32 / N_OSC as f32;
    println!("  {} oscillators, {} synapses ({:.1}/node)", N_OSC, n_syn, avg_degree);

    let init_values = values.clone();

    // ── 2. Kuramoto setup ──
    let mut phases: Vec<f32> = vec![0.0f32; N_OSC];
    let mut phases_prev = vec![0.0f32; N_OSC];
    let omegas: Vec<f32> = (0..N_OSC)
        .map(|i| OMEGA_BASE + (i as f32 / N_OSC as f32 - 0.5) * 2.0 * OMEGA_SPREAD)
        .collect();

    let mut sin_buf = vec![0.0f32; N_OSC];
    let mut cos_buf = vec![0.0f32; N_OSC];
    let mut coupling_buf = vec![0.0f32; N_OSC];
    let amps = vec![1.0f32; N_OSC];
    let mut spikes = vec![0i32; N_OSC];

    // ── 3. DopamineSTDP ──
    let stdp_dt = DT * 1000.0; // Kuramoto шаг в ms
    let mut stdp = DopamineSTDP::new(n_syn, stdp_dt);
    stdp.tau_pre = 20.0;    // ms
    stdp.tau_post = 20.0;   // ms
    stdp.tau_e = 500.0;     // ms — долгая eligibility
    stdp.a_ltp = 0.15;      // LTP amplitude
    stdp.a_ltd = 0.12;      // LTD amplitude
    stdp.w_min = W_MIN;
    stdp.w_max = W_MAX;

    let mut syn_pre_spikes = vec![0i32; n_syn];
    let mut syn_post_spikes = vec![0i32; n_syn];

    // ── 4. Tracking ──
    let mut r_history = Vec::with_capacity(N_EPISODES);
    let mut h_history = Vec::with_capacity(N_EPISODES);
    let mut delta_history = Vec::with_capacity(N_EPISODES);
    let mut r_baseline_history = Vec::with_capacity(N_EPISODES);

    // ── 5. Learning loop ──
    println!("\n─── Learning ───");
    println!("  δ = R − {:.1}·|H − {:.1}|,  η = {},  τ_e = {:.0}ms",
             ALPHA, H_TARGET, LR, stdp.tau_e);
    println!("  {} episodes, {} steps/episode, {} oscillators",
             N_EPISODES, N_STEPS, N_OSC);
    println!();
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}  {:>8}",
             "Ep", "R", "H", "δ", "mean|w|", "Δ|w|", "Time");
    println!("  {}", "─".repeat(64));

    for episode in 0..N_EPISODES {
        let t_ep = Instant::now();

        // Инициализация фаз
        let seed = episode as u32 * 7919 + 42;
        let mut rng_ep = Lcg::new(seed);
        for p in phases.iter_mut() {
            *p = rng_ep.next_f32() * 2.0 * std::f32::consts::PI;
        }
        phases_prev.copy_from_slice(&phases);
        stdp.clear_eligibility();

        let mut spike_buffer = vec![0i32; N_STEPS * N_OSC];

        // ── 5a. Kuramoto dynamics ──
        for step in 0..N_STEPS {
            kuramoto_step_sparse(
                &mut phases, &omegas, &row_ptr, &col_indices, &values,
                &mut sin_buf, &mut cos_buf, &mut coupling_buf, DT,
            );
            detect_spikes(&phases, &phases_prev, &mut spikes);
            phases_prev.copy_from_slice(&phases);

            // Per-synapse spike flags
            build_synapse_spikes(
                &mut syn_pre_spikes, &mut syn_post_spikes,
                &spikes, &row_ptr, &col_indices, N_OSC,
            );

            // STDP eligibility update
            stdp.step(&syn_pre_spikes, &syn_post_spikes);

            // Record spikes
            let base = step * N_OSC;
            for i in 0..N_OSC {
                spike_buffer[base + i] = spikes[i];
            }
        }

        // ── 5b. Phase coherence R ──
        let mut coh_out = vec![0.0f32];
        phase_coherence(&mut coh_out, &sin_buf, &cos_buf, &amps);
        let r = coh_out[0];

        // ── 5c. Spike entropy H ──
        let spike_counts = count_spikes(&spike_buffer, N_OSC, N_STEPS);
        let (h, h_penalty) = spike_entropy(&spike_counts, N_STEPS as f32, H_TARGET);

        // ── 5d. Reward signal ──
        let delta = r - ALPHA * h_penalty;

        // ── 5e. Apply dopamine weight update ──
        stdp.apply_dopamine(&mut values, delta, LR);

        // ── 5f. Baseline ──
        {
            let seed_bl = episode as u32 * 7919 + 999;
            let mut rng_bl = Lcg::new(seed_bl);
            let mut phases_bl: Vec<f32> = (0..N_OSC)
                .map(|_| rng_bl.next_f32() * 2.0 * std::f32::consts::PI)
                .collect();
            let mut sin_bl = vec![0.0f32; N_OSC];
            let mut cos_bl = vec![0.0f32; N_OSC];
            let mut coupling_bl = vec![0.0f32; N_OSC];
            let mut prev_bl = phases_bl.clone();
            let mut spikes_bl = vec![0i32; N_OSC];

            for _ in 0..N_STEPS {
                kuramoto_step_sparse(
                    &mut phases_bl, &omegas, &row_ptr, &col_indices, &init_values,
                    &mut sin_bl, &mut cos_bl, &mut coupling_bl, DT,
                );
                detect_spikes(&phases_bl, &prev_bl, &mut spikes_bl);
                prev_bl.copy_from_slice(&phases_bl);
            }
            let mut coh_bl = vec![0.0f32];
            phase_coherence(&mut coh_bl, &sin_bl, &cos_bl, &amps);
            r_baseline_history.push(coh_bl[0]);
        }

        // ── Track ──
        r_history.push(r);
        h_history.push(h);
        delta_history.push(delta);
        let mean_w = mean_abs_weight(&values);
        let delta_w = mean_w - mean_abs_weight(&init_values);

        if episode % 20 == 0 || episode == N_EPISODES - 1 {
            println!("  {:>5}  {:>8.4}  {:>8.4}  {:>8.4}  {:>10.4}  {:>+10.4}  {:>8.0}ms",
                     episode, r, h, delta, mean_w, delta_w, t_ep.elapsed().as_millis());
        }
    }

    // ── 6. Results ──
    println!("\n{}", "═".repeat(64));
    println!("  📊 Результаты");
    println!("{}", "═".repeat(64));

    let r_first = r_history.iter().take(10).sum::<f32>() / 10.0;
    let r_last = r_history.iter().rev().take(10).sum::<f32>() / 10.0;
    let r_base = r_baseline_history.iter().rev().take(10).sum::<f32>() / 10.0;
    let r_base_first = r_baseline_history.iter().take(10).sum::<f32>() / 10.0;

    let h_first = h_history.iter().take(10).sum::<f32>() / 10.0;
    let h_last = h_history.iter().rev().take(10).sum::<f32>() / 10.0;

    let delta_first = delta_history.iter().take(10).sum::<f32>() / 10.0;
    let delta_last = delta_history.iter().rev().take(10).sum::<f32>() / 10.0;

    println!("\n  ┌─────────────────────────────────────────────────────────────────┐");
    println!("  │  Метрика          │  Начало     │  Конец      │  Δ          │  Baseline │");
    println!("  ├─────────────────────────────────────────────────────────────────┤");
    println!("  │  R (когерентность) │  {:.4}    │  {:.4}    │  {:+.4}    │  {:.4}   │",
             r_first, r_last, r_last - r_first, r_base);
    println!("  │  H (энтропия)     │  {:.4}    │  {:.4}    │  {:+.4}    │  —        │",
             h_first, h_last, h_last - h_first);
    println!("  │  δ (reward)       │  {:+.4}    │  {:+.4}    │  {:+.4}    │  —        │",
             delta_first, delta_last, delta_last - delta_first);
    println!("  │  |H − H_target|   │  {:.4}    │  {:.4}    │  {:+.4}    │  —        │",
             (h_first - H_TARGET).abs(), (h_last - H_TARGET).abs(),
             (h_last - H_TARGET).abs() - (h_first - H_TARGET).abs());
    println!("  └─────────────────────────────────────────────────────────────────┘");

    // R timeline
    println!("\n  ─── R(timeline) ───");
    let step = (N_EPISODES / 12).max(1);
    for i in (0..N_EPISODES).step_by(step) {
        let bar_len = (r_history[i] * 30.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  [{:>3}] R={:.3} {}  (baseline: {:.3})",
                 i, r_history[i], bar, r_baseline_history[i]);
    }

    // H timeline
    println!("\n  ─── H(timeline) ───");
    for i in (0..N_EPISODES).step_by(step) {
        let h_val = h_history[i];
        let bar_len = (h_val / 5.0 * 25.0).min(25.0) as usize;
        let bar = "░".repeat(bar_len);
        let ok = if (h_val - H_TARGET).abs() < 0.3 { " ✓" } else { "" };
        println!("  [{:>3}] H={:.2} {}  target={:.1}{}",
                 i, h_val, bar, H_TARGET, ok);
    }

    // ── Conclusion ──
    println!("\n{}", "═".repeat(64));
    let r_up = r_last > r_first + 0.05;
    let r_vs_base = (r_last - r_first) > (r_base - r_base_first) + 0.03;
    let h_ok = (h_last - H_TARGET).abs() < 0.5;

    if r_up && h_ok {
        println!("  ✅ УСПЕХ!");
        println!("  Сеть научилась синхронизации при контролируемой энтропии!");
        println!("  R вырос на {:.3} (baseline: {:+.3})", r_last - r_first, r_base - r_base_first);
        println!("  H сошлась к H_target = {:.1} (|Δ| = {:.3})", H_TARGET, (h_last - H_TARGET).abs());
    } else if r_vs_base && h_ok {
        println!("  ✅ ЧАСТИЧНЫЙ УСПЕХ:");
        println!("  H сошлась к цели, R лучше baseline, но абсолютный рост мал.");
    } else if r_up {
        println!("  ⚠️  R растёт, но H далека от H_target = {:.1}", H_TARGET);
        println!("  Попробуй уменьшить ALPHA или увеличить N_EPISODES");
    } else {
        println!("  ⚠️  Обучение идёт, но эффект пока слабый.");
        println!("  R: {:+.4} (baseline: {:+.4})", r_last - r_first, r_base - r_base_first);
        println!("  H_target: {:.1}, |H - target|: {:.3}", H_TARGET, (h_last - H_TARGET).abs());
    }

    println!("\n  Параметры: α={:.1}, η={}, τ_e={:.0}ms, эпизод={}",
             ALPHA, LR, stdp.tau_e, N_EPISODES);
    println!("  Время: {:.2}s", t_start.elapsed().as_secs_f64());
    println!("{}", "═".repeat(64));
}
