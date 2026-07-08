//! # 🏓 Pong Chaos vs Pattern — VAE-генерация + Kuramoto + dopamine-STDP
//!
//! ## Что делает
//!
//! 1. **ConnectomeGPT VAE** → генерирует случайный мозг (600-800 нейронов)
//! 2. **Pong** — среда с мячом и ракеткой
//! 3. **Kuramoto осцилляторы** — динамика на коннектоме с phase coherence метрикой
//! 4. **dopamine-STDP** — 3-factor plasticity: δ = R − α·|H − H_target|
//! 5. **Chaos vs Pattern**: сравниваем coherence при паттерне (игра) vs хаосе (шум)
//!
//! ## Запуск
//!
//! ```bash
//! cargo run --release --example pong_chaos_pattern -p connectome-gpt
//! ```
//!
//! ## Архитектура
//!
//! ```text
//! VAE → ConnectomeData → Kuramoto oscillators → spike detection → MN readout → Pong action
//!                                   ↕                              ↕
//!                           DopamineSTDP eligibility       CEM readout weights
//!                                   ↕
//!                           δ = reward + β·(R_pat − α·|H − H_target|)
//!                                   ↕
//!                           apply_dopamine → Δw = η·δ·e
//! ```

use std::fs;
use std::time::Instant;

use taas_core::kuramoto::{kuramoto_step_sparse, phase_coherence};
use taas_core::neuron::DopamineSTDP;
use taas_kernels::spike_entropy;

// ═══════════════════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════════════════

const VAE_WEIGHTS_PATH: &str = "crates/connectome-gpt/models/vae_weights.bin";
const VAE_NORM_PATH: &str = "crates/connectome-gpt/models/vae_norm.json";

// Kuramoto
const DT: f32 = 0.01;            // timestep (s)
const N_KURAMOTO_STEPS: usize = 20; // steps per game tick (200ms brain time)
const SPIKE_THRESHOLD: f32 = 2.5;  // phase threshold for spike detection
const OMEGA_BASE: f32 = 3.0;
const OMEGA_SPREAD: f32 = 1.5;

// Dopamine-STDP
const LR: f32 = 0.05;             // learning rate
const ALPHA: f32 = 0.3;           // entropy penalty weight
const BETA: f32 = 0.5;            // coherence bonus weight
const H_TARGET: f32 = 3.0;        // target entropy
const W_MIN: f32 = 0.01;
const W_MAX: f32 = 3.0;

// Game
const MAX_GAME_TICKS: usize = 500; // max steps per episode
const PADDLE_SPEED: f32 = 0.03;    // paddle movement per tick
const BALL_SPEED: f32 = 0.02;      // ball base speed

// CEM readout training
const CEM_POP: usize = 80;
const CEM_ELITE: usize = 16;
const CEM_ITERS: usize = 30;
const CEM_NOISE: f32 = 0.4;

// Dopamine-STDP reservoir training
const N_EPISODES: usize = 80;

// VAE decoder
const VAE_SEED: u64 = 42;

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
    fn randn(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (6.2831853 * u2).cos()
    }
}

// ═══════════════════════════════════════════════════════════
//  Pong Game
// ═══════════════════════════════════════════════════════════

struct Pong {
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    paddle_y: f32,
    hits: usize,
    misses: usize,
    ticks: usize,
    hit_this_tick: bool,
}

impl Pong {
    fn new() -> Self {
        let mut rng = Lcg::new(12345);
        Self {
            ball_x: 0.5,
            ball_y: 0.5,
            ball_vx: BALL_SPEED * if rng.next_f32() < 0.5 { 1.0 } else { -1.0 },
            ball_vy: BALL_SPEED * (rng.next_f32() - 0.5) * 2.0,
            paddle_y: 0.5,
            hits: 0,
            misses: 0,
            ticks: 0,
            hit_this_tick: false,
        }
    }

    fn reset(&mut self) {
        let mut rng = Lcg::new(98765 + self.misses as u32 * 17);
        self.ball_x = 0.5;
        self.ball_y = 0.5;
        self.ball_vx = BALL_SPEED * if rng.next_f32() < 0.5 { 1.0 } else { -1.0 };
        self.ball_vy = BALL_SPEED * (rng.next_f32() - 0.5) * 2.0;
        self.paddle_y = 0.5;
        self.ticks = 0;
        self.hit_this_tick = false;
    }

    /// State: [ball_x, ball_y, ball_vx, ball_vy, paddle_y]
    /// All normalized to [-1, 1]
    fn state(&self) -> [f32; 5] {
        [
            (self.ball_x - 0.5) * 2.0,    // -1..1
            (self.ball_y - 0.5) * 2.0,    // -1..1
            self.ball_vx / BALL_SPEED,     // normalized speed
            self.ball_vy / BALL_SPEED,     // normalized speed
            (self.paddle_y - 0.5) * 2.0,   // -1..1
        ]
    }

    /// Step: action = 1 (up) or -1 (down)
    fn step(&mut self, action: f32) {
        self.hit_this_tick = false;
        self.ticks += 1;

        // Move paddle
        if action > 0.0 {
            self.paddle_y = (self.paddle_y - PADDLE_SPEED).max(0.05);
        } else {
            self.paddle_y = (self.paddle_y + PADDLE_SPEED).min(0.95);
        }

        // Move ball
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        // Bounce off top/bottom
        if self.ball_y <= 0.0 || self.ball_y >= 1.0 {
            self.ball_vy = -self.ball_vy;
            self.ball_y = self.ball_y.clamp(0.001, 0.999);
        }

        // Paddle hit detection (ball reaches paddle x ~0.95)
        if self.ball_x >= 0.92 && self.ball_vx > 0.0 {
            let paddle_top = self.paddle_y - 0.08;
            let paddle_bot = self.paddle_y + 0.08;
            if self.ball_y >= paddle_top && self.ball_y <= paddle_bot {
                // Hit! Bounce back
                self.ball_vx = -self.ball_vx;
                // Add slight angle based on where ball hit paddle
                let hit_offset = (self.ball_y - self.paddle_y) / 0.08; // -1..1
                self.ball_vy += hit_offset * BALL_SPEED * 0.3;
                self.ball_vy = self.ball_vy.clamp(-BALL_SPEED * 2.0, BALL_SPEED * 2.0);
                self.ball_x = 0.91; // prevent re-hit
                self.hits += 1;
                self.hit_this_tick = true;
            }
        }

        // Ball past paddle → miss
        if self.ball_x > 1.1 {
            self.misses += 1;
            self.reset();
        }
        // Ball past left wall → bounce
        if self.ball_x < -0.1 && self.ball_vx < 0.0 {
            self.ball_vx = -self.ball_vx;
            // Give a small reward for surviving
        }
    }

    fn done(&self) -> bool {
        self.ticks >= MAX_GAME_TICKS || self.misses >= 5
    }
}

// ═══════════════════════════════════════════════════════════
//  Kuramoto helpers
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
            if spike_buffer[base + i] != 0 { counts[i] += 1.0; }
        }
    }
    counts
}

fn build_synapse_spikes(
    syn_pre: &mut [i32], syn_post: &mut [i32],
    spikes: &[i32], row_ptr: &[i32], col_indices: &[i32], n: usize,
) {
    let max_syn = col_indices.len().min(syn_pre.len()).min(syn_post.len());
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let pre_val = if i < spikes.len() { spikes[i] } else { 0 };
        for idx in start..end {
            let j = col_indices[idx] as usize;
            syn_pre[idx] = pre_val;
            syn_post[idx] = if j < spikes.len() { spikes[j] } else { 0 };
        }
    }
}

/// Run Kuramoto dynamics for `n_steps` with input modulation.
/// Returns (spike_buffer, final_sin_buf, final_cos_buf, coherence)
fn run_kuramoto_episode(
    n: usize, omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    n_steps: usize, seed: u32,
    input_drive: &[f32],  // per-neuron omega modulation
) -> (Vec<i32>, Vec<f32>, Vec<f32>, f32) {
    let mut rng = Lcg::new(seed);
    let mut phases: Vec<f32> = (0..n)
        .map(|_| rng.next_f32() * std::f32::consts::TAU)
        .collect();
    let mut phases_prev = phases.clone();

    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];
    let mut coupling_buf = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut spike_buf = vec![0i32; n_steps * n];

    for step in 0..n_steps {
        // Modulate omegas with input drive
        let mut omega_mod = omegas.to_vec();
        for i in 0..n.min(input_drive.len()) {
            omega_mod[i] += input_drive[i];
        }

        kuramoto_step_sparse(
            &mut phases, &omega_mod,
            row_ptr, col_indices, values,
            &mut sin_buf, &mut cos_buf, &mut coupling_buf, DT,
        );

        detect_spikes(&phases, &phases_prev, &mut spikes);
        phases_prev.copy_from_slice(&phases);

        let base = step * n;
        for i in 0..n {
            spike_buf[base + i] = spikes[i];
        }
    }

    let amps = vec![1.0f32; n];
    let mut coh_out = vec![0.0f32];
    phase_coherence(&mut coh_out, &sin_buf, &cos_buf, &amps);
    let coherence = coh_out[0];

    (spike_buf, sin_buf, cos_buf, coherence)
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    println!("{}", "═".repeat(66));
    println!("  🏓 Pong Chaos vs Pattern — VAE + Kuramoto + dopamine-STDP");
    println!("{}", "═".repeat(66));
    println!();

    // ── 1. VAE → Connectome ──
    println!("▶ Generating brain from ConnectomeGPT VAE...");
    let t1 = Instant::now();

    let weights = fs::read(VAE_WEIGHTS_PATH)
        .map_err(|_| "VAE weights not found. Run `cargo run --release --example test_vae -p connectome-gpt` first.")?;
    let norm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(VAE_NORM_PATH)?)?;

    let mut vae = connectome_gpt::GraphVAE::new(connectome_gpt::VaeConfig {
        input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: VAE_SEED,
    });
    vae.load(&weights);
    let mut raw = vae.generate();

    // De-normalize
    {
        let m: Vec<f32> = norm["mean"][0].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let s: Vec<f32> = norm["std"][0].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        for (x, (mv, sv)) in raw.iter_mut().zip(m.iter().zip(s.iter())) {
            *x = *x * sv + mv;
        }
    }

    // Decode to connectome
    let cfg = connectome_gpt::decoder::DecoderConfig {
        n_neurons: raw[0].round() as usize,
        density: raw[2] / 100.0,
        ei_ratio: raw[5].max(1.0),
        mean_weight: (raw[6] / 100.0).max(0.1),
        weight_noise: 0.05,
        n_communities: raw[8].round() as usize,
        community_strength: 8.0,
        sensory_fraction: 0.05,
        dn_fraction: 0.02,
        mn_fraction: 0.005,
        stdp_weights: true,
        compute_depth: true,
        refine_epochs: 0,
        seed: VAE_SEED,
    };

    let sample = connectome_gpt::CsrDecoder::new(cfg).generate(&raw);
    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();

    let n = sample.n_neurons;
    let n_syn = sample.values.len();
    println!("  VAE brain: {} neurons, {} synapses ({:.1}% density)",
        n, n_syn, n_syn as f64 / (n as f64 * n as f64) * 100.0);

    // Find groups
    let mut sensory_idx: Vec<usize> = Vec::new();
    let mut mn_idx: Vec<usize> = Vec::new();
    let mut dn_idx: Vec<usize> = Vec::new();
    for (name, indices) in &groups {
        match *name {
            "sensory" => sensory_idx = indices.clone(),
            "mn" => mn_idx = indices.clone(),
            "dn" => dn_idx = indices.clone(),
            _ => {}
        }
    }

    // Fallback if groups are missing
    if sensory_idx.is_empty() {
        sensory_idx = (0..n / 20).collect();
    }
    if mn_idx.is_empty() {
        mn_idx = (n - n / 50..n).collect();
    }
    if dn_idx.is_empty() {
        dn_idx = (n / 20..n / 10).collect();
    }

    println!("  Groups: {} sensory, {} DN, {} MN, {} other",
        sensory_idx.len(), dn_idx.len(), mn_idx.len(),
        n - sensory_idx.len() - dn_idx.len() - mn_idx.len());
    println!("  VAE time: {:.2}s", t1.elapsed().as_secs_f32());
    println!();

    // ── 2. Setup ──
    println!("▶ Setting up Kuramoto + dopamine-STDP...");

    // Omegas
    let omegas: Vec<f32> = (0..n)
        .map(|i| OMEGA_BASE + (i as f32 / n as f32 - 0.5) * 2.0 * OMEGA_SPREAD)
        .collect();

    // Input projection: 5 state dims × n → drive per neuron
    let mut rng = Lcg::new(42);
    let input_scale = 8.0;
    let mut w_in = vec![0.0f32; 5 * n];
    for w in w_in.iter_mut() {
        *w = (rng.next_f32() - 0.5) * 2.0 * input_scale;
    }

    // DopamineSTDP
    let stdp_dt = DT * 1000.0;
    let mut stdp = DopamineSTDP::new(n_syn, stdp_dt);
    stdp.tau_pre = 20.0;
    stdp.tau_post = 20.0;
    stdp.tau_e = 500.0;
    stdp.a_ltp = 0.15;
    stdp.a_ltd = 0.12;
    stdp.w_min = W_MIN;
    stdp.w_max = W_MAX;

    let mut values = sample.values.clone();

    let mut syn_pre = vec![0i32; n_syn];
    let mut syn_post = vec![0i32; n_syn];

    // ── 3. Baseline eval (random weights) ──
    println!("▶ Baseline evaluation (random readout)...");
    let mut baseline_scores: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let mut w_out = vec![0.0f32; 2 * mn_idx.len()];
        for w in w_out.iter_mut() { *w = (rng.next_f32() - 0.5) * 0.5; }
        let score = play_pong_episode(
            n, &omegas, &sample.row_ptr, &sample.col_indices, &values,
            &w_in, &w_out, &sensory_idx, &mn_idx,
        );
        baseline_scores.push(score);
    }
    let baseline_avg: f32 = baseline_scores.iter().sum::<f32>() / 10.0;
    println!("  Baseline avg: {:.1} (hits - misses)", baseline_avg);
    println!();

    // ── 4. PHASE 1: CEM on readout weights ──
    println!("{}", "─".repeat(66));
    println!("  PHASE 1: CEM on readout ({} iters, pop={}, elite={})",
        CEM_ITERS, CEM_POP, CEM_ELITE);
    println!("{}", "─".repeat(66));

    let t_cem = Instant::now();
    let mut w_out = vec![0.0f32; 2 * mn_idx.len()];
    let mut best_cem_score = 0.0f32;

    for iter in 0..CEM_ITERS {
        let mut candidates: Vec<(f32, Vec<f32>)> = Vec::with_capacity(CEM_POP);

        for _ in 0..CEM_POP {
            let mut w = Vec::with_capacity(2 * mn_idx.len());
            for &b in w_out.iter() {
                w.push(b + CEM_NOISE * rng.randn());
            }
            let score = play_pong_episode(
                n, &omegas, &sample.row_ptr, &sample.col_indices, &values,
                &w_in, &w, &sensory_idx, &mn_idx,
            );
            candidates.push((score, w));
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let elite: Vec<&[f32]> = candidates.iter().take(CEM_ELITE).map(|(_, w)| w.as_slice()).collect();
        let n_elite = elite.len() as f32;
        for i in 0..2 * mn_idx.len() {
            w_out[i] = elite.iter().map(|w| w[i]).sum::<f32>() / n_elite;
        }

        let best = candidates[0].0;
        if best > best_cem_score { best_cem_score = best; }

        if (iter + 1) % 5 == 0 || iter == CEM_ITERS - 1 {
            let elite_mean: f32 = candidates.iter().take(CEM_ELITE).map(|(r, _)| r).sum::<f32>() / CEM_ELITE as f32;
            println!("  iter={:2}  best={:.0}  elite_mean={:.0}",
                iter + 1, candidates[0].0, elite_mean);
        }
    }

    // Post-CEM eval
    let mut post_cem_scores: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let score = play_pong_episode(
            n, &omegas, &sample.row_ptr, &sample.col_indices, &values,
            &w_in, &w_out, &sensory_idx, &mn_idx,
        );
        post_cem_scores.push(score);
    }
    let post_cem_avg: f32 = post_cem_scores.iter().sum::<f32>() / 10.0;
    println!("  CEM done: avg={:.1}  best={:.0}  Δ={:+.1}",
        post_cem_avg, best_cem_score, post_cem_avg - baseline_avg);
    println!("  Time: {:.2}s", t_cem.elapsed().as_secs_f64());
    println!();

    // ── 5. PHASE 2: Chaos/Pattern dopamine-STDP ──
    println!("{}", "─".repeat(66));
    println!("  PHASE 2: Chaos vs Pattern dopamine-STDP reservoir training");
    println!("  δ = reward + {:.1}·(R_pat − {:.1}·|H_pat − {:.1}|)", BETA, ALPHA, H_TARGET);
    println!("    └ − (R_chaos − {:.1}·|H_chaos − {:.1}|)", ALPHA, H_TARGET);
    println!("  {} episodes, η={}, τ_e={:.0}ms", N_EPISODES, LR, stdp.tau_e);
    println!("{}", "─".repeat(66));

    let t_dopa = Instant::now();
    let mut chaos_scores: Vec<f32> = Vec::new();
    let mut pattern_scores: Vec<f32> = Vec::new();
    let mut delta_history: Vec<f32> = Vec::new();
    let mut r_pat_history: Vec<f32> = Vec::new();
    let mut r_chaos_history: Vec<f32> = Vec::new();

    println!();
    println!("  {:>4}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Ep", "R_pat", "R_chaos", "H_pat", "Score", "δ", "Time");
    println!("  {}", "─".repeat(66));

    let mut best_score = post_cem_avg;

    for episode in 0..N_EPISODES {
        let t_ep = Instant::now();
        let seed_base = episode as u32 * 7919 + VAE_SEED as u32;

        // ── 5a. PATTERN phase: Play Pong ──
        stdp.clear_eligibility();

        let (total_reward, pattern_r, pattern_h) = play_pong_with_stdp(
            n, &omegas, &sample.row_ptr, &sample.col_indices, &mut values,
            &w_in, &w_out, &sensory_idx, &mn_idx,
            &mut stdp, &mut syn_pre, &mut syn_post,
            seed_base,
        );

        // ── 5b. CHAOS phase: random noise ──
        let mut rng_noise = Lcg::new(seed_base + 555);
        let noise_drive: Vec<f32> = (0..n).map(|_| rng_noise.next_f32_range(-5.0, 5.0)).collect();

        let (chaos_spike_buf, _, _, chaos_r) = run_kuramoto_episode(
            n, &omegas, &sample.row_ptr, &sample.col_indices, &values,
            N_KURAMOTO_STEPS, seed_base + 999, &noise_drive,
        );
        let chaos_counts = count_spikes(&chaos_spike_buf, n, N_KURAMOTO_STEPS);
        let (chaos_h, _) = spike_entropy(&chaos_counts, N_KURAMOTO_STEPS as f32, H_TARGET);

        // ── 5c. Compute reward signal δ ──
        let pat_utility = pattern_r - ALPHA * (pattern_h - H_TARGET).abs();
        let chaos_utility = chaos_r - ALPHA * (chaos_h - H_TARGET).abs();
        let coherence_delta = pat_utility - chaos_utility;

        // δ = game_reward + β * coherence_delta
        let delta = total_reward + BETA * coherence_delta;

        // ── 5d. Apply dopamine weight update ──
        stdp.apply_dopamine(&mut values, delta, LR);

        // ── Track ──
        pattern_scores.push(total_reward);
        chaos_scores.push(chaos_r);
        delta_history.push(delta);
        r_pat_history.push(pattern_r);
        r_chaos_history.push(chaos_r);

        if total_reward > best_score { best_score = total_reward; }

        if episode % 5 == 0 || episode == N_EPISODES - 1 {
            println!("  {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>+8.0}  {:>+8.3}  {:>8.0}ms",
                episode, pattern_r, chaos_r, pattern_h, total_reward, delta,
                t_ep.elapsed().as_millis());
        }
    }

    let t_dopa_elapsed = t_dopa.elapsed().as_secs_f64();

    // Post-dopamine eval
    let mut post_dopa_scores: Vec<f32> = Vec::new();
    for _ in 0..10 {
        let score = play_pong_episode(
            n, &omegas, &sample.row_ptr, &sample.col_indices, &values,
            &w_in, &w_out, &sensory_idx, &mn_idx,
        );
        post_dopa_scores.push(score);
    }
    let post_dopa_avg: f32 = post_dopa_scores.iter().sum::<f32>() / 10.0;

    println!();
    println!("  Dopamine time: {:.2}s", t_dopa_elapsed);
    println!("  Dopamine avg:  {:.1}", post_dopa_avg);

    // ── 6. FINAL RESULTS ──
    println!("\n{}", "═".repeat(66));
    println!("  📊 FINAL RESULTS: VAE Pong Chaos vs Pattern");
    println!("{}", "═".repeat(66));

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────────┐");
    println!("  │  Method           │  Avg Score   │  Best Score  │  Δ vs Random │");
    println!("  ├──────────────────────────────────────────────────────────────────┤");
    println!("  │  Random           │  {:.1}       │  —           │  —           │",
        baseline_avg);
    println!("  │  CEM readout      │  {:.1}       │  {:.0}        │  {:+.1}       │",
        post_cem_avg, best_cem_score, post_cem_avg - baseline_avg);
    println!("  │  CEM + dopamine   │  {:.1}       │  {:.0}        │  {:+.1}       │",
        post_dopa_avg, best_score, post_dopa_avg - baseline_avg);
    println!("  └──────────────────────────────────────────────────────────────────┘");

    // Coherence analysis
    let r_pat_avg: f32 = r_pat_history.iter().sum::<f32>() / r_pat_history.len() as f32;
    let r_chaos_avg: f32 = r_chaos_history.iter().sum::<f32>() / r_chaos_history.len() as f32;
    let delta_avg: f32 = delta_history.iter().sum::<f32>() / delta_history.len() as f32;
    let delta_first5: f32 = delta_history.iter().take(5).sum::<f32>() / 5.0;
    let delta_last5: f32 = delta_history.iter().rev().take(5).sum::<f32>() / 5.0;

    println!("\n  ─── Chaos vs Pattern coherence analysis ───");
    println!("  R_pattern avg:  {:.4}", r_pat_avg);
    println!("  R_chaos avg:    {:.4}", r_chaos_avg);
    println!("  R_gap:          {:+.4}", r_pat_avg - r_chaos_avg);
    println!("  δ avg:          {:+.4}", delta_avg);
    println!("  δ first 5 ep:   {:+.4}", delta_first5);
    println!("  δ last 5 ep:    {:+.4}  ({:+.4})", delta_last5, delta_last5 - delta_first5);

    // Timeline
    println!("\n  ─── Score timeline ───");
    let step = (N_EPISODES / 15).max(1);
    for i in (0..N_EPISODES).step_by(step) {
        let score = pattern_scores[i];
        let bar_len = ((score + 5.0) / 10.0 * 25.0).clamp(0.0, 25.0) as usize;
        let bar = "██".repeat(bar_len / 2);
        println!("  [{:>3}] score={:>+4.0}  {}  δ={:+.3}",
            i, score, bar, delta_history[i]);
    }

    // ── Conclusion ──
    println!("\n{}", "═".repeat(66));
    let improved = post_dopa_avg > post_cem_avg * 1.1;
    let beats_baseline = post_dopa_avg > baseline_avg * 1.5;
    let r_gap_positive = r_pat_avg > r_chaos_avg;
    let delta_up = delta_last5 > delta_first5;

    if improved && beats_baseline {
        println!("  ✅ УСПЕХ! VAE-сгенерированный мозг научился играть в Pong!");
        println!("  Score: {:.0} → {:.0} (baseline: {:.0})",
            baseline_avg, post_dopa_avg, baseline_avg);
        if r_gap_positive {
            println!("  🧠 Chaos/pattern: R_pat > R_chaos — мозг различает паттерн и хаос!");
        }
        if delta_up {
            println!("  📈 δ растёт — dopamine signal увеличивается с обучением.");
        }
    } else if beats_baseline {
        println!("  📈 ПРОГРЕСС: VAE-мозг играет в Pong лучше random, но dopamine не дал значительного улучшения.");
        println!("  Baseline: {:.0} → CEM: {:.0} → Dopamine: {:.0}",
            baseline_avg, post_cem_avg, post_dopa_avg);
    } else if improved {
        println!("  📈 СЛАБЫЙ ПРОГРЕСС: CEM улучшил игру, dopamine дал небольшой +.");
    } else {
        println!("  ⚠️  Обучение идёт, но score не вырос значительно.");
        println!("  Baseline: {:.0}, CEM: {:.0}, Dopamine: {:.0}",
            baseline_avg, post_cem_avg, post_dopa_avg);
        println!("  Возможно, нужно больше эпизодов или другой LR.");
    }

    println!("\n  Параметры: α={:.1}, β={:.1}, η={}, τ_e={:.0}ms", ALPHA, BETA, LR, stdp.tau_e);
    println!("  VAE: {} нейронов, {} синапсов", n, n_syn);
    println!("  Pong: {} ticks макс, paddle speed={}", MAX_GAME_TICKS, PADDLE_SPEED);
    println!("  Sensory: {}, MN: {}", sensory_idx.len(), mn_idx.len());
    println!("  Всего: {:.2}s", t0.elapsed().as_secs_f64());
    println!("{}", "═".repeat(66));

    Ok(())
}

// ═══════════════════════════════════════════════════════════
//  Pong play helpers
// ═══════════════════════════════════════════════════════════

/// Play one Pong episode with given weights.
/// Uses CONTINUOUS Kuramoto phases across all game ticks.
/// Returns score (hits - misses).
fn play_pong_episode(
    n: usize, omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    w_in: &[f32], w_out: &[f32],
    sensory_idx: &[usize], mn_idx: &[usize],
) -> f32 {
    let mut pong = Pong::new();
    let n_mn = mn_idx.len();

    // Continuous Kuramoto state across ALL game ticks
    let mut rng = Lcg::new(pong.ticks as u32 + 42);
    let mut phases: Vec<f32> = (0..n)
        .map(|_| rng.next_f32() * std::f32::consts::TAU)
        .collect();
    let mut phases_prev = phases.clone();
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];
    let mut coupling_buf = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];

    while !pong.done() {
        // Encode state → drives
        let state = pong.state();
        let mut drive = vec![0.0f32; n];
        for &si in sensory_idx {
            if si < n {
                for d in 0..5 {
                    drive[si] += state[d] * w_in[d * n + si];
                }
            }
        }

        // Accumulate MN spike counts across Kuramoto steps
        let mut mn_counts = vec![0.0f32; n_mn];

        for _ in 0..N_KURAMOTO_STEPS {
            let mut omega_mod = omegas.to_vec();
            for i in 0..n.min(drive.len()) {
                omega_mod[i] += drive[i];
            }

            kuramoto_step_sparse(
                &mut phases, &omega_mod,
                row_ptr, col_indices, values,
                &mut sin_buf, &mut cos_buf, &mut coupling_buf, DT,
            );

            detect_spikes(&phases, &phases_prev, &mut spikes);
            phases_prev.copy_from_slice(&phases);

            for (mi, &m) in mn_idx.iter().enumerate() {
                if m < n && spikes[m] != 0 {
                    mn_counts[mi] += 1.0;
                }
            }
        }

        // Readout: compare up vs down action values
        let up = mn_counts.iter().zip(w_out.iter())
            .map(|(&c, &w)| c * w).sum::<f32>();
        let down = mn_counts.iter().zip(w_out[n_mn..].iter())
            .map(|(&c, &w)| c * w).sum::<f32>();

        let action = if up > down { 1.0 } else { -1.0 };
        pong.step(action);
    }

    pong.hits as f32 - pong.misses as f32 * 0.5
}

/// Play Pong episode WITH STDP eligibility accumulation.
/// Uses CONTINUOUS Kuramoto phases across all game ticks.
/// Coherence R and entropy H are measured from ACTUAL gameplay dynamics!
/// Returns (score, coherence_r, entropy_h).
fn play_pong_with_stdp(
    n: usize, omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], values: &mut [f32],
    w_in: &[f32], w_out: &[f32],
    sensory_idx: &[usize], mn_idx: &[usize],
    stdp: &mut DopamineSTDP, syn_pre: &mut [i32], syn_post: &mut [i32],
    seed: u32,
) -> (f32, f32, f32) {
    let mut pong = Pong::new();
    let n_mn = mn_idx.len();
    let mut total_reward = 0.0f32;
    let mut prev_hits = 0usize;
    let mut prev_misses = 0usize;

    // CONTINUOUS Kuramoto state across the ENTIRE episode
    let mut rng = Lcg::new(seed);
    let mut phases: Vec<f32> = (0..n)
        .map(|_| rng.next_f32() * std::f32::consts::TAU)
        .collect();
    let mut phases_prev = phases.clone();

    // Accumulate sin/cos buffers across ALL game ticks for final coherence
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];
    let mut coupling_buf = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];

    // Accumulate spike counts for ALL steps across the episode
    let mut total_spike_counts = vec![0.0f32; n];
    let mut total_steps = 0usize;

    while !pong.done() {
        let state = pong.state();
        let mut drive = vec![0.0f32; n];
        for &si in sensory_idx {
            if si < n {
                for d in 0..5 {
                    drive[si] += state[d] * w_in[d * n + si];
                }
            }
        }

        let mut mn_counts = vec![0.0f32; n_mn];

        for _ in 0..N_KURAMOTO_STEPS {
            let mut omega_mod = omegas.to_vec();
            for i in 0..n.min(drive.len()) {
                omega_mod[i] += drive[i];
            }

            kuramoto_step_sparse(
                &mut phases, &omega_mod,
                row_ptr, col_indices, values,
                &mut sin_buf, &mut cos_buf, &mut coupling_buf, DT,
            );

            detect_spikes(&phases, &phases_prev, &mut spikes);
            phases_prev.copy_from_slice(&phases);

            // Accumulate spike counts from ACTUAL gameplay
            for i in 0..n {
                if spikes[i] != 0 { total_spike_counts[i] += 1.0; }
            }
            total_steps += 1;

            // STDP eligibility from actual spikes
            build_synapse_spikes(syn_pre, syn_post, &spikes, row_ptr, col_indices, n);
            stdp.step(syn_pre, syn_post);

            // Track MN spikes for readout
            for (mi, &m) in mn_idx.iter().enumerate() {
                if m < n && spikes[m] != 0 {
                    mn_counts[mi] += 1.0;
                }
            }
        }

        // Read MN → action
        let up = mn_counts.iter().zip(w_out.iter())
            .map(|(&c, &w)| c * w).sum::<f32>();
        let down = mn_counts.iter().zip(w_out[n_mn..].iter())
            .map(|(&c, &w)| c * w).sum::<f32>();

        let action = if up > down { 1.0 } else { -1.0 };
        pong.step(action);

        // Game reward tracking
        if pong.hits > prev_hits { total_reward += 1.0; prev_hits = pong.hits; }
        if pong.misses > prev_misses { total_reward -= 0.5; prev_misses = pong.misses; }
    }

    // Compute phase coherence from ACTUAL gameplay buffers
    let amps = vec![1.0f32; n];
    let mut coh_out = vec![0.0f32];
    phase_coherence(&mut coh_out, &sin_buf, &cos_buf, &amps);
    let coherence_r = coh_out[0];

    // Compute entropy from ACTUAL gameplay spike counts
    let (entropy_h, _) = spike_entropy(&total_spike_counts, total_steps as f32, H_TARGET);

    (total_reward, coherence_r, entropy_h)
}
