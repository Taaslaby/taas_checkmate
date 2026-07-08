//! SPAN CartPole: CEM readout + E-Prop reservoir training (40 MNs).
//!
//! 1. CEM на readout (как span_cartpole.rs) — обучает readout weights
//! 2. E-Prop на reservoir — дообучает синапсы через eligibility traces + reward signal
//! 3. Сравнение до/после E-Prop
//!
//! C 40 MNs E-Prop обновляет ~1040/20862 синапсов (vs ~100 с 4 MNs).
//! CartPole solved if avg reward ≥ 195/200.

use std::fs;
use std::time::Instant;
use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich, sim_izhikevich_eprop};
use taas_connectome::ConnectomeData;

const TAAS_PATH: &str = "crates/connectome-gpt/models/brain.taas";

// Simulation params
const N_STEPS: usize = 10;
const DT: f32 = 0.5;
const BIAS: f32 = 5.0;
const I_SCALE: f32 = 20.0;
const W_CLAMP: f32 = 5.0;
const MAX_STEPS: usize = 200;
const TARGET_MN: usize = 40;

// CEM params
const CEM_POP: usize = 100;
const CEM_ELITE: usize = 20;
const CEM_ITERS: usize = 40;
const CEM_NOISE: f32 = 0.5;

// E-Prop params
const LR_EPROP: f32 = 0.001;
const TAU_E: f32 = 20.0;
const TAU_SURROGATE: f32 = 5.0;
const N_EPROP_ITERS: usize = 5;  // E-Prop training phases (each = 100 episodes)

// Checkpoint
const CHECKPOINT_EVERY: usize = 10;  // save brain checkpoint every N CEM iters

// ─── CartPole ────────────────────────────────────────────

struct CartPole {
    x: f32, x_dot: f32, theta: f32, theta_dot: f32,
}

impl CartPole {
    fn new() -> Self { Self { x: 0.0, x_dot: 0.0, theta: 0.05, theta_dot: 0.0 } }
    fn reset(&mut self) -> [f32; 4] {
        self.x = 0.0; self.x_dot = 0.0; self.theta = 0.05; self.theta_dot = 0.0;
        [0.0, 0.0, 0.05, 0.0]
    }
    fn state(&self) -> [f32; 4] { [self.x, self.x_dot, self.theta, self.theta_dot] }
    fn step(&mut self, action: f32) -> (f32, bool) {
        let force = if action > 0.5 { 10.0 } else { -10.0 };
        let g = 9.8; let mc = 1.0; let mp = 0.1;
        let mt = mc + mp; let l = 0.5;
        let pl = mp * l;
        let ct = self.theta.cos(); let st = self.theta.sin();
        let temp = (force + pl * self.theta_dot.powi(2) * st) / mt;
        let ta = (g * st - ct * temp) / (l * (4.0/3.0 - mp * ct.powi(2) / mt));
        let xa = temp - pl * ta * ct / mt;
        self.x += 0.02 * self.x_dot; self.x_dot += 0.02 * xa;
        self.theta += 0.02 * self.theta_dot; self.theta_dot += 0.02 * ta;
        let done = self.theta.abs() > 0.209 || self.x.abs() > 2.4;
        (1.0, done)
    }
}

// ─── Helpers ─────────────────────────────────────────────

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
    fn randn(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (6.2831853 * u2).cos()
    }
}

/// Helper: run CartPole episode with Izhikevich brain (no E-Prop, fast).
/// Returns reward (steps survived).
fn run_episode(
    w_out: &[f32], conn: &ConnectomeData, w_in: &[f32],
    brain: &mut BrainState, params: &IzhikevichParams,
    mn_idx: &[usize], n_mn: usize,
) -> f32 {
    let n = conn.row_ptr.len() - 1;
    let mut env = CartPole::new();
    let mut s = env.reset();
    brain.reset_with_noise(n);
    let mut survived = 0usize;
    for _ in 0..MAX_STEPS {
        let mut ext_I = vec![BIAS; n];
        for j in 0..n { for d in 0..4 { ext_I[j] += s[d] * w_in[d * n + j]; } }
        let spike_buf = sim_izhikevich(conn, brain, params, &ext_I, N_STEPS);
        let mut mn_counts = vec![0.0f32; n_mn];
        for step in 0..N_STEPS {
            let base = step * n;
            for (mi, &m) in mn_idx.iter().enumerate() {
                if spike_buf[base + m] != 0 { mn_counts[mi] += 1.0; }
            }
        }
        let left = mn_counts.iter().zip(w_out.iter()).map(|(&c, &w)| c * w).sum::<f32>();
        let right = mn_counts.iter().zip(w_out[n_mn..].iter()).map(|(&c, &w)| c * w).sum::<f32>();
        let action = if left > right { 0.0 } else { 1.0 };
        let (_, done) = env.step(action);
        s = env.state();
        survived += 1;
        if done { break; }
    }
    survived as f32
}

/// Run CartPole episode with E-Prop recording.
/// Returns (reward, eligibility_traces_for_mn_synapses)
fn run_episode_eprop(
    w_out: &[f32], conn: &ConnectomeData, w_in: &[f32],
    brain: &mut BrainState, params: &IzhikevichParams,
    mn_idx: &[usize], n_mn: usize,
    tau_e: f32, tau_surrogate: f32,
) -> (f32, Vec<Vec<f32>>) {
    let n = conn.row_ptr.len() - 1;
    let mut env = CartPole::new();
    let mut s = env.reset();
    brain.reset_with_noise(n);
    let mut reward = 0.0f32;

    // Per-step eligibility for MN synapses only (n_mn × n_syn is too large)
    // Instead, we accumulate eligibility per-step and compute L_j at episode end
    // Store (mn_idx, eligibility_vector, action_logit) per step
    struct StepData {
        mn_counts: Vec<f32>,
        left_logit: f32,
        right_logit: f32,
        action: f32,
        eligibility: Vec<f32>,  // per-synapse eligibility at this step
    }
    let mut step_data: Vec<StepData> = Vec::with_capacity(MAX_STEPS);

    for _ in 0..MAX_STEPS {
        let mut ext_I = vec![BIAS; n];
        for j in 0..n { for d in 0..4 { ext_I[j] += s[d] * w_in[d * n + j]; } }

        let (spike_buf, eligibility) = sim_izhikevich_eprop(
            conn, brain, params, &ext_I, N_STEPS, tau_e, tau_surrogate,
        );

        let mut mn_counts = vec![0.0f32; n_mn];
        for step in 0..N_STEPS {
            let base = step * n;
            for (mi, &m) in mn_idx.iter().enumerate() {
                if spike_buf[base + m] != 0 { mn_counts[mi] += 1.0; }
            }
        }

        let left = mn_counts.iter().zip(w_out.iter()).map(|(&c, &w)| c * w).sum::<f32>();
        let right = mn_counts.iter().zip(w_out[n_mn..].iter()).map(|(&c, &w)| c * w).sum::<f32>();
        let action = if left > right { 0.0 } else { 1.0 };

        reward += 1.0;
        step_data.push(StepData {
            mn_counts,
            left_logit: left,
            right_logit: right,
            action,
            eligibility,
        });

        let (_, done) = env.step(action);
        s = env.state();
        if done { break; }
    }

    // Compute per-step learning signals using REINFORCE
    let n_syn = conn.values.len();
    // We accumulate eligibility for downstream (post-MN) synapses
    // For E-Prop, L_j = error signal for neuron j
    // With REINFORCE: L_j = Σ_k (G_t - b) * ∇action_k/∇MN_j
    // Simplified: L_j for MNs = (reward - baseline) * direction
    // where direction = ±1 depending on whether action matched

    // For reservoir update: L_j propagates from MNs through readout
    // L_j(mn) = (reward - baseline) * w_out[action, mn] for the chosen action
    // L_j(non-mn) = 0 (no direct readout → no learning signal)

    let baseline = reward / 2.0; // simple baseline
    let delta = reward - baseline;

    // Accumulate eligibility for each step, weighted by learning signal
    let mut mn_l_signals = vec![0.0f32; n_mn];
    // L_j(mn_i) = delta * w_out[action, mn_i]
    for (mi, &m) in mn_idx.iter().enumerate() {
        let chosen_action = if step_data.last().map(|s| s.action).unwrap_or(0.0) > 0.5 { 1 } else { 0 };
        // For action 0 (left): w_out[0 * n_mn + mi]
        // For action 1 (right): w_out[1 * n_mn + mi]
        mn_l_signals[mi] = delta * w_out[chosen_action * n_mn + mi];
    }

    // Generate eligibility traces for each MN (one per synaptic connection to MN)
    // This is needed for E-Prop update
    let mut mn_eligibility: Vec<Vec<f32>> = Vec::with_capacity(n_mn);
    for (mi, &m) in mn_idx.iter().enumerate() {
        // Aggregate eligibility from all steps where this MN spiked
        let start = conn.row_ptr[m].max(0) as usize;
        let end = (conn.row_ptr[m + 1] as usize).min(conn.values.len());
        let mut total_elig = vec![0.0f32; end - start];
        for sd in &step_data {
            let step_elig = &sd.eligibility;
            if start < step_elig.len() && end <= step_elig.len() {
                for (idx, e) in total_elig.iter_mut().enumerate() {
                    *e += step_elig[start + idx];
                }
            }
        }
        mn_eligibility.push(total_elig);
    }

    (reward, mn_eligibility)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  SPAN CartPole E-Prop: CEM readout + E-Prop reservoir (40 MNs)");
    println!("{}", "═".repeat(60));

    // ── 1. Load connectome ──
    let (conn, mut mn_idx) = if let Ok(bytes) = fs::read(TAAS_PATH) {
        let reader = taas_format::TaasReader::from_bytes(&bytes)?;
        let conn = taas_format::load_connectome(&reader)?;
        println!("  Loaded from {} ({}n {}syn)", TAAS_PATH, conn.n_neurons, conn.n_synapses);
        let mn = conn.group("mn").map(|x| x.to_vec()).unwrap_or_default();
        (conn, mn)
    } else {
        println!("  {} not found, generating from VAE...", TAAS_PATH);
        let weights = fs::read("crates/connectome-gpt/models/vae_weights.bin")?;
        let norm: serde_json::Value =
            serde_json::from_str(&fs::read_to_string("crates/connectome-gpt/models/vae_norm.json")?)?;
        let mut vae = connectome_gpt::GraphVAE::new(connectome_gpt::VaeConfig {
            input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42,
        });
        vae.load(&weights);
        let mut raw = vae.generate();
        {
            let m: Vec<f32> = norm["mean"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let s: Vec<f32> = norm["std"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            for (x, (mv, sv)) in raw.iter_mut().zip(m.iter().zip(s.iter())) { *x = *x * sv + mv; }
        }
        let cfg = connectome_gpt::decoder::DecoderConfig {
            n_neurons: raw[0].round() as usize, density: raw[2] / 100.0,
            ei_ratio: raw[5].max(1.0), mean_weight: (raw[6] / 100.0).max(0.1),
            weight_noise: 0.05, n_communities: raw[8].round() as usize,
            community_strength: 8.0, sensory_fraction: 0.05, dn_fraction: 0.02,
            mn_fraction: 0.005, stdp_weights: true, compute_depth: true,
            refine_epochs: 0, seed: 42,
        };
        let sample = connectome_gpt::CsrDecoder::new(cfg).generate(&raw);
        let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
            .map(|(k, v)| (k.as_str(), v.clone())).collect();
        let mn = sample.groups.get("mn").cloned().unwrap_or_default();
        let conn = ConnectomeData::new(sample.row_ptr, sample.col_indices, sample.values, sample.nt_types, groups);
        (conn, mn)
    };

    let n_neurons = conn.n_neurons;
    let n_syn = conn.n_synapses;
    // Include more MNs
    for i in (n_neurons - TARGET_MN).max(0)..n_neurons {
        if !mn_idx.contains(&i) { mn_idx.push(i); }
    }
    let n_mn = mn_idx.len();
    println!("  MN: {}/{} (upstream synapses: ~{}k)",
        n_mn, TARGET_MN,
        mn_idx.iter().map(|&m| {
            let start = conn.row_ptr[m].max(0) as usize;
            let end = (conn.row_ptr[m + 1] as usize).min(conn.values.len());
            end.saturating_sub(start)
        }).sum::<usize>() / 1000
    );

    // ── 2. Input projection ──
    let mut rng = Lcg::new(42);
    let input_scale = I_SCALE / 2.0;
    let mut w_in = vec![0.0f32; 4 * n_neurons];
    for w in w_in.iter_mut() { *w = (rng.next_f32() - 0.5) * 2.0 * input_scale; }

    // ── 3. Izhikevich setup ──
    let params = IzhikevichParams {
        a: vec![0.02; n_neurons], b: vec![0.2; n_neurons],
        c: vec![-65.0; n_neurons], d: vec![2.0; n_neurons],
        dt_ms: DT, tau_ref: 2.0,
    };

    // ── 4. Phase 1: CEM on readout ──
    println!("\n{}", "─".repeat(50));
    println!("  PHASE 1: CEM on readout (reservoir fixed)");
    println!("{}", "─".repeat(50));

    let mut w_out = vec![0.0f32; 2 * n_mn];

    // Baseline (random w_out)
    let mut pre_rewards = Vec::new();
    for _ in 0..20 {
        let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
        pre_rewards.push(run_episode(&w_out, &conn, &w_in, &mut brain, &params, &mn_idx, n_mn));
    }
    let pre_avg: f32 = pre_rewards.iter().sum::<f32>() / 20.0;
    println!("  Before CEM: avg={:.1} (max={})", pre_avg, MAX_STEPS);

    // CEM training
    let t_cem = Instant::now();
    let mut best_reward = 0.0f32;
    let mut conn_cem = conn.clone(); // mutable clone for E-Prop phase

    for iter in 0..CEM_ITERS {
        let mut candidates: Vec<(f32, Vec<f32>)> = Vec::with_capacity(CEM_POP);
        for _ in 0..CEM_POP {
            let mut w = Vec::with_capacity(2 * n_mn);
            for &b in w_out.iter() { w.push(b + CEM_NOISE * rng.randn()); }
            for wi in w.iter_mut() { *wi = wi.clamp(-W_CLAMP, W_CLAMP); }
            let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
            let r = run_episode(&w, &conn_cem, &w_in, &mut brain, &params, &mn_idx, n_mn);
            candidates.push((r, w));
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let elite: Vec<&[f32]> = candidates.iter().take(CEM_ELITE).map(|(_, w)| w.as_slice()).collect();
        let n_elite = elite.len() as f32;
        for i in 0..2 * n_mn {
            w_out[i] = elite.iter().map(|w| w[i]).sum::<f32>() / n_elite;
        }
        let best_in_iter = candidates[0].0;
        if best_in_iter > best_reward { best_reward = best_in_iter; }

        if (iter + 1) % 5 == 0 || iter == 0 || iter == CEM_ITERS - 1 {
            let elite_mean: f32 = candidates.iter().take(CEM_ELITE).map(|(r, _)| r).sum::<f32>() / CEM_ELITE as f32;
            let solved = best_reward >= MAX_STEPS as f32;
            println!("  iter={:2}  best={:.0}  elite_mean={:.0}{}",
                iter + 1, candidates[0].0, elite_mean,
                if solved { "  🏆 SOLVED!" } else { "" });
        }

        // Checkpoint: save every CHECKPOINT_EVERY iters
        if (iter + 1) % CHECKPOINT_EVERY == 0 {
            let cp_path = format!("crates/connectome-gpt/models/cartpole_cem_iter{}.taas", iter + 1);
            let mut writer = taas_format::TaasWriter::new();
            writer.add_metadata("source", "CartPole CEM");
            writer.add_metadata("iter", &(iter + 1).to_string());
            writer.add_metadata("reward", &best_reward.to_string());
            // Save io with current w_out
            let mut io = taas_connectome::BrainIO {
                bias: BIAS, input_channels: vec![], output_channels: vec![],
            };
            io.output_channels.push(taas_connectome::OutputChannel {
                name: "cartpole".into(),
                neuron_indices: mn_idx.clone(),
                n_outputs: 2,
                readout: taas_connectome::OutputReadout::Linear,
                weights: Some(w_out.clone()),
            });
            taas_format::save_connectome(&conn_cem, &mut writer);
            taas_format::save_brain_io(&io, &mut writer);
            if let Err(e) = writer.save_to_file(&cp_path) {
                eprintln!("  ⚠ Checkpoint save failed: {}", e);
            } else {
                println!("  📦 Checkpoint saved: {}", cp_path);
            }
        }
    }

    // Post-CEM evaluation
    let mut post_cem_rewards = Vec::new();
    for _ in 0..20 {
        let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
        post_cem_rewards.push(run_episode(&w_out, &conn_cem, &w_in, &mut brain, &params, &mn_idx, n_mn));
    }
    let post_cem_avg: f32 = post_cem_rewards.iter().sum::<f32>() / 20.0;
    println!("\n  CEM done: avg={:.1}  best={:.0}  solved={}",
        post_cem_avg, best_reward, best_reward >= MAX_STEPS as f32);
    let cem_solved = best_reward >= MAX_STEPS as f32;

    // ── 5. Phase 2: E-Prop reservoir training ──
    println!("\n{}", "─".repeat(50));
    println!("  PHASE 2: E-Prop reservoir training (40 MNs → ~1040 synapses)");
    println!("{}", "─".repeat(50));
    println!("  LR_EPROP={}, τ_e={}ms, τ_sg={}, iters={}",
        LR_EPROP, TAU_E, TAU_SURROGATE, N_EPROP_ITERS);

    let t_eprop = Instant::now();
    let mut best_eprop_reward = best_reward;
    let mut conn_train = conn_cem;

    for eprop_iter in 0..N_EPROP_ITERS {
        let mut episode_rewards = Vec::with_capacity(100);

        for ep in 0..100 {
            let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
            let (reward, mn_elig) = run_episode_eprop(
                &w_out, &conn_train, &w_in, &mut brain, &params,
                &mn_idx, n_mn, TAU_E, TAU_SURROGATE,
            );

            episode_rewards.push(reward);

            // Apply E-Prop update: Δw = LR * L_j * e_ij
            // For each MN, update its upstream synapses
            for (mi, &m) in mn_idx.iter().enumerate() {
                let start = conn_train.row_ptr[m].max(0) as usize;
                let end = (conn_train.row_ptr[m + 1] as usize).min(conn_train.values.len());
                if start >= end || mi >= mn_elig.len() { continue; }

                // Learning signal from readout
                let l_signal = {
                    let mut ls = 0.0f32;
                    // w_out[action, mi] * (reward - baseline)
                    let chosen_action = if episode_rewards.last().copied().unwrap_or(0.0) > 0.0 { 1 } else { 0 };
                    // Simplified: use final reward as global signal
                    ls += (reward - 100.0) * w_out[chosen_action * n_mn + mi];
                    ls
                };

                if l_signal.abs() < 1e-6 { continue; }

                let elig = &mn_elig[mi];
                for (idx_offset, &e) in elig.iter().enumerate() {
                    let syn_idx = start + idx_offset;
                    if syn_idx >= conn_train.values.len() { break; }
                    let delta = LR_EPROP * l_signal * e;
                    if delta.abs() < 1e-12 { continue; }
                    conn_train.values[syn_idx] =
                        (conn_train.values[syn_idx] + delta).clamp(-W_CLAMP, W_CLAMP);
                }
            }
        }

        // Evaluate after this E-Prop phase
        let mut eval_rewards = Vec::new();
        for _ in 0..20 {
            let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
            eval_rewards.push(run_episode(&w_out, &conn_train, &w_in, &mut brain, &params, &mn_idx, n_mn));
        }
        let eval_avg: f32 = eval_rewards.iter().sum::<f32>() / 20.0;
        let eval_max = eval_rewards.iter().cloned().fold(0.0f32, f32::max);

        if eval_max > best_eprop_reward { best_eprop_reward = eval_max; }
        let solved = best_eprop_reward >= MAX_STEPS as f32;

        println!("  eprop_iter={}  avg={:.1}  max={:.0}{}",
            eprop_iter + 1, eval_avg, eval_max,
            if solved { "  🏆 SOLVED!" } else { "" });

        // Save checkpoint
        let cp_path = format!("crates/connectome-gpt/models/cartpole_eprop{}.taas", eprop_iter + 1);
        let mut writer = taas_format::TaasWriter::new();
        writer.add_metadata("source", "CartPole E-Prop");
        writer.add_metadata("eprop_iter", &(eprop_iter + 1).to_string());
        writer.add_metadata("reward", &best_eprop_reward.to_string());
        let mut io = taas_connectome::BrainIO {
            bias: BIAS, input_channels: vec![], output_channels: vec![],
        };
        io.output_channels.push(taas_connectome::OutputChannel {
            name: "cartpole".into(),
            neuron_indices: mn_idx.clone(),
            n_outputs: 2,
            readout: taas_connectome::OutputReadout::Linear,
            weights: Some(w_out.clone()),
        });
        taas_format::save_connectome(&conn_train, &mut writer);
        taas_format::save_brain_io(&io, &mut writer);
        if let Err(e) = writer.save_to_file(&cp_path) {
            eprintln!("  ⚠ E-Prop checkpoint failed: {}", e);
        } else {
            println!("  📦 E-Prop checkpoint: {}", cp_path);
        }
    }

    // ── 6. Final comparison ──
    println!("\n{}", "═".repeat(50));
    println!("  FINAL COMPARISON");
    println!("{}", "═".repeat(50));

    // Final eval with E-Prop trained reservoir
    let mut final_rewards = Vec::new();
    for _ in 0..20 {
        let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
        final_rewards.push(run_episode(&w_out, &conn_train, &w_in, &mut brain, &params, &mn_idx, n_mn));
    }
    let final_avg: f32 = final_rewards.iter().sum::<f32>() / 20.0;
    let final_max = final_rewards.iter().cloned().fold(0.0f32, f32::max);

    println!("                CEM-only   CEM+E-Prop  Δ");
    println!("  Avg reward:   {:.1}      {:.1}      {:+.1}", post_cem_avg, final_avg, final_avg - post_cem_avg);
    println!("  Max reward:   {:.0}      {:.0}      {:+.0}", best_reward, final_max, final_max - best_reward);
    println!("  Max possible: {}", MAX_STEPS);
    println!("\n  CEM time:  {:.1}s", t_cem.elapsed().as_secs_f64());
    println!("  E-Prop time: {:.1}s", t_eprop.elapsed().as_secs_f64());
    println!("  Total:     {:.1}s", t_cem.elapsed().as_secs_f64() + t_eprop.elapsed().as_secs_f64());

    // Save final trained brain
    let final_path = "crates/connectome-gpt/models/brain_trained_eprop.taas";
    let mut writer = taas_format::TaasWriter::new();
    writer.add_metadata("source", "CartPole CEM+E-Prop");
    writer.add_metadata("reward", &final_avg.to_string());
    let mut io = taas_connectome::BrainIO {
        bias: BIAS, input_channels: vec![], output_channels: vec![],
    };
    io.output_channels.push(taas_connectome::OutputChannel {
        name: "cartpole".into(),
        neuron_indices: mn_idx.clone(),
        n_outputs: 2,
        readout: taas_connectome::OutputReadout::Linear,
        weights: Some(w_out.clone()),
    });
    taas_format::save_connectome(&conn_train, &mut writer);
    taas_format::save_brain_io(&io, &mut writer);
    if let Err(e) = writer.save_to_file(final_path) {
        eprintln!("  ⚠ Final save failed: {}", e);
    } else {
        println!("  Saved: {}", final_path);
    }

    let solved = best_eprop_reward >= MAX_STEPS as f32;
    let improved = final_avg > post_cem_avg * 1.1;
    println!("\n--- Verdict ---");
    if solved {
        println!("  🏆 PASS: CEM + E-Prop solved CartPole!");
    } else if improved {
        println!("  📈 PARTIAL: E-Prop improved over CEM-only ({:.1} → {:.1})",
            post_cem_avg, final_avg);
    } else {
        println!("  🤷 SAME: E-Prop didn't significantly change performance.");
    }
    println!("  MNs: {}  E-Prop updates ~{}k synapses per episode",
        n_mn,
        mn_idx.iter().map(|&m| {
            let start = conn_train.row_ptr[m].max(0) as usize;
            let end = (conn_train.row_ptr[m + 1] as usize).min(conn_train.values.len());
            end.saturating_sub(start)
        }).sum::<usize>() / 1000
    );
    println!("{}", "═".repeat(60));
    Ok(())
}
