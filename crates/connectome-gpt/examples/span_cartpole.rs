//! SPAN CartPole: Izhikevich brain → 40 MN → left/right action.
//!
//! 1. Load connectome from .taas (or VAE → CSR if not found)
//! 2. CartPole state (4D) → random projection → input
//! 3. MN spike counts → 2 actions via linear readout
//! 4. Cross-Entropy Method (CEM) on readout weights (reservoir fixed!)
//! 5. Сравнение до/после обучения

use std::fs;
use std::time::Instant;
use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich};
use taas_connectome::ConnectomeData;

const TAAS_PATH: &str = "crates/connectome-gpt/models/brain.taas";

const N_STEPS: usize = 10;
const DT: f32 = 0.5;
const BIAS: f32 = 5.0;
const I_SCALE: f32 = 20.0;
const W_CLAMP: f32 = 5.0;

const CEM_POP: usize = 100;
const CEM_ELITE: usize = 20;
const CEM_ITERS: usize = 40;
const CEM_NOISE: f32 = 0.5;
const MAX_STEPS: usize = 200;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  SPAN CartPole: Izhikevich brain -> 40 MN -> left/right (CEM)");
    println!("{}", "═".repeat(60));

    // ── 1. Load connectome (from .taas or VAE) ──
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
    // include more MNs for CartPole: last 40 neurons
    let target_mn = 40;
    for i in (n_neurons - target_mn).max(0)..n_neurons {
        if !mn_idx.contains(&i) { mn_idx.push(i); }
    }
    let n_mn = mn_idx.len();
    println!("  MN indices ({}/{}): {:?}", n_mn, target_mn, mn_idx);

    // ── 2. Input projection: 4D state → n_neurons ──
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

    // ── 4. Baseline (random w_out) ──
    println!("\n--- Before training (random readout) ---");
    let mut w_out = vec![0.0f32; 2 * n_mn];
    let mut pre_rewards = Vec::new();
    for _ in 0..20 {
        let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
        pre_rewards.push(run_episode(&w_out, &conn, &w_in, &mut brain, &params, &mn_idx, n_mn));
    }
    let pre_avg: f32 = pre_rewards.iter().sum::<f32>() / 20.0;
    let pre_max = pre_rewards.iter().cloned().fold(0.0f32, f32::max);
    println!("  Avg reward: {:.1}  Max: {:.0}  (max: {})", pre_avg, pre_max, MAX_STEPS);

    // ── 5. CEM training ──
    println!("\n--- CEM training (pop={}, elite={}, iters={}) ---", CEM_POP, CEM_ELITE, CEM_ITERS);
    let t_train = Instant::now();
    let mut best_reward = 0.0f32;

    for iter in 0..CEM_ITERS {
        let mut candidates: Vec<(f32, Vec<f32>)> = Vec::with_capacity(CEM_POP);
        for _ in 0..CEM_POP {
            let mut w = Vec::with_capacity(2 * n_mn);
            for &b in w_out.iter() {
                w.push(b + CEM_NOISE * rng.randn());
            }
            for wi in w.iter_mut() { *wi = wi.clamp(-W_CLAMP, W_CLAMP); }
            let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
            let r = run_episode(&w, &conn, &w_in, &mut brain, &params, &mn_idx, n_mn);
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
            println!("  iter={:2}  best={:.0}  elite_mean={:.0}{}",
                iter + 1, candidates[0].0, elite_mean,
                if best_reward >= MAX_STEPS as f32 { "  SOLVED!" } else { "" });
        }
    }

    // ── 6. Post-training ──
    println!("\n--- After training ---");
    let mut post_rewards = Vec::new();
    for _ in 0..20 {
        let mut brain = BrainState::new(n_neurons, NeuronModel::Izhikevich);
        post_rewards.push(run_episode(&w_out, &conn, &w_in, &mut brain, &params, &mn_idx, n_mn));
    }
    let post_avg: f32 = post_rewards.iter().sum::<f32>() / 20.0;
    let post_max = post_rewards.iter().cloned().fold(0.0f32, f32::max);

    println!("\n--- Comparison ---");
    println!("                Before   After   Delta");
    println!("  Avg reward:  {:.1}    {:.1}   {:+.1}", pre_avg, post_avg, post_avg - pre_avg);
    println!("  Max reward:  {:.0}    {:.0}   {:+.0}", pre_max, post_max, post_max - pre_max);
    println!("  Max poss:    {}", MAX_STEPS);

    // Save trained brain to .taas
    let save_path = "crates/connectome-gpt/models/brain_trained.taas";
    let io = taas_format::load_brain_io(&taas_format::TaasReader::from_file(TAAS_PATH)?).unwrap_or(
        taas_connectome::BrainIO {
            bias: BIAS, input_channels: vec![], output_channels: vec![],
        }
    );
    let mut writer = taas_format::TaasWriter::new();
    writer.add_metadata("source", "CartPole CEM");
    writer.add_metadata("reward", &post_avg.to_string());
    // Save updated connectome (with trained weights unchanged — reservoir fixed)
    // Actually we need to save the ORIGINAL connectome since reservoir was fixed.
    // Read original: the loaded conn was cloned into conn_train but NOT modified.
    let reader = taas_format::TaasReader::from_file(TAAS_PATH)?;
    let orig_conn = taas_format::load_connectome(&reader)?;
    taas_format::save_connectome(&orig_conn, &mut writer);
    // Save updated w_out in IO metadata
    let mut io = io;
    if let Some(ch) = io.output_channels.first_mut() {
        ch.weights = Some(w_out.clone());
    }
    taas_format::save_brain_io(&io, &mut writer);
    writer.save_to_file(save_path)?;
    println!("  Saved trained brain to {}", save_path);

    let solved = best_reward >= MAX_STEPS as f32;
    let improved = post_avg > pre_avg * 1.2;
    println!("\n--- Verdict ---");
    if solved {
        println!("  PASS: CEM solved CartPole. MN patterns are informative enough!");
    } else if improved {
        println!("  PARTIAL: CEM improved ({:.0} -> {:.0}) but not solved.", pre_avg, post_avg);
    } else {
        println!("  FAIL: CEM did not help. MN patterns insufficient for CartPole control.");
    }
    println!("  CEM: pop={}, elite={}, iters={}", CEM_POP, CEM_ELITE, CEM_ITERS);
    println!("  Time: {:.1}s", t_train.elapsed().as_secs_f64());
    println!("{}", "═".repeat(60));
    Ok(())
}
