//! Ablation: SBM vs no-SBM на VAE-коннектоме → MNIST.
//!
//! Генерируем 2 коннектома (один с community_strength=8, второй с strength=1)
//! и сравниваем accuracy на 10K train / 10K test.

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich};
use taas_connectome::ConnectomeData;

const N_STEPS: usize = 30;
const DT: f32 = 0.5;
const I_BIAS: f32 = 7.0;
const I_SCALE: f32 = 30.0;
const LR_OUT: f32 = 0.01;
const W_CLAMP: f32 = 5.0;
const N_OUTPUT: usize = 10;
const N_TRAIN: usize = 10_000;
const N_TEST: usize = 10_000;
const DATA_DIR: &str = "examples/data";

fn load_images(path: &str) -> Vec<Vec<f32>> {
    let data = fs::read(path).unwrap();
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let pixel_count = rows * cols;
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let off = 16 + i * pixel_count;
        let mut img = Vec::with_capacity(pixel_count);
        for p in 0..pixel_count { img.push(data[off + p] as f32 / 255.0); }
        images.push(img);
    }
    images
}

fn load_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).unwrap();
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    data[8..8 + n].to_vec()
}

fn check_file(path: &str) {
    if Path::new(path).exists() { return; }
    panic!("Missing: {path}. Run: python3 examples/python/download_mnist.py");
}

fn fast_exp(x: f32) -> f32 {
    let log2e = 1.4426950408889634f32;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;
    let c = [0.6930321207, 0.2413797629, 0.0520972777, 0.0093553389];
    let poly = ((c[3] * f + c[2]) * f + c[1]) * f + c[0];
    let exp2f = poly * f + 1.0;
    let biased = n.wrapping_add(127);
    let exp2n = if biased <= 0 { 0.0 }
        else if biased >= 255 { f32::INFINITY }
        else { f32::from_bits((biased as u32) << 23) };
    exp2n * exp2f
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter()
        .map(|&x| fast_exp((x - max_v).max(-50.0).min(50.0))).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 1e-20 { for e in &mut exps { *e /= sum; } }
    exps
}

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
}

fn run_mnist(
    sample: &connectome_gpt::ConnectomeSample,
    train_images: &[Vec<f32>],
    train_labels: &[u8],
    test_images: &[Vec<f32>],
    test_labels: &[u8],
    label: &str,
) -> (f64, f64) {
    let n_neurons = sample.n_neurons;
    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    let conn = ConnectomeData::new(
        sample.row_ptr.clone(), sample.col_indices.clone(),
        sample.values.clone(), sample.nt_types.clone(), groups,
    );

    // Input projection (deterministic seed — same for both)
    let ni = 784;
    let mut rng = Lcg::new(123);
    let input_scale = I_SCALE / (ni as f32).sqrt();
    let mut w_in = vec![0.0f32; ni * n_neurons];
    for w in w_in.iter_mut() { *w = (rng.next_f32() - 0.5) * 2.0 * input_scale; }

    let mut w_out = vec![0.0f32; N_OUTPUT * n_neurons];
    let mut state = BrainState::new(n_neurons, NeuronModel::Izhikevich);
    state.reset_with_noise(n_neurons);
    let params = IzhikevichParams {
        a: vec![0.02; n_neurons], b: vec![0.2; n_neurons],
        c: vec![-65.0; n_neurons], d: vec![2.0; n_neurons],
        dt_ms: DT, tau_ref: 2.0,
    };

    fn forward(
        img: &[f32], w_in: &[f32], conn: &ConnectomeData,
        state: &mut BrainState, params: &IzhikevichParams,
        n: usize, steps: usize,
    ) -> Vec<f32> {
        let mut ext = vec![I_BIAS; n];
        for j in 0..n {
            let mut s = 0.0f32;
            for i in 0..img.len() { s += img[i] * w_in[i * n + j]; }
            ext[j] += s;
        }
        let buf = sim_izhikevich(conn, state, params, &ext, steps);
        let mut counts = vec![0.0f32; n];
        for step in 0..steps {
            let base = step * n;
            for j in 0..n { if buf[base + j] != 0 { counts[j] += 1.0; } }
        }
        counts
    }

    // Train
    let t0 = Instant::now();
    let mut correct = 0usize;
    for idx in 0..N_TRAIN {
        let spikes = forward(&train_images[idx], &w_in, &conn, &mut state, &params, n_neurons, N_STEPS);
        let label = train_labels[idx] as usize;
        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let ro = k * n_neurons;
            let mut s = 0.0f32;
            for j in 0..n_neurons { s += w_out[ro + j] * spikes[j]; }
            logits[k] = s;
        }
        let probs = softmax(&logits);
        let mut error = vec![0.0f32; N_OUTPUT];
        error[label] = 1.0;
        for k in 0..N_OUTPUT { error[k] -= probs[k]; }
        for k in 0..N_OUTPUT {
            let ro = k * n_neurons;
            let scale = LR_OUT * error[k];
            if scale.abs() > 1e-10 {
                for j in 0..n_neurons { w_out[ro + j] += scale * spikes[j]; }
            }
        }
        for w in w_out.iter_mut() { *w = w.clamp(-W_CLAMP, W_CLAMP); }
        let pred = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        if pred == label { correct += 1; }
    }
    let train_acc = correct as f64 / N_TRAIN as f64 * 100.0;
    let train_t = t0.elapsed();

    // Test
    let t1 = Instant::now();
    correct = 0;
    for idx in 0..N_TEST {
        let spikes = forward(&test_images[idx], &w_in, &conn, &mut state, &params, n_neurons, N_STEPS);
        let label = test_labels[idx] as usize;
        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let ro = k * n_neurons;
            let mut s = 0.0f32;
            for j in 0..n_neurons { s += w_out[ro + j] * spikes[j]; }
            logits[k] = s;
        }
        let probs = softmax(&logits);
        let pred = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        if pred == label { correct += 1; }
    }
    let test_acc = correct as f64 / N_TEST as f64 * 100.0;
    let test_t = t1.elapsed();

    println!("  {label}: train={train_acc:.1}% test={test_acc:.1}%  ({:.1}s train, {:.1}s test)",
        train_t.as_secs_f64(), test_t.as_secs_f64());
    (train_acc, test_acc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  ABLATION: SBM vs no-SBM  (10K train / 10K test)");
    println!("{}", "═".repeat(60));

    // ── Load VAE ──
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
    println!("\nVAE features: n={:.0} syn={:.0} dens={:.2}% deg={:.1} ei={:.1}",
        raw[0], raw[1], raw[2], raw[3], raw[5]);

    // ── Generate 3 connectomes ──
    let base_cfg = || -> connectome_gpt::DecoderConfig {
        connectome_gpt::DecoderConfig {
            n_neurons: raw[0].round() as usize,
            density: raw[2] / 100.0,
            ei_ratio: raw[5].max(1.0),
            mean_weight: (raw[6] / 100.0).max(0.1),
            weight_noise: 0.05,
            n_communities: raw[8].round() as usize,
            community_strength: 8.0,
            sensory_fraction: 0.05, dn_fraction: 0.02, mn_fraction: 0.005,
            stdp_weights: true, compute_depth: true, refine_epochs: 0, seed: 42,
        }
    };

    // 1) Full: SBM + groups + depth + STDP
    let sample_full = {
        let mut d = connectome_gpt::CsrDecoder::new(base_cfg());
        d.generate(&raw)
    };
    println!("\n[Full]    {}n {}syn ({:.1}/n)  SBM+groups+depth+STDP",
        sample_full.n_neurons, sample_full.n_synapses,
        sample_full.n_synapses as f32 / sample_full.n_neurons as f32);

    // 2) No groups: groups=0
    let sample_nogroups = {
        let cfg = connectome_gpt::DecoderConfig {
            sensory_fraction: 0.0, dn_fraction: 0.0, mn_fraction: 0.0,
            ..base_cfg()
        };
        let mut d = connectome_gpt::CsrDecoder::new(cfg);
        d.generate(&raw)
    };
    println!("[NoGroup] {}n {}syn ({:.1}/n)  groups=0",
        sample_nogroups.n_neurons, sample_nogroups.n_synapses,
        sample_nogroups.n_synapses as f32 / sample_nogroups.n_neurons as f32);

    // 3) Random: no SBM, no groups, no depth, uniform weights
    let sample_random = {
        let cfg = connectome_gpt::DecoderConfig {
            community_strength: 1.0,
            sensory_fraction: 0.0, dn_fraction: 0.0, mn_fraction: 0.0,
            stdp_weights: false, compute_depth: false,
            ..base_cfg()
        };
        let mut d = connectome_gpt::CsrDecoder::new(cfg);
        d.generate(&raw)
    };
    println!("[Random]  {}n {}syn ({:.1}/n)  no SBM, groups, depth, uniform weights",
        sample_random.n_neurons, sample_random.n_synapses,
        sample_random.n_synapses as f32 / sample_random.n_neurons as f32);

    // ── Load MNIST ──
    println!("\n─── Loading MNIST ───");
    check_file(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    check_file(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    check_file(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    check_file(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));
    let ti = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let tl = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let ei = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let el = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));
    println!("  {} train, {} test", ti.len(), ei.len());

    // ── Run all 3 ──
    println!("\n─── MNIST Results (10K train, 10K test) ───");
    let (full_tr, full_te) = run_mnist(&sample_full, &ti, &tl, &ei, &el, "Full   ");
    let (ng_tr, ng_te) = run_mnist(&sample_nogroups, &ti, &tl, &ei, &el, "NoGroup");
    let (rnd_tr, rnd_te) = run_mnist(&sample_random, &ti, &tl, &ei, &el, "Random ");

    println!("\n─── Comparison ───");
    println!("  Full:    train={full_tr:.1}%  test={full_te:.1}%  (SBM+groups+depth+STDP)");
    println!("  NoGroup: train={ng_tr:.1}%  test={ng_te:.1}%  (only SBM+depth+STDP)");
    println!("  Random:  train={rnd_tr:.1}%  test={rnd_te:.1}%  (no structure, same density/EI)");
    println!("  Full − Random: +{:.1}% train  +{:.1}% test",
        full_tr - rnd_tr, full_te - rnd_te);
    println!("{}", "═".repeat(60));
    Ok(())
}
