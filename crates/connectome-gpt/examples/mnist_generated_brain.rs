//! MNIST classification on a VAE-generated connectome (Liquid State Machine).
//!
//! 1. VAE → decoder → 800‑neuron connectome (SBM, E/I, groups, STDP weights)
//! 2. Each MNIST pixel → random projection → all 800 neurons as constant bias
//! 3. Izhikevich simulation (CSR propagation, 30 steps)
//! 4. Readout: spike counts × W_out → 10 classes (trained via SGD)
//!
//! Usage:
//!   python3 examples/python/download_mnist.py
//!   cargo run --release -p connectome-gpt --example mnist_generated_brain

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich};
use taas_connectome::ConnectomeData;

// ─── Constants ────────────────────────────────────────────

const N_STEPS: usize = 30;
const DT: f32 = 0.5; // ms
const I_BIAS: f32 = 7.0;
const I_SCALE: f32 = 30.0;
const LR_OUT: f32 = 0.01;
const W_CLAMP: f32 = 5.0;
const N_OUTPUT: usize = 10;

const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;
const BATCH_REPORT: usize = 500;
const DATA_DIR: &str = "examples/data";

// ─── IDX parser ──────────────────────────────────────────

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
        for p in 0..pixel_count {
            img.push(data[off + p] as f32 / 255.0);
        }
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

// ─── helpers ──────────────────────────────────────────────

fn fast_exp(x: f32) -> f32 {
    let log2e = 1.4426950408889634f32;
    let t = x * log2e;
    let n = t.floor() as i32;
    let f = t - n as f32;
    let c1 = 0.6930321207;
    let c2 = 0.2413797629;
    let c3 = 0.0520972777;
    let c4 = 0.0093553389;
    let poly = ((c4 * f + c3) * f + c2) * f + c1;
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

// simple LCG
struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
}

// ─── Main ─────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  VAE-Generated Connectome → MNIST (Liquid State Machine)");
    println!("{}", "═".repeat(60));

    // 1. Generate brain from VAE
    println!("\n─── Generating connectome from VAE ───");
    let t_gen = Instant::now();
    let weights = fs::read("crates/connectome-gpt/models/vae_weights.bin")?;
    let norm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("crates/connectome-gpt/models/vae_norm.json")?)?;

    let mut vae = connectome_gpt::GraphVAE::new(connectome_gpt::VaeConfig {
        input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42,
    });
    vae.load(&weights);

    let mut raw = vae.generate();
    // unnormalize
    {
        let m: Vec<f32> = norm["mean"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let s: Vec<f32> = norm["std"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        for (x, (mv, sv)) in raw.iter_mut().zip(m.iter().zip(s.iter())) { *x = *x * sv + mv; }
    }

    println!("  VAE features: n={:.0} syn={:.0} dens={:.2}% deg={:.1} ei={:.1}",
        raw[0], raw[1], raw[2], raw[3], raw[5]);

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
        seed: 42,
    };
    let mut decoder = connectome_gpt::CsrDecoder::new(cfg);
    let sample = decoder.generate(&raw);
    println!("  Connectome: {}n {}syn ({:.1}/n) density={:.2}%",
        sample.n_neurons, sample.n_synapses,
        sample.n_synapses as f32 / sample.n_neurons as f32,
        sample.n_synapses as f64 / (sample.n_neurons as f64).powi(2) * 100.0);
    println!("  Groups: {:?}", sample.groups.keys().map(|k| (k.as_str(), sample.groups[k].len())).collect::<Vec<_>>());
    println!("  Gen time: {:.3}s", t_gen.elapsed().as_secs_f64());

    let n_neurons = sample.n_neurons;

    // 2. Convert to ConnectomeData
    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    let conn = ConnectomeData::new(
        sample.row_ptr.clone(), sample.col_indices.clone(),
        sample.values.clone(), sample.nt_types.clone(),
        groups,
    );

    // 3. Random input projection: 784 pixels → all n_neurons
    let ni = 784;
    let mut rng = Lcg::new(123);
    let input_scale = I_SCALE / (ni as f32).sqrt();
    let mut w_in = vec![0.0f32; ni * n_neurons];
    for w in w_in.iter_mut() {
        *w = (rng.next_f32() - 0.5) * 2.0 * input_scale;
    }

    // 4. Output weights: all neurons → 10 classes
    let mut w_out = vec![0.0f32; N_OUTPUT * n_neurons];

    // 5. Simulation state (reused)
    let mut state = BrainState::new(n_neurons, NeuronModel::Izhikevich);
    state.reset_with_noise(n_neurons);
    let params = IzhikevichParams {
        a: vec![0.02; n_neurons],
        b: vec![0.2; n_neurons],
        c: vec![-65.0; n_neurons],
        d: vec![2.0; n_neurons],
        dt_ms: DT, tau_ref: 2.0,
    };

    // 6. Load MNIST
    println!("\n─── Loading MNIST ───");
    check_file(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    check_file(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    check_file(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    check_file(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));

    let train_images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let train_labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let test_images = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let test_labels = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));
    println!("  {} train, {} test", train_images.len(), test_images.len());

    // 7. Reservoir forward pass: returns spike counts [n_neurons]
    fn reservoir_forward(
        img: &[f32],
        w_in: &[f32],
        conn: &ConnectomeData,
        state: &mut BrainState,
        params: &IzhikevichParams,
        n_neurons: usize,
        n_steps: usize,
    ) -> Vec<f32> {
        // Compute external current from image
        let mut ext_I = vec![I_BIAS; n_neurons];
        for j in 0..n_neurons {
            let mut sum = 0.0f32;
            for i in 0..img.len() {
                sum += img[i] * w_in[i * n_neurons + j];
            }
            ext_I[j] += sum;
        }

        // Run simulation
        let spike_buf = sim_izhikevich(
            conn, state, params, &ext_I, n_steps,
        );

        // Count spikes per neuron
        let mut counts = vec![0.0f32; n_neurons];
        for step in 0..n_steps {
            let base = step * n_neurons;
            for j in 0..n_neurons {
                if spike_buf[base + j] != 0 {
                    counts[j] += 1.0;
                }
            }
        }
        counts
    }

    // 8. Train
    println!("\n─── Training (reservoir: fixed, readout: SGD) ───");
    let t_train = Instant::now();
    let mut total_correct = 0usize;
    let mut total_loss = 0.0f32;

    for img_idx in 0..N_TRAIN {
        // Forward through reservoir
        let spike_counts = reservoir_forward(
            &train_images[img_idx], &w_in, &conn, &mut state, &params,
            n_neurons, N_STEPS,
        );

        // Output: logits_k = Σ_j w_out[k, j] · spike_counts[j]
        let label = train_labels[img_idx] as usize;
        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let row_off = k * n_neurons;
            let mut sum = 0.0f32;
            for j in 0..n_neurons {
                sum += w_out[row_off + j] * spike_counts[j];
            }
            logits[k] = sum;
        }

        // Softmax + cross-entropy
        let probs = softmax(&logits);
        let loss = -(probs[label].max(1e-10)).ln();

        // Error gradient
        let mut error = vec![0.0f32; N_OUTPUT];
        error[label] = 1.0;
        for k in 0..N_OUTPUT { error[k] -= probs[k]; }

        // Update output weights
        for k in 0..N_OUTPUT {
            let row_off = k * n_neurons;
            let scale = LR_OUT * error[k];
            if scale.abs() > 1e-10 {
                for j in 0..n_neurons {
                    w_out[row_off + j] += scale * spike_counts[j];
                }
            }
        }
        for w in w_out.iter_mut() { *w = w.clamp(-W_CLAMP, W_CLAMP); }

        // Stats
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let correct = pred == label;
        if correct { total_correct += 1; }
        total_loss += loss;

        if (img_idx + 1) % BATCH_REPORT == 0 {
            let acc = total_correct as f64 / (img_idx + 1) as f64 * 100.0;
            let avg_loss = total_loss / (img_idx + 1) as f32;
            let elapsed = t_train.elapsed().as_secs_f64();
            let ips = (img_idx + 1) as f64 / elapsed;
            println!("  [{:5}/{}] loss={:.4} acc={:.1}% | {:.0} img/s | {:.1}s",
                img_idx + 1, N_TRAIN, avg_loss, acc, ips, elapsed);
        }
    }

    let train_acc = total_correct as f64 / N_TRAIN as f64 * 100.0;
    println!("\n─── Training Done ───");
    println!("  Train accuracy: {train_acc:.1}% ({total_correct}/{N_TRAIN})");

    // 9. Test
    println!("\n─── Testing ───");
    let t_test = Instant::now();
    let mut test_correct = 0usize;
    let mut pred_dist = [0usize; N_OUTPUT];

    for img_idx in 0..N_TEST {
        let spike_counts = reservoir_forward(
            &test_images[img_idx], &w_in, &conn, &mut state, &params,
            n_neurons, N_STEPS,
        );
        let label = test_labels[img_idx] as usize;
        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let row_off = k * n_neurons;
            let mut sum = 0.0f32;
            for j in 0..n_neurons {
                sum += w_out[row_off + j] * spike_counts[j];
            }
            logits[k] = sum;
        }
        let probs = softmax(&logits);
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        if pred == label { test_correct += 1; }
        pred_dist[pred] += 1;
    }

    let test_acc = test_correct as f64 / N_TEST as f64 * 100.0;
    println!("  Test accuracy: {test_acc:.1}% ({test_correct}/{N_TEST})");
    println!("  Test time: {:.1}s ({:.0} img/s)",
        t_test.elapsed().as_secs_f64(),
        N_TEST as f64 / t_test.elapsed().as_secs_f64());
    println!("\n  Prediction distribution:");
    for k in 0..N_OUTPUT {
        println!("    {k}: {} ({:.1}%)", pred_dist[k],
            pred_dist[k] as f64 / N_TEST as f64 * 100.0);
    }

    println!("\n─── Summary ───");
    println!("  Reservoir: VAE-generated connectome ({}n, {}syn)",
        n_neurons, sample.n_synapses);
    println!("  Neuron:    Izhikevich, {} steps × {}ms", N_STEPS, DT);
    println!("  Readout:   {}→10 linear + SGD (lr={})", n_neurons, LR_OUT);
    println!("  Train acc: {train_acc:.1}%");
    println!("  Test acc:  {test_acc:.1}%");
    println!("{}", "═".repeat(60));
    Ok(())
}
