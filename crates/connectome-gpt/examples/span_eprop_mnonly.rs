//! SPAN MN-only: раздражитель → MN спайки → классификация.
//!
//! Readout ТОЛЬКО с 4 MNs (не 800). E-Prop учит reservoir так,
//! чтобы разные цифры давали разные MN паттерны.
//!
//! 1. VAE → CSR → Izhikevich + eligibility
//! 2. Readout: 4 MNs → 10 классов (не 800→10!)
//! 3. E-Prop: L_j только для MNs → веса к MNs учатся
//! 4. Сравнение MN паттернов до/после

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich_eprop};
use taas_connectome::ConnectomeData;

const N_STEPS: usize = 30;
const DT: f32 = 0.5;
const I_BIAS: f32 = 7.0;
const I_SCALE: f32 = 30.0;
const LR_OUT: f32 = 0.05;
const LR_EPROP: f32 = 0.005;
const W_CLAMP: f32 = 5.0;
const N_OUTPUT: usize = 10;
const TAU_E: f32 = 20.0;
const TAU_SURROGATE: f32 = 5.0;

const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;
const BATCH_REPORT: usize = 2000;
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

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    dot / ((na * nb).sqrt() + 1e-10)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  SPAN MN-only: E-Prop только на 4 MNs");
    println!("  Классификация только по MN спайкам");
    println!("{}", "═".repeat(60));

    // ── 1. VAE → connectome ──
    println!("\n─── Generating connectome from VAE ───");
    let t0 = Instant::now();
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

    let n_neurons = sample.n_neurons;
    let n_syn = sample.values.len();
    let mn_idx: Vec<usize> = sample.groups.get("mn").cloned().unwrap_or_else(||
        ((n_neurons - 4).max(0)..n_neurons).collect());
    println!("  {}n {}syn | MN: {:?}", n_neurons, n_syn, mn_idx);

    // Convert to ConnectomeData
    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    let conn = ConnectomeData::new(
        sample.row_ptr.clone(), sample.col_indices.clone(),
        sample.values.clone(), sample.nt_types.clone(),
        groups,
    );

    // ── 2. Input projection (784 → all neurons) ──
    let ni = 784;
    let mut rng = Lcg::new(123);
    let input_scale = I_SCALE / (ni as f32).sqrt();
    let mut w_in = vec![0.0f32; ni * n_neurons];
    for w in w_in.iter_mut() { *w = (rng.next_f32() - 0.5) * 2.0 * input_scale; }

    // ── 3. Readout: ONLY 4 MNs → 10 classes ──
    let n_mn = mn_idx.len();
    let mut w_out = vec![0.0f32; N_OUTPUT * n_mn];

    // ── 4. Izhikevich setup ──
    let mut state = BrainState::new(n_neurons, NeuronModel::Izhikevich);
    let params = IzhikevichParams {
        a: vec![0.02; n_neurons],
        b: vec![0.2; n_neurons],
        c: vec![-65.0; n_neurons],
        d: vec![2.0; n_neurons],
        dt_ms: DT, tau_ref: 2.0,
    };

    // ── 5. Load MNIST ──
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

    // ── 6. Measure MN patterns (baseline) ──
    fn measure_patterns(
        conn: &ConnectomeData, state: &mut BrainState, params: &IzhikevichParams,
        w_in: &[f32], n_neurons: usize, mn_idx: &[usize],
        images: &[Vec<f32>], labels: &[u8], n_samples: usize,
    ) -> Vec<Vec<f32>> {
        let mut patterns = vec![vec![0.0f32; mn_idx.len()]; 10];
        let mut counts = vec![0usize; 10];
        for idx in 0..n_samples {
            let label = labels[idx] as usize;
            if counts[label] >= 10 { continue; }
            state.reset_with_noise(n_neurons);

            let mut ext_I = vec![I_BIAS; n_neurons];
            for j in 0..n_neurons {
                let mut sum = 0.0f32;
                for i in 0..images[idx].len() { sum += images[idx][i] * w_in[i * n_neurons + j]; }
                ext_I[j] += sum;
            }

            let spike_buf = taas_connectome::sim::sim_izhikevich(conn, state, params, &ext_I, N_STEPS);
            for (mi, &m) in mn_idx.iter().enumerate() {
                let mut cnt = 0.0;
                for step in 0..N_STEPS { if spike_buf[step * n_neurons + m] != 0 { cnt += 1.0; } }
                patterns[label][mi] += cnt;
            }
            counts[label] += 1;
        }
        for d in 0..10 {
            if counts[d] > 0 { let inv = 1.0 / counts[d] as f32; for mi in 0..mn_idx.len() { patterns[d][mi] *= inv; } }
        }
        patterns
    }

    println!("\n─── MN patterns BEFORE training ───");
    let pat_before = measure_patterns(
        &conn, &mut state.clone(), &params, &w_in, n_neurons, &mn_idx,
        &train_images, &train_labels, 200,
    );

    println!("  Digit → [4 MN avg spikes]:");
    for (d, p) in pat_before.iter().enumerate() {
        println!("    {}: [{:.2} {:.2} {:.2} {:.2}]", d, p[0], p[1], p[2], p[3]);
    }

    let cross_before: f32 = (0..10).flat_map(|a| (0..10).filter(move |&b| b != a).map(move |b| (a, b)))
        .map(|(a, b)| cosine_sim(&pat_before[a], &pat_before[b])).sum::<f32>() / 90.0;
    let margin_before: f32 = 1.0 - cross_before;
    println!("  Cross-sim: {:.4}  Margin: {:.4}", cross_before, margin_before);

    // ── 7. Helper: run E-Prop forward, return counts + eligibility ──
    fn eprop_forward(
        img: &[f32], w_in: &[f32], conn: &ConnectomeData,
        state: &mut BrainState, params: &IzhikevichParams,
        n_neurons: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut ext_I = vec![I_BIAS; n_neurons];
        for j in 0..n_neurons {
            let mut sum = 0.0f32;
            for i in 0..img.len() { sum += img[i] * w_in[i * n_neurons + j]; }
            ext_I[j] += sum;
        }
        let (spike_buf, eligibility) = sim_izhikevich_eprop(conn, state, params, &ext_I, N_STEPS, TAU_E, TAU_SURROGATE);
        let mut counts = vec![0.0f32; n_neurons];
        for step in 0..N_STEPS {
            let base = step * n_neurons;
            for j in 0..n_neurons { if spike_buf[base + j] != 0 { counts[j] += 1.0; } }
        }
        (counts, eligibility)
    }

    // ── 8. Training loop ──
    println!("\n─── E-Prop training (readout: 4 MN → 10, reservoir: E-Prop) ───");
    let t_train = Instant::now();
    let mut total_correct = 0usize;
    let mut total_loss = 0.0f32;
    let mut conn_train = conn.clone();

    for img_idx in 0..N_TRAIN {
        let label = train_labels[img_idx] as usize;

        // Forward + eligibility
        state.reset_with_noise(n_neurons);
        let (counts, eligibility) = eprop_forward(
            &train_images[img_idx], &w_in, &conn_train, &mut state, &params, n_neurons,
        );

        // Readout: ONLY from MNs
        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let row_off = k * n_mn;
            let mut sum = 0.0f32;
            for (mi, &m) in mn_idx.iter().enumerate() { sum += w_out[row_off + mi] * counts[m]; }
            logits[k] = sum;
        }
        let probs = softmax(&logits);
        let loss = -(probs[label].max(1e-10)).ln();

        let mut error = vec![0.0f32; N_OUTPUT];
        error[label] = 1.0;
        for k in 0..N_OUTPUT { error[k] -= probs[k]; }

        let pred = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        let correct = pred == label;
        if correct { total_correct += 1; }
        total_loss += loss;

        // ── Update w_out (MN → 10) ──
        for k in 0..N_OUTPUT {
            let row_off = k * n_mn;
            let scale = LR_OUT * error[k];
            if scale.abs() > 1e-10 {
                for (mi, &m) in mn_idx.iter().enumerate() { w_out[row_off + mi] += scale * counts[m]; }
            }
        }
        for w in w_out.iter_mut() { *w = w.clamp(-W_CLAMP, W_CLAMP); }

        // ── E-Prop: update reservoir via MN learning signal ──
        // L_j for non-MN = 0 (no readout weight → no signal)
        // L_j for MNs = Σ_k error[k] * w_out[k, mi]
        let max_err = error.iter().map(|e| e.abs()).fold(0.0f32, f32::max);
        if max_err > 0.01 {
            let mut l_signal = vec![0.0f32; n_neurons];
            for (mi, &m) in mn_idx.iter().enumerate() {
                let mut ls = 0.0f32;
                for k in 0..N_OUTPUT { ls += error[k] * w_out[k * n_mn + mi]; }
                l_signal[m] = ls;
            }

            // Apply E-Prop: update synapses → MN and synapses → other (if L_j != 0)
            for j in 0..n_neurons {
                let ls = l_signal[j];
                if ls.abs() < 1e-6 { continue; }
                let start = conn_train.row_ptr[j].max(0) as usize;
                let end = (conn_train.row_ptr[j + 1] as usize).min(conn_train.values.len());
                for idx in start..end {
                    let delta = -LR_EPROP * ls * eligibility[idx];
                    if delta.abs() < 1e-12 { continue; }
                    conn_train.values[idx] = (conn_train.values[idx] + delta).clamp(-W_CLAMP, W_CLAMP);
                }
            }
        }

        if (img_idx + 1) % BATCH_REPORT == 0 {
            let acc = total_correct as f64 / (img_idx + 1) as f64 * 100.0;
            let avg_loss = total_loss / (img_idx + 1) as f32;
            let ips = (img_idx + 1) as f64 / t_train.elapsed().as_secs_f64();
            println!("  [{:5}/{}] loss={:.4} acc={:.1}% | {:.0} img/s",
                img_idx + 1, N_TRAIN, avg_loss, acc, ips);
        }
    }

    let train_acc = total_correct as f64 / N_TRAIN as f64 * 100.0;
    println!("\n─── Training done ───");
    println!("  Train accuracy: {:.1}% ({}/{})", train_acc, total_correct, N_TRAIN);

    // ── 9. Test ──
    println!("\n─── Testing ───");
    let mut test_correct = 0usize;
    for img_idx in 0..N_TEST {
        state.reset_with_noise(n_neurons);
        let (counts, _) = eprop_forward(
            &test_images[img_idx], &w_in, &conn_train, &mut state, &params, n_neurons,
        );

        let mut logits = vec![0.0f32; N_OUTPUT];
        for k in 0..N_OUTPUT {
            let row_off = k * n_mn;
            let mut sum = 0.0f32;
            for (mi, &m) in mn_idx.iter().enumerate() { sum += w_out[row_off + mi] * counts[m]; }
            logits[k] = sum;
        }
        let probs = softmax(&logits);
        let pred = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        if pred == test_labels[img_idx] as usize { test_correct += 1; }
    }
    let test_acc = test_correct as f64 / N_TEST as f64 * 100.0;
    println!("  Test accuracy: {:.1}% ({}/{})", test_acc, test_correct, N_TEST);

    // ── 10. MN patterns AFTER training ──
    println!("\n─── MN patterns AFTER training ───");
    let pat_after = measure_patterns(
        &conn_train, &mut state, &params, &w_in, n_neurons, &mn_idx,
        &train_images, &train_labels, 200,
    );

    println!("  Digit → [4 MN avg spikes]:");
    for (d, p) in pat_after.iter().enumerate() {
        println!("    {}: [{:.2} {:.2} {:.2} {:.2}]", d, p[0], p[1], p[2], p[3]);
    }

    let cross_after: f32 = (0..10).flat_map(|a| (0..10).filter(move |&b| b != a).map(move |b| (a, b)))
        .map(|(a, b)| cosine_sim(&pat_after[a], &pat_after[b])).sum::<f32>() / 90.0;
    let margin_after: f32 = 1.0 - cross_after;

    // Print similarity matrix
    println!("\n  Cosine sim matrix (after):");
    println!("        {}", (0..10).map(|d| format!("{:4}", d)).collect::<Vec<_>>().join(" "));
    for a in 0..10 {
        let row: Vec<String> = (0..10).map(|b| format!("{:.2}", cosine_sim(&pat_after[a], &pat_after[b]))).collect();
        println!("    {}: {}", a, row.join(" "));
    }

    println!("\n─── Comparison ───");
    println!("                    Before    After     Δ");
    println!("  Cross-sim:       {:.4}    {:.4}    {:+.4}", cross_before, cross_after, cross_after - cross_before);
    println!("  Margin:          {:.4}    {:.4}    {:+.4}", margin_before, margin_after, margin_after - margin_before);

    // Nearest-template accuracy
    let n_template = 10.min(N_TEST);
    let mut template_correct = 0usize;
    for img_idx in 0..n_template {
        state.reset_with_noise(n_neurons);
        let (counts, _) = eprop_forward(
            &test_images[img_idx], &w_in, &conn_train, &mut state, &params, n_neurons,
        );
        let mn_vec: Vec<f32> = mn_idx.iter().map(|&m| counts[m]).collect();
        let pred = pat_after.iter().enumerate()
            .max_by(|(_, a), (_, b)| cosine_sim(a, &mn_vec).partial_cmp(&cosine_sim(b, &mn_vec)).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == test_labels[img_idx] as usize { template_correct += 1; }
    }
    println!("  Nearest-template accuracy: {}/{} = {:.1}%",
        template_correct, n_template, template_correct as f64 / n_template as f64 * 100.0);

    println!("\n─── Summary ───");
    println!("  Reservoir: VAE connectome ({}n, {}syn)", n_neurons, n_syn);
    println!("  Readout: 4 MNs → 10 (LR_OUT={})", LR_OUT);
    println!("  E-Prop: LR={}, τ_e={}ms, τ_sg={}", LR_EPROP, TAU_E, TAU_SURROGATE);
    println!("  Train acc: {:.1}%", train_acc);
    println!("  Test acc:  {:.1}%", test_acc);
    println!("  Time: {:.1}s", t_train.elapsed().as_secs_f64());
    println!("{}", "═".repeat(60));
    Ok(())
}
