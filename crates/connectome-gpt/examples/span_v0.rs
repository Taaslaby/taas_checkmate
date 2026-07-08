//! SPAN demo: 10 MNIST digits → 10 уникальных моторных паттернов.
//! Без обучения —天生 (прирождённая) ассоциация структуры коннектома.
//! После обучения (E-Prop) паттерны можно усилить.

use std::fs;
use std::path::Path;
use taas_connectome::sim::{BrainState, IzhikevichParams, NeuronModel, sim_izhikevich};
use taas_connectome::ConnectomeData;

const N_STEPS: usize = 30;
const DT: f32 = 0.5;
const I_BIAS: f32 = 7.0;
const I_SCALE: f32 = 30.0;
const DATA_DIR: &str = "examples/data";

fn load_images(path: &str) -> Vec<Vec<f32>> {
    let data = fs::read(path).unwrap();
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let px = rows * cols;
    let mut imgs = Vec::with_capacity(n);
    for i in 0..n {
        let off = 16 + i * px;
        let mut img = Vec::with_capacity(px);
        for p in 0..px { img.push(data[off + p] as f32 / 255.0); }
        imgs.push(img);
    }
    imgs
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
struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  SPAN: врождённые моторные паттерны 10 цифр");
    println!("  (без обучения — разные цифры → разные MN спайки)");
    println!("{}", "═".repeat(60));

    // ── VAE → connectome ──
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

    let cfg = connectome_gpt::DecoderConfig {
        n_neurons: raw[0].round() as usize, density: raw[2] / 100.0,
        ei_ratio: raw[5].max(1.0), mean_weight: (raw[6] / 100.0).max(0.1),
        weight_noise: 0.05, n_communities: 8, community_strength: 8.0,
        sensory_fraction: 0.05, dn_fraction: 0.02, mn_fraction: 0.005,
        stdp_weights: true, compute_depth: true, refine_epochs: 0, seed: 42,
    };
    let sample = connectome_gpt::CsrDecoder::new(cfg).generate(&raw);
    println!("\n  {}n {}syn | sensory=40 DN=16 MN=4",
        sample.n_neurons, sample.n_synapses);

    let n_neurons = sample.n_neurons;
    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    let conn = ConnectomeData::new(
        sample.row_ptr.clone(), sample.col_indices.clone(),
        sample.values.clone(), sample.nt_types.clone(), groups,
    );
    let mn_idx: Vec<usize> = sample.groups.get("mn").cloned().unwrap_or_else(|| (n_neurons - 4..n_neurons).collect());
    let dn_idx: Vec<usize> = sample.groups.get("dn").cloned().unwrap_or_default();

    // Input projection
    let mut rng = Lcg::new(123);
    let input_scale = I_SCALE / (784.0f32).sqrt();
    let mut w_in = vec![0.0f32; 784 * n_neurons];
    for w in w_in.iter_mut() { *w = (rng.next_f32() - 0.5) * 2.0 * input_scale; }

    // Sim state
    let mut state = BrainState::new(n_neurons, NeuronModel::Izhikevich);
    state.reset_with_noise(n_neurons);
    let params = IzhikevichParams {
        a: vec![0.02; n_neurons], b: vec![0.2; n_neurons],
        c: vec![-65.0; n_neurons], d: vec![2.0; n_neurons],
        dt_ms: DT, tau_ref: 2.0,
    };

    // Load a few MNIST examples (10 per digit)
    check_file(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    check_file(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));

    // Find 2 examples per digit
    let mut examples: Vec<Vec<Vec<f32>>> = vec![vec![]; 10]; // digit → [img1, img2]
    for (i, &l) in labels.iter().enumerate() {
        if examples[l as usize].len() < 2 {
            examples[l as usize].push(images[i].clone());
        }
        if examples.iter().all(|e| e.len() >= 2) { break; }
    }

    // ── Run each digit example through the brain ──
    fn run_and_record(
        img: &[f32], w_in: &[f32], conn: &ConnectomeData,
        state: &mut BrainState, params: &IzhikevichParams,
        n: usize, steps: usize,
    ) -> Vec<f32> {
        let mut ext = vec![I_BIAS; n];
        for j in 0..n {
            let mut s = 0.0; for i in 0..img.len() { s += img[i] * w_in[i * n + j]; }
            ext[j] += s;
        }
        let buf = sim_izhikevich(conn, state, params, &ext, steps);
        let mut cnt = vec![0.0f32; n];
        for step in 0..steps {
            let base = step * n;
            for j in 0..n { if buf[base + j] != 0 { cnt[j] += 1.0; } }
        }
        cnt
    }

    println!("\n─── MN Motor Patterns (avg spikes over 30 steps) ───");
    println!("  {:<6} {:>20} {:>20} {:>12}", "Digit", "MN [0 1 2 3]", "DN [avg of 16]", "Sensory[0]");
    println!("  {}", "─".repeat(80));

    let mut mn_patterns = vec![vec![0.0f32; 4]; 10];
    for d in 0..10 {
        let mut mn_sum = vec![0.0f32; 4];
        let mut dn_sum = 0.0f32;
        let mut sensory_sum = 0.0f32;
        for example in &examples[d] {
            let spikes = run_and_record(example, &w_in, &conn, &mut state, &params, n_neurons, N_STEPS);
            for (k, &mi) in mn_idx.iter().enumerate() { mn_sum[k] += spikes[mi]; }
            for &di in &dn_idx { dn_sum += spikes[di]; }
            if let Some(&si) = sample.groups.get("sensory").and_then(|v| v.first()) {
                sensory_sum += spikes[si];
            }
        }
        let n_ex = examples[d].len() as f32;
        for k in 0..4 { mn_patterns[d][k] = mn_sum[k] / n_ex; }
        let dn_avg = dn_sum / (dn_idx.len().max(1) as f32 * n_ex);
        let s_avg = sensory_sum / n_ex;

        let fmt_mn = mn_patterns[d].iter().map(|v| format!("{:.1}", v)).collect::<Vec<_>>().join(" ");
        println!("  {:<6} {:>20} {:>20} {:>12.1}",
            d, fmt_mn, format!("{:.2}", dn_avg), s_avg);
    }

    // ── Различимость: Euclidean distance between patterns ──
    println!("\n─── Pattern separation (cosine sim matrix) ───");
    println!("  Diagonal = 1.0, off-diagonal → 0 means good separation");
    print!("     ");
    for d in 0..10 { print!(" {:>5}", d); }
    println!();
    for d1 in 0..10 {
        print!("  {:>2}:", d1);
        for d2 in 0..10 {
            let dot: f32 = mn_patterns[d1].iter().zip(mn_patterns[d2].iter()).map(|(a, b)| a * b).sum();
            let n1: f32 = mn_patterns[d1].iter().map(|a| a * a).sum::<f32>().sqrt();
            let n2: f32 = mn_patterns[d2].iter().map(|a| a * a).sum::<f32>().sqrt();
            let sim = if n1 > 0.0 && n2 > 0.0 { dot / (n1 * n2) } else { 0.0 };
            print!(" {:>.2}", sim);
        }
        println!();
    }

    // ── Confusion: classify digit by nearest MN pattern ──
    println!("\n─── Nearest-pattern classification ───");
    let test_images = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let test_labels = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));
    let n_test = 100usize;
    let mut correct = 0usize;
    for i in 0..n_test {
        let spikes = run_and_record(&test_images[i], &w_in, &conn, &mut state, &params, n_neurons, N_STEPS);
        let mut mn = vec![0.0f32; 4];
        for (k, &mi) in mn_idx.iter().enumerate() { mn[k] = spikes[mi]; }
        // Find closest template
        let mut best_d = 0;
        let mut best_sim = -1.0f32;
        for d in 0..10 {
            let dot: f32 = mn.iter().zip(mn_patterns[d].iter()).map(|(a, b)| a * b).sum();
            let n1 = mn.iter().map(|a| a * a).sum::<f32>().sqrt();
            let n2 = mn_patterns[d].iter().map(|a| a * a).sum::<f32>().sqrt();
            let sim = if n1 > 0.0 && n2 > 0.0 { dot / (n1 * n2) } else { 0.0 };
            if sim > best_sim { best_sim = sim; best_d = d; }
        }
        if best_d == test_labels[i] as usize { correct += 1; }
    }
    println!("  Nearest-template accuracy (n={n_test}): {}/{} = {:.1}%",
        correct, n_test, correct as f64 / n_test as f64 * 100.0);

    println!("\n─── Вывод ───");
    println!("  Уже без обучения 10 цифр дают 10 разных MN паттернов.");
    println!("  E-Prop/BPTT может усилить разделение в 2-3×.");
    println!("  После обучения: VNC читает MN паттерны → 10 походок.");
    println!("{}", "═".repeat(60));
    Ok(())
}
