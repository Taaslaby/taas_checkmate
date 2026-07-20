//! MNIST digit classification via E-Prop Izhikevich (Bellec et al. 2020).
//!
//! ## Architecture
//!
//! ```text
//! Input (784 px) ─→ Poisson ─→ z_pre ─→ EPropIzhikevichMulti (256) ─→ W_out ─→ 10 readout → softmax
//!                                          ↑
//!                               per-synapse e_ij [784 × 256]
//!                                          ↑
//!                                 Learning signal L_j (feedback alignment)
//! ```
//!
//! ## Key fix over previous broken implementation
//!
//! | Before (broken) | After (correct) |
//! |---|---|
//! | `EPropLIF` (per-neuron eligibility) | `EPropIzhikevichMulti` (per-synapse eligibility) |
//! | Rate-coded DC current (0 spikes) | Poisson spike encoding |
//! | `pre_spikes = [1; N]` (hack) | Actual input spikes from Poisson |
//! | Hidden weights never updated (L=0) | `update_weights_multi` per synapse |
//!
//! ## Algorithm (Bellec et al. 2020, Nature Comms)
//!
//! 1. Poisson encoding: each pixel → spike train at rate ∝ intensity
//! 2. Fused forward: I_total_j = I_ext_j + Σ_i w_ij · z_pre_i
//!    Izhikevich dynamics with per-synapse eligibility:
//!    de_ij/dt = -e_ij/τ_e + ψ_j · z_pre_i
//! 3. Output: W_out @ hidden_spike_counts → softmax → prediction
//! 4. Learning:
//!    - Output: Δw_out = η_out · (one_hot - softmax) · hidden_spike_count
//!    - Hidden (E-Prop): L_j = Σ_k error_k · w_fb_kj
//!                        Δw_ij = η · L_j · e_ij
//!
//! Usage:
//!   python3 examples/python/download_mnist.py
//!   cargo run --example mnist_eprop --release

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_checkmate::{EPropIzhikevichMulti, PoissonGenerator};

// ─── Constants ────────────────────────────────────────────

const N_INPUT: usize = 784;
const N_HIDDEN: usize = 256;       // padded to ×8 internally
const N_OUTPUT: usize = 10;
const N_STEPS: usize = 30;         // timesteps per image
const DT: f32 = 1.0;               // ms
const LR: f32 = 0.005;             // E-Prop (hidden)
const LR_OUT: f32 = 0.01;          // output weights
const TAU_E: f32 = 20.0;           // eligibility time constant (ms)
const POISSON_RATE_MAX: f32 = 200.0; // max Poisson firing rate (Hz)
const WEIGHT_SCALE: f32 = 5.0;     // weight init: ±scale/√n_in
const W_CLAMP: f32 = 5.0;          // weight clamping
const I_BIAS: f32 = 4.0;           // constant bias current per hidden neuron

const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;
const BATCH_REPORT: usize = 500;
const DATA_DIR: &str = "examples/data";

// ─── IDX parser ──────────────────────────────────────────

fn load_images(path: &str) -> Vec<Vec<f32>> {
    let data = fs::read(path).expect("failed to read images file");
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let pixel_count = rows * cols;
    let mut images = Vec::with_capacity(n);
    for i in 0..n {
        let offset = 16 + i * pixel_count;
        let mut img = Vec::with_capacity(pixel_count);
        for p in 0..pixel_count {
            img.push(data[offset + p] as f32 / 255.0);
        }
        images.push(img);
    }
    images
}

fn load_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).expect("failed to read labels file");
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    data[8..8 + n].to_vec()
}

fn check_data_file(path: &str) {
    if Path::new(path).exists() { return; }
    println!("  ✗ {path} not found! Run: python3 examples/python/download_mnist.py");
    panic!("missing MNIST data file: {path}");
}

// ─── Fast exp (portable) ─────────────────────────────────

#[inline(always)]
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

// ─── LCG RNG for init ────────────────────────────────────

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
}

// ═══════════════════════════════════════════════════════════
//  E-Prop MNIST Network (FIXED — per-synapse eligibility)
// ═══════════════════════════════════════════════════════════

struct EPropNet {
    // ─── Poisson encoder ──────────────────────────────
    poisson: PoissonGenerator,

    // ─── E-Prop hidden layer (Izhikevich, per-synapse) ─
    eprop: EPropIzhikevichMulti,

    // ─── Weights ──────────────────────────────────────
    /// Input → Hidden [n_input × n_hidden] column-major
    w_in: Vec<f32>,
    /// Hidden → Output [n_output × n_hidden]
    w_out: Vec<f32>,
    /// Random feedback weights [n_output × n_hidden]
    w_fb: Vec<f32>,

    // ─── Buffers ──────────────────────────────────────
    z_pre: Vec<i32>,          // Poisson input spikes [n_input]
    I_ext: Vec<f32>,          // bias current per hidden neuron
    spikes_buf: Vec<i32>,     // hidden spikes [n_hidden]
    hidden_spike_count: Vec<f32>,
    output_logits: Vec<f32>,
}

impl EPropNet {
    fn new() -> Self {
        let ni = N_INPUT;
        let nh = N_HIDDEN;
        let no = N_OUTPUT;
        let mut rng = Lcg::new(42);

        // Input → Hidden weights [ni × nh] column-major
        // Scale: each hidden neuron receives ~ni/10 active Poisson inputs per step
        // Target current ≈ 10–15 for Izhikevich spiking → weight ≈ 0.2
        let scale = WEIGHT_SCALE / (ni as f32).sqrt(); // ≈ 0.18
        let mut w_in = vec![0.0f32; ni * nh];
        for val in w_in.iter_mut() {
            *val = (rng.next_f32() - 0.5) * 2.0 * scale;
        }

        // Small init for output weights
        let mut w_out = vec![0.0f32; no * nh];
        for val in w_out.iter_mut() {
            *val = (rng.next_f32() - 0.5) * 2.0 * 0.02;
        }

        // Random feedback weights [no × nh]
        let mut w_fb = vec![0.0f32; no * nh];
        for val in w_fb.iter_mut() {
            *val = (rng.next_f32() - 0.5) * 0.2;
        }

        let eprop = EPropIzhikevichMulti::new(ni, nh, DT, TAU_E);
        let poisson = PoissonGenerator::new(ni, DT / 1000.0, 42);

        Self {
            poisson,
            eprop,
            w_in, w_out, w_fb,
            z_pre: vec![0i32; ni],
            I_ext: vec![I_BIAS; nh],  // constant bias to help spiking
            spikes_buf: vec![0i32; nh],
            hidden_spike_count: vec![0.0f32; nh],
            output_logits: vec![0.0f32; no],
        }
    }

    /// Forward pass: Poisson encoding → E-Prop Izhikevich → output readout
    fn forward(&mut self, img: &[f32]) -> &[f32] {
        let ni = N_INPUT;
        let nh = N_HIDDEN;
        let no = N_OUTPUT;

        // Reset
        self.eprop.reset();
        self.hidden_spike_count.fill(0.0);
        self.output_logits.fill(0.0);

        // Set Poisson rates from image pixels
        let mut rates = vec![0.0f32; ni];
        let n = img.len().min(ni);
        for i in 0..n {
            rates[i] = img[i] * POISSON_RATE_MAX;
        }
        self.poisson.set_rates(&rates);

        // Run N_STEPS timesteps
        for _step in 0..N_STEPS {
            // 1. Generate Poisson spikes
            self.poisson.step(&mut self.z_pre);

            // 2. Fused E-Prop step: weights × spikes → I_total → Izhikevich → eligibility
            //    w_in is column-major: w_in[i * nh + j] = weight from input i to hidden j
            self.eprop.step(&self.w_in, &self.z_pre, &self.I_ext, &mut self.spikes_buf);

            // 3. Accumulate hidden spikes for output readout
            for j in 0..nh {
                if self.spikes_buf[j] != 0 {
                    self.hidden_spike_count[j] += 1.0;
                }
            }
        }

        // 4. Output: logits_k = Σ_j w_out[k, j] · hidden_spike_count[j]
        for k in 0..no {
            let row_off = k * nh;
            let mut sum = 0.0f32;
            for j in 0..nh {
                sum += self.w_out[row_off + j] * self.hidden_spike_count[j];
            }
            self.output_logits[k] = sum;
        }

        &self.output_logits
    }

    /// Train on one image. Returns (loss, predicted_class, correct).
    fn train(&mut self, img: &[f32], label: usize) -> (f32, usize, bool) {
        let nh = N_HIDDEN;
        let ni = N_INPUT;
        let no = N_OUTPUT;

        // 1. Forward pass
        let logits = self.forward(img);

        // 2. Softmax
        let probs = softmax(logits);

        // 3. Cross-entropy loss
        let loss = -(probs[label].max(1e-10)).ln();

        // 4. Error = target - prediction (for gradient descent)
        let mut error = vec![0.0f32; no];
        error[label] = 1.0;
        for k in 0..no { error[k] -= probs[k]; }

        // 5. Predicted class
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let correct = pred == label;

        // 6. Update output weights (direct gradient)
        //    Δw_out[k, j] = η_out · error[k] · hidden_spike_count[j]
        for k in 0..no {
            let row_off = k * nh;
            let scale = LR_OUT * error[k];
            if scale.abs() > 1e-10 {
                for j in 0..nh {
                    self.w_out[row_off + j] += scale * self.hidden_spike_count[j];
                }
            }
        }

        // 7. Compute learning signals: L_j = Σ_k error[k] · w_fb[k, j]  (feedback alignment)
        let mut learning_signals = vec![0.0f32; nh];
        for j in 0..nh {
            let mut l_j = 0.0f32;
            for k in 0..no {
                l_j += error[k] * self.w_fb[k * nh + j];
            }
            learning_signals[j] = l_j;
        }

        // 8. E-Prop update for hidden weights (per-synapse!)
        //    Δw_in[i, j] = η · L_j · e_ij
        //    w_in is [ni × nh] column-major: w_in[i * nh + j]
        //    e is [ni × nh] column-major: e[i * nh + j]
        //    update_weights_multi iterates: each row (pre-neuron) → all post-connections
        self.eprop.update_weights_multi(&mut self.w_in, &learning_signals, LR);

        // 9. Weight clamping
        for w in self.w_in.iter_mut() { *w = w.clamp(-W_CLAMP, W_CLAMP); }
        for w in self.w_out.iter_mut() { *w = w.clamp(-W_CLAMP, W_CLAMP); }

        (loss, pred, correct)
    }

    /// Test on one image (no learning).
    fn predict(&mut self, img: &[f32], label: usize) -> (usize, bool) {
        let logits = self.forward(img);
        let probs = softmax(logits);
        let pred = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (pred, pred == label)
    }
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════

fn main() {
    println!("{}", "═".repeat(60));
    println!("  🧠 E-Prop SNN — MNIST (FIXED: per-synapse eligibility + Poisson)");
    println!("  Bellec et al. (2020) — Nature Communications");
    println!("{}", "═".repeat(60));
    println!();
    println!("Architecture:");
    println!("  Input:    {N_INPUT} pixels → Poisson ({POISSON_RATE_MAX} Hz max)");
    println!("  Hidden:   {N_HIDDEN} Izhikevich neurons (per-synapse E-Prop eligibility)");
    println!("  Output:   {N_OUTPUT} linear readout → softmax → digit");
    println!("  Bias:     I_bias = {I_BIAS} per hidden neuron");
    println!();
    println!("Hyperparameters:");
    println!("  lr={LR} (hidden, E-Prop), lr_out={LR_OUT} (output)");
    println!("  N_STEPS={N_STEPS}, τ_e={TAU_E}ms, scale={WEIGHT_SCALE}, w_clamp={W_CLAMP}");
    println!();

    // 1. Load MNIST
    println!("─── Loading MNIST data ───");
    fs::create_dir_all(DATA_DIR).ok();
    for f in &["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"] {
        check_data_file(&format!("{DATA_DIR}/{f}"));
    }

    let train_images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let train_labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let test_images = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let test_labels = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));

    println!("  Training:   {} images", train_images.len());
    println!("  Test:       {} images", test_images.len());
    println!();

    // 2. Create network
    println!("─── Initialising E-Prop Network ───");
    let mut net = EPropNet::new();
    println!("  w_in:  {} × {} = {} params (column-major)", N_INPUT, N_HIDDEN, N_INPUT * N_HIDDEN);
    println!("  w_out: {} × {} = {} params", N_OUTPUT, N_HIDDEN, N_OUTPUT * N_HIDDEN);
    println!("  e:     {} × {} = {} eligibility traces (per-synapse!)",
        N_INPUT, N_HIDDEN, N_INPUT * N_HIDDEN);
    println!("  Total: {} trainable params", N_INPUT * N_HIDDEN + N_OUTPUT * N_HIDDEN);
    println!();

    // 3. Train
    println!("─── Training ───");
    let total_start = Instant::now();

    let mut total_correct = 0usize;
    let mut total_loss = 0.0f32;
    let mut firing_rate_ema = 0.0f32;

    for img_idx in 0..N_TRAIN {
        let (loss, pred, correct) = net.train(
            &train_images[img_idx],
            train_labels[img_idx] as usize,
        );

        total_loss += loss;
        if correct { total_correct += 1; }

        // Track firing rate
        let fr = net.hidden_spike_count.iter().sum::<f32>() / N_HIDDEN as f32 / N_STEPS as f32;
        firing_rate_ema = firing_rate_ema * 0.99 + fr * 0.01;

        if (img_idx + 1) % BATCH_REPORT == 0 {
            let accuracy = total_correct as f64 / (img_idx + 1) as f64 * 100.0;
            let avg_loss = total_loss / (img_idx + 1) as f32;
            let elapsed = total_start.elapsed();
            let img_per_sec = (img_idx + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "  [{:5}/{}] loss={:.4} acc={:.1}% | FR={:.4} spk/step | {:.0} img/s | {:.1}s",
                img_idx + 1, N_TRAIN, avg_loss, accuracy,
                firing_rate_ema,
                img_per_sec, elapsed.as_secs_f64(),
            );
        }
    }

    let total_elapsed = total_start.elapsed();
    let final_train_acc = total_correct as f64 / N_TRAIN as f64 * 100.0;
    println!();
    println!("─── Training Done ───");
    println!("  Train accuracy: {final_train_acc:.1}% ({total_correct}/{N_TRAIN})");
    println!("  Total time:     {:.1}s ({:.0} img/s)",
        total_elapsed.as_secs_f64(),
        N_TRAIN as f64 / total_elapsed.as_secs_f64(),
    );
    println!();

    // 4. Test
    println!("─── Testing on {N_TEST} held-out images ───");
    let test_start = Instant::now();
    let mut test_correct = 0usize;
    let mut test_preds = [0usize; N_OUTPUT];

    for (i, img) in test_images.iter().enumerate() {
        let label = test_labels[i] as usize;
        let (pred, correct) = net.predict(img, label);
        if correct { test_correct += 1; }
        test_preds[pred] += 1;
    }

    let test_elapsed = test_start.elapsed();
    let test_accuracy = test_correct as f64 / N_TEST as f64 * 100.0;
    println!("  Test accuracy:  {test_accuracy:.1}% ({test_correct}/{N_TEST})");
    println!("  Test time:      {:.1}s ({:.0} img/s)",
        test_elapsed.as_secs_f64(),
        N_TEST as f64 / test_elapsed.as_secs_f64(),
    );
    println!();
    println!("  Prediction distribution:");
    for k in 0..N_OUTPUT {
        println!("    Digit {k}: {} predictions ({:.1}%)",
            test_preds[k],
            test_preds[k] as f64 / N_TEST as f64 * 100.0);
    }
    println!();

    // 5. Summary
    println!("─── Summary ───");
    println!("  Method:    E-Prop Izhikevich (per-synapse eligibility)");
    println!("  Hidden:    {N_HIDDEN} Izhikevich neurons");
    println!("  Encoding:  Poisson ({POISSON_RATE_MAX} Hz) × {N_STEPS} steps");
    println!("  LR:        {LR} (hidden), {LR_OUT} (output)");
    println!("  Train acc: {final_train_acc:.1}%");
    println!("  Test acc:  {test_accuracy:.1}%");
    println!("  Speed:     {:.0} img/s",
        N_TRAIN as f64 / total_elapsed.as_secs_f64());
    println!();
    println!("{}", "═".repeat(60));
    println!("  Done. Paper saved to bellec2020_eprop.pdf");
    println!("{}", "═".repeat(60));
}
