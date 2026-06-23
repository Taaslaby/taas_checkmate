//! MNIST digit classification via spiking Actor-Critic (contextual bandit).
//!
//! Frames MNIST as an RL problem:
//!   State  — 28×28 pixel values (Poisson-encoded)
//!   Action — predicted digit (0-9)
//!   Reward — +1 correct, -0.5 wrong
//!
//! The agent learns through dopamine-modulated STDP — no backprop, no autograd.
//!
//! Usage:
//!   cargo run --example mnist_train --no-default-features
//!
//!
//! Data is downloaded by running:
//!   python3 examples/download_mnist.py

use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_checkmate::rl::ActorCriticAgent;

// ─── Constants ────────────────────────────────────────────

const N_INPUT: usize = 784;   // 28 × 28 pixels
const N_HIDDEN: usize = 64;   // Izhikevich neurons (padded to 64 → ×8)
const N_ACTIONS: usize = 10;  // digits 0-9
const N_STEPS: usize = 20;    // internal simulation steps
const DT: f32 = 1.0;          // ms
const SEED: u32 = 42;

const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;

const BATCH_SIZE: usize = 500;  // report accuracy every N images

const DATA_DIR: &str = "examples/data";

// ─── IDX format parser ───────────────────────────────────

/// Parse MNIST images file (IDX3 → n × rows × cols f32).
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
            img.push(data[offset + p] as f32 / 255.0);  // normalize to [0, 1]
        }
        images.push(img);
    }
    images
}

/// Parse MNIST labels file (IDX1 → n labels).
fn load_labels(path: &str) -> Vec<u8> {
    let data = fs::read(path).expect("failed to read labels file");
    let n = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    data[8..8 + n].to_vec()
}

/// Check that a file exists; if not, print instructions for manual download.
fn check_data_file(path: &str) {
    if Path::new(path).exists() {
        return;
    }
    println!("  ✗ {path} not found!");
    println!();
    println!("  Run the download script first:");
    println!("    python3 examples/download_mnist.py");
    panic!("missing MNIST data file: {path}");
}

// ─── Training ─────────────────────────────────────────────

fn main() {
    println!("═══ MNIST via Spiking Actor-Critic (contextual bandit) ═══");
    println!();
    println!("Architecture:");
    println!("  Input:    {N_INPUT} pixels (Poisson-encoded)");
    println!("  Hidden:   {N_HIDDEN} Izhikevich neurons (SIMD f32x8)");
    println!("  Output:   {N_ACTIONS} LIF readout → softmax → digit");
    println!("  Plasticity: dopamine-STDP (3-factor, Izhikevich 2007)");
    println!("  Internal sim: {N_STEPS} steps × DT = {} ms", N_STEPS as f32 * DT);
    println!();

    // 1. Download / load MNIST
    println!("─── Loading MNIST data ───");
    fs::create_dir_all(DATA_DIR).ok();

    for f in &["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"] {
        let path = format!("{DATA_DIR}/{f}");
        check_data_file(&path);
    }

    let train_images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let train_labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    let test_images = load_images(&format!("{DATA_DIR}/t10k-images-idx3-ubyte"));
    let test_labels = load_labels(&format!("{DATA_DIR}/t10k-labels-idx1-ubyte"));

    println!("  Training:   {} images", train_images.len());
    println!("  Test:       {} images", test_images.len());
    println!();

    // 2. Create agent
    println!("─── Creating Agent ───");
    let mut agent = ActorCriticAgent::new(
        N_INPUT, N_HIDDEN, N_ACTIONS, N_STEPS, DT, SEED,
    );
    // Tune hyperparams for MNIST
    agent.lr_actor = 0.02;
    agent.lr_critic = 0.02;
    agent.gamma = 0.9;
    agent.tau_trace = 20.0;
    agent.tau_e = 800.0;
    agent.a_ltp = 0.08;
    agent.a_ltd = 0.10;
    agent.w_min = 0.0;
    agent.w_max = 3.0;
    println!("  lr_actor={}, gamma={}, tau_e={}", agent.lr_actor, agent.gamma, agent.tau_e);
    println!("  lr={}, a_ltp={}, a_ltd={}, w_max={}", agent.lr_actor, agent.a_ltp, agent.a_ltd, agent.w_max);
    println!();

    // 3. Train
    println!("─── Training ───");
    let total_start = Instant::now();

    let mut episode = 0;
    let mut total_correct = 0usize;
    let mut total_reward = 0.0f32;
    let mut avg_reward_ema = 0.0f32;

    // Train on all 60K images, each as one RL episode
    for img_idx in 0..N_TRAIN {
        let img = &train_images[img_idx];
        let label = train_labels[img_idx] as usize;

        // Forward pass
        agent.reset_episode();
        let (action, _) = agent.act(img);

        // Reward
        let reward = if action == label { 1.0 } else { -0.5 };
        let td = agent.learn(reward, 0.0, true);

        total_reward += reward;
        if action == label { total_correct += 1; }
        episode += 1;

        // EMA of average reward
        let avg = total_reward as f32 / episode as f32;
        avg_reward_ema = avg_reward_ema * 0.99 + avg * 0.01;

        // Report
        if episode % BATCH_SIZE == 0 {
            let accuracy = total_correct as f64 / episode as f64 * 100.0;
            let elapsed = total_start.elapsed();
            let img_per_sec = episode as f64 / elapsed.as_secs_f64();
            println!(
                "  [{episode:5}/{N_TRAIN}] acc={accuracy:.1}% | avg_r={avg:.3} | td={td:.3} | {img_per_sec:.0} img/s | {:.1}s",
                elapsed.as_secs_f64(),
            );
        }

        // Decay learning rate
        if img_idx > 0 && img_idx % 10_000 == 0 {
            agent.lr_actor *= 0.7;
            agent.lr_critic *= 0.7;
            println!("  → lr decayed to {:.4}", agent.lr_actor);
        }
    }

    let total_elapsed = total_start.elapsed();
    let final_accuracy = total_correct as f64 / N_TRAIN as f64 * 100.0;
    println!();
    println!("─── Training Done ───");
    println!("  Final accuracy: {final_accuracy:.1}% ({total_correct}/{N_TRAIN})");
    println!("  Total time: {:.1}s ({:.0} img/s)",
        total_elapsed.as_secs_f64(),
        N_TRAIN as f64 / total_elapsed.as_secs_f64(),
    );

    // 4. Test on held-out set
    println!();
    println!("─── Testing on {N_TEST} held-out images ───");
    let test_start = Instant::now();
    let mut test_correct = 0usize;
    for (i, img) in test_images.iter().enumerate() {
        agent.reset_episode();
        let (action, _) = agent.act(img);
        if action == test_labels[i] as usize {
            test_correct += 1;
        }
    }
    let test_elapsed = test_start.elapsed();
    let test_accuracy = test_correct as f64 / N_TEST as f64 * 100.0;
    println!("  Test accuracy: {test_accuracy:.1}% ({test_correct}/{N_TEST})");
    println!("  Test time: {:.1}s ({:.0} img/s)",
        test_elapsed.as_secs_f64(),
        N_TEST as f64 / test_elapsed.as_secs_f64(),
    );
    println!();
    println!("═══ Done ═══");
}
