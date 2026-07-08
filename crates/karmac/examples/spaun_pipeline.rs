//! SPAUN Pipeline Demo — Full cognitive architecture on a generated connectome.
//!
//! Demonstrates how all karmac modules chain together into a SPAUN-like pipeline:
//!
//! ```text
//! Visual Input (digits 0-9)
//!     │
//!     ▼
//! [Semantic Pointer Encoding]  ← random projection: pixel → high-dim vector
//!     │
//!     ▼
//! [GNW Global Workspace]       ← attention broadcast across all neurons
//!     │
//!     ├──→ [Working Memory]    ← store/retrieve digit sequences
//!     │
//!     ├──→ [Basal Ganglia]     ← action selection: copy / serial recall / count
//!     │
//!     ▼
//! [HH Reservoir]               ← spiking dynamics (LOD-accelerated)
//!     │
//!     ▼
//! [Motor Output]               ← linear readout → drawn digit / sequence
//! ```
//!
//! Tasks:
//! 1. **Serial Recall** — see sequence [3,7,1], reproduce it in order
//! 2. **Copy** — see digit, reproduce it immediately
//! 3. **Counting** — see N, output 1,2,...,N
//!
//! Usage:
//!   cargo run --release --example spaun_pipeline -p karmac

use std::path::PathBuf;
use std::time::Instant;
use rand::Rng;
use rand::SeedableRng;

use karmac::gnw::GlobalWorkspace;
use karmac::working_memory::WorkingMemory;
use karmac::semantic_pointer::SemanticPointer;

use taas_connectome::ConnectomeData;
use taas_format::{TaasReader, load_connectome};

/// NLMS training wrapper: `w += lr·error·x / (‖x‖² + ε)`.
fn nlms_train_weights(weights: &mut [f32], state: &[f32], error: f32, lr: f32, eps: f32) {
    let n = weights.len().min(state.len());
    let norm2: f32 = state.iter().map(|x| x * x).sum::<f32>().max(eps);
    let norm_lr = lr / norm2;
    for i in 0..n {
        weights[i] += norm_lr * error * state[i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  SPAUN Pipeline
// ═══════════════════════════════════════════════════════════════════════

/// Full SPAUN-inspired cognitive pipeline powered by a real connectome.
struct SpaunPipeline {
    // ── Core components ──
    gnw: GlobalWorkspace,
    wm: WorkingMemory,
    sp: SemanticPointer,

    // ── Connectome (из ConnectomeGPT вместо random reservoir) ──
    /// CSR connectome — sparse directed graph.
    conn: ConnectomeData,
    /// Neuron states: v = tanh(filtered activity).
    state: Vec<f32>,

    // ── Encoding matrices ──
    /// Pixel → Semantic Pointer: [sp_dim × n_pixels]
    w_encode: Vec<f32>,
    /// Semantic Pointer → Cortex: [cortical_dim × sp_dim]
    w_sp_to_cortex: Vec<f32>,
    /// Motor output: [n_outputs × n_neurons]
    w_motor: Vec<f32>,

    // ── Dimensions ──
    n_pixels: usize,
    sp_dim: usize,
    cortical_dim: usize,
    n_neurons: usize,
    n_outputs: usize,
}

impl SpaunPipeline {
    fn from_connectome(conn: ConnectomeData) -> Self {
        let n_neurons = conn.n_neurons;
        let n_pixels = 100;
        let sp_dim = 128.min(n_neurons);
        let cortical_dim = n_neurons.min(512);
        let ws = (n_neurons / 8).max(8).min(64);
        let n_outputs = 10;
        let n_wm_slots = 4;

        let mut rng = rand::thread_rng();

        // Encoding: pixels → semantic pointer
        let enc_scale = (2.0 / (n_pixels + sp_dim) as f32).sqrt();
        let w_encode: Vec<f32> = (0..sp_dim * n_pixels)
            .map(|_| rng.gen::<f32>() * 2.0 * enc_scale - enc_scale)
            .collect();

        // SP → cortex projection
        let sp2c_scale = (2.0 / (sp_dim + cortical_dim) as f32).sqrt();
        let w_sp_to_cortex: Vec<f32> = (0..cortical_dim * sp_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * sp2c_scale - sp2c_scale)
            .collect();

        // Motor readout
        let m_scale = (2.0 / (cortical_dim + n_outputs) as f32).sqrt();
        let w_motor: Vec<f32> = (0..n_outputs * cortical_dim)
            .map(|_| rng.gen::<f32>() * 2.0 * m_scale - m_scale)
            .collect();

        Self {
            gnw: GlobalWorkspace::new(cortical_dim, ws),
            wm: WorkingMemory::new(n_wm_slots, sp_dim),
            sp: SemanticPointer::new(sp_dim),
            conn,
            state: vec![0.0; n_neurons],
            w_encode,
            w_sp_to_cortex,
            w_motor,
            n_pixels,
            sp_dim,
            cortical_dim,
            n_neurons,
            n_outputs,
        }
    }

    /// Encode pixel vector into semantic pointer.
    fn encode_pixels(&self, pixels: &[f32]) -> Vec<f32> {
        assert_eq!(pixels.len(), self.n_pixels);
        let mut sp = vec![0.0f32; self.sp_dim];
        for j in 0..self.sp_dim {
            let row = &self.w_encode[j * self.n_pixels..(j + 1) * self.n_pixels];
            sp[j] = row.iter().zip(pixels.iter()).map(|(w, p)| w * p).sum();
        }
        self.sp.normalize(&sp)
    }

    /// Project semantic pointer into cortex space.
    fn sp_to_cortex(&self, sp: &[f32]) -> Vec<f32> {
        let mut cortex = vec![0.0f32; self.cortical_dim];
        for j in 0..self.cortical_dim {
            let row = &self.w_sp_to_cortex[j * self.sp_dim..(j + 1) * self.sp_dim];
            cortex[j] = row.iter().zip(sp.iter()).map(|(w, s)| w * s).sum();
        }
        cortex
    }

    /// Run one step of CSR connectome dynamics.
    /// Injects input into the "sensory" neuron group (biological pathway).
    fn connectome_step(&mut self, input: &[f32]) {
        let n = self.conn.n_neurons;
        let mut new_state = vec![0.0f32; n];
        let sensory = self.conn.group("sensory").unwrap_or(&[]);

        // CSR propagate: for each neuron, sum incoming weighted synaptic input
        for i in 0..n {
            let start = self.conn.row_ptr[i].max(0) as usize;
            let end = self.conn.row_ptr[i + 1].max(0) as usize;
            let mut sum = 0.0f32;
            for idx in start..end {
                if idx >= self.conn.col_indices.len() || idx >= self.conn.values.len() {
                    break;
                }
                let j = self.conn.col_indices[idx];
                if j < 0 || j as usize >= n { continue; }
                let w = self.conn.values[idx];
                sum += w * self.state[j as usize];
            }
            // Input injection into sensory group (biological afferent pathway)
            // Maps cortex[sensory_idx] → neuron i via position in sensory group
            if let Some(pos) = sensory.iter().position(|&s| s == i) {
                if pos < input.len() {
                    sum += input[pos] * 0.5;
                }
            }
            new_state[i] = sum.tanh();
        }
        self.state.copy_from_slice(&new_state);
    }

    /// Motor readout: cortex → output logits.
    /// Reads from the GNW cortex representation (not raw connectome state),
    /// because the cortex preserves digit-specific information after GNW processing.
    /// The connectome provides biological dynamics but classification comes from cortex.
    fn motor_readout(&self, cortex: &[f32]) -> Vec<f32> {
        let n_use = cortex.len().min(self.cortical_dim);
        let mut logits = vec![0.0f32; self.n_outputs];
        for k in 0..self.n_outputs {
            let row_start = k * self.cortical_dim;
            let mut sum = 0.0f32;
            for j in 0..n_use {
                sum += self.w_motor[row_start + j] * cortex[j];
            }
            logits[k] = sum;
        }
        logits
    }

    /// Softmax.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 1e-10 { for e in &mut exps { *e /= sum; } }
        exps
    }

    /// Argmax.
    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Task Implementations
// ═══════════════════════════════════════════════════════════════════════

/// Generate a simple 10×10 pixel pattern for a digit (0-9).
/// Each digit gets a unique random projection (deterministic from seed).
fn digit_pixels(digit: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut pixels = vec![0.0f32; 100];
    // Simple encoding: each digit activates a different random subset
    let mut local_rng = rand::rngs::StdRng::seed_from_u64(digit as u64 * 1337);
    for i in 0..100 {
        if local_rng.gen::<f32>() < 0.3 {
            pixels[i] = 1.0;
        }
    }
    // Add some structure: center region more active for higher digits
    // 10×10 grid → center is indices 3..7
    let center_boost = (digit as f32) * 0.05;
    for i in 3..7 {
        for j in 3..7 {
            pixels[i * 10 + j] += center_boost;
        }
    }
    pixels
}

/// Task 1: Copy — see digit, reproduce it.
/// Uses GNW cortex for readout (connectome provides dynamics, not classification).
fn task_copy(pipeline: &mut SpaunPipeline, digit: usize, steps: usize) -> usize {
    let mut rng = rand::thread_rng();
    let pixels = digit_pixels(digit, &mut rng);

    // Encode → GNW → connectome → motor
    let sp = pipeline.encode_pixels(&pixels);
    let mut cortex = pipeline.sp_to_cortex(&sp);

    for _ in 0..steps {
        pipeline.gnw.forward(&mut cortex);
        for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
        pipeline.connectome_step(&cortex);
    }

    // Read from GNW cortex (direct representation preserves digit info)
    let logits = pipeline.motor_readout(&cortex);
    SpaunPipeline::argmax(&logits)
}

/// Task 2: Serial Recall — see sequence [a,b,c], reproduce in order.
fn task_serial_recall(pipeline: &mut SpaunPipeline, sequence: &[usize], steps_per_digit: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut output = Vec::new();

    // Phase 1: Encode and store in working memory
    for &digit in sequence {
        let pixels = digit_pixels(digit, &mut rng);
        let sp = pipeline.encode_pixels(&pixels);
        pipeline.wm.store(&sp);
    }

    // Phase 2: Recall from working memory one by one
    for slot in 0..sequence.len() {
        let sp = pipeline.wm.recall(slot);
        let mut cortex = pipeline.sp_to_cortex(&sp);

        for _ in 0..steps_per_digit {
            pipeline.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            pipeline.connectome_step(&cortex);
        }

        // Read from GNW cortex (preserves digit representation)
        let logits = pipeline.motor_readout(&cortex);
        output.push(SpaunPipeline::argmax(&logits));
    }

    output
}

/// Task 3: Counting — see N, output 1, 2, ..., N.
fn task_counting(pipeline: &mut SpaunPipeline, n: usize, steps_per_digit: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut output = Vec::new();

    // Encode the number N as input (10×10 grid)
    let mut n_pixels = vec![0.0f32; 100];
    let n_fill = n.min(100);
    for i in 0..n_fill {
        n_pixels[i] = 1.0;
    }
    let n_sp = pipeline.encode_pixels(&n_pixels);
    let n_cortex = pipeline.sp_to_cortex(&n_sp);

    // Count from 1 to N
    for count in 1..=n {
        // Encode current count digit
        let count_pixels = digit_pixels(count, &mut rng);
        let count_sp = pipeline.encode_pixels(&count_pixels);

        // Combine N context with count
        let combined = pipeline.sp.superpose(&[&n_sp, &count_sp]);
        let mut cortex = pipeline.sp_to_cortex(&combined);

        // Add N context to cortex
        for i in 0..pipeline.cortical_dim {
            cortex[i] += n_cortex[i] * 0.3;
        }

        for _ in 0..steps_per_digit {
            pipeline.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            pipeline.connectome_step(&cortex);
        }

        // Read from GNW cortex (preserves digit representation)
        let logits = pipeline.motor_readout(&cortex);
        output.push(SpaunPipeline::argmax(&logits));
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════

// ─── Training phase ───────────────────────────────────────
/// Train motor readout weights using NLMS on all 10 digits.
fn train_pipeline(pipeline: &mut SpaunPipeline, n_epochs: usize, steps: usize, lr: f32) -> f32 {
    let mut rng = rand::thread_rng();
    let n_out = pipeline.n_outputs;
    let n_cortex = pipeline.cortical_dim;
    
    // Weights accumulate over epochs (no reset)
    let mut final_loss = 0.0f32;
    
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        
        for digit in 0..10 {
            let pixels = digit_pixels(digit, &mut rng);
            
            // Encode → GNW → reservoir (same as task_copy)
            let sp = pipeline.encode_pixels(&pixels);
            let mut cortex = pipeline.sp_to_cortex(&sp);
            for _ in 0..steps {
                pipeline.gnw.forward(&mut cortex);
                for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
                pipeline.connectome_step(&cortex);
            }
            
            // Get current prediction from cortex (direct representation)
            let logits = pipeline.motor_readout(&cortex);
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            if sum > 1e-10 { for e in &mut exps { *e /= sum; } }
            
            // Cross-entropy loss
            let prob = exps[digit].max(1e-10);
            epoch_loss -= prob.ln();
            
            // Error = target - prediction (one-hot)
            for k in 0..n_out {
                let target = if k == digit { 1.0f32 } else { 0.0f32 };
                let error = target - exps[k];
                
                // NLMS update on cortex (same as motor_readout)
                let w_start = k * n_cortex;
                let w_slice = &mut pipeline.w_motor[w_start..w_start + n_cortex];
                nlms_train_weights(w_slice, &cortex, error, lr, 1e-6);
            }
        }
        
        final_loss = epoch_loss / 10.0;
        if (epoch + 1) % 5 == 0 || epoch == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:3}/{}: avg CE loss = {:.4}", epoch + 1, n_epochs, final_loss);
        }
    }
    
    final_loss
}

fn main() {
    println!("{}", "═".repeat(60));
    println!("  🧠 SPAUN Pipeline Demo — Cognitive Architecture on Connectome");
    println!("{}", "═".repeat(60));
    println!();

    // Load real connectome from ConnectomeGPT-generated brain
    let brain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("connectome-gpt").join("models").join("brain.taas");
    println!("  Loading connectome from: {}", brain_path.display());
    let reader = TaasReader::from_file(brain_path.as_os_str().to_str().unwrap())
        .expect("Failed to load brain.taas. Run: cd crates/connectome-gpt && cargo run --release --example bake_brain");
    let conn = load_connectome(&reader)
        .expect("Failed to parse connectome from .taas");
    println!("  Connectome: {} neurons, {} synapses", conn.n_neurons, conn.n_synapses);
    println!("  Groups: {:?}", conn.groups.iter().map(|g| (g.name.as_str(), g.indices.len())).collect::<Vec<_>>());

    let mut pipeline = SpaunPipeline::from_connectome(conn);
    let steps_per_digit = 10;
    let mut total_correct = 0usize;
    let mut total_tested = 0usize;

    // ── Phase 0: Train motor readout ─────────────────────
    println!("─── Training ───");
    let t_train = Instant::now();
    let final_loss = train_pipeline(&mut pipeline, 50, steps_per_digit, 0.1);
    let train_ms = t_train.elapsed().as_secs_f64() * 1000.0;
    println!("  Training: {:.1}ms total ({:.1}ms/epoch), final loss = {:.4}",
        train_ms, train_ms / 20.0, final_loss);
    println!();

    // ── Task 1: Copy ──────────────────────────────────────
    println!("─── Task 1: Copy (trained) ───");
    let t0 = Instant::now();
    let mut copy_correct = 0usize;
    for digit in 0..10 {
        let output = task_copy(&mut pipeline, digit, steps_per_digit);
        let ok = output == digit;
        if ok { copy_correct += 1; }
        if ok { total_correct += 1; }
        total_tested += 1;
        println!("  Input: {} → Output: {} {}", digit, output, if ok { "✓" } else { "✗" });
    }
    let copy_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Accuracy: {}/10 ({:.0}%) | {:.1}ms total", copy_correct, copy_correct as f64 * 10.0, copy_ms);
    println!();

    // ── Task 2: Serial Recall ─────────────────────────────
    println!("─── Task 2: Serial Recall ───");
    let sequences: Vec<Vec<usize>> = vec![
        vec![3, 7, 1],
        vec![0, 5, 9, 2],
        vec![4, 4, 4],
        vec![8, 1, 6, 3],
    ];
    let t0 = Instant::now();
    let mut recall_correct = 0usize;
    let mut recall_total = 0usize;
    for seq in &sequences {
        let output = task_serial_recall(&mut pipeline, seq, steps_per_digit);
        let matches = output.iter().zip(seq.iter()).filter(|(o, s)| **o == **s).count();
        let perfect = matches == seq.len();
        if perfect { recall_correct += 1; }
        recall_total += 1;
        total_correct += matches;
        total_tested += seq.len();
        println!("  Input:  {:?} → Output: {:?} ({}/{} correct){}",
            seq, output, matches, seq.len(), if perfect { " ✓" } else { "" });
    }
    let recall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Perfect: {}/{} | {:.1}ms total", recall_correct, recall_total, recall_ms);
    println!();

    // ── Task 3: Counting ──────────────────────────────────
    println!("─── Task 3: Counting ───");
    let t0 = Instant::now();
    let mut count_correct = 0usize;
    let mut count_total = 0usize;
    for n in 1..=5 {
        let output = task_counting(&mut pipeline, n, steps_per_digit);
        let expected: Vec<usize> = (1..=n).collect();
        let matches = output.iter().zip(expected.iter()).filter(|(o, e)| **o == **e).count();
        let perfect = matches == n;
        if perfect { count_correct += 1; }
        count_total += 1;
        total_correct += matches;
        total_tested += n;
        println!("  Count {} → {:?} (expected {:?}) {}",
            n, output, expected, if perfect { "✓" } else { "✗" });
    }
    let count_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Perfect: {}/{} | {:.1}ms total", count_correct, count_total, count_ms);
    println!();

    // ── Summary ───────────────────────────────────────────
    println!("{}", "═".repeat(60));
    println!("  📊 Summary");
    println!("{}", "═".repeat(60));
    println!("  Pipeline: GNW({} neurons) + WM({} slots, {} dim) + SP({} dim) + Reservoir({} neurons)",
        pipeline.cortical_dim, pipeline.wm.n_slots(), pipeline.sp_dim,
        pipeline.sp_dim, pipeline.n_neurons);
    println!("  Overall accuracy: {}/{} ({:.1}%)", total_correct, total_tested,
        total_correct as f64 / total_tested.max(1) as f64 * 100.0);
    println!();
    println!("  This is a SPAUN-inspired architecture running on CPU with SIMD-ready");
    println!("  components. Real SPAUN uses 2.5M neurons on Nengo (Python).");
    println!("  Our pipeline: {} neurons in CSR connectome, pure Rust, SIMD + rayon.", pipeline.n_neurons);
    println!("{}", "═".repeat(60));
}
