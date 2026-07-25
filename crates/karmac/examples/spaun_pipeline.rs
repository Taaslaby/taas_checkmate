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
use karmac::basal_ganglia::BasalGanglia;

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
    bg: BasalGanglia,

    // ── Connectome (из ConnectomeGPT вместо random reservoir) ──
    /// CSR connectome — sparse directed graph.
    conn: ConnectomeData,
    /// Neuron states: v = tanh(filtered activity).
    state: Vec<f32>,
    /// Pre-allocated buffer for connectome_step (avoid allocation per call).
    new_state: Vec<f32>,

    // ── Encoding matrices ──
    /// Pixel → Semantic Pointer: [sp_dim × n_pixels]
    w_encode: Vec<f32>,
    /// Semantic Pointer → Cortex: [cortical_dim × sp_dim]
    w_sp_to_cortex: Vec<f32>,
    /// Motor output: [n_outputs × n_cortex]
    w_motor: Vec<f32>,

    // ── SP Templates (stateful representation) ──
    /// Stored semantic pointer for each digit 0-9.
    sp_templates: Vec<Vec<f32>>,
    /// Stored initial cortex (pre-GNW) for each digit, used to train w_sp_to_cortex.
    target_cortex: Vec<Vec<f32>>,

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

        // BG with hardcoded diagonal weights: action k responds to channel k
        let mut bg_weights = vec![0.0f32; 3 * 3];
        for k in 0..3 {
            bg_weights[k * 3 + k] = 5.0; // strong self-connection
            for j in 0..3 {
                if j != k { bg_weights[k * 3 + j] = -2.0; } // inhibit cross-talk
            }
        }

        Self {
            gnw: GlobalWorkspace::new(cortical_dim, ws),
            wm: WorkingMemory::new(n_wm_slots, sp_dim),
            sp: SemanticPointer::new(sp_dim),
            bg: BasalGanglia::from_weights(bg_weights, 3, 3),
            conn,
            state: vec![0.0; n_neurons],
            new_state: vec![0.0; n_neurons],
            w_encode,
            w_sp_to_cortex,
            w_motor,
            sp_templates: vec![vec![0.0; sp_dim]; 10],
            target_cortex: vec![vec![0.0; cortical_dim]; 10],
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
    /// Uses pre-allocated `new_state` buffer — zero heap allocations per call.
    fn connectome_step(&mut self, input: &[f32]) {
        let n = self.conn.n_neurons;
        let new_state = &mut self.new_state;
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

    /// Decode a semantic pointer to digit via NLMS-trained cortex projection.
    /// Uses full pipeline: sp_to_cortex → GNW → connectome → motor_readout.
    /// This is the SPAUN way — recalled SP goes through the entire cognitive
    /// architecture, not just template matching.
    fn decode_sp_full_pipeline(&mut self, sp: &[f32], steps: usize) -> usize {
        let mut cortex = self.sp_to_cortex(sp);
        for _ in 0..steps {
            self.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            self.connectome_step(&cortex);
        }
        let logits = self.motor_readout(&cortex);
        Self::argmax(&logits)
    }

    /// Store a semantic pointer and its initial cortex as templates for training.
    fn store_template(&mut self, digit: usize, sp: &[f32], cortex: &[f32]) {
        if digit < self.sp_templates.len() {
            self.sp_templates[digit].copy_from_slice(sp);
            let n = cortex.len().min(self.cortical_dim);
            for j in 0..n {
                self.target_cortex[digit][j] = cortex[j];
            }
        }
    }

    /// Reset connectome reservoir state to zeros (for clean recall).
    fn reset_state(&mut self) {
        self.state.fill(0.0);
        self.new_state.fill(0.0);
    }

    /// Select and execute a task via Basal Ganglia action selection.
    ///
    /// Architecture (SPAUN-aligned):
    ///   task cue → SP → cortex → GNW → downsample to 3 BG channels → BG → action
    ///
    /// `task_cue` — 3 floats [copy, recall, count] ∈ [-1, 1] that seed the SP.
    /// Returns (action_index, task_result).
    fn bg_select_and_execute(
        &mut self,
        task_cue: &[f32; 3],
        task_arg: usize,
        steps: usize,
    ) -> (usize, usize) {
        // 1. Encode task cue as a semantic pointer that activates the corresponding
        //    cortical region. Each channel index maps to a distinct SP.
        let cue_idx = task_cue.iter().position(|&c| c > 0.5).unwrap_or(0);
        let cue_sp = self.sp.from_index(cue_idx * 50 + 10);
        
        // 2. Project SP → cortex
        let mut cortex = self.sp_to_cortex(&cue_sp);
        
        // 3. GNW processes the task cue through the workspace
        //    This is the SPAUN way: cortex goes through Global Workspace
        //    before reaching Basal Ganglia for action selection.
        for _ in 0..3 {
            self.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
        }
        
        // 4. Downsample GNW-processed 512-dim cortex → 3-dim BG input
        //    Each BG channel averages 1/3 of the cortex dimensions.
        let third = self.cortical_dim / 3;
        let bg_input: [f32; 3] = [
            cortex[..third].iter().sum::<f32>() / third.max(1) as f32,
            cortex[third..2*third].iter().sum::<f32>() / third.max(1) as f32,
            cortex[2*third..].iter().sum::<f32>() / (self.cortical_dim - 2*third).max(1) as f32,
        ];
        
        // 5. BG selects action based on GNW-processed cortical activity
        let action = self.bg.step(&bg_input);
        
        // Execute the chosen action
        let result = match action {
            0 => task_copy(self, task_arg, steps),
            1 => {
                let seq = match task_arg {
                    0 => vec![3, 7, 1],
                    1 => vec![0, 5, 9, 2],
                    2 => vec![4, 4, 4],
                    _ => vec![8, 1, 6, 3],
                };
                let out = task_serial_recall(self, &seq, steps);
                // Return number of correct matches
                out.iter().zip(seq.iter()).filter(|(o, s)| **o == **s).count()
            },
            2 => {
                let out = task_counting(self, task_arg + 1, steps);
                // Return number of correct matches
                let expected: Vec<usize> = (1..=task_arg + 1).collect();
                out.iter().zip(expected.iter()).filter(|(o, e)| **o == **e).count()
            },
            _ => unreachable!(),
        };
        
        (action, result)
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
fn digit_pixels(digit: usize) -> Vec<f32> {
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
fn task_copy(pipeline: &mut SpaunPipeline, digit: usize, steps: usize) -> usize {
    let pixels = digit_pixels(digit);

    // Encode → GNW → connectome → motor
    let sp = pipeline.encode_pixels(&pixels);
    let cortex = pipeline.sp_to_cortex(&sp);
    // Store template for w_sp_to_cortex training
    pipeline.store_template(digit, &sp, &cortex);
    let mut c = cortex;

    for _ in 0..steps {
        pipeline.gnw.forward(&mut c);
        for cj in c.iter_mut() { *cj = cj.clamp(-5.0, 5.0); }
        pipeline.connectome_step(&c);
    }

    let logits = pipeline.motor_readout(&c);
    SpaunPipeline::argmax(&logits)
}

/// Task 2: Serial Recall — see sequence [a,b,c], reproduce in order.
/// Uses the full SPAUN pipeline: recalled SP → sp_to_cortex (trained) →
/// GNW → connectome → motor_readout. Reset connectome state before recall
/// for consistent dynamics.
fn task_serial_recall(pipeline: &mut SpaunPipeline, sequence: &[usize], steps_per_digit: usize) -> Vec<usize> {
    let mut output = Vec::new();

    // Phase 1: Encode and store in working memory
    for &digit in sequence {
        let pixels = digit_pixels(digit);
        let sp = pipeline.encode_pixels(&pixels);
        pipeline.wm.store(&sp);
    }

    // Phase 2: Reset connectome state for clean recall dynamics
    pipeline.reset_state();

    // Phase 3: Recall from WM through full pipeline
    for slot in 0..sequence.len() {
        let sp = pipeline.wm.recall(slot);
        // Full pipeline: SP → trained cortex → GNW → connectome → motor_readout
        let digit = pipeline.decode_sp_full_pipeline(&sp, steps_per_digit);
        output.push(digit);
    }

    output
}

/// Task 3: Counting — see N, output 1, 2, ..., N.
fn task_counting(pipeline: &mut SpaunPipeline, n: usize, steps_per_digit: usize) -> Vec<usize> {
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
        let count_pixels = digit_pixels(count);
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
/// Train motor readout + w_sp_to_cortex using NLMS on all 10 digits.
///
/// Phase 1: Train w_motor (cortex → digit)
/// Phase 2: Record (SP, initial_cortex) pairs, train w_sp_to_cortex via NLMS
///          to produce stable initial_cortex values for each SP.
fn train_pipeline(pipeline: &mut SpaunPipeline, n_epochs: usize, steps: usize, lr: f32) -> f32 {
    let n_out = pipeline.n_outputs;
    let n_cortex = pipeline.cortical_dim;
    let sp_dim = pipeline.sp_dim;
    
    // Weights accumulate over epochs (no reset)
    let mut final_loss = 0.0f32;
    
    // ── Phase 1: Train w_motor ──────────────────────
    for epoch in 0..n_epochs {
        let mut epoch_loss = 0.0f32;
        
        for digit in 0..10 {
            let pixels = digit_pixels(digit);
            
            // Encode → GNW → reservoir
            let sp = pipeline.encode_pixels(&pixels);
            let mut cortex = pipeline.sp_to_cortex(&sp);
            // Store (SP, initial_cortex) pair from the LAST epoch
            if epoch == n_epochs - 1 {
                pipeline.store_template(digit, &sp, &cortex);
            }
            
            for _ in 0..steps {
                pipeline.gnw.forward(&mut cortex);
                for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
                pipeline.connectome_step(&cortex);
            }
            
            // Get current prediction
            let logits = pipeline.motor_readout(&cortex);
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            if sum > 1e-10 { for e in &mut exps { *e /= sum; } }
            
            // Cross-entropy loss
            let prob = exps[digit].max(1e-10);
            epoch_loss -= prob.ln();
            
            // NLMS update on w_motor
            for k in 0..n_out {
                let target = if k == digit { 1.0f32 } else { 0.0f32 };
                let error = target - exps[k];
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
    
    // ── Phase 2: Train w_sp_to_cortex via NLMS ─────
    // Teacher-forcing on WM-projected SPs (one at a time, clean WM per digit).
    // This ensures w_sp_to_cortex learns to handle the W_in projection distortion
    // that WorkingMemory applies during store().
    println!("  Training w_sp_to_cortex via NLMS (WM-projected SP, clean per digit)...");
    let sp2c_lr = 0.05;
    for _ in 0..10 {
        for digit in 0..10 {
            // Fresh WM per digit: store one SP, recall from slot 0
            pipeline.wm.clear();
            let pixels = digit_pixels(digit);
            let sp = pipeline.encode_pixels(&pixels);
            pipeline.wm.store(&sp);
            let projected_sp = pipeline.wm.recall(0);
            
            let target = &pipeline.target_cortex[digit];
            
            // Train w_sp_to_cortex on PROJECTED SP → target cortex
            for j in 0..n_cortex {
                let mut pred = 0.0f32;
                for i in 0..sp_dim {
                    pred += pipeline.w_sp_to_cortex[j * sp_dim + i] * projected_sp[i];
                }
                let error = target[j] - pred;
                nlms_train_weights(
                    &mut pipeline.w_sp_to_cortex[j * sp_dim..(j + 1) * sp_dim],
                    &projected_sp, error, sp2c_lr, 1e-6,
                );
            }
        }
    }
    pipeline.wm.clear();
    
    // Verify: clean WM per digit, same as Serial Recall would use
    println!("  Verifying on clean WM-projected SPs...");
    pipeline.reset_state();
    let mut sp2c_correct = 0usize;
    for digit in 0..10 {
        pipeline.wm.clear();
        let pixels = digit_pixels(digit);
        let sp = pipeline.encode_pixels(&pixels);
        pipeline.wm.store(&sp);
        let projected_sp = pipeline.wm.recall(0);
        
        let mut cortex = pipeline.sp_to_cortex(&projected_sp);
        for _ in 0..steps {
            pipeline.gnw.forward(&mut cortex);
            for c in cortex.iter_mut() { *c = c.clamp(-5.0, 5.0); }
            pipeline.connectome_step(&cortex);
        }
        let logits = pipeline.motor_readout(&cortex);
        let pred = SpaunPipeline::argmax(&logits);
        if pred == digit { sp2c_correct += 1; }
    }
    pipeline.wm.clear();
    println!("  w_sp_to_cortex post-train accuracy (post-WM): {}/10 ({:.0}%)",
        sp2c_correct, sp2c_correct as f64 * 10.0);
    
    final_loss
}

fn main() {
    println!("{}", "═".repeat(60));
    println!("  🧠 SPAUN Pipeline Demo — Cognitive Architecture on Connectome");
    println!("{}", "═".repeat(60));
    println!();

    // Load connectome from examples/data
    let brain_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("examples").join("data").join("brain.taas");
    println!("  Loading connectome from: {}", brain_path.display());
    let reader = TaasReader::from_file(brain_path.as_os_str().to_str().unwrap())
        .expect("Failed to load brain.taas. Generate one with --features connectome");
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
        train_ms, train_ms / 50.0, final_loss);
    println!();

    // ── BG Action Selection Test ────────────────────────
    println!("─── Basal Ganglia Action Selection ───");
    println!("  BG: 3 cortical channels → 3 actions (Copy=0, SerialRecall=1, Count=2)");
    println!();
    
    // Test Copy task via BG
    println!("  --- Cue: Copy (task_cue = [1.0, 0.0, 0.0]) ---");
    let t0 = Instant::now();
    let mut bg_copy_correct = 0usize;
    for digit in 0..10 {
        pipeline.bg.reset();
        let (action, result) = pipeline.bg_select_and_execute(
            &[1.0, 0.0, 0.0], digit, steps_per_digit,
        );
        let ok = result == digit;
        if ok { bg_copy_correct += 1; }
        if ok { total_correct += 1; }
        total_tested += 1;
        let task_name = match action { 0 => "Copy", 1 => "Recall", 2 => "Count", _ => "?" };
        println!("  Digit {} → BG chose {} ({}), output: {} {}",
            digit, action, task_name, result,
            if ok { "✓" } else { "✗" });
    }
    let bg_copy_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  BG Copy accuracy: {}/10 ({:.0}%) | {:.1}ms",
        bg_copy_correct, bg_copy_correct as f64 * 10.0, bg_copy_ms);
    println!();
    
    // Test Serial Recall via BG
    println!("  --- Cue: Serial Recall (task_cue = [0.0, 1.0, 0.0]) ---");
    let t0 = Instant::now();
    let recall_seq: [&[usize]; 4] = [
        &[3, 7, 1],
        &[0, 5, 9, 2],
        &[4, 4, 4],
        &[8, 1, 6, 3],
    ];
    for (seq_idx, seq) in recall_seq.iter().enumerate() {
        pipeline.bg.reset();
        let (action, matches) = pipeline.bg_select_and_execute(
            &[0.0, 1.0, 0.0], seq_idx, steps_per_digit,
        );
        let perfect = matches == seq.len();
        total_correct += matches;
        total_tested += seq.len();
        let task_name = match action { 0 => "Copy", 1 => "Recall", 2 => "Count", _ => "?" };
        println!("  Sequence {:?} → BG chose {} ({}), matches: {}/{}{}",
            seq, action, task_name, matches, seq.len(),
            if perfect { " ✓" } else { "" });
    }
    let bg_recall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  BG Serial Recall: {:.1}ms", bg_recall_ms);
    println!();
    
    // Test Counting via BG
    println!("  --- Cue: Counting (task_cue = [0.0, 0.0, 1.0]) ---");
    let t0 = Instant::now();
    for n in 1..=5 {
        pipeline.bg.reset();
        let (action, matches) = pipeline.bg_select_and_execute(
            &[0.0, 0.0, 1.0], n - 1, steps_per_digit,
        );
        let perfect = matches == n;
        total_correct += matches;
        total_tested += n;
        let task_name = match action { 0 => "Copy", 1 => "Recall", 2 => "Count", _ => "?" };
        println!("  Count to {} → BG chose {} ({}), matches: {}/{}{}",
            n, action, task_name, matches, n,
            if perfect { " ✓" } else { "" });
    }
    let bg_count_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  BG Counting: {:.1}ms", bg_count_ms);
    println!();

    // ── Summary ───────────────────────────────────────────
    println!("{}", "═".repeat(60));
    println!("  📊 Summary");
    println!("{}", "═".repeat(60));
    println!("  Pipeline: GNW({}) + WM({}×{}) + SP({}) + BG(3→3) + Reservoir({})",
        pipeline.cortical_dim, pipeline.wm.n_slots(), pipeline.sp_dim,
        pipeline.sp_dim, pipeline.n_neurons);
    println!("  Overall accuracy: {}/{} ({:.1}%)", total_correct, total_tested,
        total_correct as f64 / total_tested.max(1) as f64 * 100.0);
    println!();
    println!("  Basal Ganglia routes action selection: task cue → SP → GNW → BG → action.");
    println!("  BG weights are random — selection accuracy depends on cortico-striatal mapping.");
    println!("  Future: train BG via dopamine-STDP for task-switching learning.");
    println!("{}", "═".repeat(60));
}
