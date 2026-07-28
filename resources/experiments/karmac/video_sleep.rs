/// 🧠 Video → Wake → Sleep → Recall: Sleep Consolidation Experiment
///
/// ## Pipeline
///
/// 1. **Encode**: MNIST digit → Poisson spike trains (32 input channels)
/// 2. **WAKE**: Inject spikes into SBM connectome (256 LIF neurons) → record raster
/// 3. **SLEEP**: Replay recorded spikes × N_EPOCHS with STDP plasticity
/// 4. **RECALL**: Show SAME digit + DIFFERENT digit → compare before/after
///
/// ## Hypothesis
///
/// After sleep replay with STDP, the network should show:
/// - Higher firing rate for the seen digit (consolidation)
/// - Lower firing rate for unseen digits (pattern separation)
/// - More stable representation (less variance across recall trials)
///
/// Run: cargo run --package karmac --example video_sleep --release
use std::fs;
use std::path::Path;
use std::time::Instant;

use taas_connectome::ConnectomeData;
use taas_connectome::sim::{
    BrainState, NeuronModel,
    csr_propagate_from_queue, push_spikes_to_queue, record_spikes,
};

// ═══════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════

const N_NEURONS: usize = 256;      // total neurons
const N_INPUT: usize = 32;         // visual input group
const N_BLOCKS: usize = 4;         // SBM blocks
const WITHIN_PROB: f32 = 0.50;     // intra-block connection prob (dense modules)
const BETWEEN_PROB: f32 = 0.03;    // inter-block connection prob

const DT: f32 = 0.5;                // ms per step
const TAU_REF: f32 = 2.0;           // refractory period (ms)
const TAU_M: f32 = 10.0;            // LIF time constant
const V_REST: f32 = -65.0;          // resting potential
const V_THRESH: f32 = -50.0;        // spike threshold
const V_RESET: f32 = -70.0;         // reset potential

const N_WAKE_STEPS: usize = 200;    // 100 ms
const N_SLEEP_EPOCHS: usize = 8;    // number of sleep replay epochs
const N_SLEEP_STEPS: usize = 200;   // 100 ms per epoch (full replay of wake)
const N_RECALL_STEPS: usize = 200;  // 100 ms

/// Direct current scaling: I_input[i] = pixel_intensity[i] × WAKE_I_SCALE
/// Only pixels > INPUT_THRESH get current — creates sparse input patterns.
/// I=60, threshold=0.3: bright pixels get I=60, dim/subthreshold get 0.
/// With LIF: v_step = 60 * 0.0488 = 2.93 mV, fires in ~3 steps.
const WAKE_I_SCALE: f32 = 60.0;
const INPUT_THRESH: f32 = 0.3;    // ignore pixels below this intensity
/// Replay input strength (maps saved 0/1 spikes to current)
const REPLAY_STRENGTH: f32 = 60.0; // Tonic strength: rate_hz * 60 / 1000 = µA. At 10 Hz → 0.6 µA, at 50 Hz → 3.0 µA

// STDP parameters
const TAU_PRE: f32 = 20.0;    // ms — pre-synaptic trace time constant
const TAU_POST: f32 = 20.0;   // ms — post-synaptic trace time constant
const LR_LTP: f32 = 0.005;    // LTP learning rate
const LR_LTD: f32 = 0.005;    // LTD learning rate
const W_MIN: f32 = 0.0;       // min synaptic weight
const W_MAX: f32 = 10.0;      // max synaptic weight (must be > max init weight ~8.0)

const DATA_DIR: &str = "examples/data";

// ═══════════════════════════════════════════════════════════
//  MNIST Loading
// ═══════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════
//  Poisson Encoding
// ═══════════════════════════════════════════════════════════

/// Build external_I array from pixel intensities (direct current injection).
///
/// Each input neuron gets I[i] = pixel_intensity[i] × I_SCALE for all steps,
/// but ONLY if intensity > INPUT_THRESH. This creates sparse input patterns
/// that make STDP learning more effective.
fn build_input_current(
    pixel_intensities: &[f32],
    n_neurons: usize,
    n_steps: usize,
    i_scale: f32,
) -> Vec<Vec<f32>> {
    let n = pixel_intensities.len().min(n_neurons);
    let mut per_step = Vec::with_capacity(n_steps);
    for _step in 0..n_steps {
        let mut cur = vec![0.0f32; n_neurons];
        for i in 0..n {
            if pixel_intensities[i] > INPUT_THRESH {
                cur[i] = pixel_intensities[i] * i_scale;
            }
        }
        per_step.push(cur);
    }
    per_step
}

// ═══════════════════════════════════════════════════════════
//  SBM Connectome Generator
// ═══════════════════════════════════════════════════════════

fn generate_sbm_connectome(
    n: usize,
    n_blocks: usize,
    exc_ratio: f32,   // fraction of excitatory neurons (e.g. 0.8)
) -> ConnectomeData {
    let mut rng_state = 42u32;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_state >> 8) as f32 / 16777216.0
    };

    // Assign blocks and E/I types
    let block_of: Vec<usize> = (0..n).map(|i| (i * n_blocks / n) % n_blocks).collect();
    let nt_types: Vec<f32> = (0..n).map(|i| if next_f32() < exc_ratio { 1.0 } else { -1.0 }).collect();

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptr.push(0);
    for i in 0..n {
        let block_i = block_of[i];
        let mut n_syn = 0;
        for j in 0..n {
            if i == j { continue; }  // no self-connections
            let block_j = block_of[j];
            let prob = if block_i == block_j { WITHIN_PROB } else { BETWEEN_PROB };
            if next_f32() < prob {
                // Weight based on E/I: positive for excitatory, negative from inhibitory
                // Stronger weights: input neuron spikes at 200 Hz → hidden neurons should fire
                // With w=5 avg: v_jump = 5 * 0.0488 = 0.24 mV/spike
                // 10 spikes from pre-neuron → 2.4 mV, approaching threshold
                let w = if nt_types[i] > 0.0 {
                    next_f32() * 6.0 + 2.0    // [2.0, 8.0] excitatory
                } else {
                    -(next_f32() * 6.0 + 2.0) // [-8.0, -2.0] inhibitory
                };
                col_indices.push(j as i32);
                values.push(w);
                n_syn += 1;
            }
        }
        row_ptr.push(row_ptr[i] + n_syn);
    }

    let n_synapses = values.len();
    println!("  SBM: {n} neurons, {n_synapses} synapses");

    ConnectomeData::new(
        row_ptr,
        col_indices,
        values,
        nt_types,
        vec![
            ("input", (0..N_INPUT).collect()),
            ("hidden", (N_INPUT..n).collect()),
        ],
    )
}

// ═══════════════════════════════════════════════════════════
//  STDP — Simple per-neuron trace-based STDP
// ═══════════════════════════════════════════════════════════

struct Stdplr {
    /// Per-neuron pre-synaptic trace (decays, increments on neuron's own spikes)
    trace_pre: Vec<f32>,
    /// Per-neuron post-synaptic trace (decays, increments when neuron receives input)
    trace_post: Vec<f32>,

    decay_pre: f32,   // exp(-dt / tau_pre)
    decay_post: f32,  // exp(-dt / tau_post)

    lr_ltp: f32,
    lr_ltd: f32,
    w_min: f32,
    w_max: f32,
}

impl Stdplr {
    fn new(n: usize, dt: f32) -> Self {
        let decay_pre = (-dt / TAU_PRE).exp();
        let decay_post = (-dt / TAU_POST).exp();
        Self {
            trace_pre: vec![0.0; n],
            trace_post: vec![0.0; n],
            decay_pre,
            decay_post,
            lr_ltp: LR_LTP,
            lr_ltd: LR_LTD,
            w_min: W_MIN,
            w_max: W_MAX,
        }
    }

    /// One step of STDP: update traces, then modify weights for spiked neurons.
    ///
    /// `spikes[i]`: 1 if neuron i spiked this step, 0 otherwise.
    /// `weights`: connectome values (synaptic weights), modified in-place.
    /// `row_ptr`, `col_indices`: CSR structure (immutable references to connections).
    fn step(
        &mut self,
        spikes: &[i32],
        weights: &mut [f32],
        row_ptr: &[i32],
        col_indices: &[i32],
    ) {
        let n = self.trace_pre.len().min(spikes.len());

        // 1. Decay traces
        for i in 0..n {
            self.trace_pre[i] *= self.decay_pre;
            self.trace_post[i] *= self.decay_post;
        }

        // 2. Spike-triggered trace increment
        for i in 0..n {
            if spikes[i] != 0 {
                self.trace_pre[i] += 1.0;
                // Post-synaptic: when a neuron spikes, its post-synaptic trace
                // captures "receiving a spike" for feedback to incoming synapses
                self.trace_post[i] += 1.0;
            }
        }

        // 3. Weight update: for each spiked neuron, iterate its outgoing synapses
        let max_syn = col_indices.len().min(weights.len());
        for i in 0..n {
            if spikes[i] == 0 { continue; }

            let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
            let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);

            // STDP rule (Bi & Poo 1998):
            //   Pre-spike  → LTD: Δw = -A_ltd · trace_post[post]  (post fired before pre)
            //   Post-spike → LTP: Δw = +A_ltp · trace_pre[pre]   (pre fired before post)

            let mut j = start;
            while j + 4 <= end {
                let p0 = col_indices[j].max(0) as usize;
                let p1 = col_indices[j + 1].max(0) as usize;
                let p2 = col_indices[j + 2].max(0) as usize;
                let p3 = col_indices[j + 3].max(0) as usize;

                // Pre-spike → LTD: weight -= A_ltd · trace_post[post]
                if p0 < n { weights[j] -= self.lr_ltd * self.trace_post[p0]; }
                if p1 < n { weights[j + 1] -= self.lr_ltd * self.trace_post[p1]; }
                if p2 < n { weights[j + 2] -= self.lr_ltd * self.trace_post[p2]; }
                if p3 < n { weights[j + 3] -= self.lr_ltd * self.trace_post[p3]; }

                j += 4;
            }
            while j < end {
                let post = col_indices[j].max(0) as usize;
                if post < n {
                    weights[j] -= self.lr_ltd * self.trace_post[post];
                }
                j += 1;
            }
        }

        // Post-spike events: when neuron j spikes, its incoming synapses
        // from neurons that spiked recently get LTP
        for j in 0..n {
            if spikes[j] == 0 { continue; }
            // Post-spike → LTP: weight += A_ltp · trace_pre[pre]
            for pre in 0..n {
                let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
                let end = (row_ptr.get(pre + 1).copied().unwrap_or(0) as usize).min(max_syn);
                for si in start..end {
                    if col_indices[si].max(0) as usize == j {
                        weights[si] += self.lr_ltp * self.trace_pre[pre];
                    }
                }
            }
        }

        // 4. Clamp weights
        for w in weights.iter_mut() {
            *w = w.clamp(self.w_min, self.w_max);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Custom Sleep Simulation with STDP
// ═══════════════════════════════════════════════════════════

/// Build LIF params packed layout for `taas_kernels::safe::lif_step`.
fn build_lif_packed(n: usize, dt: f32) -> Vec<f32> {
    let n_padded = n.div_ceil(8) * 8;
    let mut packed = vec![0.0f32; n_padded * 5];
    packed[0] = dt;
    for i in 0..n {
        packed[n_padded + i] = TAU_M;
        packed[2 * n_padded + i] = V_REST;
        packed[3 * n_padded + i] = V_THRESH;
        packed[4 * n_padded + i] = V_RESET;
    }
    for i in n..n_padded {
        packed[n_padded + i] = TAU_M;
        packed[2 * n_padded + i] = V_REST;
        packed[3 * n_padded + i] = V_THRESH;
        packed[4 * n_padded + i] = V_RESET;
    }
    packed
}

fn sleep_epoch(
    state: &mut BrainState,
    packed_params: &[f32],
    tonic_current: &[f32],
    n_steps: usize,
    weights: &mut [f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    nt_types: &[f32],
    stdp: &mut Stdplr,
    scale: f32,
    mut spike_buf: Option<&mut [i32]>,  // optional spike recording buffer [n_steps × n]
) -> u32 {
    let n = state.n_neurons;
    let n_padded = state.n_padded;

    let mut total_spikes = 0u32;

    for step in 0..n_steps {
        // 1. Add tonic current (constant throughout epoch)
        for i in 0..n.min(tonic_current.len()) {
            state.I_buf[i] += tonic_current[i];
        }

        // 2. LIF step
        taas_kernels::safe::lif_step(
            &mut state.v,
            &mut state.I_buf,
            &mut state.spikes,
            packed_params,
            &mut state.refractory,
            TAU_REF,
            None::<&[f32]>,
        );

        // 3. Push spikes to queue & record
        push_spikes_to_queue(&state.spikes, n, &mut state.queue);

        // 4. CSR propagate through connectome
        csr_propagate_from_queue(
            row_ptr, col_indices, weights, nt_types,
            &mut state.queue, &mut state.I_buf, scale,
        );

        // 5. STDP update
        stdp.step(&state.spikes[..n], weights, row_ptr, col_indices);

        // 6. Record to buffer if provided
        if let Some(buf) = spike_buf.as_mut() {
            record_spikes(&state.spikes, n, buf, step);
        }

        // 7. Clear I_buf for next step
        state.I_buf[..n].fill(0.0);
        for i in n..n_padded {
            state.I_buf[i] = 0.0;
        }

        // Count spikes
        total_spikes += state.spikes[..n].iter().filter(|&&s| s > 0).count() as u32;
    }

    total_spikes
}

// ═══════════════════════════════════════════════════════════
//  Analysis
// ═══════════════════════════════════════════════════════════

/// Compute population firing rate from spike buffer.
fn pop_rate(spike_buffer: &[i32], n_neurons: usize, n_steps: usize) -> f32 {
    let mut total = 0u32;
    for step in 0..n_steps {
        let off = step * n_neurons;
        for i in 0..n_neurons {
            if spike_buffer[off + i] > 0 { total += 1; }
        }
    }
    total as f32 / n_steps as f32 / n_neurons as f32 * 1000.0 / DT // Hz
}

/// Compute pattern similarity between two spike buffers.
/// Uses population vector correlation: for each neuron, rate over entire trial.
fn pattern_similarity(buf1: &[i32], buf2: &[i32], n: usize, n_steps: usize) -> f32 {
    // Compute per-neuron rates
    let mut r1 = vec![0.0f32; n];
    let mut r2 = vec![0.0f32; n];
    for step in 0..n_steps {
        let off1 = step * n;
        let off2 = step * n;
        for i in 0..n {
            if buf1[off1 + i] > 0 { r1[i] += 1.0; }
            if buf2[off2 + i] > 0 { r2[i] += 1.0; }
        }
    }
    // Normalize to rates
    let t = (n_steps as f32 * DT / 1000.0).max(1e-6);
    for r in r1.iter_mut() { *r /= t; }
    for r in r2.iter_mut() { *r /= t; }

    // Cosine similarity
    let dot: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| a * b).sum();
    let n1: f32 = r1.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-10);
    let n2: f32 = r2.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-10);
    dot / (n1 * n2)
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════

fn main() {
    let t0 = Instant::now();
    println!("{}", "═".repeat(72));
    println!("  🧠 VIDEO → WAKE → SLEEP → RECALL");
    println!("  Sleep Consolidation with STDP on LIF Connectome");
    println!("{}", "═".repeat(72));
    println!();

    // ─── 0. Load MNIST ───
    println!("─── 0. Loading MNIST data ───");
    fs::create_dir_all(DATA_DIR).ok();
    for f in &["train-images-idx3-ubyte", "train-labels-idx1-ubyte"] {
        check_data_file(&format!("{DATA_DIR}/{f}"));
    }

    let train_images = load_images(&format!("{DATA_DIR}/train-images-idx3-ubyte"));
    let train_labels = load_labels(&format!("{DATA_DIR}/train-labels-idx1-ubyte"));
    println!("  {} training images", train_images.len());
    println!();

    // Find first image of digit 3 and digit 7
    let img_idx_3 = train_labels.iter().position(|&l| l == 3).unwrap_or(0);
    let img_idx_7 = train_labels.iter().position(|&l| l == 7).unwrap_or(1);
    let digit_3_pixels = &train_images[img_idx_3];
    let digit_7_pixels = &train_images[img_idx_7];

    println!("  Using digit 3 (img {img_idx_3}) and digit 7 (img {img_idx_7})");
    println!();

    // ─── 1. Build connectome ───
    println!("─── 1. Building SBM Connectome ───");
    let mut conn = generate_sbm_connectome(N_NEURONS, N_BLOCKS, 0.8);
    let n = conn.n_neurons;
    let n_syn = conn.n_synapses;
    println!("  {n} neurons, {n_syn} synapses");
    println!("  Input group: {} neurons (indices 0..{})", N_INPUT, N_INPUT);
    println!();

    // ─── 2. Encode MNIST digits as Poisson spike trains ───
    println!("─── 2. Poisson encoding ───");

    // Downsample 784 pixels → N_INPUT channels by averaging blocks
    let block_size = 784 / N_INPUT; // 24.5 → 24 pixels per block
    let mut encode_digit = |pixels: &[f32]| -> Vec<f32> {
        let mut rates = vec![0.0f32; N_INPUT];
        for ch in 0..N_INPUT {
            let start = ch * block_size;
            let end = ((ch + 1) * block_size).min(784).min(pixels.len());
            let sum: f32 = pixels[start..end].iter().sum();
            rates[ch] = sum / (end - start).max(1) as f32; // mean intensity
        }
        rates
    };

    let rates_3 = encode_digit(digit_3_pixels);
    let rates_7 = encode_digit(digit_7_pixels);

    // Show input patterns
    println!("  Digit 3 input rates:");
    for (ch, &r) in rates_3.iter().enumerate().step_by(4) {
        print!("    ch{ch:>2}: {r:.2}");
        if ch + 1 < N_INPUT { print!("  ch{}: {:.2}", ch + 1, rates_3[ch + 1]); }
        if ch + 2 < N_INPUT { print!("  ch{}: {:.2}", ch + 2, rates_3[ch + 2]); }
        if ch + 3 < N_INPUT { print!("  ch{}: {:.2}", ch + 3, rates_3[ch + 3]); }
        println!();
    }
    println!("  Digit 7 input rates:");
    for (ch, &r) in rates_7.iter().enumerate().step_by(4) {
        print!("    ch{ch:>2}: {r:.2}");
        if ch + 1 < N_INPUT { print!("  ch{}: {:.2}", ch + 1, rates_7[ch + 1]); }
        if ch + 2 < N_INPUT { print!("  ch{}: {:.2}", ch + 2, rates_7[ch + 2]); }
        if ch + 3 < N_INPUT { print!("  ch{}: {:.2}", ch + 3, rates_7[ch + 3]); }
        println!();
    }
    println!();

    // Build input currents (direct injection, one per step)
    let wake_input_3 = build_input_current(&rates_3, N_INPUT, N_WAKE_STEPS, WAKE_I_SCALE);

    let mean_input_current: f32 = wake_input_3.iter()
        .flat_map(|I| I.iter().take(N_INPUT))
        .sum::<f32>() / (N_WAKE_STEPS * N_INPUT) as f32;
    println!("  Input current (digit 3): mean = {mean_input_current:.2} µA/cm² (N_INPUT={N_INPUT})");
    println!();

    // ─── 3. WAKE: first simulation (before sleep) ───
    println!("─── 3. WAKE (before sleep) ───");

    let packed_params = build_lif_packed(n, DT);
    let mut state = BrainState::new(n, NeuronModel::LIF);
    let n_padded = state.n_padded;

    let mut spike_buf_pre = vec![0i32; N_WAKE_STEPS * n];

    for step in 0..N_WAKE_STEPS {
        // 1. Add direct current input to input neurons
        for i in 0..N_INPUT {
            state.I_buf[i] += wake_input_3[step][i];
        }

        // 2. LIF step
        taas_kernels::safe::lif_step(
            &mut state.v,
            &mut state.I_buf,
            &mut state.spikes,
            &packed_params,
            &mut state.refractory,
            TAU_REF,
            None::<&[f32]>,
        );

        // 3. Push to queue & propagate
        push_spikes_to_queue(&state.spikes, n, &mut state.queue);
        csr_propagate_from_queue(
            &conn.row_ptr, &conn.col_indices, &conn.values,
            &conn.nt_types, &mut state.queue, &mut state.I_buf, 1.0,
        );

        // 4. Record spikes
        record_spikes(&state.spikes, n, &mut spike_buf_pre, step);

        // 5. Clear I_buf
        state.I_buf[..n].fill(0.0);
        for i in n..n_padded { state.I_buf[i] = 0.0; }
    }

    let pre_spikes_total: i32 = spike_buf_pre.iter().sum();
    let pre_rate = pop_rate(&spike_buf_pre, n, N_WAKE_STEPS);
    println!("  Total spikes: {pre_spikes_total} (pop rate: {pre_rate:.1} Hz)");

    // Count input neurons activity
    let pre_input_rate: f32 = {
        let mut input_spks = 0u32;
        for step in 0..N_WAKE_STEPS {
            for i in 0..N_INPUT.min(n) {
                if spike_buf_pre[step * n + i] > 0 { input_spks += 1; }
            }
        }
        input_spks as f32 / N_INPUT as f32 / (N_WAKE_STEPS as f32 * DT / 1000.0)
    };
    println!("  Input neuron rate: {pre_input_rate:.1} Hz");

    // Sleep doesn't need state snapshot — each epoch starts fresh (like wake did)

    // ─── 4. SLEEP: replay with STDP ───
    println!();
    println!("─── 4. SLEEP ({N_SLEEP_EPOCHS} epochs of replay) ───");

    // Build tonic current for replay: ALL wake-active neurons get a fixed current.
    // This mimics NREM sleep slow oscillations: UP states uniformly depolarize
    // recently active populations regardless of their exact firing rates.
    // Fixed current = WAKE_I_SCALE * 0.75 = 45 µA — V_inf = -20 > -50 guarantees spiking
    let replay_current = WAKE_I_SCALE * 0.75;
    let mut tonic_current = vec![0.0f32; n];
    for step in 0..N_WAKE_STEPS {
        let off = step * n;
        for i in 0..n {
            if spike_buf_pre[off + i] > 0 {
                tonic_current[i] = replay_current;
            }
        }
    }
    let n_tonic = tonic_current.iter().filter(|&&c| c > 0.0).count();
    println!("  Tonic replay: {n_tonic} neurons get fixed I = {replay_current:.0} µA");

    let mut stdp = Stdplr::new(n, DT);
    let mut sleep_weights = conn.values.clone();  // modifiable copy

    let mut epoch_spikes = Vec::new();
    let mut epoch_rates = Vec::new();

    // Buffer to record the LAST sleep epoch (for dream injection)
    let mut sleep_raster = vec![0i32; N_SLEEP_STEPS * n];

    for epoch in 0..N_SLEEP_EPOCHS {
        let mut sleep_state = BrainState::new(n, NeuronModel::LIF);

        // Record spikes on the last epoch for dream injection
        let record_buf = if epoch == N_SLEEP_EPOCHS - 1 {
            Some(&mut sleep_raster[..])
        } else {
            None
        };

        let spk = sleep_epoch(
            &mut sleep_state, &packed_params,
            &tonic_current, N_SLEEP_STEPS,
            &mut sleep_weights, &conn.row_ptr, &conn.col_indices, &conn.nt_types,
            &mut stdp, 1.0,
            record_buf,
        );

        epoch_spikes.push(spk);
        let rate = spk as f32 / n as f32 / (N_SLEEP_STEPS as f32 * DT / 1000.0);
        epoch_rates.push(rate);

        let weight_mean = sleep_weights.iter().sum::<f32>() / sleep_weights.len() as f32;
        let weight_std = {
            let m = weight_mean;
            (sleep_weights.iter().map(|w| (w - m).powi(2)).sum::<f32>() / sleep_weights.len() as f32).sqrt()
        };

        println!("  Epoch {:2}: {:5} spikes ({:.1} Hz) | weights: μ={:.4} σ={:.4}",
            epoch + 1, spk, rate, weight_mean, weight_std);
    }

    // ─── 5. DREAM: inject sleep spikes into fresh brain ───
    println!();
    println!("─── 5. DREAM: inject sleep raster into fresh brain ───");

    let sleep_total: i32 = sleep_raster.iter().sum();
    let sleep_rate = pop_rate(&sleep_raster, n, N_SLEEP_STEPS);
    println!("  Sleep raster: {sleep_total} spikes ({sleep_rate:.1} Hz)");

    // Build tonic current from sleep raster (same logic as wake→tonic)
    let mut dream_current = vec![0.0f32; n];
    let dream_inject_strength = WAKE_I_SCALE * 0.5; // 30 µA — moderate drive
    for step in 0..N_SLEEP_STEPS {
        let off = step * n;
        for i in 0..n {
            if sleep_raster[off + i] > 0 {
                dream_current[i] = dream_inject_strength;
            }
        }
    }
    let n_dream = dream_current.iter().filter(|&&c| c > 0.0).count();
    println!("  Dream injection: {n_dream} neurons get I={dream_inject_strength:.0} µA (from sleep raster)");

    // Run fresh brain with dream input (no STDP, original weights)
    let mut dream_state = BrainState::new(n, NeuronModel::LIF);
    let mut spike_buf_dream = vec![0i32; N_SLEEP_STEPS * n];

    for step in 0..N_SLEEP_STEPS {
        for i in 0..n.min(dream_current.len()) {
            dream_state.I_buf[i] += dream_current[i];
        }

        taas_kernels::safe::lif_step(
            &mut dream_state.v, &mut dream_state.I_buf, &mut dream_state.spikes,
            &packed_params, &mut dream_state.refractory, TAU_REF,
            None::<&[f32]>,
        );

        push_spikes_to_queue(&dream_state.spikes, n, &mut dream_state.queue);
        csr_propagate_from_queue(
            &conn.row_ptr, &conn.col_indices, &sleep_weights,
            &conn.nt_types, &mut dream_state.queue, &mut dream_state.I_buf, 1.0,
        );

        record_spikes(&dream_state.spikes, n, &mut spike_buf_dream, step);

        dream_state.I_buf[..n].fill(0.0);
        for i in n..n_padded { dream_state.I_buf[i] = 0.0; }
    }

    let dream_total: i32 = spike_buf_dream.iter().sum();
    let dream_rate = pop_rate(&spike_buf_dream, n, N_SLEEP_STEPS);
    println!("  Dream response: {dream_total} spikes ({dream_rate:.1} Hz)");

    let sim_dream_vs_wake = pattern_similarity(&spike_buf_pre, &spike_buf_dream, n,
        N_WAKE_STEPS.min(N_SLEEP_STEPS));
    println!("  Pattern similarity: sleep raster vs wake raster = {sim_dream_vs_wake:.4}");

    let mean_weight_change = {
        let orig: f32 = conn.values.iter().sum();
        let new: f32 = sleep_weights.iter().sum();
        (new - orig) / orig.max(1e-10) * 100.0
    };
    println!("  Total weight change: {mean_weight_change:.1}%");

    // ─── 6. RECALL: same digit, after sleep ───
    println!();
    println!("─── 6. RECALL (same digit 3, after sleep) ───");

    let mut state_post = BrainState::new(n, NeuronModel::LIF);
    let mut spike_buf_post_same = vec![0i32; N_RECALL_STEPS * n];
    let recall_input_3 = build_input_current(&rates_3, N_INPUT, N_RECALL_STEPS, WAKE_I_SCALE);

    for step in 0..N_RECALL_STEPS {
        for i in 0..N_INPUT {
            state_post.I_buf[i] += recall_input_3[step][i];
        }

        taas_kernels::safe::lif_step(
            &mut state_post.v, &mut state_post.I_buf, &mut state_post.spikes,
            &packed_params, &mut state_post.refractory, TAU_REF,
            None::<&[f32]>,
        );

        push_spikes_to_queue(&state_post.spikes, n, &mut state_post.queue);
        csr_propagate_from_queue(
            &conn.row_ptr, &conn.col_indices, &sleep_weights,
            &conn.nt_types, &mut state_post.queue, &mut state_post.I_buf, 1.0,
        );

        record_spikes(&state_post.spikes, n, &mut spike_buf_post_same, step);

        state_post.I_buf[..n].fill(0.0);
        for i in n..n_padded { state_post.I_buf[i] = 0.0; }
    }

    let post_same_total: i32 = spike_buf_post_same.iter().sum();
    let post_same_rate = pop_rate(&spike_buf_post_same, n, N_RECALL_STEPS);
    println!("  Total spikes: {post_same_total} (pop rate: {post_same_rate:.1} Hz)");

    // ─── 7. RECALL: different digit 7 ───
    println!();
    println!("─── 7. RECALL (different digit 7, after sleep) ───");

    let mut state_diff = BrainState::new(n, NeuronModel::LIF);
    let mut spike_buf_post_diff = vec![0i32; N_RECALL_STEPS * n];
    let recall_input_7 = build_input_current(&rates_7, N_INPUT, N_RECALL_STEPS, WAKE_I_SCALE);

    for step in 0..N_RECALL_STEPS {
        for i in 0..N_INPUT {
            state_diff.I_buf[i] += recall_input_7[step][i];
        }

        taas_kernels::safe::lif_step(
            &mut state_diff.v, &mut state_diff.I_buf, &mut state_diff.spikes,
            &packed_params, &mut state_diff.refractory, TAU_REF,
            None::<&[f32]>,
        );

        push_spikes_to_queue(&state_diff.spikes, n, &mut state_diff.queue);
        csr_propagate_from_queue(
            &conn.row_ptr, &conn.col_indices, &sleep_weights,
            &conn.nt_types, &mut state_diff.queue, &mut state_diff.I_buf, 1.0,
        );

        record_spikes(&state_diff.spikes, n, &mut spike_buf_post_diff, step);

        state_diff.I_buf[..n].fill(0.0);
        for i in n..n_padded { state_diff.I_buf[i] = 0.0; }
    }

    let post_diff_total: i32 = spike_buf_post_diff.iter().sum();
    let post_diff_rate = pop_rate(&spike_buf_post_diff, n, N_RECALL_STEPS);
    println!("  Total spikes: {post_diff_total} (pop rate: {post_diff_rate:.1} Hz)");

    // ─── 8. ANALYSIS ───
    println!();
    println!("{}", "═".repeat(72));
    println!("  📊 ANALYSIS");
    println!("{}", "═".repeat(72));
    println!();

    // Rate comparison
    let rate_change_same = if pre_rate > 0.0 {
        (post_same_rate - pre_rate) / pre_rate * 100.0
    } else {
        0.0
    };

    println!("  Population rates:");
    println!("    Before sleep:   {pre_rate:.2} Hz (digit 3)");
    println!("    After sleep:    {post_same_rate:.2} Hz (digit 3)");
    println!("    Change:         {rate_change_same:+.1}%");
    println!("    Different (7):  {post_diff_rate:.2} Hz");
    println!("    Dream response: {dream_rate:.2} Hz");

    // Pattern similarity: same digit pre vs post
    let sim_same = pattern_similarity(&spike_buf_pre, &spike_buf_post_same, n,
        N_WAKE_STEPS.min(N_RECALL_STEPS));
    println!();
    println!("  Pattern similarity (cosine):");
    println!("    Same digit (3):  {sim_same:.4}");

    // Similarity: same digit post vs different digit
    let sim_diff = pattern_similarity(&spike_buf_post_same, &spike_buf_post_diff, n,
        N_RECALL_STEPS.min(N_RECALL_STEPS));
    println!("    Same vs diff:   {sim_diff:.4}");

    println!("    Wake vs Dream:  {sim_dream_vs_wake:.4}");

    // Selectivity
    let selectivity_post = sim_same - sim_diff;
    println!();
    println!("  Selectivity (same - diff): {selectivity_post:.4}");

    // Is dream more like wake or like different?
    let dream_vs_wake_gap = sim_dream_vs_wake - sim_diff;
    println!("  Dream is {:.1}% more like the seen digit than the unseen one",
        dream_vs_wake_gap * 100.0);

    // STDP statistics
    let n_strengthened = sleep_weights.iter().zip(conn.values.iter())
        .filter(|(new, old)| *new > *old).count();
    let n_weakened = sleep_weights.iter().zip(conn.values.iter())
        .filter(|(new, old)| *new < *old).count();
    println!();
    println!("  STDP weight changes:");
    println!("    Strengthened: {n_strengthened}/{n_syn} ({:.1}%)",
        n_strengthened as f32 / n_syn as f32 * 100.0);
    println!("    Weakened:     {n_weakened}/{n_syn} ({:.1}%)",
        n_weakened as f32 / n_syn as f32 * 100.0);
    println!("    Total change: {mean_weight_change:.1}%");

    // Conclusion
    println!();
    println!("{}", "═".repeat(72));
    if rate_change_same > 5.0 {
        println!("  ✅ SLEEP CONSOLIDATION DETECTED!");
        println!("  Population rate increased by {:.1}% after sleep replay.", rate_change_same);
        println!("  The network shows stronger response to the seen stimulus.");
    } else if rate_change_same > 0.0 {
        println!("  ⚡ Weak consolidation: rate increased by {:.1}%", rate_change_same);
    } else if rate_change_same > -5.0 {
        println!("  ➖ No significant consolidation (rate change: {:.1}%)", rate_change_same);
    } else {
        println!("  ❌ Rate DECREASED by {:.1}% after sleep.", -rate_change_same);
        println!("  Sleep may have caused synaptic depression (homeostatic).");
    }
    if sim_dream_vs_wake > 0.5 {
        println!("  💭 DREAM SIMILAR TO WAKE! (similarity = {:.4})", sim_dream_vs_wake);
        println!("  The brain dreams of what it saw.");
    } else {
        println!("  💤 Dream differs from wake (similarity = {:.4})", sim_dream_vs_wake);
        println!("  Sleep generates its own activity patterns.");
    }
    if selectivity_post > 0.1 {
        println!("  ✅ Pattern separation: same vs diff similarity gap = {:.4}", selectivity_post);
    } else {
        println!("  ➖ No pattern separation (gap = {:.4})", selectivity_post);
    }
    println!("{}", "═".repeat(72));
    println!();
    println!("  Experiment: {:.2}s", t0.elapsed().as_secs_f64());
}
