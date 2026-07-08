//! # FlyWire Chaos vs Pattern — dopamine-STDP на реальном коннектоме
//!
//! ## Эксперимент
//!
//! Берём **реальный FlyWire коннектом** (connections_princeton + visual_neuron_types),
//! выделяем подграф визуальных нейронов (~1000), и обучаем его через dopamine-STDP
//! различать **looming (хаос)** и **coherent motion (паттерн)**:
//!
//! - **Looming stimulus** (CHAOS): случайный шум в LC4/LPLC2 нейроны → высокая энтропия
//! - **Motion stimulus** (PATTERN): когерентная осцилляция в T4/T5 нейроны → низкая энтропия
//!
//! Награда δ = R − α·|H − H_target| максимизируется для паттерна.
//! Синапсы, способствующие синхронизации при паттерне, укрепляются.
//! Синапсы, усиливающие хаос при looming, ослабляются.
//!
//! ## Запуск
//!
//! ```bash
//! # Подготовка данных (один раз, скачать с https://codex.flywire.ai):
//! # zcat /path/to/connections_princeton.csv.gz > /tmp/conns.csv
//! # zcat /path/to/visual_neuron_types.csv.gz > /tmp/visual.csv
//!
//! # Запуск (1000 нейронов, 100 эпизод):
//! cargo run --release --example flywire_chaos_pattern -p connectome-gpt -- \
//!   --connections /tmp/conns.csv --visual-types /tmp/visual.csv \
//!   --max-neurons 1000 --episodes 100
//!
//! # Быстрый тест (200 нейронов, 20 эпизод):
//! cargo run --example flywire_chaos_pattern -p connectome-gpt -- \
//!   --connections /tmp/conns.csv --visual-types /tmp/visual.csv \
//!   --max-neurons 200 --episodes 20
//! ```

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

use taas_core::kuramoto::{kuramoto_step_sparse, phase_coherence};
use taas_core::neuron::DopamineSTDP;
use taas_kernels::spike_entropy;

// ═══════════════════════════════════════════════════════════
//  Config (CLI args override these)
// ═══════════════════════════════════════════════════════════

const DT: f32 = 0.01;              // Kuramoto timestep (s)
const N_OSCILLATOR_STEPS: usize = 200;  // steps per condition per episode
const ALPHA: f32 = 0.3;            // entropy penalty weight
const H_TARGET: f32 = 3.0;         // target entropy (~5% firing for 1000 neurons)
const LR: f32 = 0.1;               // dopamine-STDP learning rate
const W_MIN: f32 = 0.0;
const W_MAX: f32 = 3.0;
const OMEGA_BASE: f32 = 3.0;       // base Kuramoto frequency (rad/s)
const OMEGA_SPREAD: f32 = 1.0;     // frequency spread
const SPIKE_THRESHOLD: f32 = 2.5;  // phase threshold for "spike" detection

// ═══════════════════════════════════════════════════════════
//  CSV streaming
// ═══════════════════════════════════════════════════════════

fn stream_csv<F>(path: &str, mut cb: F) -> Result<usize, String>
where F: FnMut(&[&str]) -> bool {
    let f = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let buf = BufReader::new(f);
    let mut count = 0;
    for line in buf.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        let l = line.trim().to_string();
        if l.is_empty() { continue; }
        let cols: Vec<&str> = l.split(',').collect();
        if !cb(&cols) { break; }
        count += 1;
    }
    Ok(count)
}

// ═══════════════════════════════════════════════════════════
//  LCG RNG (zero-dep)
// ═══════════════════════════════════════════════════════════

struct Lcg { state: u32 }
impl Lcg {
    fn new(seed: u32) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 / 16777216.0
    }
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

// ═══════════════════════════════════════════════════════════
//  Visual type detection
// ═══════════════════════════════════════════════════════════

/// Определяет, является ли тип нейрона looming-чувствительным (LC4, LPLC2, LC11, LC6).
fn is_looming_type(ntype: &str) -> bool {
    matches!(ntype, "LC4" | "LPLC2" | "LC11" | "LC6" | "LC5" | "LC10a" | "LC12" | "LC15")
}

/// Определяет, является ли тип нейрона motion-чувствительным (T4, T5, VS, HS).
fn is_motion_type(ntype: &str) -> bool {
    ntype.starts_with("T4") || ntype.starts_with("T5")
        || ntype.starts_with("VS") || ntype.starts_with("HS")
        || ntype.starts_with("HSE") || ntype.starts_with("HSN") || ntype.starts_with("HSS")
}

/// Определяет, является ли семейство визуальным (для общего пула).
fn is_visual_family(family: &str) -> bool {
    matches!(family,
        "Photo Receptors" | "Lamina Monopolar" | "Medulla Intrinsic"
        | "Transmedullary" | "Transmedullary Y" | "Distal Medulla"
        | "T4 Neuron" | "T5 Neuron" | "LC" | "Lobula Columnar"
        | "Centrifugal" | "Serpentine Medulla" | "T2 Neuron" | "T3 Neuron"
        | "Y Neuron" | "T1 Neuron" | "TmY Neuron"
        | "Lobula Plate" | "Lobula Intrinsic" | "LPTC"
    )
}

// ═══════════════════════════════════════════════════════════
//  Kuramoto helpers
// ═══════════════════════════════════════════════════════════

fn detect_spikes(phases: &[f32], phases_prev: &[f32], spikes: &mut [i32]) {
    for i in 0..phases.len().min(spikes.len()) {
        spikes[i] = if phases[i] > SPIKE_THRESHOLD && phases_prev[i] <= SPIKE_THRESHOLD {
            1
        } else {
            0
        };
    }
}

fn count_spikes(spike_buffer: &[i32], n: usize, n_steps: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; n];
    for step in 0..n_steps {
        let base = step * n;
        for i in 0..n {
            if spike_buffer[base + i] != 0 { counts[i] += 1.0; }
        }
    }
    counts
}

fn build_synapse_spikes(
    syn_pre: &mut [i32], syn_post: &mut [i32],
    spikes: &[i32], row_ptr: &[i32], col_indices: &[i32], n: usize,
) {
    let max_syn = col_indices.len().min(syn_pre.len()).min(syn_post.len());
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        let pre_val = spikes.get(i).copied().unwrap_or(0);
        for idx in start..end {
            let j = col_indices[idx] as usize;
            syn_pre[idx] = pre_val;
            syn_post[idx] = spikes.get(j).copied().unwrap_or(0);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════

fn main() {
    let t0 = Instant::now();

    // ── Parse CLI args ──
    let mut args = std::env::args().skip(1);
    let mut conns_path = String::new();
    let mut visual_path = String::new();
    let mut max_neurons = 500usize;
    let mut episodes = 50usize;
    let mut stim_freq = 1.0f32;
    let mut stim_amp = 8.0f32;
    let mut lr = LR;
    let mut alpha = ALPHA;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--connections" => conns_path = args.next().unwrap_or_default(),
            "--visual-types" => visual_path = args.next().unwrap_or_default(),
            "--max-neurons" => max_neurons = args.next().and_then(|s| s.parse().ok()).unwrap_or(500),
            "--episodes" => episodes = args.next().and_then(|s| s.parse().ok()).unwrap_or(50),
            "--stim-freq" => stim_freq = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0),
            "--stim-amp" => stim_amp = args.next().and_then(|s| s.parse().ok()).unwrap_or(8.0),
            "--lr" => lr = args.next().and_then(|s| s.parse().ok()).unwrap_or(LR),
            "--alpha" => alpha = args.next().and_then(|s| s.parse().ok()).unwrap_or(ALPHA),
            _ => { eprintln!("Usage: --connections <file> --visual-types <file> [--max-neurons N] [--episodes N] [--stim-freq F] [--stim-amp F] [--lr F] [--alpha F]"); return; }
        }
    }

    if conns_path.is_empty() || visual_path.is_empty() {
        eprintln!("Error: --connections and --visual-types are required");
        return;
    }

    println!("{}", "═".repeat(66));
    println!("  🪰 FlyWire Chaos vs Pattern — dopamine-STDP на реальном коннектоме");
    println!("{}", "═".repeat(66));
    println!();
    println!("  ⚙️  Config: neurons≤{}, episodes={}, stim={}Hz×{}, α={}, η={}",
             max_neurons, episodes, stim_freq, stim_amp, alpha, lr);
    println!();

    // ── Step 1: Load visual neuron types ──
    println!("▶ Loading visual neuron types from: {}", visual_path);
    let t1 = Instant::now();
    // root_id → (type, family, subsystem)
    let mut visual_types: HashMap<u64, (String, String, String)> = HashMap::new();
    // Separate indices for looming vs motion neurons
    let mut looming_root_ids: Vec<u64> = Vec::new();
    let mut motion_root_ids: Vec<u64> = Vec::new();
    let mut other_visual_ids: Vec<u64> = Vec::new();

    let line_count = stream_csv(&visual_path, |cols| {
        if cols.len() < 4 || cols[0] == "root_id" { return true; }
        if let Ok(id) = cols[0].trim().parse::<u64>() {
            let ntype = cols.get(1).map(|s| s.trim()).unwrap_or("");
            let family = cols.get(2).map(|s| s.trim()).unwrap_or("");
            let subsystem = cols.get(3).map(|s| s.trim()).unwrap_or("");
            visual_types.insert(id, (ntype.to_string(), family.to_string(), subsystem.to_string()));

            if is_looming_type(ntype) {
                looming_root_ids.push(id);
            } else if is_motion_type(ntype) {
                motion_root_ids.push(id);
            } else if is_visual_family(family) || !ntype.is_empty() {
                other_visual_ids.push(id);
            }
        }
        true
    }).expect("Failed to read visual types");

    println!("   {} visual neurons loaded ({:.2}s)", line_count, t1.elapsed().as_secs_f32());
    println!("   Looming-sensitive (LC4/LPLC2/LC11/LC6): {}", looming_root_ids.len());
    println!("   Motion-sensitive (T4/T5/VS/HS):         {}", motion_root_ids.len());
    println!("   Other visual:                             {}", other_visual_ids.len());
    println!();

    if looming_root_ids.is_empty() || motion_root_ids.is_empty() {
        eprintln!("⚠️  Missing looming or motion neuron types. Need LC4/LPLC2 and T4/T5 neurons.");
        eprintln!("   Looming: {}", looming_root_ids.len());
        eprintln!("   Motion:  {}", motion_root_ids.len());
        return;
    }

    // ── Step 2: Load connections and build subgraph ──
    println!("▶ Loading connections from: {}", conns_path);
    let t2 = Instant::now();

    // PASS 1: count connections per neuron (pre + post weight) to find top-connected
    println!("   Pass 1: counting connections...");
    let mut conn_counts: HashMap<u64, usize> = HashMap::new();
    let mut read_progress = 0usize;
    stream_csv(&conns_path, |cols| {
        if cols.len() < 5 || cols[0] == "pre_root_id" { return true; }
        let pre = match cols[0].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        let post = match cols[1].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        let nt = cols[4].trim();
        // Only count known NT types
        match nt { "ACH" | "GLUT" | "GABA" | "DA" | "SER" | "OCT" => {}
            _ => return true
        }
        *conn_counts.entry(pre).or_insert(0) += 1;
        *conn_counts.entry(post).or_insert(0) += 1;
        read_progress += 1;
        if read_progress % 500_000 == 0 {
            eprint!("\r   ...{}M connections scanned", read_progress / 1_000_000);
        }
        true
    }).expect("Failed to read connections");
    eprintln!("\r   ...{}M connections scanned ✓", read_progress / 1_000_000);
    
    // Pick top max_neurons by connection count
    let mut ranked: Vec<(u64, usize)> = conn_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    let n = max_neurons.min(ranked.len());
    let selected_root_ids: Vec<u64> = ranked.iter().take(n).map(|(id, _)| *id).collect();
    let selected: HashSet<u64> = selected_root_ids.iter().copied().collect();
    println!("   Top-{} neurons by connection count", n);
    
    // Count how many of the top neurons have known visual types
    let visual_in_selection = selected_root_ids.iter()
        .filter(|id| visual_types.contains_key(id)).count();
    println!("   Of these, {} have visual type labels", visual_in_selection);

    // PASS 2: read edges between selected neurons
    println!("   Pass 2: building subgraph edges...");
    let mut raw_edges: Vec<(u64, u64, f32)> = Vec::new();
    let mut total_conns = 0usize;
    let mut nt_counts: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut edge_progress = 0usize;

    stream_csv(&conns_path, |cols| {
        if cols.len() < 5 || cols[0] == "pre_root_id" { return true; }
        let pre = match cols[0].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        let post = match cols[1].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        if !selected.contains(&pre) || !selected.contains(&post) { return true; }
        let syn = match cols[3].trim().parse::<u32>() { Ok(v) => v, _ => return true };
        let nt = cols[4].trim();
        total_conns += 1;
        let sign = match nt {
            "ACH" | "GLUT" => 1.0,
            "GABA" => -1.0,
            "DA" | "SER" | "OCT" => -1.0,
            _ => return true,
        };
        let w = (syn as f32).sqrt().min(3.0) / 5.0 * sign;
        raw_edges.push((pre, post, w));
        let entry = nt_counts.entry(pre).or_insert((0, 0));
        if sign > 0.0 { entry.0 += 1; } else { entry.1 += 1; }
        edge_progress += 1;
        if edge_progress % 10_000 == 0 {
            eprint!("\r   ...{}K edges found", edge_progress / 1_000);
        }
        true
    }).expect("Failed to read connections");
    eprintln!("\r   ...{} edges found ✓", edge_progress);

    println!("   Subgraph: {} edges from {} neurons", raw_edges.len(), n);

    if raw_edges.len() < 100 {
        eprintln!("⚠️  Too few edges ({}) for {n} neurons. Subgraph too sparse.", raw_edges.len());
        return;
    }

    // ── Step 3: Build CSR ──
    println!("▶ Building CSR...");
    let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut nt_types = vec![0.0f32; n];
    let mut looming_idx = Vec::new();
    let mut motion_idx = Vec::new();

    for (i, &root_id) in selected_root_ids.iter().take(n).enumerate() {
        id_to_idx.insert(root_id, i);
        let (exc, inh) = nt_counts.get(&root_id).copied().unwrap_or((1, 0));
        nt_types[i] = if exc >= inh { 1.0 } else { -1.0 };
        if let Some((type_name, _, _)) = visual_types.get(&root_id) {
            if is_looming_type(type_name) { looming_idx.push(i); }
            if is_motion_type(type_name) { motion_idx.push(i); }
        }
    }

    // Build CSR sorted by pre-neuron
    let mut sub_edges: Vec<(usize, usize, f32)> = raw_edges.iter()
        .filter_map(|(pre, post, w)| Some((*id_to_idx.get(pre)?, *id_to_idx.get(post)?, *w)))
        .collect();
    sub_edges.sort_by_key(|(pre, _, _)| *pre);

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::with_capacity(sub_edges.len());
    let mut values = Vec::with_capacity(sub_edges.len());
    let mut e = 0;
    for i in 0..n {
        while e < sub_edges.len() && sub_edges[e].0 == i {
            col_indices.push(sub_edges[e].1 as i32);
            values.push(sub_edges[e].2);
            e += 1;
        }
        row_ptr[i + 1] = col_indices.len() as i32;
    }

    let n_syn = values.len();
    let n_exc = nt_types.iter().filter(|&&t| t > 0.0).count();
    let n_inh = nt_types.iter().filter(|&&t| t < 0.0).count();
    let mean_w = values.iter().map(|w| w.abs()).sum::<f32>() / n_syn as f32;

    println!("   CSR: {} neurons, {} synapses", n, n_syn);
    println!("   E/I: {} / {} (ratio {:.2})", n_exc, n_inh, n_exc as f32 / n_inh.max(1) as f32);
    println!("   Mean |weight|: {:.4}", mean_w);
    println!("   Time: {:.2}s", t2.elapsed().as_secs_f32());
    println!();

    if looming_idx.is_empty() && !looming_root_ids.is_empty() {
        // Fallback: use first neurons as looming
        println!("⚠️  No looming neurons in subgraph. Using first 5% as looming.");
        looming_idx = (0..n / 20).collect();
    }
    if motion_idx.is_empty() && !motion_root_ids.is_empty() {
        println!("⚠️  No motion neurons in subgraph. Using next 5% as motion.");
        motion_idx = (n / 20..n / 10).collect();
    }
    println!("📊 Groups: {} looming, {} motion, {} other",
             looming_idx.len(), motion_idx.len(), n - looming_idx.len() - motion_idx.len());
    println!();

    // ── Step 4: Chaos vs Pattern experiment ──
    println!("{}", "─".repeat(66));
    println!("  🧪 Chaos vs Pattern dopamine-STDP experiment");
    println!("  δ = R − {:.1}·|H − {:.1}|,  η = {}", alpha, H_TARGET, lr);
    println!("  {} episodes, {} steps/condition", episodes, N_OSCILLATOR_STEPS);
    println!("{}", "─".repeat(66));

    // Setup Kuramoto state
    let omegas: Vec<f32> = (0..n)
        .map(|i| OMEGA_BASE + (i as f32 / n as f32 - 0.5) * 2.0 * OMEGA_SPREAD)
        .collect();

    // DopamineSTDP for weight updates
    let stdp_dt = DT * 1000.0;
    let mut stdp = DopamineSTDP::new(n_syn, stdp_dt);
    stdp.tau_pre = 20.0;
    stdp.tau_post = 20.0;
    stdp.tau_e = 500.0;
    stdp.a_ltp = 0.15;
    stdp.a_ltd = 0.12;
    stdp.w_min = W_MIN;
    stdp.w_max = W_MAX;

    let mut mut_values = values.clone();

    // Tracking
    let mut r_chaos_history: Vec<f32> = Vec::with_capacity(episodes);
    let mut r_pattern_history: Vec<f32> = Vec::with_capacity(episodes);
    let mut h_chaos_history: Vec<f32> = Vec::with_capacity(episodes);
    let mut h_pattern_history: Vec<f32> = Vec::with_capacity(episodes);
    let mut delta_history: Vec<f32> = Vec::with_capacity(episodes);
    let mut syn_pre = vec![0i32; n_syn];
    let mut syn_post = vec![0i32; n_syn];

    println!();
    println!("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
             "Ep", "R_chaos", "R_pat", "H_chaos", "H_pat", "δ", "Time");
    println!("  {}", "─".repeat(66));

    for episode in 0..episodes {
        let t_ep = Instant::now();
        let seed_base = episode as u32 * 1337;

        // ── Chaos condition: looming (random noise → desynchronization) ──
        let mut rng_c = Lcg::new(seed_base + 42);
        let mut phases_c: Vec<f32> = (0..n).map(|_| rng_c.next_f32() * std::f32::consts::TAU).collect();
        let mut prev_c = phases_c.clone();
        let mut sin_buf = vec![0.0f32; n];
        let mut cos_buf = vec![0.0f32; n];
        let mut coupling_buf = vec![0.0f32; n];
        let amps = vec![1.0f32; n];
        let mut spikes = vec![0i32; n];
        let mut spike_buf_c = vec![0i32; N_OSCILLATOR_STEPS * n];

        // Pre-generate noise for looming stimulus
        let mut rng_noise = Lcg::new(seed_base + 123);
        let noise: Vec<f32> = (0..N_OSCILLATOR_STEPS).map(|_| rng_noise.next_f32_range(-5.0, 5.0)).collect();

        for step in 0..N_OSCILLATOR_STEPS {
            // Looming stimulus: inject random noise into each looming neuron
            let mut omegas_copy = omegas.clone();
            for &i in &looming_idx {
                omegas_copy[i] += noise[step % noise.len()];
            }
            kuramoto_step_sparse(
                &mut phases_c, &omegas_copy, &row_ptr, &col_indices, &mut_values,
                &mut sin_buf, &mut cos_buf, &mut coupling_buf, DT,
            );
            detect_spikes(&phases_c, &prev_c, &mut spikes);
            prev_c.copy_from_slice(&phases_c);
            let base = step * n;
            for i in 0..n { spike_buf_c[base + i] = spikes[i]; }
        }

        let mut coh_c = vec![0.0f32];
        phase_coherence(&mut coh_c, &sin_buf, &cos_buf, &amps);
        let r_chaos = coh_c[0];
        let (h_chaos, _) = spike_entropy(
            &count_spikes(&spike_buf_c, n, N_OSCILLATOR_STEPS),
            N_OSCILLATOR_STEPS as f32, H_TARGET,
        );

        // ── Pattern condition: coherent motion (sine wave → synchronization) ──
        let mut rng_p = Lcg::new(seed_base + 999);
        let mut phases_p: Vec<f32> = (0..n).map(|_| rng_p.next_f32() * std::f32::consts::TAU).collect();
        let mut prev_p = phases_p.clone();
        let mut sin_buf_p = vec![0.0f32; n];
        let mut cos_buf_p = vec![0.0f32; n];
        let mut coupling_buf_p = vec![0.0f32; n];
        let mut spike_buf_p = vec![0i32; N_OSCILLATOR_STEPS * n];
        let mut spikes_p = vec![0i32; n];
        stdp.clear_eligibility();

        for step in 0..N_OSCILLATOR_STEPS {
            let t = step as f32 * DT;
            let mut omegas_copy = omegas.clone();
            // Motion stimulus: coherent sine wave into each motion neuron
            for &i in &motion_idx {
                let diversity = (i as f32 * 0.1).sin() * 0.3;
                let phase = t * stim_freq * std::f32::consts::TAU;
                omegas_copy[i] += stim_amp * (phase + diversity).sin();
            }
            kuramoto_step_sparse(
                &mut phases_p, &omegas_copy, &row_ptr, &col_indices, &mut_values,
                &mut sin_buf_p, &mut cos_buf_p, &mut coupling_buf_p, DT,
            );
            detect_spikes(&phases_p, &prev_p, &mut spikes_p);
            prev_p.copy_from_slice(&phases_p);
            let base = step * n;
            for i in 0..n { spike_buf_p[base + i] = spikes_p[i]; }
            // STDP eligibility accumulation
            build_synapse_spikes(&mut syn_pre, &mut syn_post, &spikes_p, &row_ptr, &col_indices, n);
            stdp.step(&syn_pre, &syn_post);
        }

        let mut coh_p = vec![0.0f32];
        phase_coherence(&mut coh_p, &sin_buf_p, &cos_buf_p, &amps);
        let r_pattern = coh_p[0];
        let (h_pattern, h_penalty) = spike_entropy(
            &count_spikes(&spike_buf_p, n, N_OSCILLATOR_STEPS),
            N_OSCILLATOR_STEPS as f32, H_TARGET,
        );

        // ── Reward signal ──
        // δ = R_pattern - α·|H_pattern - H_target| - (R_chaos - α·|H_chaos - H_target|)
        // = ΔR - α·Δ|H - H_target|
        // Positive δ means pattern condition is better, so reinforce pattern-driving weights
        let chaos_err = (h_chaos - H_TARGET).abs();
        let pattern_err = h_penalty;
        let delta = (r_pattern - r_chaos) - alpha * (pattern_err - chaos_err);

        // ── Apply dopamine-STDP ──
        // Positive δ → strengthen synapses that fired during pattern
        // Negative δ → weaken them
        stdp.apply_dopamine(&mut mut_values, delta, lr);

        // ── Track ──
        r_chaos_history.push(r_chaos);
        r_pattern_history.push(r_pattern);
        h_chaos_history.push(h_chaos);
        h_pattern_history.push(h_pattern);
        delta_history.push(delta);

        if episode % 5 == 0 || episode == episodes - 1 {
            println!("  {:>5}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>+8.4}  {:>8.0}ms",
                     episode, r_chaos, r_pattern, h_chaos, h_pattern, delta,
                     t_ep.elapsed().as_millis());
        }
    }

    // ── Results ──
    println!("\n{}", "═".repeat(66));
    println!("  📊 Результаты: FlyWire Chaos vs Pattern");
    println!("{}", "═".repeat(66));

    let take = 10.min(episodes);
    let r_chaos_first = r_chaos_history.iter().take(take).sum::<f32>() / take as f32;
    let r_chaos_last = r_chaos_history.iter().rev().take(take).sum::<f32>() / take as f32;
    let r_pat_first = r_pattern_history.iter().take(take).sum::<f32>() / take as f32;
    let r_pat_last = r_pattern_history.iter().rev().take(take).sum::<f32>() / take as f32;
    let h_chaos_avg = h_chaos_history.iter().sum::<f32>() / episodes as f32;
    let h_pat_avg = h_pattern_history.iter().sum::<f32>() / episodes as f32;
    let delta_first = delta_history.iter().take(take).sum::<f32>() / take as f32;
    let delta_last = delta_history.iter().rev().take(take).sum::<f32>() / take as f32;

    println!("\n  ┌──────────────────────────────────────────────────────────────────┐");
    println!("  │  Метрика              │  Начало       │  Конец       │  Δ         │");
    println!("  ├──────────────────────────────────────────────────────────────────┤");
    println!("  │  R (CHAOS = looming)  │  {:.4}      │  {:.4}      │  {:+.4}    │",
             r_chaos_first, r_chaos_last, r_chaos_last - r_chaos_first);
    println!("  │  R (PATTERN = motion)  │  {:.4}      │  {:.4}      │  {:+.4}    │",
             r_pat_first, r_pat_last, r_pat_last - r_pat_first);
    println!("  │  ΔR = R_pat − R_chaos │  {:+.4}      │  {:+.4}      │  {:+.4}    │",
             r_pat_first - r_chaos_first, r_pat_last - r_chaos_last,
             (r_pat_last - r_chaos_last) - (r_pat_first - r_chaos_first));
    println!("  ├──────────────────────────────────────────────────────────────────┤");
    println!("  │  δ (reward signal)    │  {:+.4}      │  {:+.4}      │  {:+.4}    │",
             delta_first, delta_last, delta_last - delta_first);
    println!("  ├──────────────────────────────────────────────────────────────────┤");
    println!("  │  H_chaos avg          │  {:.4}                                    │", h_chaos_avg);
    println!("  │  H_pattern avg        │  {:.4}                                    │", h_pat_avg);
    println!("  └──────────────────────────────────────────────────────────────────┘");

    // R gap analysis
    let gap_first = r_pat_first - r_chaos_first;
    let gap_last = r_pat_last - r_chaos_last;
    let gap_change = (r_pat_last - r_chaos_last) - (r_pat_first - r_chaos_first);
    println!("\n  ─── Gap analysis: R_pattern − R_chaos ───");
    println!("  Initial gap: {:.4}", gap_first);
    println!("  Final gap:   {:.4}", gap_last);
    println!("  Δgap:        {:+.4}", gap_change);

    // R timeline
    let step = (episodes / 10).max(1);
    println!("\n  ─── R timeline ───");
    println!("  Ep    R_chaos    R_pattern   Gap");
    for i in (0..episodes).step_by(step.max(1)) {
        let gap = r_pattern_history[i] - r_chaos_history[i];
        let bar_len = (r_pattern_history[i] * 20.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  [{:>3}]  {:.4}   {:.4}   {:+.4}  {}",
                 i, r_chaos_history[i], r_pattern_history[i], gap, bar);
    }

    // ── Conclusion ──
    println!("\n{}", "═".repeat(66));
    let r_gap_grew = gap_change > 0.02;
    let delta_grew = delta_last > delta_first + 0.02;
    let pattern_beats_chaos = r_pat_last > r_chaos_last + 0.05;

    if r_gap_grew && pattern_beats_chaos {
        println!("  ✅ УСПЕХ!");
        println!("  FlyWire визуальный подграф научился различать хаос и паттерн!");
        println!("  Разрыв R_pattern − R_chaos вырос на {:.3}", gap_change);
    } else if delta_grew {
        println!("  ✅ ЧАСТИЧНЫЙ УСПЕХ:");
        println!("  δ растёт (обучение идёт), но разрыв R_pat − R_chaos пока мал.");
        println!("  Δδ = {:+.4}, Δgap = {:+.4}", delta_last - delta_first, gap_change);
    } else if pattern_beats_chaos {
        println!("  ✅ PATTERN > CHAOS изначально:");
        println!("  FlyWire сеть уже показывает большую когерентность при motion vs looming (без обучения).");
    } else {
        println!("  ⚠️  Разрыв R_pat − R_chaos не вырос.");
        println!("  Δgap = {:+.4}. Попробуй увеличить N_EPISODES или α.", gap_change);
        println!("  FlyWire подграф может требовать больше эпизод для обучения.");
    }

    println!("\n  Параметры: α={:.1}, η={}, stim={}Hz×{}, episodes={}, neurons={}",
             alpha, lr, stim_freq, stim_amp, episodes, n);
    println!("  Synapses: {}, E/I ratio: {:.2}, mean|w|: {:.4}",
             n_syn, n_exc as f32 / n_inh.max(1) as f32, mean_w);
    println!("  Looming types: LC4/LPLC2/LC11/LC6 ({} neurons)", looming_idx.len());
    println!("  Motion types: T4/T5/VS/HS ({} neurons)", motion_idx.len());
    println!("  Всего: {:.2}s", t0.elapsed().as_secs_f64());
    println!("{}", "═".repeat(66));
}
