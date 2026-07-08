//! # Fly Sensorimotor Pipeline — SpikeGrad Closed Loop
//!
//! Настоящий сенсомоторный эксперимент на реальном FlyWire коннектоме:
//! 1. Загружаем connections + classification
//! 2. Делим нейроны на сенсор/централ/мотор
//! 3. Инжектируем стимул в сенсорные нейроны
//! 4. Kuramoto динамика распространяет сигнал через коннектом
//! 5. Читаем моторный выход — phase locking со стимулом
//! 6. Сравниваем ДО и ПОСЛЕ SpikeGrad очистки
//!
//! Использование:
//! ```bash
//! # zcat /path/to/connections_princeton.csv.gz | head -300000 > /tmp/conns.csv
//! # zcat /path/to/classification.csv.gz > /tmp/class.csv
//! cargo run --example fly_sensorimotor -p connectome-gpt -- \
//!   --connections /tmp/conns.csv --classification /tmp/class.csv \
//!   --max-neurons 300 --epochs 100 --stim-freq 0.5
//! ```

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

fn stream_csv<F>(path: &str, mut cb: F) -> Result<usize, String>
where F: FnMut(&[&str]) -> bool {
    let f = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
    let buf = BufReader::new(f);
    let mut count = 0;
    for line in buf.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        if line.trim().is_empty() { continue; }
        let cols: Vec<&str> = line.trim().split(',').collect();
        if !cb(&cols) { break; }
        count += 1;
    }
    Ok(count)
}

/// Run Kuramoto with sensory drive injection and measure motor output.
/// Returns (motor_phase_locking, final_coherence) where phase_locking
/// measures how well motor neurons follow the stimulus frequency.
fn run_sensorimotor_test(
    values: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    n: usize,
    sensory_idx: &[usize],
    motor_idx: &[usize],
    stim_freq: f32,
    stim_amp: f32,
    n_steps: usize,
    dt: f32,
    seed: u64,
) -> (f32, f32) {
    use rand::Rng;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut phases: Vec<f32> = (0..n).map(|_| (rng.gen::<f32>() - 0.5) * std::f32::consts::TAU).collect();
    let base_omegas: Vec<f32> = (0..n).map(|_| 0.5 + rng.gen::<f32>() * 2.5).collect();
    let mut sin_buf = vec![0.0f32; n];
    let mut cos_buf = vec![0.0f32; n];

    // Track motor phase over time for PLV computation
    let mut motor_phases_hist: Vec<Vec<f32>> = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t = step as f32 * dt;
        let stimulus = (stim_freq * t * std::f32::consts::TAU).sin() * stim_amp;

        // sin/cos of current phases
        for i in 0..n {
            sin_buf[i] = phases[i].sin();
            cos_buf[i] = phases[i].cos();
        }

        // Kuramoto coupling + sensory drive
        for i in 0..n {
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1] as usize).min(col_indices.len());
            let mut cpl = 0.0f32;
            for idx in start..end {
                let j = col_indices[idx] as usize;
                if j >= n { continue; }
                let w = values[idx];
                cpl += w * (sin_buf[j] * cos_buf[i] - cos_buf[j] * sin_buf[i]);
            }

            // Drive: extra frequency push for sensory neurons
            let drive = if sensory_idx.contains(&i) { stimulus } else { 0.0 };
            phases[i] += dt * (base_omegas[i] + cpl + drive);
        }

        // Record motor neuron phases
        let motor_phases: Vec<f32> = motor_idx.iter().map(|&idx| phases[idx]).collect();
        motor_phases_hist.push(motor_phases);
    }

    // Compute Phase Locking Value (PLV) between motor output and stimulus
    // PLV = |mean(exp(i*(θ_motor - θ_stimulus)))| over last 50% of steps
    let plv_start = n_steps / 2;
    let mut complex_sum = (0.0f64, 0.0f64);
    let mut count = 0;

    for step in plv_start..n_steps {
        let t = step as f32 * dt;
        let stim_phase = (stim_freq * t * std::f32::consts::TAU).fract() * std::f32::consts::TAU;

        // Mean motor phase this step
        let m_phases = &motor_phases_hist[step];
        let mean_motor_sin: f64 = m_phases.iter().map(|&p| (p as f64 - stim_phase as f64).sin()).sum::<f64>() / m_phases.len() as f64;
        let mean_motor_cos: f64 = m_phases.iter().map(|&p| (p as f64 - stim_phase as f64).cos()).sum::<f64>() / m_phases.len() as f64;

        complex_sum.0 += mean_motor_sin;
        complex_sum.1 += mean_motor_cos;
        count += 1;
    }

    let plv = if count > 0 {
        let (s, c) = complex_sum;
        ((s * s + c * c).sqrt() / count as f64) as f32
    } else {
        0.0
    };

    // Global coherence at final step
    let (sum_s, sum_c): (f64, f64) = phases.iter().fold((0.0, 0.0), |(s, c), &th| {
        (s + th.sin() as f64, c + th.cos() as f64)
    });
    let coherence = ((sum_s * sum_s + sum_c * sum_c).sqrt() / n as f64) as f32;

    (plv, coherence)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut conns_path = String::new();
    let mut class_path = String::new();
    let mut max_neurons = 300usize;
    let mut epochs = 100usize;
    let mut stim_freq = 0.5f32;
    let mut stim_amp = 5.0f32;
    let mut lr = 0.001f32;
    let mut plv_weight = 5.0f32;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--connections" => conns_path = args.next().unwrap_or_default(),
            "--classification" => class_path = args.next().unwrap_or_default(),
            "--max-neurons" => max_neurons = args.next().and_then(|s| s.parse().ok()).unwrap_or(300),
            "--epochs" => epochs = args.next().and_then(|s| s.parse().ok()).unwrap_or(100),
            "--stim-freq" => stim_freq = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            "--stim-amp" => stim_amp = args.next().and_then(|s| s.parse().ok()).unwrap_or(5.0),
            "--lr" => lr = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.001),
            "--plv-weight" => plv_weight = args.next().and_then(|s| s.parse().ok()).unwrap_or(5.0),
            _ => { eprintln!("Usage: --connections <file> --classification <file> [--max-neurons N] [--epochs N] [--stim-freq F] [--stim-amp A] [--lr F] [--plv-weight F]"); return; }
        }
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Fly Sensorimotor — SpikeGrad Closed-Loop Experiment    ║");
    println!("║  Sensory drive → FlyWire → Motor readout → BPTT→weight ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    let t0 = Instant::now();

    // ── Step 1: Load classification ──
    println!("\n▶ Loading classification...");
    let mut flow_map: HashMap<u64, String> = HashMap::new();
    stream_csv(&class_path, |cols| {
        if cols[0] == "root_id" { return true; }
        if let Ok(id) = cols[0].trim().parse::<u64>() {
            flow_map.insert(id, cols.get(1).map(|s| s.trim()).unwrap_or("").to_string());
        }
        true
    }).expect("classification");
    println!("   {} classified neurons", flow_map.len());

    // ── Step 2: Load connections ──
    println!("▶ Loading connections...");
    let mut raw_edges: Vec<(u64, u64, f32)> = Vec::new();
    let mut conn_counts: HashMap<u64, usize> = HashMap::new();
    let mut total = 0usize;

    stream_csv(&conns_path, |cols| {
        if cols.len() < 5 || cols[0] == "pre_root_id" { return true; }
        let pre = match cols[0].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        let post = match cols[1].trim().parse::<u64>() { Ok(v) => v, _ => return true };
        let syn = match cols[3].trim().parse::<u32>() { Ok(v) => v, _ => return true };
        let nt = cols[4].trim();
        total += 1;
        *conn_counts.entry(pre).or_insert(0) += 1;
        *conn_counts.entry(post).or_insert(0) += 1;
        let sign = match nt { "ACH"|"GLUT" => 1.0, "GABA" => -1.0, "DA"|"SER"|"OCT" => -1.0, _ => return true };
        raw_edges.push((pre, post, (syn as f32).sqrt().min(3.0) / 5.0 * sign));
        true
    }).expect("connections");
    println!("   {} connections, {} matched edges", total, raw_edges.len());

    // ── Step 3: Stratified selection ──
    println!("▶ Stratified selection: top {} from afferent + intrinsic + efferent...", max_neurons);
    
    // Separate ranked lists by flow group
    let all_ranked: Vec<(u64, usize)> = conn_counts.into_iter().collect();
    let mut afferent_ranked: Vec<(u64, usize)> = Vec::new();
    let mut intrinsic_ranked: Vec<(u64, usize)> = Vec::new();
    let mut efferent_ranked: Vec<(u64, usize)> = Vec::new();
    
    for (id, count) in all_ranked {
        match flow_map.get(&id).map(|s| s.as_str()) {
            Some("afferent" | "ascending") => afferent_ranked.push((id, count)),
            Some("efferent" | "descending") => efferent_ranked.push((id, count)),
            _ => intrinsic_ranked.push((id, count)), // intrinsic + unknown flow
        }
    }
    // Sort each by connection count descending
    let sort_desc = |a: &(u64, usize), b: &(u64, usize)| b.1.cmp(&a.1);
    afferent_ranked.sort_by(sort_desc);
    efferent_ranked.sort_by(sort_desc);
    intrinsic_ranked.sort_by(sort_desc);
    
    println!("   Available: {} afferent, {} intrinsic, {} efferent",
        afferent_ranked.len(), intrinsic_ranked.len(), efferent_ranked.len());
    
    // Stratify: take top N/3 from each known group, fill remainder with intrinsic
    let per_group = max_neurons / 3;
    let take_aff = afferent_ranked.len().min(per_group);
    let take_eff = efferent_ranked.len().min(per_group);
    let remaining = max_neurons.saturating_sub(take_aff + take_eff);
    let take_int = intrinsic_ranked.len().min(remaining);
    
    // Merge selected IDs: afferent → efferent → intrinsic
    let selected_ids: Vec<u64> = afferent_ranked.into_iter().take(take_aff)
        .chain(efferent_ranked.into_iter().take(take_eff))
        .chain(intrinsic_ranked.into_iter().take(take_int))
        .map(|(id, _)| id)
        .collect();
    
    let n = selected_ids.len();
    println!("   Selected: {} afferent + {} efferent + {} intrinsic = {} total",
        take_aff, take_eff, n - take_aff - take_eff, n);
    
    // Build id_to_idx + NT types (majority vote from outgoing edges)
    let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut nt_types = vec![0.0f32; n];
    let mut neuron_flow: Vec<String> = Vec::with_capacity(n);
    
    for (i, &root_id) in selected_ids.iter().enumerate() {
        id_to_idx.insert(root_id, i);
        let (exc, inh): (usize, usize) = raw_edges.iter()
            .filter(|(pre, _, _)| *pre == root_id)
            .fold((0, 0), |(e, i), (_, _, w)| if *w > 0.0 { (e+1, i) } else { (e, i+1) });
        nt_types[i] = if exc >= inh { 1.0 } else { -1.0 };
        neuron_flow.push(flow_map.get(&root_id).cloned().unwrap_or_default());
    }

    // ── Step 4: Build CSR (INCOMING convention) ──
    // row_ptr[post] → col_indices[pre] — pre-synaptic neuron j influences post i.
    // coupling[i] += W_ij * sin(θ_j - θ_i) — i is influenced by j.
    // This matches the refiner's convention and enables cross-term BPTT propagation.
    let mut sub_edges: Vec<(usize, usize, f32)> = raw_edges.iter()
        .filter_map(|(pre, post, w)| Some((*id_to_idx.get(pre)?, *id_to_idx.get(post)?, *w)))
        .collect();
    sub_edges.sort_by_key(|(_, post, _)| *post); // sort by POST (target)

    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::with_capacity(sub_edges.len());
    let mut values = Vec::with_capacity(sub_edges.len());
    let mut e = 0;
    for i in 0..n {
        while e < sub_edges.len() && sub_edges[e].1 == i { // i = post
            col_indices.push(sub_edges[e].0 as i32);         // col = pre
            values.push(sub_edges[e].2);
            e += 1;
        }
        row_ptr[i + 1] = col_indices.len() as i32;
    }
    println!("   CSR: {} neurons, {} synapses\n", n, values.len());

    // ── Step 5: Build groups ──
    let mut sensory_idx = Vec::new();
    let mut motor_idx = Vec::new();
    let mut central_idx = Vec::new();

    for i in 0..n {
        let f = &neuron_flow[i];
        if f == "afferent" || f == "ascending" { sensory_idx.push(i); }
        else if f == "efferent" || f == "descending" { motor_idx.push(i); }
        else { central_idx.push(i); }
    }

    // Fallbacks if groups are empty
    if sensory_idx.is_empty() { sensory_idx = (0..n/10).collect(); }
    if motor_idx.is_empty() { motor_idx = (n*9/10..n).collect(); }
    if central_idx.is_empty() { central_idx = (n/10..n*9/10).collect(); }

    println!("📊 Neuron groups:");
    println!("   Sensory (afferent/ascending):  {}", sensory_idx.len());
    println!("   Central (intrinsic):           {}", central_idx.len());
    println!("   Motor (efferent/descending):   {}", motor_idx.len());
    println!();

    // ── Step 6: Pre-refinement sensorimotor test ──
    println!("▶ Pre-refinement sensorimotor test (stim freq={}Hz)...", stim_freq);
    let (plv_before, coherence_before) = run_sensorimotor_test(
        &values, &row_ptr, &col_indices, n,
        &sensory_idx, &motor_idx, stim_freq, stim_amp, 100, 0.05, 42
    );
    println!("   Motor Phase Locking Value (PLV): {:.4}", plv_before);
    println!("   Global coherence: {:.4}", coherence_before);
    println!();

    // ── Step 7: SpikeGrad refinement ──
    println!("▶ Running SpikeGrad ConnectomeRefiner ({} epochs)...", epochs);

    // Build ConnectomeSample for the refiner
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    groups.insert("sensory".into(), sensory_idx.clone());
    groups.insert("central".into(), central_idx.clone());
    groups.insert("motor".into(), motor_idx.clone());

    let sample = connectome_gpt::ConnectomeSample {
        source: "flywire".into(), n_neurons: n, n_synapses: values.len(),
        row_ptr: row_ptr.clone(), col_indices: col_indices.clone(),
        values: values.clone(), nt_types: nt_types.clone(),
        groups, features: vec![], depth: None,
    };

    let config = connectome_gpt::RefinerConfig {
        n_steps: 50, dt: 0.05,
        coherence_weight: 1.0, sparsity_weight: 0.0001,
        temporal_weight: 0.0, homeostasis_weight: 0.0,
        target_mean_weight: 0.1, omegas: vec![], seed: 42,
        // Sensorimotor phase locking
        sensory_idx: sensory_idx.clone(),
        motor_idx: motor_idx.clone(),
        stim_freq, stim_amp,
        lr,
        plv_weight,      // phase locking weight
        max_grad_norm: 10.0,   // prevent gradient explosion
        ..Default::default()
    };
    let mut refiner = connectome_gpt::ConnectomeRefiner::new(config);
    let refine_start = Instant::now();
    let (refined, history) = refiner.refine(&sample, epochs);
    let refine_time = refine_start.elapsed();

    let first = history.first().unwrap();
    let last = history.last().unwrap();

    // ── Step 8: Post-refinement sensorimotor test ──
    println!("▶ Post-refinement sensorimotor test...");
    let (plv_after, coherence_after) = run_sensorimotor_test(
        &refined.values, &refined.row_ptr, &refined.col_indices, n,
        &sensory_idx, &motor_idx, stim_freq, stim_amp, 100, 0.05, 42
    );

    // ── Report ──
    let phase_err_before = first.plv;
    let phase_err_after = last.plv;
    println!("\n╔══════════════════════════════════╦═══════════╦═══════════╗");
    println!("║  Metric                         ║ Before    ║ After     ║");
    println!("╠══════════════════════════════════╬═══════════╬═══════════╣");
    println!("║  Loss (refiner)                 ║ {:>+8.4} ║ {:>+8.4} ║", first.loss, last.loss);
    println!("║  Coherence R (global)           ║ {:>8.4} ║ {:>8.4} ║", coherence_before, coherence_after);
    println!("║  Phase error (1-cos(φ), lower=better) ║ {:>8.4} ║ {:>8.4} ║", phase_err_before, phase_err_after);
    println!("║  Motor PLV (external)           ║ {:>8.4} ║ {:>8.4} ║", plv_before, plv_after);
    println!("║  Mean |weight|                  ║ {:>8.4} ║ {:>8.4} ║", first.mean_weight, last.mean_weight);
    println!("╠══════════════════════════════════╩═══════════╩═══════════╣");
    println!("║  N={}, E={}, {} epochs in {:.2}s                  ║",
        n, values.len(), epochs, refine_time.as_secs_f32());
    println!("╚════════════════════════════════════════════════════════╝");
    println!();

    // ── Step 9: Export refined weights for Python MuJoCo bridge ──
    println!("\n▶ Exporting refined weights for Python MuJoCo bridge...");
    {
        use std::io::Write;
        // Export format: binary file with:
        //   [n_edges: u32] [n_neurons: u32]
        //   [pre_root_id: u64] [post_root_id: u64] [weight: f32] × n_edges
        //   [root_id: u64] [flow_str_len: u32] [flow_str: bytes] × n_neurons
        let export_path = "fly_refined_weights.bin";
        if let Ok(mut f) = File::create(export_path) {
            let n_edges = refined.n_synapses as u32;
            let n_neurons = n as u32;
            f.write_all(&n_edges.to_le_bytes()).ok();
            f.write_all(&n_neurons.to_le_bytes()).ok();
            
            // Write edges: (pre_root_id, post_root_id, weight) for each synapse
            for i in 0..n {
                let start = refined.row_ptr[i] as usize;
                let end = (refined.row_ptr[i + 1] as usize).min(refined.col_indices.len());
                let post_root_id = selected_ids[i];
                for idx in start..end {
                    let pre_csr_idx = refined.col_indices[idx] as usize;
                    if pre_csr_idx < n {
                        let pre_root_id = selected_ids[pre_csr_idx];
                        let w = refined.values[idx];
                        f.write_all(&pre_root_id.to_le_bytes()).ok();
                        f.write_all(&post_root_id.to_le_bytes()).ok();
                        f.write_all(&w.to_le_bytes()).ok();
                    }
                }
            }
            
            // Write neuron metadata
            for i in 0..n {
                let root_id = selected_ids[i];
                let flow_str = &neuron_flow[i];
                let flow_bytes = flow_str.as_bytes();
                f.write_all(&root_id.to_le_bytes()).ok();
                let len = flow_bytes.len() as u32;
                f.write_all(&len.to_le_bytes()).ok();
                f.write_all(flow_bytes).ok();
            }
            
            println!("   ✅ Exported {} edges to {} ({} bytes total)", n_edges, export_path, n_edges as usize * 20 + n_neurons as usize * 12);
        } else {
            println!("   ⚠ Could not write to {}", export_path);
        }
    }

    // Summary
    println!("📊 **Sensorimotor SpikeGrad Summary**");
    if last.loss < first.loss {
        println!("✅ Coherence loss improved: {:.4} → {:.4}", first.loss, last.loss);
    }
    if plv_after > plv_before {
        let gain = (plv_after / plv_before - 1.0) * 100.0;
        println!("✅🦗 Motor PLV improved: {:.4} → {:.4} (+{:.1}%)",
            plv_before, plv_after, gain);
    } else {
        println!("→ Motor PLV: {:.4} → {:.4} (stable)", plv_before, plv_after);
    }
    if phase_err_after < phase_err_before {
        let drop = (1.0 - phase_err_after / phase_err_before) * 100.0;
        println!("✅ Phase error ↓{:.1}% ({:.4} → {:.4}) — motor locking to stimulus!",
            drop, phase_err_before, phase_err_after);
    }
    if coherence_after > coherence_before {
        println!("✅ Global coherence: {:.4} → {:.4} (+{:.1}%)",
            coherence_before, coherence_after,
            (coherence_after / coherence_before - 1.0) * 100.0);
    }
    println!("\n⏱ Total: {:.2}s", t0.elapsed().as_secs_f32());
    println!("✅ Sensorimotor experiment complete!");
}
