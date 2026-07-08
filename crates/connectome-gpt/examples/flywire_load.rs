//! # FlyWire Connectome Loader + Refiner Demo (v2)
//!
//! Строит коннектом ТОЛЬКО из connections CSV — не требует neurons файл.
//! NT типы определяются из колонки nt_type в соединениях.
//!
//! Использование:
//! ```bash
//! # zcat /path/to/connections_princeton.csv.gz | head -500000 > /tmp/conns.csv
//! cargo run --example flywire_load -p connectome-gpt -- \
//!   --connections /tmp/conns.csv --max-neurons 500 --epochs 100
//! ```

use connectome_gpt::{ConnectomeSample, ConnectomeRefiner, RefinerConfig};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::fs::File;
use std::time::Instant;

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

fn main() {
    let mut args = std::env::args().skip(1);
    let mut conns_path = String::new();
    let mut max_neurons = 500usize;
    let mut epochs = 100usize;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--connections" => conns_path = args.next().unwrap_or_default(),
            "--max-neurons" => max_neurons = args.next().and_then(|s| s.parse().ok()).unwrap_or(500),
            "--epochs" => epochs = args.next().and_then(|s| s.parse().ok()).unwrap_or(100),
            _ => { eprintln!("Usage: --connections <file> [--max-neurons N] [--epochs N]"); return; }
        }
    }

    println!("╔════════════════════════════════════════════════════╗");
    println!("║    FlyWire Princeton + ConnectomeRefiner v2       ║");
    println!("║    Building connectome FROM CONNECTIONS only      ║");
    println!("╚════════════════════════════════════════════════════╝");
    println!();

    // ── Single pass: read connections, build subgraph ──
    println!("▶ Loading connections and building subgraph...");
    let t0 = Instant::now();

    // neuron_id → nt_sign (aggregated from outgoing connections)
    let mut nt_map: HashMap<u64, f32> = HashMap::new();
    // pre → [(post, weight)]
    let mut edges: Vec<(u64, u64, f32)> = Vec::new();
    // Count exc/inh per neuron for majority vote
    let mut exc_count: HashMap<u64, usize> = HashMap::new();
    let mut inh_count: HashMap<u64, usize> = HashMap::new();
    let mut total = 0usize;
    let mut skipped = 0usize;

    stream_csv(&conns_path, |cols| {
        if cols.len() < 5 { skipped += 1; return true; }
        if cols[0] == "pre_root_id" { return true; }
        let pre = match cols[0].trim().parse::<u64>() { Ok(v) => v, _ => { skipped += 1; return true; }};
        let post = match cols[1].trim().parse::<u64>() { Ok(v) => v, _ => { skipped += 1; return true; }};
        let syn = match cols[3].trim().parse::<u32>() { Ok(v) => v, _ => { skipped += 1; return true; }};
        let nt_label = cols[4].trim();
        total += 1;

        // Determine sign from NT type
        let sign = match nt_label {
            "ACH" | "GLUT" => 1.0,
            "GABA" => -1.0,
            "DA" | "SER" | "OCT" => -1.0,
            _ => { skipped += 1; return true; } // unknown NT
        };

        // Majority vote: count exc vs inh per pre-neuron
        if sign > 0.0 { *exc_count.entry(pre).or_insert(0) += 1; }
        else { *inh_count.entry(pre).or_insert(0) += 1; }

        // Store edge
        let weight = (syn as f32).sqrt().min(3.0) / 5.0;
        edges.push((pre, post, weight));
        true
    }).expect("Failed to read connections");

    // Build nt_map from majority vote
    for pre in exc_count.keys().chain(inh_count.keys()) {
        let e = exc_count.get(pre).copied().unwrap_or(0);
        let i = inh_count.get(pre).copied().unwrap_or(0);
        nt_map.insert(*pre, if e >= i { 1.0 } else { -1.0 });
    }

    println!("   Total connections: {}", total);
    println!("   Unique pre-neurons: {}", nt_map.len());
    println!("   Skipped: {}", skipped);
    println!("   Stored edges: {}", edges.len());
    println!();

    // ── Select top N neurons by connection count ──
    println!("▶ Selecting top {} neurons...", max_neurons);
    // Count both pre AND post frequencies for connected subgraph
    let mut conn_counts: HashMap<u64, usize> = HashMap::new();
    for (pre, post, _) in &edges {
        *conn_counts.entry(*pre).or_insert(0) += 1;
        *conn_counts.entry(*post).or_insert(0) += 1;
    }
    let mut ranked: Vec<(u64, usize)> = conn_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    let n = max_neurons.min(ranked.len());
    let selected: HashSet<u64> = ranked.iter().take(n).map(|(id, _)| *id).collect();
    println!("   Selected {} pre-neurons with most outgoing connections", n);

    // ── Build CSR for selected neurons ──
    println!("▶ Building CSR...");
    let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut nt_types = vec![0.0f32; n];
    let mut idx = 0usize;
    for &root_id in &selected {
        id_to_idx.insert(root_id, idx);
        nt_types[idx] = *nt_map.get(&root_id).unwrap_or(&1.0);
        idx += 1;
    }

    // Filter edges: keep only those where pre is selected AND post is ALSO selected (subgraph)
    let mut sub_edges: Vec<(usize, usize, f32)> = Vec::new();
    for (pre, post, w) in &edges {
        if let (Some(&pre_idx), Some(&post_idx)) = (id_to_idx.get(pre), id_to_idx.get(post)) {
            let sign = nt_types[pre_idx];
            sub_edges.push((pre_idx, post_idx, w * sign));
        }
    }
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

    println!("   CSR built: {} synapses", values.len());
    println!("   Time: {:.2}s", t0.elapsed().as_secs_f32());
    println!();

    if values.is_empty() {
        println!("⚠️  No edges in subgraph — try with more connections (head -500000) or fewer neurons.");
        println!("   The selected neurons may not connect to each other.");
        return;
    }

    let n_exc = nt_types.iter().filter(|&&t| t > 0.0).count();
    let n_inh = nt_types.iter().filter(|&&t| t < 0.0).count();

    // Create functional groups
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    groups.insert("selected".into(), (0..n).collect());
    // Split into E/I groups
    let exc_group: Vec<usize> = (0..n).filter(|&i| nt_types[i] > 0.0).collect();
    let inh_group: Vec<usize> = (0..n).filter(|&i| nt_types[i] < 0.0).collect();
    if !exc_group.is_empty() { groups.insert("excitatory".into(), exc_group); }
    if !inh_group.is_empty() { groups.insert("inhibitory".into(), inh_group); }

    // Stats for top-pre by connection count
    let top5: Vec<String> = ranked.iter().take(5)
        .map(|(id, cnt)| format!("{} ({})", id, cnt))
        .collect();
    println!("📊 FlyWire subgraph:");
    println!("   Neurons: {}, Synapses: {}", n, values.len());
    println!("   E/I ratio: {:.2} ({} exc / {} inh)",
        n_exc as f32 / n_inh.max(1) as f32, n_exc, n_inh);
    println!("   Mean |weight|: {:.4}",
        values.iter().map(|w| w.abs()).sum::<f32>() / values.len() as f32);
    println!("   Top 5 pre-neurons by connections: {:?}", top5);
    println!();

    let sample = ConnectomeSample {
        source: "flywire_princeton_v2".into(), n_neurons: n, n_synapses: values.len(),
        row_ptr, col_indices, values, nt_types, groups, features: vec![], depth: None,
    };

    // ── Run ConnectomeRefiner ──
    println!("▶ Running ConnectomeRefiner ({} epochs)...", epochs);
    let config = RefinerConfig {
        n_steps: 30, dt: 0.05, lr: 0.005,
        coherence_weight: 1.0, sparsity_weight: 0.001,
        temporal_weight: 0.05, homeostasis_weight: 0.01,
        target_mean_weight: 0.2, omegas: vec![], seed: 42,
        ..Default::default()
    };
    let mut refiner = ConnectomeRefiner::new(config);
    let refine_start = Instant::now();
    let (_refined, history) = refiner.refine(&sample, epochs);
    let refine_elapsed = refine_start.elapsed();

    let first = history.first().unwrap();
    let last = history.last().unwrap();

    // NaN check
    if !last.loss.is_finite() {
        let n_valid = history.iter().take_while(|r| r.loss.is_finite()).count();
        eprintln!("⚠️  NaN after {} valid epochs (lr may be too high for this graph)", n_valid);
        if n_valid > 0 {
            println!("   First {} epochs stable: loss {:.4} → {:.4}",
                n_valid, history[0].loss, history[n_valid-1].loss);
        }
    }

    println!();
    println!("╔══════════════════════════════╦═══════════╦═══════════╗");
    println!("║  Metric                     ║ Before    ║ After     ║");
    println!("╠══════════════════════════════╬═══════════╬═══════════╣");
    println!("║  Loss                       ║ {:>+8.4} ║ {:>+8.4} ║", first.loss, last.loss);
    println!("║  Coherence R                ║ {:>8.4} ║ {:>8.4} ║", first.coherence, last.coherence);
    println!("║  Sparsity penalty (L1 mean) ║ {:>8.4} ║ {:>8.4} ║", first.sparsity_penalty, last.sparsity_penalty);
    println!("║  Temporal penalty           ║ {:>8.6} ║ {:>8.6} ║", first.temporal_penalty, last.temporal_penalty);
    println!("║  Mean |weight|              ║ {:>8.4} ║ {:>8.4} ║", first.mean_weight, last.mean_weight);
    println!("╠══════════════════════════════╩═══════════╩═══════════╣");
    println!("║  N={}, E={}, {} epochs, {:.2}s              ║",
        sample.n_neurons, sample.n_synapses, epochs, refine_elapsed.as_secs_f32());
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    if last.loss.is_finite() && last.loss < first.loss {
        println!("✅ Loss: {:.4} → {:.4} (improved via BPTT)", first.loss, last.loss);
    }
    if last.coherence.is_finite() && last.coherence > first.coherence {
        let gain = (last.coherence / first.coherence - 1.0) * 100.0;
        println!("✅ Coherence: {:.4} → {:.4} (+{:.1}%)", first.coherence, last.coherence, gain);
    }
    println!("✅ FlyWire connections-only loader complete!");
}
