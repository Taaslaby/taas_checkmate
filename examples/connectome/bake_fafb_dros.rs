/// Build Drosophila-aware receptor_types for Princeton connectome.
///
/// Uses `connections_princeton.csv.gz` which has FlyWire long IDs with
/// NT type labels (ACH, GABA, GLUT, DA, SER, OCT) per synapse.
///
/// Builds per-synapse receptor_types with Drosophila-appropriate mapping:
///   ACH  → nAChR (4, fast exc)
///   GABA → GABA_A (2, fast inh)
///   GLUT → GluCl (5, fast INHIBITORY — Cl⁻ channel in insects!)
///   DA/SER/OCT → mAChR (6, slow mod)
///
/// Run:
///   cargo run --features connectome --example bake_fafb_dros --release
///
/// Requires: connections_princeton.csv.gz in the data directory.
#[cfg(feature = "connectome")]
mod inner {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::Instant;
    use taas_connectome::ConnectomeData;

    fn data(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("examples/data");
        p.push(name);
        p
    }

    const PRINCETON_CSV: &str = "/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/datasets/connections_princeton.csv.gz";

    /// Map NT label to Drosophila receptor type:
    ///   ACH  → nAChR (4, fast exc)
    ///   GABA → GABA_A (2, fast inh)
    ///   GLUT → GluCl (5, fast inh — Cl⁻ channel!)
    ///   DA/SER/OCT → mAChR (6, slow mod)
    fn nt_to_receptor(nt: &str) -> u8 {
        match nt {
            "GABA" => 2, // GABA_A
            "GLUT" => 5, // GluCl (inhibitory in insects!)
            "DA" => 6,   // mAChR-like (slow mod)
            "SER" => 6,  // mAChR-like
            "OCT" => 6,  // mAChR-like
            _ => 4,      // ACH or unknown → nAChR (fast exc)
        }
    }

    pub fn run() {
        let t0 = Instant::now();
        println!();
        println!("╔═══ Drosophila Princeton Bake ═══╗");
        println!();

        // ─── Step 1: Stream CSV, build graph + NT mapping ─────────────
        println!("1. Streaming {} ...", PRINCETON_CSV);
        let child = Command::new("zcat")
            .arg(PRINCETON_CSV)
            .stdout(Stdio::piped())
            .spawn()
            .expect("zcat failed");
        let reader = BufReader::new(child.stdout.unwrap());
        let mut iter = reader.lines();
        iter.next(); // skip header

        let mut root_to_idx: HashMap<u64, usize> = HashMap::new();
        let mut next_idx = 0usize;
        let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(5_500_000);
        // Per-neuron NT label store: after first pass, we know which NT each neuron uses
        let mut neuron_nt: HashMap<u64, String> = HashMap::new();

        let mut line_count = 0usize;
        for line in iter {
            let line = line.expect("read error");
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 5 {
                continue;
            }
            line_count += 1;

            let pre: u64 = fields[0].parse().unwrap_or(0);
            let post: u64 = fields[1].parse().unwrap_or(0);
            let w: f32 = fields[3].parse().unwrap_or(1.0);
            let nt = fields[4].trim().to_uppercase();

            let pi = *root_to_idx.entry(pre).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            });
            let pj = *root_to_idx.entry(post).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            });

            edges.push((pi as u32, pj as u32, w));

            // Store NT label for pre-neuron (use first seen)
            neuron_nt.entry(pre).or_insert_with(|| nt.clone());

            if line_count % 1_000_000 == 0 {
                println!("   {line_count} lines, {next_idx} neurons...");
            }
        }

        let n = root_to_idx.len();
        let n_syn = edges.len();
        println!(
            "   {n} neurons, {n_syn} synapses in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        // ─── Step 2: Build CSR matrix ─────────────────────────────────
        println!("2. Building CSR matrix ...");
        let t1 = Instant::now();
        let mut row_counts = vec![0usize; n + 1];
        for &(pre, _, _) in &edges {
            row_counts[pre as usize] += 1;
        }

        let mut row_ptr = vec![0i32; n + 1];
        let mut cum = 0i32;
        for i in 0..n {
            row_ptr[i] = cum;
            cum += row_counts[i] as i32;
        }
        row_ptr[n] = cum;
        drop(row_counts);

        let mut col_indices = vec![0i32; n_syn];
        let mut values = vec![0.0f32; n_syn];
        // Also track receptor type per synapse
        let mut receptor_types = vec![0u8; n_syn];
        let mut pos = vec![0usize; n];
        // Per-neuron nt_type sign (+1 exc, -1 inh)
        let mut nt_types = vec![1.0f32; n];
        // Per-neuron receptor type from NT label
        let mut per_neuron_label: Vec<String> = vec![String::new(); n];

        // Build root_id → idx reverse mapping
        let mut idx_to_root: Vec<u64> = vec![0u64; n];
        for (&root_id, &idx) in &root_to_idx {
            if idx < n {
                idx_to_root[idx] = root_id;
            }
        }
        // Build per-neuron NT labels and nt_types
        for i in 0..n {
            let root_id = idx_to_root[i];
            if let Some(label) = neuron_nt.get(&root_id) {
                per_neuron_label[i] = label.clone();
                // nt_type sign: GABA/GLUT/DA/SER/OCT → -1 (inhibitory in insects)
                // ACH/unknown → +1 (excitatory)
                match label.as_str() {
                    "GABA" | "GLUT" | "DA" | "SER" | "OCT" => nt_types[i] = -1.0,
                    _ => nt_types[i] = 1.0,
                }
            }
        }

        // Build CSR with per-synapse receptor_types
        let mut syn_idx = 0usize;
        for &(pre, post, w) in &edges {
            let p = pos[pre as usize];
            let idx = row_ptr[pre as usize] as usize + p;
            col_indices[idx] = post as i32;
            values[idx] = w * 1e-12; // scale raw synapse count to nA
                                     // Receptor type from pre-neuron's NT label → Drosophila-appropriate
            let label = &per_neuron_label[pre as usize];
            receptor_types[syn_idx] = nt_to_receptor(label);
            pos[pre as usize] = p + 1;
            syn_idx += 1;
        }
        drop(edges);
        drop(pos);

        println!("   CSR built in {:.1}s", t1.elapsed().as_secs_f64());

        // ─── Step 3: NT distribution stats ────────────────────────────
        let mut nt_counts: HashMap<&str, usize> = HashMap::new();
        for label in &per_neuron_label {
            let key = if label.is_empty() {
                "unknown"
            } else {
                label.as_str()
            };
            *nt_counts.entry(key).or_insert(0) += 1;
        }
        let mut nt_sorted: Vec<_> = nt_counts.iter().collect();
        nt_sorted.sort_by(|a, b| b.1.cmp(a.1));
        println!();
        println!("   NT distribution (per neuron):");
        for (label, count) in nt_sorted {
            println!(
                "     {label:>8}: {count:>6} ({:.1}%)",
                *count as f64 / n as f64 * 100.0
            );
        }

        // ─── Step 4: Receptor type stats ──────────────────────────────
        let nachr = receptor_types.iter().filter(|&&r| r == 4).count();
        let glucl = receptor_types.iter().filter(|&&r| r == 5).count();
        let machr = receptor_types.iter().filter(|&&r| r == 6).count();
        let gaba_a = receptor_types.iter().filter(|&&r| r == 2).count();
        let ampa = receptor_types.iter().filter(|&&r| r == 0).count();
        let _nmda = receptor_types.iter().filter(|&&r| r == 1).count();
        let _gaba_b = receptor_types.iter().filter(|&&r| r == 3).count();
        println!();
        println!("   Receptor types (per synapse):");
        println!(
            "     nAChR (ACH):  {nachr:>8} ({:.1}%)",
            nachr as f64 / n_syn as f64 * 100.0
        );
        println!(
            "     GluCl (GLUT): {glucl:>8} ({:.1}%) — inhibitory in insects! ⭐",
            glucl as f64 / n_syn as f64 * 100.0
        );
        println!(
            "     mAChR (DA/O): {machr:>8} ({:.1}%)",
            machr as f64 / n_syn as f64 * 100.0
        );
        println!(
            "     GABA_A:       {gaba_a:>8} ({:.1}%)",
            gaba_a as f64 / n_syn as f64 * 100.0
        );
        println!(
            "     AMPA:         {ampa:>8} ({:.1}%)",
            ampa as f64 / n_syn as f64 * 100.0
        );

        let exc = nachr + ampa;
        let inh = glucl + gaba_a + machr;
        println!(
            "     Excitatory: {exc} ({:.1}%)",
            exc as f64 / n_syn as f64 * 100.0
        );
        println!(
            "     Inhibitory: {inh} ({:.1}%)",
            inh as f64 / n_syn as f64 * 100.0
        );

        // ─── Step 5: Build ConnectomeData ─────────────────────────────
        let conn = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values,
            nt_types,
            receptor_types,
            groups: vec![],
        };

        // ─── Step 6: Save ─────────────────────────────────────────────
        let out_path = data("princeton_dros.taas");
        println!("3. Saving to {} ...", out_path.display());
        let mut f = File::create(&out_path).unwrap();
        conn.save_baked(&mut f).unwrap();
        drop(f);

        let file_size = std::fs::metadata(&out_path).unwrap().len();
        println!("   ✓ Saved {:.1}MB", file_size as f64 / 1_048_576.0);

        // ─── Step 7: Round-trip ──────────────────────────────────────
        println!("4. Verifying round-trip ...");
        let mut f = File::open(&out_path).unwrap();
        let loaded = ConnectomeData::load_baked(&mut f).unwrap();
        drop(f);

        assert_eq!(loaded.n_neurons, conn.n_neurons);
        assert_eq!(loaded.n_synapses, conn.n_synapses);
        assert_eq!(
            loaded.receptor_types, conn.receptor_types,
            "receptor_types mismatch after round-trip"
        );
        println!("   ✓ All arrays match");

        let elapsed = t0.elapsed().as_secs_f64();
        println!();
        println!("╔═══ Summary ═══════════════════════════════════════╗");
        println!("║  Output:  {}", out_path.display());
        println!("║  Size:    {:.1}MB", file_size as f64 / 1_048_576.0);
        println!("║  Neurons: {n}");
        println!("║  Synapses: {n_syn}");
        println!("║  Time:    {elapsed:.1}s");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
        println!("Run biology_check on it:");
        println!(
            "  cargo run --features connectome --example biology_check --release -- {}",
            out_path.display()
        );
    }
}

#[cfg(not(feature = "connectome"))]
mod inner {
    pub fn run() {
        println!("This example requires the `connectome` feature.");
        println!("  cargo run --features connectome --example bake_fafb_dros --release");
    }
}

fn main() {
    inner::run();
}
