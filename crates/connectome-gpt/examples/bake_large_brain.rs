//! Fast connectome generator for benchmarking.
//!
//! Generates a synthetic connectome with biological group structure
//! (sensory, DN, MN, VNC) and saves to .taas format.
//!
//! Usage:
//!   cargo run --release --example bake_large_brain -p connectome-gpt -- [n_neurons] [output.taas]
//!
//! Examples:
//!   cargo run --release --example bake_large_brain -p connectome-gpt -- 10000 ./10k.taas
//!   cargo run --release --example bake_large_brain -p connectome-gpt -- 1000000 ./1m.taas

use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_neurons: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let out_path = std::env::args().nth(2).unwrap_or_else(|| {
        let tmp = std::env::temp_dir();
        format!("{}brain_{}k.taas", tmp.display(), n_neurons / 1000)
    });

    println!("Generating {} neurons -> {}", n_neurons, out_path);
    let t0 = std::time::Instant::now();

    // ─── Fast CSR generator (O(n × avg_deg), no HashSet) ───
    let avg_deg = 38.min(n_neurons / 10).max(4);
    let n_exc = (n_neurons as f32 * 0.8) as usize;
    let mut rng: u64 = 42;
    let mut rand_f32 = move || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 32) as f32) / 4294967296.0
    };
    let mut rand_usize = move |max: usize| -> usize {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 32) as usize) % max
    };

    let mut row_ptr = Vec::with_capacity(n_neurons + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut nt_types = Vec::with_capacity(n_neurons);

    row_ptr.push(0);
    for i in 0..n_neurons {
        // Excitatory/inhibitory
        nt_types.push(if i < n_exc { 1.0 } else { -1.0 });

        let deg = (avg_deg as f32 * (0.5 + rand_f32() * 0.5)).ceil() as usize;
        let deg = deg.max(1).min(n_neurons - 1);

        let sign = nt_types[i];
        for _ in 0..deg {
            let j = rand_usize(n_neurons);
            if j == i { continue; }  // no self-loops
            col_indices.push(j as i32);
            let w = (rand_f32() * 0.5 + 0.1).abs();
            values.push(if sign > 0.0 { w } else { -w });
        }
        row_ptr.push(col_indices.len() as i32);
    }

    // ─── Biological groups ───
    let n_sensory = (n_neurons as f32 * 0.05) as usize;
    let n_dn = (n_neurons as f32 * 0.02) as usize;
    let n_mn = (n_neurons as f32 * 0.005) as usize;
    let n_vnc = (n_neurons as f32 * 0.10) as usize;

    let mut indices: Vec<usize> = (0..n_neurons).collect();
    // Fisher-Yates shuffle
    for i in (1..n_neurons).rev() {
        let j = rand_usize(i + 1);
        indices.swap(i, j);
    }

    let mut p = 0;
    let sensory: Vec<usize> = indices[p..p + n_sensory.min(n_neurons - p)].to_vec(); p += n_sensory.min(n_neurons - p);
    let dn: Vec<usize> = indices[p..p + n_dn.min(n_neurons - p)].to_vec(); p += n_dn.min(n_neurons - p);
    let mn: Vec<usize> = indices[p..p + n_mn.min(n_neurons - p)].to_vec(); p += n_mn.min(n_neurons - p);
    let vnc: Vec<usize> = indices[p..p + n_vnc.min(n_neurons - p)].to_vec(); p += n_vnc.min(n_neurons - p);

    // ─── Build ConnectomeData ───
    let groups: Vec<(&str, Vec<usize>)> = vec![
        ("sensory", sensory),
        ("dn", dn),
        ("mn", mn),
        ("vnc", vnc),
    ];

    let conn = taas_connectome::ConnectomeData::new(row_ptr, col_indices, values, nt_types, groups);

    let gen_time = t0.elapsed();
    println!("  {} neurons, {} synapses", conn.n_neurons, conn.n_synapses);
    for g in &conn.groups {
        println!("  group '{}': {} neurons", g.name, g.indices.len());
    }
    println!("  Generated in {:.3}s", gen_time.as_secs_f64());

    // ─── Save as .taas ───
    if let Some(parent) = PathBuf::from(&out_path).parent() {
        fs::create_dir_all(parent)?;
    }

    let mut writer = taas_format::TaasWriter::new();
    writer.add_metadata("source", "bake_large_brain");
    writer.add_metadata("n_neurons", &conn.n_neurons.to_string());
    writer.add_metadata("n_synapses", &conn.n_synapses.to_string());
    writer.add_metadata("seed", "42");
    writer.add_metadata("description", &format!("Synthetic {} neuron connectome", conn.n_neurons));
    taas_format::save_connectome(&conn, &mut writer);

    writer.save_to_file(&out_path)?;
    let file_size = fs::metadata(&out_path)?.len();
    println!("  Saved {} ({:.1} MB) in {:.1}s",
        out_path, file_size as f64 / 1_048_576.0, t0.elapsed().as_secs_f64() - gen_time.as_secs_f64());

    Ok(())
}
