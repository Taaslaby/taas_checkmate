/// Bake FAFB connectome CSV → .taas binary format.
///
/// Однократный парсинг 80M синапсов, затем моментальная загрузка.
///
/// Запуск: cargo run --features cuda --example bake_fafb --release
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;
use taas_connectome::ConnectomeData;

/// Resolve a path under `examples/data/` (works from any cwd via CARGO_MANIFEST_DIR).
fn data(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("examples/data");
    p.push(name);
    p
}

const FAFB_CSV: &str = "/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/datasets/fafb_v783_princeton_synapse_table.csv.gz";
const PRINCETON_CSV: &str = "/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/datasets/connections_princeton.csv.gz";

/// Stream through zcat, calling `f` on each parsed line.
fn stream_csv<F>(path: &str, n_fields: usize, f: F)
where
    F: FnMut(&[&str]),
{
    let child = Command::new("zcat")
        .arg(path)
        .stdout(Stdio::piped())
        .spawn()
        .expect("zcat failed");
    let reader = BufReader::new(child.stdout.unwrap());
    let mut iter = reader.lines();
    iter.next(); // skip header

    let mut g = f;
    for line in iter {
        let line = line.expect("read error");
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < n_fields { continue; }
        g(&fields);
    }
}

fn bake_fafb() {
    println!("FAFB: parsing {} ...", FAFB_CSV);
    let t0 = Instant::now();
    let mut neuron_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut next_idx = 0usize;
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(80_000_000);

    stream_csv(FAFB_CSV, 13, |fields| {
        let pre: u64 = fields[10].parse().unwrap_or(0);
        let post: u64 = fields[11].parse().unwrap_or(0);
        let w: f32 = fields[9].parse().unwrap_or(1.0);
        let pi = *neuron_to_idx.entry(pre).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
        let pj = *neuron_to_idx.entry(post).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
        edges.push((pi as u32, pj as u32, w));
    });

    let n = neuron_to_idx.len();
    let n_syn = edges.len();
    println!("   {n} neurons, {n_syn} synapses in {:.1}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let mut row_counts = vec![0usize; n + 1];
    for &(pre, _, _) in &edges { row_counts[pre as usize] += 1; }

    let mut row_ptr = vec![0i32; n + 1];
    let mut cum = 0i32;
    for i in 0..n { row_ptr[i] = cum; cum += row_counts[i] as i32; }
    row_ptr[n] = cum;
    drop(row_counts);

    let mut col_indices = vec![0i32; n_syn];
    let mut values = vec![0.0f32; n_syn];
    let mut pos = vec![0usize; n];
    for &(pre, post, w) in &edges {
        let p = pos[pre as usize];
        let idx = row_ptr[pre as usize] as usize + p;
        col_indices[idx] = post as i32;
        values[idx] = w * 1e-12;
        pos[pre as usize] = p + 1;
    }
    drop(edges); drop(pos);

    let nt_types = vec![1.0f32; n];
    println!("   CSR built in {:.1}s", t1.elapsed().as_secs_f64());

    // Save to .taas
    let t2 = Instant::now();
    let conn = ConnectomeData::new(row_ptr, col_indices, values, nt_types, vec![]);
    let fafb_path = data("fafb.taas");
    let mut f = File::create(&fafb_path).unwrap();
    conn.save_baked(&mut f).unwrap();
    drop(f);

    let file_size = std::fs::metadata(&fafb_path).unwrap().len();
    println!("   Saved to {} ({:.1}MB) in {:.1}s",
             fafb_path.display(), file_size as f64 / 1_048_576.0, t2.elapsed().as_secs_f64());

    // Verify round-trip
    let t3 = Instant::now();
    let mut f = File::open(&fafb_path).unwrap();
    let loaded = ConnectomeData::load_baked(&mut f).unwrap();
    assert_eq!(loaded.n_neurons, conn.n_neurons);
    assert_eq!(loaded.n_synapses, conn.n_synapses);
    assert_eq!(loaded.row_ptr.len(), conn.row_ptr.len());
    assert_eq!(loaded.col_indices.len(), conn.col_indices.len());
    println!("   Round-trip verified in {:.3}s", t3.elapsed().as_secs_f64());
    println!("   Total bake time: {:.1}s", t0.elapsed().as_secs_f64());
}

fn bake_princeton() {
    println!("Princeton: parsing {} ...", PRINCETON_CSV);
    let t0 = Instant::now();
    let mut neuron_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut next_idx = 0usize;
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(6_000_000);

    stream_csv(PRINCETON_CSV, 5, |fields| {
        let pre: u64 = fields[0].parse().unwrap_or(0);
        let post: u64 = fields[1].parse().unwrap_or(0);
        let w: f32 = fields[3].parse().unwrap_or(1.0);
        let pi = *neuron_to_idx.entry(pre).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
        let pj = *neuron_to_idx.entry(post).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
        edges.push((pi as u32, pj as u32, w));
    });

    let n = neuron_to_idx.len();
    let n_syn = edges.len();
    println!("   {n} neurons, {n_syn} synapses in {:.1}s", t0.elapsed().as_secs_f64());

    // Count NT types per neuron
    let mut nt_signs: HashMap<u64, f32> = HashMap::new();
    stream_csv(PRINCETON_CSV, 5, |fields| {
        let pre: u64 = fields[0].parse().unwrap_or(0);
        let nt = fields[4].trim();
        let sign = match nt { "GABA" => -1.0, _ => 1.0 };
        nt_signs.insert(pre, sign);
    });

    let t1 = Instant::now();
    let mut row_counts = vec![0usize; n + 1];
    for &(pre, _, _) in &edges { row_counts[pre as usize] += 1; }

    let mut row_ptr = vec![0i32; n + 1];
    let mut cum = 0i32;
    for i in 0..n { row_ptr[i] = cum; cum += row_counts[i] as i32; }
    row_ptr[n] = cum;
    drop(row_counts);

    let mut col_indices = vec![0i32; n_syn];
    let mut values = vec![0.0f32; n_syn];
    let mut pos = vec![0usize; n];
    for &(pre, post, w) in &edges {
        let p = pos[pre as usize];
        let idx = row_ptr[pre as usize] as usize + p;
        col_indices[idx] = post as i32;
        values[idx] = w * 1e-12;
        pos[pre as usize] = p + 1;
    }
    drop(edges); drop(pos);

    // Assign NT types (default +1 if not found)
    let nt_types: Vec<f32> = (0..n).map(|i| {
        let root_id = neuron_to_idx.iter().find(|(_, &v)| v == i).map(|(k, _)| *k);
        match root_id.and_then(|id| nt_signs.get(&id)) {
            Some(&s) => s,
            None => 1.0,
        }
    }).collect();

    println!("   CSR built in {:.1}s", t1.elapsed().as_secs_f64());

    let t2 = Instant::now();
    let conn = ConnectomeData::new(row_ptr, col_indices, values, nt_types, vec![]);
    let princeton_path = data("princeton.taas");
    let mut f = File::create(&princeton_path).unwrap();
    conn.save_baked(&mut f).unwrap();
    drop(f);

    let file_size = std::fs::metadata(&princeton_path).unwrap().len();
    println!("   Saved to {} ({:.1}MB) in {:.1}s",
             princeton_path.display(), file_size as f64 / 1_048_576.0, t2.elapsed().as_secs_f64());
    println!("   Total bake time: {:.1}s", t0.elapsed().as_secs_f64());
}

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Bake connectome CSV → .taas                 ║");
    println!("╚════════════════════════════════════════════════╝");
    let t_all = Instant::now();

    bake_fafb();
    println!();
    bake_princeton();

    println!("\nDone in {:.1}s total.", t_all.elapsed().as_secs_f64());
    println!("Use: load via `ConnectomeData::load_baked(&mut file)`");

    let total_mb = std::fs::metadata(data("fafb.taas")).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0)
        + std::fs::metadata(data("princeton.taas")).map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
    println!("Total disk: {total_mb:.0}MB");
}
