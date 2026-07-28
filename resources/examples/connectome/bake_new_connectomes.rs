/// Bake NEW connectome CSVs (Buhmann, Princeton no-threshold) → .taas binary.
///
/// Однократный парсинг, затем моментальная загрузка.
///
/// Запуск:
///   cargo run --features connectome --release --example bake_new_connectomes -- \
///       /home/taaslab/Загрузки/connections_buhmann_no_threshold.csv.gz \
///       buhmann.taas
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
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < n_fields {
            continue;
        }
        g(&fields);
    }
}

fn bake_connectome(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: bake_new_connectomes <input.csv.gz> <output.taas>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    println!("Parsing {} ...", input);
    let t0 = Instant::now();
    let mut neuron_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut next_idx = 0usize;
    // Pre-allocate based on file size
    let edges_cap = match input.contains("buhmann") {
        true => 20_000_000,
        _ => 30_000_000,
    };
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(edges_cap);
    let mut nt_signs: HashMap<u64, f32> = HashMap::new();

    let has_nt = input.contains("princeton"); // Princeton has nt_type, Buhmann doesn't always

    // First pass: edges + NT types
    stream_csv(input, 5, |fields| {
        let pre: u64 = fields[0].parse().unwrap_or(0);
        let post: u64 = fields[1].parse().unwrap_or(0);
        let w: f32 = fields[3].parse().unwrap_or(1.0);
        let pi = *neuron_to_idx.entry(pre).or_insert_with(|| {
            let i = next_idx;
            next_idx += 1;
            i
        });
        let pj = *neuron_to_idx.entry(post).or_insert_with(|| {
            let i = next_idx;
            next_idx += 1;
            i
        });
        edges.push((pi as u32, pj as u32, w));
        if has_nt && fields.len() > 4 {
            let nt = fields[4].trim();
            let sign = match nt {
                "GABA" => -1.0,
                "GLUT" => -1.0, // Drosophila: GLUT → GluCl (inhibitory)
                _ => 1.0,       // ACh, etc. → excitatory
            };
            nt_signs.entry(pre).or_insert(sign);
        }
    });

    let n = neuron_to_idx.len();
    let n_syn = edges.len();
    println!(
        "   {} neurons, {} synapses in {:.1}s",
        n,
        n_syn,
        t0.elapsed().as_secs_f64()
    );
    println!("   NT types: {} unique", nt_signs.len());

    // Build CSR
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
    let mut pos = vec![0usize; n];
    for &(pre, post, w) in &edges {
        let p = pos[pre as usize];
        let idx = row_ptr[pre as usize] as usize + p;
        col_indices[idx] = post as i32;
        values[idx] = w * 1e-12;
        pos[pre as usize] = p + 1;
    }
    drop(edges);
    drop(pos);

    // Assign NT types
    let nt_types: Vec<f32> = if has_nt {
        (0..n)
            .map(|i| {
                neuron_to_idx
                    .iter()
                    .find(|(_, &v)| v == i)
                    .and_then(|(k, _)| nt_signs.get(k))
                    .copied()
                    .unwrap_or(1.0)
            })
            .collect()
    } else {
        vec![1.0f32; n]
    };
    println!("   CSR built in {:.1}s", t1.elapsed().as_secs_f64());

    // Save
    let t2 = Instant::now();
    let conn = ConnectomeData::new(row_ptr, col_indices, values, nt_types, vec![]);
    let out_path = PathBuf::from(output);
    let mut f = File::create(&out_path).unwrap();
    conn.save_baked(&mut f).unwrap();
    drop(f);
    let file_size = std::fs::metadata(&out_path).unwrap().len();
    println!(
        "   Saved {} ({:.1}MB) in {:.1}s",
        output,
        file_size as f64 / 1_048_576.0,
        t2.elapsed().as_secs_f64()
    );
    println!("   Total: {:.1}s", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        // Default: bake both datasets
        println!("Default: baking both Buhmann and Princeton no-threshold...");
        bake_connectome(&[
            "".into(),
            "/home/taaslab/Загрузки/connections_buhmann_no_threshold.csv.gz".into(),
            data("buhmann.taas").to_string_lossy().into_owned(),
        ]);
        println!();
        bake_connectome(&[
            "".into(),
            "/home/taaslab/Загрузки/connections_princeton_no_threshold.csv.gz".into(),
            data("princeton_not.taas").to_string_lossy().into_owned(),
        ]);
    } else {
        bake_connectome(&args);
    }
}
