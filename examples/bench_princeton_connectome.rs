/// Princeton mouse visual cortex connectome benchmark.
///
/// Loads real data (115K neurons, 5.34M synapses from MICrONs pipeline),
/// builds CSR, and compares CPU SIMD LIF vs GPU Izhikevich throughput.
///
/// Requires: `connections_princeton.csv.gz` from MICrONs
///   (https://www.microns-explorer.org/phase1)
///
/// Запуск: cargo run --features cuda --example bench_princeton_connectome --release
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;
use cuda_kernels::GpuContext;
use taas_connectome::{
    BrainState, ConnectomeData, LifParams, NeuronModel, sim_lif,
};

const DT_MS: f32 = 0.1;
const DT_S: f32 = 0.1e-3;
const TAU_M: f32 = 10e-3;
const V_REST: f32 = -65e-3;
const V_THRESH: f32 = -50e-3;
const V_RESET: f32 = -65e-3;
const TAU_REF: f32 = 2e-3;
const N_STEPS: usize = 100;
const N_SAMPLE: usize = 3;

/// Map Princeton neurotransmitter to our ±1 convention.
fn nt_to_sign(nt: &str) -> f32 {
    match nt {
        "GABA" => -1.0,
        _ => 1.0, // ACH (ACh), GLUT (glutamate), DA (dopamine), SER (serotonin), OCT (octopamine)
    }
}

fn load_princeton_connectome(path: &str) -> (ConnectomeData, Vec<f32>) {
    println!("  Reading {} ...", path);
    let output = Command::new("zcat")
        .arg(path)
        .output()
        .expect("failed to run zcat -- is the file gzipped?");
    assert!(output.status.success(), "zcat failed");

    let stdin_str = String::from_utf8(output.stdout).expect("invalid UTF-8 in CSV");
    let mut lines: Vec<&str> = stdin_str.lines().collect();
    let header = lines.remove(0);
    assert_eq!(header, "pre_root_id,post_root_id,neuropil,syn_count,nt_type");

    // Build neuron ID -> index mapping
    let mut neuron_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut next_idx = 0usize;

    // First pass: collect unique IDs
    for line in &lines {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        let pre: u64 = fields[0].parse().expect("pre_root_id");
        let post: u64 = fields[1].parse().expect("post_root_id");
        neuron_to_idx.entry(pre).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
        neuron_to_idx.entry(post).or_insert_with(|| { let i = next_idx; next_idx += 1; i });
    }

    let n = neuron_to_idx.len();
    println!("  {n} unique neurons, {} synapses", lines.len());

    // Build CSR: count synapses per neuron first
    let n_syn = lines.len();
    let mut row_counts = vec![0usize; n + 1]; // +1 for sentinel
    let mut synapse_list: Vec<(usize, usize, f32)> = Vec::with_capacity(n_syn);

    for line in &lines {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        let pre: u64 = fields[0].parse().unwrap();
        let post: u64 = fields[1].parse().unwrap();
        let syn_count: f32 = fields[3].parse().unwrap_or(1.0);
        let pre_idx = neuron_to_idx[&pre];
        let post_idx = neuron_to_idx[&post];
        row_counts[pre_idx] += 1;
        synapse_list.push((pre_idx, post_idx, syn_count));
    }

    // Build row_ptr (cumsum)
    let mut row_ptr = vec![0i32; n + 1];
    let mut cum = 0i32;
    for i in 0..n {
        row_ptr[i] = cum;
        cum += row_counts[i] as i32;
    }
    row_ptr[n] = cum;
    assert_eq!(cum as usize, n_syn);

    // Fill col_indices and values
    let mut col_indices = vec![0i32; n_syn];
    let mut values = vec![0.0f32; n_syn];
    let mut pos = vec![0usize; n]; // write position per row

    for &(pre_idx, post_idx, syn_w) in &synapse_list {
        let p = pos[pre_idx];
        let idx = row_ptr[pre_idx] as usize + p;
        col_indices[idx] = post_idx as i32;
        values[idx] = syn_w * 1e-12; // scale to nA
        pos[pre_idx] = p + 1;
    }

    // NT types per neuron (most common type)
    // First pass: count NT types per neuron
    let mut nt_counts: Vec<HashMap<String, usize>> = vec![HashMap::new(); n];
    for line in &lines {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        let pre: u64 = fields[0].parse().unwrap();
        let nt = fields[4].to_string();
        let pre_idx = neuron_to_idx[&pre];
        *nt_counts[pre_idx].entry(nt).or_insert(0) += 1;
    }

    let nt_types: Vec<f32> = nt_counts.iter().map(|cnts| {
        let best = cnts.iter().max_by_key(|(_, c)| *c);
        match best {
            Some((nt, _)) => nt_to_sign(nt),
            None => 1.0,
        }
    }).collect();

    let conn = ConnectomeData::new(
        row_ptr, col_indices, values, nt_types, vec![],
    );

    // External I: 2nA baseline for all
    let external_I = vec![2e-9; n];

    (conn, external_I)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Princeton mouse visual cortex connectome benchmark     ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let princeton_csv = "/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/datasets/connections_princeton.csv.gz";

    let (conn, external_I) = load_princeton_connectome(princeton_csv);
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    println!("  Neurons: {n}, Synapses: {n_syn}, Avg syn/neu: {:.1}", n_syn as f64 / n as f64);

    // Estimate VRAM
    let mem_lif = n as f64 * 44.0;
    let mem_csr = n_syn as f64 * 12.0 + (n + 1) as f64 * 4.0;
    let mem_total = mem_lif + mem_csr;
    println!("  VRAM estimate: LIF {:.1}MB + CSR {:.1}MB = {:.1}MB",
             mem_lif / 1_048_576.0, mem_csr / 1_048_576.0, mem_total / 1_048_576.0);
    println!();

    // ─── 1. CPU SIMD LIF + CSR ─────────────────────────────
    println!("─── 1. CPU SIMD LIF + CSR ───");
    let mut state = BrainState::new(n, NeuronModel::LIF);
    let lif_params = LifParams::new(
        &vec![TAU_M; n], &vec![V_REST; n], &vec![V_THRESH; n], &vec![V_RESET; n],
        DT_MS, TAU_REF,
    );

    // Warmup
    let _ = sim_lif(&conn, &mut state, &lif_params, &external_I, 10);

    let mut cpu_t_avg = 0.0;
    for _ in 0..N_SAMPLE {
        let mut s = BrainState::new(n, NeuronModel::LIF);
        let t0 = Instant::now();
        let _ = sim_lif(&conn, &mut s, &lif_params, &external_I, N_STEPS);
        cpu_t_avg += t0.elapsed().as_secs_f64();
    }
    cpu_t_avg /= N_SAMPLE as f64;
    let cpu_thr = n as f64 * N_STEPS as f64 / cpu_t_avg;
    let cpu_us = cpu_t_avg / N_STEPS as f64 * 1e6;
    println!("  SIMD LIF: {:.3}s for {N_STEPS} steps, {:.1} us/step, {:.2e} n·s/s",
             cpu_t_avg, cpu_us, cpu_thr);
    println!();

    // ─── 2. GPU Izhikevich + CSR ────────────────────────────
    println!("─── 2. GPU Izhikevich + CSR ───");
    let ctx = GpuContext::new(n, n_syn).expect("GpuContext::new");
    ctx.upload_lif_params(
        DT_S, &vec![TAU_M; n], &vec![V_REST; n], &vec![V_THRESH; n], &vec![V_RESET; n],
    ).unwrap();
    ctx.upload_I(&external_I).unwrap();

    ctx.upload_connectome(
        &conn.row_ptr, &conn.col_indices, &conn.values,
        &conn.nt_types, &external_I,
    ).unwrap();

    // Allocate GPU spike counter
    let d_spike_counter = ctx.alloc_spike_counter().unwrap();

    // Warmup
    ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
    ctx.sync().unwrap();

    let mut gpu_t_avg = 0.0;
    for _ in 0..N_SAMPLE {
        ctx.upload_I(&external_I).unwrap();
        let t0 = Instant::now();
        for _ in 0..N_STEPS {
            ctx.upload_I(&external_I).unwrap();
            ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
        }
        ctx.sync().unwrap();
        gpu_t_avg += t0.elapsed().as_secs_f64();
    }
    gpu_t_avg /= N_SAMPLE as f64;
    let gpu_thr = n as f64 * N_STEPS as f64 / gpu_t_avg;
    let gpu_us = gpu_t_avg / N_STEPS as f64 * 1e6;

    // Also benchmark GPU without the upload_I overhead (optimistic path)
    let mut gpu_t_opt = 0.0;
    for _ in 0..N_SAMPLE {
        ctx.upload_I(&external_I).unwrap();
        let t0 = Instant::now();
        for _ in 0..N_STEPS {
            ctx.connectome_step(DT_S, 0.99, Some(d_spike_counter)).unwrap();
        }
        ctx.sync().unwrap();
        gpu_t_opt += t0.elapsed().as_secs_f64();
    }
    gpu_t_opt /= N_SAMPLE as f64;
    let gpu_thr_opt = n as f64 * N_STEPS as f64 / gpu_t_opt;
    let gpu_us_opt = gpu_t_opt / N_STEPS as f64 * 1e6;

    println!("  Connectome step (with re-upload I): {:.4}s for {N_STEPS} steps, {:.1} us/step, {:.2e} n·s/s",
             gpu_t_avg, gpu_us, gpu_thr);
    println!("  Connectome step (без re-upload I):  {:.4}s for {N_STEPS} steps, {:.1} us/step, {:.2e} n·s/s",
             gpu_t_opt, gpu_us_opt, gpu_thr_opt);
    println!();

    // ─── 3. Comparison ──────────────────────────────────────
    println!("─── 3. CPU vs GPU сравнение ───");
    let cpu_sp = cpu_t_avg / N_STEPS as f64;
    let gpu_sp_opt = gpu_t_opt / N_STEPS as f64;
    let speedup = cpu_sp / gpu_sp_opt;
    println!("  CPU SIMD LIF @ {n} neu: {:.2e}s/step, {:.2e} n·s/s",
             cpu_sp, cpu_thr);
    println!("  GPU Izhikevich @ {n} neu: {:.2e}s/step, {:.2e} n·s/s",
             gpu_sp_opt, gpu_thr_opt);
    println!("  GPU vs CPU speedup: {:.0}×", speedup);
    let gpu_vram = mem_total / 1_048_576.0;
    println!("  Dataset VRAM: {:.1} MB", gpu_vram);
}
