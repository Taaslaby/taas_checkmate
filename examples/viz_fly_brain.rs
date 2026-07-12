/// Visual stimulus → FAFB photoreceptors → HH brain
use std::fs::File;
use std::io::{Write, BufWriter, Read};
use cuda_kernels::GpuContext;
use rand::prelude::*;

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 8.0;
const WEIGHT_SCALE: f32 = 0.002;
const BATCH: usize = 50;
const TOTAL_STEPS: usize = 6000;
const N_BATCHES: usize = TOTAL_STEPS / BATCH;

/// Load photoreceptor map: `[n:u32] [idx:u32, x:f32, y:f32]×n`
fn load_photo_map(path: &str) -> Vec<(usize, f32, f32)> {
    let mut f = File::open(path).unwrap();
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4).unwrap();
    let n = u32::from_le_bytes(buf4) as usize;
    let mut map = Vec::with_capacity(n);
    let mut entry = [0u8; 12]; // u32 + f32 + f32
    for _ in 0..n {
        f.read_exact(&mut entry).unwrap();
        let idx = u32::from_le_bytes(entry[0..4].try_into().unwrap()) as usize;
        let x = f32::from_le_bytes(entry[4..8].try_into().unwrap());
        let y = f32::from_le_bytes(entry[8..12].try_into().unwrap());
        map.push((idx, x, y));
    }
    map
}

/// Load indices from text file (one per line)
fn load_indices(path: &str) -> Vec<usize> {
    let content = std::fs::read_to_string(path).unwrap();
    content.lines().filter_map(|l| l.trim().parse().ok()).collect()
}

fn main() {
    let t0 = std::time::Instant::now();
    let mut rng = rand::thread_rng();

    // ─── Load connectome ───
    let mut f = File::open("fafb.taas").unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    println!("Connectome: {n} neurons, {n_syn} synapses");

    // ─── Load photoreceptor map ───
    let photo_map = load_photo_map("photo_map.bin");
    println!("Photoreceptors: {}", photo_map.len());

    // ─── Load monitor groups ───
    let lamina = load_indices("lamina_indices.txt");
    let t4 = load_indices("t4_indices.txt");
    let t5 = load_indices("t5_indices.txt");
    let medulla = load_indices("medulla_intrinsic_indices.txt");
    let distal = load_indices("distal_medulla_indices.txt");
    println!("Lamina: {}, T4: {}, T5: {}, Medulla: {}, Distal: {}",
             lamina.len(), t4.len(), t5.len(), medulla.len(), distal.len());

    // ─── GPU context ───
    let ctx = GpuContext::new(n, n_syn).unwrap();

    // Upload connectome (all excitatory for FAFB — no E/I split needed)
    let nt_types = vec![1.0f32; n];
    let pace_I = vec![0.0f32; n];
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &nt_types, &pace_I).unwrap();

    // ─── HH params ───
    const NG: usize = 4;
    let mut v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    let mut g_leak = vec![0.0f32; n];
    let mut e_leak = vec![-65.0f32; n];
    let mut c_m = vec![1.0f32; n];

    for i in 0..n {
        gates[i*NG..i*NG+4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]);
        g_leak[i] = (0.15f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 1.5)).max(0.05).min(0.6);
        c_m[i] = (1.0f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 1.5)).max(0.3).min(2.5);
        e_leak[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 6.0;
        v_init[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 6.0;
    }

    ctx.hh_upload_state(&v_init, &gates, &vec![0.0f32; n],
                        &g_leak, &e_leak, &c_m, NG).unwrap();
    ctx.hh_upload_channels(
        &vec![50.0, -77.0, 120.0],  // E_rev: Na, K, Ca
        &vec![120.0, 36.0, 0.5],    // g_max
        &vec![3u32, 1, 4, 1],       // powers [n_gates_per_ch+n_gate_types] 
        &vec![2i32, 1, 1],          // n_gates_per_ch [n_channels]
        NG, 3
    ).unwrap();

    // ─── Pre-allocate ───
    let mut spike_buf = vec![0i32; TOTAL_STEPS * n];
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];

    // ─── Stimulus params ───
    let I_bg = 0.0f32;      // no background — photoreceptors OFF in dark
    let I_amp = 10.0f32;    // strong flash amplitude (nA)
    // Moving bar: a bright bar sweeps across the retina
    let bar_width = 0.1f32;   // fraction of retina width
    let bar_speed = 16.7f32;  // full sweep in 60ms (1.0 / 0.06)

    // ─── Monitor group rates ───
    let groups = [("lamina", &lamina), ("t4", &t4), ("t5", &t5),
                  ("medulla", &medulla), ("distal", &distal)];
    let mut group_rates: Vec<Vec<f32>> = groups.iter().map(|_| Vec::with_capacity(TOTAL_STEPS)).collect();

    println!("Running {} batches × {} steps...", N_BATCHES, BATCH);
    for b in 0..N_BATCHES {
        // ─── Build external_I with visual stimulus ───
        let mut I_batch = vec![0.0f32; n];
        let t_base = (b * BATCH) as f32 * DT * 0.001; // time in seconds

        for &(idx, rx, ry) in &photo_map {
            let t_sec = t_base;
            // Moving bar: bright bar at position bar_center
            // bar_center sweeps across retina over time
            let bar_center = (t_sec * bar_speed) % 1.0f32; // wraps around
            // Distance from photoreceptor to bar center (horizontal)
            let dist = (rx - bar_center).abs();
            let s = if dist < bar_width { 1.0 } else { 0.0 };
            I_batch[idx] = I_amp * s;
        }

        let start = b * BATCH * n;
        let end = (b + 1) * BATCH * n;
        ctx.batch_hh_steps(&I_batch, &mut spike_buf[start..end],
                           &mut hv, &mut his,
                           BATCH, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();

        // ─── Record group rates ───
        let step_start = b * BATCH;
        for step_rel in 0..BATCH {
            let step_abs = step_start + step_rel;
            let off = step_abs * n;
            for (gi, (_, neurons)) in groups.iter().enumerate() {
                let count = neurons.iter()
                    .filter(|&&ni| spike_buf[off + ni] > 0)
                    .count();
                group_rates[gi].push(count as f32 / neurons.len() as f32);
            }
        }

        if (b + 1) % 20 == 0 || b == 0 {
            let pct = (b + 1) * 100 / N_BATCHES;
            let el = t0.elapsed();
            println!("  {pct}% ({:.1}s)", el.as_secs_f64());
        }
    }

    let total_spikes: i32 = spike_buf.iter().sum();
    let elapsed = t0.elapsed();
    println!("\nTotal: {total_spikes} spikes in {:.2}s ({:.0} us/step)",
             elapsed.as_secs_f64(),
             elapsed.as_secs_f64() / TOTAL_STEPS as f64 * 1e6);

    // ─── Stats ───
    let mut spike_count = vec![0usize; n];
    for step in 0..TOTAL_STEPS {
        let off = step * n;
        for i in 0..n {
            if spike_buf[off + i] > 0 { spike_count[i] += 1; }
        }
    }
    let n_fired = spike_count.iter().filter(|&&c| c > 0).count();
    let max_spk = *spike_count.iter().max().unwrap();
    let mean_spk = spike_count.iter().sum::<usize>() as f64 / n_fired.max(1) as f64;
    println!("Fired: {n_fired}/{n}, max/neur: {max_spk}, mean/active: {mean_spk:.1}");

    // Group-specific stats
    println!("\n─── Group activity ───");
    for (gi, (name, neurons)) in groups.iter().enumerate() {
        let group_total: usize = neurons.iter().map(|&ni| spike_count[ni]).sum();
        let group_fired = neurons.iter().filter(|&&ni| spike_count[ni] > 0).count();
        let rate_hz = group_rates[gi].iter().sum::<f32>() / TOTAL_STEPS as f32 * 100.0;
        println!("  {name:>8}: {group_fired}/{} fired, {group_total} spikes, {rate_hz:.1} Hz avg", neurons.len());
    }

    // ─── Dump spikes ───
    let out_name = "viz_fly.dat";
    let mut out = BufWriter::new(File::create(out_name).unwrap());
    for step in 0..TOTAL_STEPS {
        let off = step * n;
        let cnt: u32 = spike_buf[off..off + n].iter().filter(|&&s| s > 0).count() as u32;
        out.write_all(&cnt.to_le_bytes()).unwrap();
        for (i, &s) in spike_buf[off..off + n].iter().enumerate() {
            if s > 0 { out.write_all(&(i as u32).to_le_bytes()).unwrap(); }
        }
    }
    out.flush().unwrap();
    println!("\nDumped {out_name}");

    // ─── Dump group rates ───
    let mut gr_out = BufWriter::new(File::create("viz_fly_groups.csv").unwrap());
    write!(gr_out, "step").unwrap();
    for (name, _) in &groups { write!(gr_out, ",{name}").unwrap(); }
    writeln!(gr_out).unwrap();
    for s in 0..TOTAL_STEPS {
        write!(gr_out, "{s}").unwrap();
        for gi in 0..groups.len() {
            write!(gr_out, ",{:.6}", group_rates[gi][s]).unwrap();
        }
        writeln!(gr_out).unwrap();
    }
    println!("Dumped viz_fly_groups.csv");
}
