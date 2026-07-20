/// T3 BCI × HH/FAFB — optimized closed loop (pre-alloc, t3.sin1)
use std::fs::File;
use std::time::Instant;
use std::io::{Write, BufWriter};
use std::path::PathBuf;
use cuda_kernels::GpuContext;
use taas_t3::oscillator::TriOscillator;
use taas_t3::backend::Avx2Backend;
use rand::prelude::*;

/// Resolve a path under `examples/data/` (works from any cwd via CARGO_MANIFEST_DIR).
fn data(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("examples/data");
    p.push(name);
    p
}

const DT_MS: f32 = 0.01;
const DT_S: f32 = DT_MS / 1000.0;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 5.0;
const WEIGHT_SCALE: f32 = 0.005;
const N_T3: usize = 50;
const N_STEPS: usize = 6000;
const I_BASE: f32 = 0.5;
const I_AMP: f32 = 10.0;

fn main() {
    let ts = Instant::now();
    let mut f = File::open(data("fafb.taas")).unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    let grp = n / N_T3;
    let grp_rem = n - grp * N_T3; // leftover neurons go to last group
    let mut rng = thread_rng();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  T3 BCI × HH/FAFB — optimized closed loop       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  {n} neurons, {n_syn} synapses, {N_T3} T3, {grp}+{grp_rem} neur/grp");

    let ctx = GpuContext::new(n, n_syn).unwrap();
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &vec![1.0f32; n], &vec![0.0f32; n]).unwrap();

    const NG: usize = 4;
    let v = vec![-65.0f32; n];
    let mut g = vec![0.0f32; n * NG];
    for i in 0..n { g[i*NG..i*NG+4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]); }
    ctx.hh_upload_state(&v, &g, &vec![0.0f32; n],
                        &vec![0.3f32; n], &vec![-65.0f32; n], &vec![1.0f32; n], NG).unwrap();
    ctx.hh_upload_channels(&vec![50.0, -77.0, 120.0], &vec![120.0, 36.0, 0.5],
                           &vec![3u32, 1, 4, 1], &vec![2i32, 1, 1], NG, 3).unwrap();

    // T3 with coupling
    let mut t3 = TriOscillator::new(N_T3, 50.0, 500.0, Avx2Backend);
    t3.set_uniform_coupling(2.0, 1.2, 0.8);
    t3.alpha = 0.05;
    for k in 0..N_T3 {
        t3.th1[k] = rng.gen::<f32>() * 6.2832;
        t3.th2[k] = rng.gen::<f32>() * 6.2832;
        t3.th3[k] = rng.gen::<f32>() * 6.2832;
    }

    // Pre-allocated external_I buffer (mutated each step)
    let mut ext = vec![0.0f32; n];

    // Buffers
    let mut spike_buf = vec![0i32; N_STEPS * n];
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];
    let mut th1_log = vec![0.0f32; N_STEPS * N_T3];
    let mut rate_log = vec![0.0f32; N_STEPS * N_T3];

    // Pre-compute group index for each neuron (avoids division each step)
    let mut neuron_group = vec![0usize; n];
    for i in 0..n {
        neuron_group[i] = (i / grp).min(N_T3 - 1);
    }

    println!("  Running {N_STEPS} steps...");
    let tl = Instant::now();
    for step in 0..N_STEPS {
        // 1. Fill external_I: I = I_base + I_amp * sin1[k]
        //    sin1[k] = sin(th1[k]) from T3's prepare()
        for i in 0..n {
            let k = neuron_group[i];
            ext[i] = I_BASE + I_AMP * t3.sin1[k];
        }

        // 2. HH step
        let off = step * n;
        ctx.batch_hh_steps(&ext, &mut spike_buf[off..off + n],
                           &mut hv, &mut his,
                           1, DT_MS, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();

        // 3. Group spike rates → T3 drive
        let mut gspk = vec![0i32; N_T3];
        for i in 0..n {
            if spike_buf[off + i] > 0 { gspk[neuron_group[i]] += 1; }
        }
        for k in 0..N_T3 {
            let gs = if k == N_T3 - 1 { grp + grp_rem } else { grp };
            let rate = gspk[k] as f32 / DT_MS / gs as f32;
            rate_log[step * N_T3 + k] = rate;
            t3.drive1[k] = rate * 0.01;
        }

        // 4. T3 step (prepare() computes sin1 used in next iteration)
        t3.step(DT_S);
        t3.prepare();

        for k in 0..N_T3 { th1_log[step * N_T3 + k] = t3.th1[k]; }
    }
    let lt = tl.elapsed();

    let total: i32 = spike_buf.iter().sum();
    println!("  {:.3}s wall, {:.0} us/step, {:.2e} n·s/s",
             lt.as_secs_f64(), lt.as_secs_f64() / N_STEPS as f64 * 1e6,
             n as f64 * N_STEPS as f64 / lt.as_secs_f64());
    println!("  Total spikes: {total}, mean rate: {:.0} Hz",
             total as f64 / (N_STEPS as f64 * n as f64 * DT_MS as f64 / 1000.0));

    // Stats
    let mut sc = vec![0i32; n];
    for s in 0..N_STEPS {
        let o = s * n;
        for i in 0..n { if spike_buf[o + i] > 0 { sc[i] += 1; } }
    }
    let nf = sc.iter().filter(|&&c| c > 0).count();
    let mx = *sc.iter().max().unwrap();
    let mn = sc.iter().sum::<i32>() as f64 / n as f64;
    let vr: f64 = sc.iter().map(|&c| (c as f64 - mn).powi(2)).sum::<f64>() / n as f64;
    println!("  Fired: {nf}/{n}, max/neur: {mx}, mean: {mn:.1}, CV: {:.2}", vr.sqrt() / mn.max(1e-6));

    // Dump
    let mut dump = BufWriter::new(File::create("t3hh.dat").unwrap());
    for s in 0..N_STEPS {
        let o = s * n;
        let cnt = spike_buf[o..o + n].iter().filter(|&&x| x > 0).count() as u32;
        dump.write_all(&cnt.to_le_bytes()).unwrap();
        for i in 0..n { if spike_buf[o + i] > 0 { dump.write_all(&(i as u32).to_le_bytes()).unwrap(); } }
    }
    dump.flush().unwrap();

    let mut csv = BufWriter::new(File::create("t3hh_t3.csv").unwrap());
    writeln!(csv, "step,osc,th1,rate").unwrap();
    for s in 0..N_STEPS { for k in 0..N_T3 {
        writeln!(csv, "{s},{k},{},{}", th1_log[s * N_T3 + k], rate_log[s * N_T3 + k]).unwrap();
    }}

    let mut meta = BufWriter::new(File::create("t3hh_neurons.csv").unwrap());
    writeln!(meta, "neuron,spikes,group").unwrap();
    for i in 0..n { writeln!(meta, "{i},{},{}", sc[i], neuron_group[i]).unwrap(); }
    println!("  Dumped");
    println!("  Total: {:.2}s", ts.elapsed().as_secs_f64());
}
