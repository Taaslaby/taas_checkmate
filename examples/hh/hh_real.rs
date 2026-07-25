use cuda_kernels::GpuContext;
use rand::prelude::*;
/// HH heterogeneous: E/I, random params, noise — realistic raster
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Resolve a path under `examples/data/` (works from any cwd via CARGO_MANIFEST_DIR).
fn data(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("examples/data");
    p.push(name);
    p
}

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 8.0; // slower synaptic integration
const WEIGHT_SCALE: f32 = 0.002;
const NOISE_STD: f32 = 1.0;
const BATCH: usize = 50;
const TOTAL_STEPS: usize = 6000;
const N_BATCHES: usize = TOTAL_STEPS / BATCH;

fn main() {
    let t0 = std::time::Instant::now();
    let mut f = File::open(data("fafb.taas")).unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    let mut rng = rand::thread_rng();

    let ctx = GpuContext::new(n, n_syn).unwrap();

    // E/I: 50% exc, 50% inh — balanced
    let mut nt_types = vec![0.0f32; n];
    let n_pace = n / 200; // 0.5% pacemakers
    let mut pace_I = vec![0.0f32; n];
    for i in 0..n {
        nt_types[i] = if rng.gen::<f32>() < 0.5 { 1.0 } else { -1.0 };
        if i < n_pace {
            pace_I[i] = 8.0 + rng.gen::<f32>() * 4.0;
        } // 8-12 nA
    }
    ctx.upload_connectome(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        &nt_types,
        &pace_I,
    )
    .unwrap();

    // Heterogeneous HH params
    const NG: usize = 4;
    let mut v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    let mut g_leak = vec![0.0f32; n];
    let mut e_leak = vec![-65.0f32; n];
    let mut c_m = vec![0.0f32; n];

    for i in 0..n {
        gates[i * NG..i * NG + 4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]);
        g_leak[i] = (0.15f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0))
            .max(0.02)
            .min(0.8);
        c_m[i] = (1.0f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 1.8))
            .max(0.2)
            .min(3.0);
        e_leak[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 10.0;
        v_init[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 10.0;
    }

    ctx.hh_upload_state(
        &v_init,
        &gates,
        &vec![0.0f32; n],
        &g_leak,
        &e_leak,
        &c_m,
        NG,
    )
    .unwrap();
    ctx.hh_upload_channels(
        &vec![50.0, -77.0, 120.0],
        &vec![120.0, 36.0, 0.5],
        &vec![3u32, 1, 4, 1],
        &vec![2i32, 1, 1],
        NG,
        3,
    )
    .unwrap();

    let mut spike_buf = vec![0i32; TOTAL_STEPS * n];
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];

    // Pre-generate all noise vectors
    println!("Pre-generating noise...");
    let mut noise_all = vec![0.0f32; N_BATCHES * n];
    for b in 0..N_BATCHES {
        let off = b * n;
        for i in 0..n {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            noise_all[off + i] =
                (-2.0f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * NOISE_STD;
        }
    }

    println!("Running {} batches × {} steps...", N_BATCHES, BATCH);
    for b in 0..N_BATCHES {
        let mut I_batch = pace_I.clone();
        let noise_off = b * n;
        for i in 0..n {
            I_batch[i] += noise_all[noise_off + i];
            I_batch[i] = I_batch[i].max(-2.0).min(20.0);
        }

        let start = b * BATCH * n;
        let end = (b + 1) * BATCH * n;
        ctx.batch_hh_steps(
            &I_batch,
            &mut spike_buf[start..end],
            &mut hv,
            &mut his,
            BATCH,
            DT,
            TAU_REF,
            TAU_SYN,
            WEIGHT_SCALE,
        )
        .unwrap();
    }

    let total_spikes: i32 = spike_buf.iter().sum();
    let elapsed = t0.elapsed();
    println!(
        "\nTotal: {total_spikes} spikes in {:.2}s ({:.0} us/step)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / TOTAL_STEPS as f64 * 1e6
    );

    // ─── Stats ───
    let mut spike_count = vec![0i32; n];
    let mut first_step = vec![-1i32; n];
    for step in 0..TOTAL_STEPS {
        let off = step * n;
        for i in 0..n {
            if spike_buf[off + i] > 0 {
                spike_count[i] += 1;
                if first_step[i] < 0 {
                    first_step[i] = step as i32;
                }
            }
        }
    }

    let n_fired = spike_count.iter().filter(|&&c| c > 0).count();
    let n_multi = spike_count.iter().filter(|&&c| c > 1).count();
    let max_spk = *spike_count.iter().max().unwrap();
    let mean_spk = spike_count.iter().sum::<i32>() as f64 / n_fired.max(1) as f64;
    let m = spike_count.iter().sum::<i32>() as f64 / n as f64;
    let v: f64 = spike_count
        .iter()
        .map(|&c| (c as f64 - m).powi(2))
        .sum::<f64>()
        / n as f64;
    let cv = (v.sqrt() / m.max(0.001));
    println!(
        "Fired: {n_fired}/{n}, multi: {}, max/neur: {max_spk}, mean: {mean_spk:.1}, CV: {cv:.2}",
        n_multi
    );

    // E/I split
    let n_exc = nt_types.iter().filter(|&&t| t > 0.0).count();
    let n_inh = n - n_exc;
    let exc_spikes: f64 = spike_count
        .iter()
        .zip(nt_types.iter())
        .filter(|(_, &t)| t > 0.0)
        .map(|(&c, _)| c as f64)
        .sum::<f64>()
        / n_exc.max(1) as f64;
    let inh_spikes: f64 = spike_count
        .iter()
        .zip(nt_types.iter())
        .filter(|(_, &t)| t <= 0.0)
        .map(|(&c, _)| c as f64)
        .sum::<f64>()
        / n_inh.max(1) as f64;
    println!("E/I = {n_exc}/{n_inh}, mean rate: E={exc_spikes:.1}, I={inh_spikes:.1} spikes/60ms");

    // ─── Dump ───
    let out_name = "hh_real.dat";
    let mut out = BufWriter::new(File::create(out_name).unwrap());
    for step in 0..TOTAL_STEPS {
        let off = step * n;
        let cnt: u32 = spike_buf[off..off + n].iter().filter(|&&s| s > 0).count() as u32;
        out.write_all(&cnt.to_le_bytes()).unwrap();
        for (i, &s) in spike_buf[off..off + n].iter().enumerate() {
            if s > 0 {
                out.write_all(&(i as u32).to_le_bytes()).unwrap();
            }
        }
    }
    out.flush().unwrap();

    let mut meta = BufWriter::new(File::create("hh_real_neurons.csv").unwrap());
    writeln!(meta, "neuron,first,spikes,exc,pace,g_leak,c_m,e_leak").unwrap();
    for i in 0..n {
        writeln!(
            meta,
            "{i},{},{},{},{},{:.4},{:.4},{:.2}",
            first_step[i],
            spike_count[i],
            if nt_types[i] > 0.0 { 1 } else { 0 },
            if i < n_pace { 1 } else { 0 },
            g_leak[i],
            c_m[i],
            e_leak[i]
        )
        .unwrap();
    }
    println!("Dumped {out_name} + hh_real_neurons.csv");
}
