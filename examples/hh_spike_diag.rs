/// HH realistic spiking: short pulse to trigger, then network dynamics
use std::fs::File;
use std::time::Instant;
use cuda_kernels::GpuContext;

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 5.0;
const WEIGHT_SCALE: f32 = 0.01; // 2× stronger than bench

fn run(ctx: &GpuContext, external_I: &[f32], n_steps: usize,
       spike_buf: &mut [i32], host_v: &mut [f32], host_I_syn: &mut [f32]) {
    ctx.batch_hh_steps(external_I, spike_buf, host_v, host_I_syn,
                       n_steps, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();
}

fn main() {
    let mut f = File::open("fafb.taas").unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;

    println!("╔══════════════════════════════════════════╗");
    println!("║  HH realistic spiking: pulse → network  ║");
    println!("╚══════════════════════════════════════════╝");
    println!("{n} neurons, {} synapses", conn.col_indices.len());

    let n_syn = conn.col_indices.len();
    let ctx = GpuContext::new(n, n_syn).unwrap();

    // Upload connectome
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &vec![0.0f32; n]).unwrap();

    // HH setup
    const NG: usize = 4;
    let v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    for i in 0..n { gates[i*NG..i*NG+4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]); }
    ctx.hh_upload_state(&v_init, &gates, &vec![0.0f32; n],
                        &vec![0.3f32; n], &vec![-65.0f32; n], &vec![1.0f32; n], NG).unwrap();
    ctx.hh_upload_channels(&vec![50.0, -77.0, 120.0], &vec![120.0, 36.0, 0.5],
                           &vec![3u32, 1, 4, 1], &vec![2i32, 1, 1], NG, 3).unwrap();

    // Phase 1: brief high-current pulse to trigger initial spikes
    let pulse_len = 50; // 0.5ms pulse
    let mut pulse_I = vec![0.0f32; n];
    let n_sp = n / 20; // 5% get the pulse
    for i in 0..n_sp { pulse_I[i] = 15.0; } // 15nA — way above rheobase

    let mut spike_buf = vec![0i32; pulse_len * n];
    let mut host_v = vec![0.0f32; n];
    let mut host_I_syn = vec![0.0f32; n];

    let t0 = Instant::now();
    run(&ctx, &pulse_I, pulse_len, &mut spike_buf, &mut host_v, &mut host_I_syn);
    let pulse_spikes: i32 = spike_buf.iter().sum();
    let active = spike_buf.iter().filter(|&&s| s > 0).count();
    println!("\n─── Phase 1: pulse 50 steps (15nA to {n_sp} neurons) ───");
    println!("  Spikes: {pulse_spikes}, unique neurons: {active}");

    // Phase 2: background current, let network propagate
    let run_len = 2000; // 20ms biological
    let mut bg_I = vec![0.0f32; n];
    let n_bg = n / 10;
    for i in 0..n_bg { bg_I[i] = 1.0; } // 1nA — subthreshold

    let mut spike_buf2 = vec![0i32; run_len * n];
    let t0 = Instant::now();
    run(&ctx, &bg_I, run_len, &mut spike_buf2, &mut host_v, &mut host_I_syn);
    let elapsed = t0.elapsed();
    let run_spikes: i32 = spike_buf2[..run_len * n].iter().sum();
    let run_active = spike_buf2[..run_len * n].iter().filter(|&&s| s > 0).count();

    println!("\n─── Phase 2: background 1nA to {n_bg} neurons, {run_len} steps ───");
    println!("  {:.3}s — {:.0} us/step, {:.2e} n*s/s",
             elapsed.as_secs_f64(),
             elapsed.as_secs_f64() / run_len as f64 * 1e6,
             n as f64 * run_len as f64 / elapsed.as_secs_f64());
    println!("  Spikes: {run_spikes}, unique neurons: {run_active}");

    // Detailed spike stats
    let mut spike_count = vec![0i32; n];
    let mut first_step = vec![-1i32; n];
    for step in 0..run_len {
        let off = step * n;
        for i in 0..n {
            if spike_buf2[off + i] > 0 {
                spike_count[i] += 1;
                if first_step[i] < 0 { first_step[i] = step as i32; }
            }
        }
    }
    let multi = spike_count.iter().filter(|&&c| c > 1).count();
    let max_spk = spike_count.iter().max().copied().unwrap_or(0);
    let mean_fr = spike_count.iter().sum::<i32>() as f64 / run_active.max(1) as f64;
    println!("  Multi-spike neurons: {multi} (max {max_spk}/neur, mean {mean_fr:.1}/neur)");

    // Spike density over time
    println!("\n  Spike histogram (40 bins × 50 steps each):");
    let bins = 40;
    let spb = run_len / bins;
    for bi in 0..bins {
        let start = bi * spb;
        let end = start + spb;
        let count: i32 = spike_buf2[start * n..end * n].iter().sum();
        let bar = "█".repeat((count as usize / (run_spikes as usize / bins).max(1) / 5).max(1));
        println!("    {:4}ms–{:4}ms: {:6} {}", start as f32 * DT, end as f32 * DT, count, bar);
    }

    // Final state
    let v_mean = host_v.iter().sum::<f32>() / n as f32;
    println!("\n  Final V: min={:.1} max={:.1} mean={:.1} mV",
             host_v.iter().fold(f32::MAX, |a, &b| a.min(b)),
             host_v.iter().fold(f32::MIN, |a, &b| a.max(b)),
             v_mean);
}
