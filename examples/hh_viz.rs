/// HH spike data dumper for visualization
use std::fs::File;
use std::io::{Write, BufWriter};
use cuda_kernels::GpuContext;

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 5.0;
const WEIGHT_SCALE: f32 = 0.01;

fn main() {
    let mut f = File::open("fafb.taas").unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();

    let ctx = GpuContext::new(n, n_syn).unwrap();
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &vec![0.0f32; n]).unwrap();

    const NG: usize = 4;
    let v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    for i in 0..n { gates[i*NG..i*NG+4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]); }
    ctx.hh_upload_state(&v_init, &gates, &vec![0.0f32; n],
                        &vec![0.3f32; n], &vec![-65.0f32; n], &vec![1.0f32; n], NG).unwrap();
    ctx.hh_upload_channels(&vec![50.0, -77.0, 120.0], &vec![120.0, 36.0, 0.5],
                           &vec![3u32, 1, 4, 1], &vec![2i32, 1, 1], NG, 3).unwrap();

    let n_sp = n / 20;     // 5% pulse group
    let n_bg = n / 10;     // 10% total stimulated (includes pulse group)

    // Phase 1: pulse 15nA to 5%, 200 steps (2ms)
    let pulse_len = 200;
    let mut pulse_I = vec![0.0f32; n];
    for i in 0..n_sp { pulse_I[i] = 15.0; }
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];

    // Use a single combined spike buffer
    let total_steps = pulse_len + 4000; // 2ms pulse + 40ms run
    let mut spike_buf = vec![0i32; total_steps * n];

    ctx.batch_hh_steps(&pulse_I, &mut spike_buf[..pulse_len * n],
                       &mut hv, &mut his,
                       pulse_len, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();
    let p1: i32 = spike_buf[..pulse_len * n].iter().sum();
    println!("Phase 1 (pulse {pulse_len} steps): {p1} spikes");

    // Phase 2: 1.5nA to 10%, 4000 steps (40ms)
    let run_len = 4000;
    let mut bg_I = vec![0.0f32; n];
    for i in 0..n_bg { bg_I[i] = 1.5; }
    ctx.batch_hh_steps(&bg_I, &mut spike_buf[pulse_len * n..],
                       &mut hv, &mut his,
                       run_len, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();
    let p2: i32 = spike_buf[pulse_len * n..].iter().sum();
    println!("Phase 2 (background {run_len} steps): {p2} spikes");

    // ─── Dump binary spike data ───
    let out_name = "hh_spikes.dat";
    let mut out = BufWriter::new(File::create(out_name).unwrap());
    for step in 0..total_steps {
        let off = step * n;
        let count: u32 = spike_buf[off..off + n].iter().filter(|&&s| s > 0).count() as u32;
        out.write_all(&count.to_le_bytes()).unwrap();
        for (i, &s) in spike_buf[off..off + n].iter().enumerate() {
            if s > 0 { out.write_all(&(i as u32).to_le_bytes()).unwrap(); }
        }
    }
    out.flush().unwrap();
    println!("Dumped {total_steps} steps × {n} neurons → {out_name}");

    // ─── Neuron metadata ───
    let mut spike_count = vec![0i32; n];
    let mut first_step = vec![-1i32; n];
    for step in 0..total_steps {
        let off = step * n;
        for i in 0..n {
            if spike_buf[off + i] > 0 {
                spike_count[i] += 1;
                if first_step[i] < 0 { first_step[i] = step as i32; }
            }
        }
    }
    let meta_name = "hh_neurons.csv";
    let mut meta = BufWriter::new(File::create(meta_name).unwrap());
    writeln!(meta, "neuron,first_step,spike_count,group").unwrap();
    for i in 0..n {
        let grp = if i < n_sp { 1 } else if i < n_bg { 2 } else { 0 };
        writeln!(meta, "{i},{},{},{}", first_step[i], spike_count[i], grp).unwrap();
    }
    println!("Stats: total={} multi={} max_spk={}",
             spike_count.iter().sum::<i32>(),
             spike_count.iter().filter(|&&c| c > 1).count(),
             spike_count.iter().max().unwrap());
    println!("Metadata → {meta_name}");
}
