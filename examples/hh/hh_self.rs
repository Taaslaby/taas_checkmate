use cuda_kernels::GpuContext;
/// HH self-sustained: pacemaker neurons drive the connectome, no external current otherwise
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("examples/data");
    p.push(name);
    p
}

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 5.0;
const WEIGHT_SCALE: f32 = 0.01;

fn main() {
    let mut f = File::open(data("fafb.taas")).unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();

    let ctx = GpuContext::new(n, n_syn).unwrap();
    ctx.upload_connectome(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        &conn.nt_types,
        &vec![0.0f32; n],
    )
    .unwrap();

    const NG: usize = 4;
    let v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    for i in 0..n {
        gates[i * NG..i * NG + 4].copy_from_slice(&[0.05, 0.6, 0.3, 0.0]);
    }
    ctx.hh_upload_state(
        &v_init,
        &gates,
        &vec![0.0f32; n],
        &vec![0.3f32; n],
        &vec![-65.0f32; n],
        &vec![1.0f32; n],
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

    // Pacemaker: 1% of neurons get 6.5nA (slightly above HH rheobase)
    let n_pace = n / 100;
    let mut ext_I = vec![0.0f32; n];
    for i in 0..n_pace {
        ext_I[i] = 6.5;
    }

    let total_steps = 6000; // 60ms
    let mut spike_buf = vec![0i32; total_steps * n];
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];

    ctx.batch_hh_steps(
        &ext_I,
        &mut spike_buf,
        &mut hv,
        &mut his,
        total_steps,
        DT,
        TAU_REF,
        TAU_SYN,
        WEIGHT_SCALE,
    )
    .unwrap();

    let total_spikes: i32 = spike_buf.iter().sum();
    println!(
        "Total: {total_spikes} spikes in {total_steps} steps ({:.1}ms)",
        total_steps as f32 * DT
    );

    // ─── Stats ───
    let mut spike_count = vec![0i32; n];
    let mut first_step = vec![-1i32; n];
    for step in 0..total_steps {
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
    let pace_fired = spike_count[..n_pace].iter().filter(|&&c| c > 0).count();
    let rest_fired = spike_count[n_pace..].iter().filter(|&&c| c > 0).count();
    println!("Fired: {n_fired}/{n}, multi: {n_multi}, max/neur: {max_spk}, mean: {mean_spk:.1}");
    println!(
        "Pacemakers (n={n_pace}): {pace_fired} fired, rest: {rest_fired}/{r} fired",
        r = n - n_pace
    );
    println!(
        "Period estimate: {:.0}ms between waves ({} waves in {:.1}ms)",
        total_steps as f32 * DT / (max_spk as f32).max(1.0),
        max_spk,
        total_steps as f32 * DT
    );

    // ─── Per-step density ───
    let density: Vec<u32> = (0..total_steps)
        .map(|step| {
            spike_buf[step * n..(step + 1) * n]
                .iter()
                .filter(|&&s| s > 0)
                .count() as u32
        })
        .collect();

    // Count bursts (consecutive steps with >0 spikes after a gap)
    let mut bursts = 0;
    let mut in_burst = false;
    let mut burst_starts = Vec::new();
    for step in 0..total_steps {
        if density[step] > 1000 {
            // more than 1000 spikes/step = burst
            if !in_burst {
                bursts += 1;
                burst_starts.push(step);
            }
            in_burst = true;
        } else {
            in_burst = false;
        }
    }
    println!("Bursts (>1000 spk/step): {bursts}");
    if !burst_starts.is_empty() {
        let intervals: Vec<i32> = burst_starts
            .windows(2)
            .map(|w| (w[1] - w[0]) as i32)
            .collect();
        if !intervals.is_empty() {
            let mean_int = intervals.iter().sum::<i32>() as f64 / intervals.len() as f64;
            println!(
                "Inter-burst interval: {mean_int:.0} steps ({:.1}ms)",
                mean_int * DT as f64
            );
            println!("Frequency: {:.0} Hz", 1000.0 / (mean_int * DT as f64));
        }
    }

    // ─── Dump for viz ───
    let out_name = "hh_self.dat";
    let mut out = BufWriter::new(File::create(out_name).unwrap());
    for step in 0..total_steps {
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
    println!("Dumped {out_name}");

    let mut meta = BufWriter::new(File::create("hh_self_neurons.csv").unwrap());
    writeln!(meta, "neuron,first_step,spike_count,pace").unwrap();
    for i in 0..n {
        writeln!(
            meta,
            "{i},{},{},{}",
            first_step[i],
            spike_count[i],
            if i < n_pace { 1 } else { 0 }
        )
        .unwrap();
    }
    println!("Dumped hh_self_neurons.csv");
}
