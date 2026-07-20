/// T2 visual stimulus → HH on FAFB 15M connectome with E/I balance
///
/// Uses the pipeline from fly-brain-main:
///   1. Visual input → ommatidia (721 per eye)
///   2. Ommatidia → T2/T2a neuron rates (T2_MAX=120 Hz, contrast threshold 0.3)
///   3. Rate → current injection (rate * 0.05 uA/cm² + baseline 8.0)
///   4. HH + CSR propagate through connectome
///   5. DN readout (descending neurons for behavior)
use std::fs::File;
use std::io::{Write, BufWriter, Read, BufReader, BufRead};
use std::path::PathBuf;
use cuda_kernels::GpuContext;
use rand::prelude::*;

/// Resolve a path under `examples/data/` (works from any cwd via CARGO_MANIFEST_DIR).
fn data(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("examples/data");
    p.push(name);
    p
}

const DT: f32 = 0.01;
const TAU_REF: f32 = 2.0;
const TAU_SYN: f32 = 8.0;
const WEIGHT_SCALE: f32 = 5_000_000_000.0;
const BATCH: usize = 50;
const TOTAL_STEPS: usize = 6000;
const N_BATCHES: usize = TOTAL_STEPS / BATCH;

/// T2 vision parameters (from fly-brain-main visual_system.py)
const T2_MAX_HZ: f32 = 120.0;      // max T2 firing rate
const CONTRAST_THRESH: f32 = 0.3;   // contrast threshold
const BASELINE_UA: f32 = 10.0;      // baseline current (uA/cm²)
const RATE_TO_CURRENT: f32 = 0.08;  // Hz → uA/cm²
const N_OMMATIDIA: usize = 721;     // ommatidia per eye
const BAR_SPEED: f32 = 16.7;        // retina sweep per second

fn load_indices(path: impl AsRef<std::path::Path>) -> Vec<usize> {
    let f = File::open(path).unwrap();
    let reader = BufReader::new(f);
    reader.lines().filter_map(|l| l.ok().and_then(|s| s.trim().parse().ok())).collect()
}

/// Build left/right T2 → ommatidium mapping
/// Each T2 neuron maps to one of 721 ommatidia, preserving rough retinotopy
fn build_t2_omm_mapping(t2_indices: &[usize], left_indices: &[usize], right_indices: &[usize])
    -> Vec<(usize, usize)> // (t2_idx_in_connectome, ommatidium_idx)
{
    let mut mapping = Vec::with_capacity(t2_indices.len());
    // Left eye T2s
    for (i, &idx) in left_indices.iter().enumerate() {
        let omm = i * N_OMMATIDIA / left_indices.len();
        mapping.push((idx, omm));
    }
    // Right eye T2s
    for (i, &idx) in right_indices.iter().enumerate() {
        let omm = i * N_OMMATIDIA / right_indices.len();
        mapping.push((idx, omm));
    }
    mapping
}

/// Compute visual stimulus: moving bar
/// Returns T2 rates for each T2 neuron
fn compute_t2_rates(t2_omm: &[(usize, usize)], t_sec: f32) -> Vec<f32> {
    let bar_center = (t_sec * BAR_SPEED) % 1.0f32;
    let bar_width = 0.1f32;

    // For each ommatidium, compute brightness (0-1)
    // Moving bright bar: ommatidia near bar center get brightness=1, others=0
    let mut omm_brightness = vec![0.0f32; N_OMMATIDIA];
    for omm in 0..N_OMMATIDIA {
        let pos = omm as f32 / N_OMMATIDIA as f32;
        let dist = (pos - bar_center).abs();
        omm_brightness[omm] = if dist < bar_width { 1.0 } else { 0.03 }; // slight background
    }

    // Global mean brightness
    let mean_bright: f32 = omm_brightness.iter().sum::<f32>() / N_OMMATIDIA as f32;
    let mean_bright = mean_bright.max(0.05);

    // Compute T2 rates from contrast (OFF pathway: respond to darkness)
    let mut rates = Vec::with_capacity(t2_omm.len());
    for &(_, omm) in t2_omm {
        let bright = omm_brightness[omm];
        // OFF contrast: positive when ommatidium is darker than background
        let raw_contrast = (mean_bright - bright) / mean_bright;
        let contrast = (raw_contrast - CONTRAST_THRESH).max(0.0).min(1.0);
        rates.push(contrast * T2_MAX_HZ);
    }
    rates
}

fn main() {
    let t0 = std::time::Instant::now();
    let mut rng = rand::thread_rng();

    // ─── Load 80M FAFB connectome ───
    let mut f = File::open(data("fafb.taas")).unwrap();
    let conn = taas_connectome::ConnectomeData::load_baked(&mut f).unwrap();
    let n = conn.n_neurons;
    let n_syn = conn.col_indices.len();
    println!("Connectome: {n} neurons, {n_syn} synapses");

    // ─── Load T2 indices ───
    let t2_all = load_indices(&data("t2_indices_80m.txt"));
    println!("T2/T2a neurons: {}", t2_all.len());

    // Split by left/right (from annotations: first 1649 ≈ left, rest ≈ right)
    // Actually we should load from side-specific files. For now, simulate:
    let t2_left: Vec<usize> = t2_all.iter().take(1649).copied().collect();
    let t2_right: Vec<usize> = t2_all.iter().skip(1649).copied().collect();
    let t2_omm = build_t2_omm_mapping(&t2_all, &t2_left, &t2_right);
    println!("  Left: {}, Right: {}", t2_left.len(), t2_right.len());

    // ─── Load DN indices ───
    let dn_p9 = load_indices(&data("P6_8P9_indices.txt"));
    let dn_dna01 = load_indices(&data("DNa01_indices.txt"));
    let dn_dna02 = load_indices(&data("DNa02_indices.txt"));
    let dn_mdn = load_indices(&data("MDN_indices.txt"));
    let dn_lplc2 = load_indices(&data("LPLC2_indices.txt"));
    let dn_lc4 = load_indices(&data("LC4_indices.txt"));
    println!("DNs: P9={}, DNa01={}, DNa02={}, MDN={}, LPLC2={}, LC4={}",
             dn_p9.len(), dn_dna01.len(), dn_dna02.len(), dn_mdn.len(),
             dn_lplc2.len(), dn_lc4.len());

    // ─── Upload connectome ───
    let ctx = GpuContext::new(n, n_syn).unwrap();
    // The .taas already has nt_types from the baking (excitatory/inhibitory from data)
    let pace_I = vec![0.0f32; n];
    ctx.upload_connectome(&conn.row_ptr, &conn.col_indices, &conn.values,
                          &conn.nt_types, &pace_I).unwrap();

    // ─── HH state ───
    const NG: usize = 3;
    let mut v_init = vec![-65.0f32; n];
    let mut gates = vec![0.0f32; n * NG];
    let mut g_leak = vec![0.0f32; n];
    let mut e_leak = vec![-65.0f32; n];
    let mut c_m = vec![1.0f32; n];

    for i in 0..n {
        gates[i*NG..i*NG+3].copy_from_slice(&[0.05, 0.6, 0.3]);
        g_leak[i] = (0.15f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 1.5)).max(0.05).min(0.6);
        c_m[i] = (1.0f32 * (1.0 + (rng.gen::<f32>() - 0.5) * 1.5)).max(0.3).min(2.5);
        e_leak[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 6.0;
        v_init[i] = -65.0 + (rng.gen::<f32>() - 0.5) * 6.0;
    }

    ctx.hh_upload_state(&v_init, &gates, &vec![0.0f32; n],
                        &g_leak, &e_leak, &c_m, NG).unwrap();
    // Only NaV (m³h) + KV (n⁴) — NO CaV-T (would make all neurons pacemakers)
    ctx.hh_upload_channels(
        &vec![50.0, -77.0],
        &vec![120.0, 36.0],
        &vec![3u32, 1, 4],
        &vec![2i32, 1],
        3, 2
    ).unwrap();

    // ─── Buffers ───
    let mut spike_buf = vec![0i32; TOTAL_STEPS * n];
    let mut hv = vec![0.0f32; n];
    let mut his = vec![0.0f32; n];

    // ─── DN rate tracking ───
    let dn_groups: Vec<(&str, &[usize])> = vec![
        ("P9", &dn_p9), ("DNa01", &dn_dna01), ("DNa02", &dn_dna02),
        ("MDN", &dn_mdn), ("LPLC2", &dn_lplc2), ("LC4", &dn_lc4),
    ];
    let mut dn_rates: Vec<Vec<f32>> = dn_groups.iter().map(|_| Vec::with_capacity(TOTAL_STEPS)).collect();

    println!("Running {} batches × {} steps...", N_BATCHES, BATCH);
    for b in 0..N_BATCHES {
        // ─── Build external_I: only T2 neurons get visual drive ───
        let mut I_batch = vec![0.0f32; n];
        let t_base = (b * BATCH) as f32 * DT * 0.001;

        // Compute T2 rates from visual stimulus
        let t2_rates = compute_t2_rates(&t2_omm, t_base);

        // Inject into T2 neurons: strong visual drive
        for (&(idx, _), &rate_hz) in t2_omm.iter().zip(t2_rates.iter()) {
            let i_vis = BASELINE_UA + rate_hz * RATE_TO_CURRENT;
            I_batch[idx] = i_vis.max(0.0).min(20.0);
        }

        // Clamp to reasonable range
        for v in I_batch.iter_mut() {
            *v = v.max(-2.0).min(20.0);
        }

        let start = b * BATCH * n;
        let end = (b + 1) * BATCH * n;
        ctx.batch_hh_steps(&I_batch, &mut spike_buf[start..end],
                           &mut hv, &mut his,
                           BATCH, DT, TAU_REF, TAU_SYN, WEIGHT_SCALE).unwrap();

        // ─── Record DN rates ───
        for step_rel in 0..BATCH {
            let step_abs = b * BATCH + step_rel;
            let off = step_abs * n;
            for (gi, (_, neurons)) in dn_groups.iter().enumerate() {
                let count = neurons.iter().filter(|&&ni| spike_buf[off + ni] > 0).count();
                dn_rates[gi].push(count as f32 / neurons.len() as f32);
            }
        }

        if (b + 1) % 20 == 0 || b == 0 {
            let el = t0.elapsed();
            println!("  {}% ({:.1}s)", (b + 1) * 100 / N_BATCHES, el.as_secs_f64());
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
        for i in 0..n { if spike_buf[off + i] > 0 { spike_count[i] += 1; } }
    }
    let n_fired = spike_count.iter().filter(|&&c| c > 0).count();
    let max_spk = *spike_count.iter().max().unwrap();
    let mean_spk = spike_count.iter().sum::<usize>() as f64 / n_fired.max(1) as f64;
    println!("Fired: {n_fired}/{n}, max/neur: {max_spk}, mean/active: {mean_spk:.1}");

    // DN stats
    println!("\n─── DN activity ───");
    for (gi, (name, _)) in dn_groups.iter().enumerate() {
        let total_dn: usize = dn_groups[gi].1.iter().map(|&ni| spike_count[ni]).sum();
        let rate = dn_rates[gi].iter().sum::<f32>() / TOTAL_STEPS as f32 * 100.0;
        println!("  {name:>8}: {total_dn} spikes total, {rate:.2} Hz avg");
    }

    // E/I split stats
    let n_exc = conn.nt_types.iter().filter(|&&t| t > 0.0).count();
    let n_inh = n - n_exc;
    let exc_spikes: f64 = spike_count.iter().zip(conn.nt_types.iter())
        .filter(|(_, &t)| t > 0.0).map(|(&c, _)| c as f64).sum::<f64>() / n_exc.max(1) as f64;
    let inh_spikes: f64 = spike_count.iter().zip(conn.nt_types.iter())
        .filter(|(_, &t)| t <= 0.0).map(|(&c, _)| c as f64).sum::<f64>() / n_inh.max(1) as f64;
    println!("\n  E/I: {n_exc}/{n_inh}, mean spikes: E={exc_spikes:.1}, I={inh_spikes:.1}");

    // ─── Save spikes ───
    let out_name = "t2_vision.dat";
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

    // ─── Save DN rates ───
    let mut gr_out = BufWriter::new(File::create("t2_vision_dn.csv").unwrap());
    write!(gr_out, "step").unwrap();
    for (name, _) in &dn_groups { write!(gr_out, ",{name}").unwrap(); }
    writeln!(gr_out).unwrap();
    for s in 0..TOTAL_STEPS {
        write!(gr_out, "{s}").unwrap();
        for gi in 0..dn_groups.len() {
            write!(gr_out, ",{:.6}", dn_rates[gi][s]).unwrap();
        }
        writeln!(gr_out).unwrap();
    }

    println!("\nSaved {out_name} + t2_vision_dn.csv");
}
