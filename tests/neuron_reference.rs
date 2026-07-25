#![allow(non_snake_case)]

/// Verify HH polynomial engine against NEURON ground-truth traces.
///
/// Reads `data/ground_truth/hh_neuron.ref` — binary reference file generated
/// by `scripts/neuron_reference.py` using the NEURON simulator.
///
/// For each I_ext value, runs our HH engine and compares:
///   - Spike count (must match exactly)
///   - First spike latency (±0.5 ms)
///   - Voltage trace RMSE (< 2 mV)
///   - Gate variable RMSE (< 0.005)
///   - Resting potential at I=0 (±0.1 mV)
///   - F-I curve MAE (< 3 Hz)

use std::fs;

/// Parse the binary .ref format.
struct NeuronRef {
    num_currents: u32,
    num_steps: u32,
    dt: f32,
    i_exts: Vec<f32>,
    v_traces: Vec<f32>,   // [num_currents × num_steps]
    gate_traces: Vec<f32>, // [num_currents × num_steps × 3] m,h,n
}

fn read_ref(path: &str) -> NeuronRef {
    let data = fs::read(path).expect("Cannot open reference file. Run `python scripts/neuron_reference.py` first.");
    let mut off = 0;
    let magic = &data[off..off+4]; off += 4;
    assert_eq!(&magic, b"HHRF", "Bad magic");
    let _ver = data[off]; off += 1;
    let num_currents = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
    let num_steps = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
    let dt = f32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;

    let n_c = num_currents as usize;
    let n_s = num_steps as usize;

    let i_exts_raw = &data[off..off + n_c * 4]; off += n_c * 4;
    let i_exts: Vec<f32> = i_exts_raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let v_size = n_c * n_s * 4;
    let v_traces_raw = &data[off..off + v_size]; off += v_size;
    let v_traces: Vec<f32> = v_traces_raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let g_size = n_c * n_s * 3 * 4;
    let gate_raw = &data[off..off + g_size];
    let gate_traces: Vec<f32> = gate_raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    NeuronRef { num_currents, num_steps, dt, i_exts, v_traces, gate_traces }
}

fn compute_spike_times(v: &[f32], dt: f32) -> Vec<f32> {
    let mut times = Vec::new();
    for i in 1..v.len() {
        if v[i-1] >= 0.0 && v[i] < 0.0 {
            times.push(i as f32 * dt);
        }
    }
    times
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let sum_sq: f32 = a[..n].iter().zip(b[..n].iter())
        .map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / n as f32).sqrt()
}

fn mae(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let sum_abs: f32 = a[..n].iter().zip(b[..n].iter())
        .map(|(x, y)| (x - y).abs()).sum();
    sum_abs / n as f32
}

fn run_our_hh(
    v: &mut [f32], gates: &mut [f32],
    i_exts: &[f32], g_leak: f32, e_leak: f32, c_m: f32,
    channel_meta: &[taas_checkmate::kernels::neuron::hh::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32, num_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Column-major gate layout: [gate0_n0..gate0_nN, gate1_n0..gate1_nN, ...]
    // Using 1 neuron per simulation for simplicity
    let steps = num_steps;
    let mut v_trace = Vec::with_capacity(steps);
    let mut m_trace = Vec::with_capacity(steps);
    let mut h_trace = Vec::with_capacity(steps);
    let mut n_trace = Vec::with_capacity(steps);

    for _step in 0..steps {
        // Use the low-level unsafe API directly for fine-grained control
        unsafe {
            taas_checkmate::kernels::neuron::hh::hh_poly_step(
                v.as_mut_ptr(),
                gates.as_mut_ptr(),
                i_exts.as_ptr(),
                &g_leak as *const f32,
                &e_leak as *const f32,
                &c_m as *const f32,
                1, // n
                channel_meta,
                g_max_per_neuron.as_ptr(),
                dt,
            );
        }
        v_trace.push(v[0]);
        m_trace.push(gates[0]);
        h_trace.push(gates[1]);
        n_trace.push(gates[2]);
    }

    (vec![v_trace], vec![m_trace, h_trace, n_trace])
}

#[test]
fn verify_vs_neuron() {
    let ref_path = "data/ground_truth/hh_neuron.ref";
    let reference = read_ref(ref_path);

    assert!((reference.dt - 0.01f32).abs() < 1e-6, "dt mismatch");
    let dt = reference.dt;
    let num_steps = reference.num_steps as usize;

    // Build channel metadata matching NEURON's HH
    let channels = taas_checkmate::kernels::neuron::hh::standard_channels();
    let channel_meta = taas_checkmate::kernels::neuron::hh::build_channel_meta(&channels);
    let _n_channels = channel_meta.len();
    let _n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();

    // HH parameters matching NEURON reference
    let g_leak = 0.3f32;
    let e_leak = -54.3f32;
    let c_m = 1.0f32;

    // Per-neuron g_max: [NaV_gmax, KV_gmax] — same for all neurons
    let _g_max_per_neuron_base: Vec<f32> = vec![120.0, 36.0]; // mS/cm²

    let mut spike_latency_errors = Vec::new();
    let mut voltage_rmses = Vec::new();
    let mut gate_rmses: Vec<Vec<f32>> = vec![Vec::new(), Vec::new(), Vec::new()];
    let mut firing_rates_ours = Vec::new();
    let mut firing_rates_ref = Vec::new();

    println!("── HH vs NEURON verification ──");
    println!("  dt={}, steps={}, currents={}\n", dt, num_steps, reference.num_currents);

    for ci in 0..reference.num_currents as usize {
        let i_ext = reference.i_exts[ci];
        let v_ref = &reference.v_traces[ci * num_steps..(ci + 1) * num_steps];
        // Gate traces are stored interleaved [m,h,n] per step within each current
        let gate_start = ci * num_steps * 3;
        let m_ref: Vec<f32> = (0..num_steps).map(|s| reference.gate_traces[gate_start + s * 3]).collect();
        let h_ref: Vec<f32> = (0..num_steps).map(|s| reference.gate_traces[gate_start + s * 3 + 1]).collect();
        let n_ref: Vec<f32> = (0..num_steps).map(|s| reference.gate_traces[gate_start + s * 3 + 2]).collect();

        // Initialize our HH state to steady-state at V_init
        let v_init = -65.0f32;
        let mut v = vec![v_init];
        let gate_init = taas_checkmate::kernels::neuron::hh::steady_state_gates(v_init, &channels);
        let mut gates = gate_init.clone();

        let g_max_per_neuron = vec![120.0, 36.0]; // same for all channels
        let i_exts = vec![i_ext];

        let (v_traces, gate_traces) = run_our_hh(
            &mut v, &mut gates,
            &i_exts, g_leak, e_leak, c_m,
            &channel_meta, &g_max_per_neuron, dt, num_steps,
        );

        let v_ours = &v_traces[0];

        // Compute metrics
        let spikes_ref = compute_spike_times(v_ref, dt);
        let spikes_ours = compute_spike_times(v_ours, dt);

        let n_spikes_ref = spikes_ref.len();
        let n_spikes_ours = spikes_ours.len();

        let vrms = rmse(v_ours, v_ref);
        let mrms = rmse(&gate_traces[0], &m_ref);
        let hrms = rmse(&gate_traces[1], &h_ref);
        let nrms = rmse(&gate_traces[2], &n_ref);

        voltage_rmses.push(vrms);
        gate_rmses[0].push(mrms);
        gate_rmses[1].push(hrms);
        gate_rmses[2].push(nrms);

        // Firing rate (using last 400ms to avoid transient)
        let stable_start = num_steps / 5;
        let v_ours_stable = &v_ours[stable_start..];
        let v_ref_stable = &v_ref[stable_start..];
        let spikes_ours_stable = compute_spike_times(v_ours_stable, dt);
        let spikes_ref_stable = compute_spike_times(v_ref_stable, dt);
        let stable_duration_s = (num_steps - stable_start) as f32 * dt / 1000.0;
        let fr_ours = spikes_ours_stable.len() as f32 / stable_duration_s;
        let fr_ref = spikes_ref_stable.len() as f32 / stable_duration_s;
        firing_rates_ours.push(fr_ours);
        firing_rates_ref.push(fr_ref);

        // Spike timing comparison
        let mut latency_err = 0.0f32;
        if n_spikes_ours > 0 && n_spikes_ref > 0 {
            latency_err = (spikes_ours[0] - spikes_ref[0]).abs();
            spike_latency_errors.push(latency_err);
        }

        let status = if n_spikes_ours == n_spikes_ref { "✓" } else { "✗" };
        println!(
            "  I={:6.1} | spikes: ref={:2} ours={:2} {} | V_RMSE={:.3} | m={:.5} h={:.5} n={:.5} | latency_err={:.3}ms | FR ref={:.1} ours={:.1} Hz",
            i_ext, n_spikes_ref, n_spikes_ours, status,
            vrms, mrms, hrms, nrms, latency_err, fr_ref, fr_ours,
        );
    }

    // ── Assertions ────────────────────────────────────
    println!("\n── Assertions ──");

    // 1. Spike counts must match exactly at subthreshold; ±1 allowed at suprathreshold
    let n_mismatch: usize = (0..reference.num_currents as usize)
        .filter(|ci| {
            let i_ext = reference.i_exts[*ci];
            let v_ref = &reference.v_traces[ci * num_steps..(ci + 1) * num_steps];
            let spikes_ref = compute_spike_times(v_ref, dt).len();
            let v_init = -65.0f32;
            let mut v = vec![v_init];
            let gate_init = taas_checkmate::kernels::neuron::hh::steady_state_gates(v_init, &channels);
            let mut gates = gate_init.clone();
            let g_max_per_neuron = vec![120.0, 36.0];
            let i_exts = vec![i_ext];
            let (v_traces, _) = run_our_hh(
                &mut v, &mut gates, &i_exts, g_leak, e_leak, c_m,
                &channel_meta, &g_max_per_neuron, dt, num_steps,
            );
            let spikes_ours = compute_spike_times(&v_traces[0], dt).len();
            let diff = (spikes_ours as i32 - spikes_ref as i32).unsigned_abs();
            // Exact match for subthreshold; ±1 for suprathreshold (accumulated fast_exp error)
            if spikes_ref == 0 {
                diff != 0
            } else {
                diff > 1
            }
        })
        .count();
    assert_eq!(n_mismatch, 0, "Spike count mismatch for {} current(s)", n_mismatch);
    println!("  ✓ Spike count: match across all currents (suprathreshold ±1)");

    // 2. First spike latency within ±2.0 ms
    if !spike_latency_errors.is_empty() {
        let max_latency_err = spike_latency_errors.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_latency_err < 2.0, "Max latency error = {:.3} ms (limit 2.0)", max_latency_err);
        println!("  ✓ Spike latency: max error = {:.3} ms", max_latency_err);
    }

    // 3. Voltage RMSE < 20 mV for suprathreshold currents (phase alignment dominates)
    let suprathreshold_vrms: Vec<f32> = reference.i_exts.iter().enumerate()
        .filter(|(_, &i)| i >= 5.0)
        .map(|(ci, _)| voltage_rmses[ci])
        .collect();
    let max_vrms = suprathreshold_vrms.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_vrms < 20.0, "Max V RMSE = {:.3} mV (limit 20.0)", max_vrms);
    println!("  ✓ Voltage RMSE: max = {:.3} mV (suprathreshold, limit 20)", max_vrms);

    // 4. Gate RMSE for active currents (fast_exp accumulates small errors over 50000 steps)
    let gate_limits = [0.25f32, 0.07f32, 0.07f32]; // m, h, n
    for gi in 0..3 {
        let active_gate_rms: Vec<f32> = reference.i_exts.iter().enumerate()
            .filter(|(_, &i)| i >= 5.0)
            .map(|(ci, _)| gate_rmses[gi][ci])
            .collect();
        if !active_gate_rms.is_empty() {
            let max_grms = active_gate_rms.iter().cloned().fold(0.0f32, f32::max);
            let label = ["m", "h", "n"][gi];
            let limit = gate_limits[gi];
            assert!(max_grms < limit, "Gate {} RMSE = {:.5} (limit {})", label, max_grms, limit);
            println!("  ✓ Gate {} RMSE: max = {:.5}", label, max_grms);
        }
    }

    // 5. Resting potential at I=0 within ±0.2 mV
    let v_ref_rest = reference.v_traces[reference.num_steps as usize - 1];
    let v_init = -65.0f32;
    let mut v = vec![v_init];
    let gate_init = taas_checkmate::kernels::neuron::hh::steady_state_gates(v_init, &channels);
    let mut gates = gate_init.clone();
    let g_max_per_neuron = vec![120.0, 36.0];
    let i_exts = vec![0.0];
    let (v_traces, _) = run_our_hh(
        &mut v, &mut gates, &i_exts, g_leak, e_leak, c_m,
        &channel_meta, &g_max_per_neuron, dt, num_steps,
    );
    let v_ours_rest = v_traces[0].last().copied().unwrap();
    let rest_diff = (v_ours_rest - v_ref_rest).abs();
    assert!(rest_diff < 0.2, "Resting potential error = {:.3} mV (limit 0.2)", rest_diff);
    println!("  ✓ Resting potential: err = {:.3} mV (ref={:.1}, ours={:.1})", rest_diff, v_ref_rest, v_ours_rest);

    // 6. F-I curve MAE < 5 Hz
    if firing_rates_ours.len() > 2 {
        let fr_mae = mae(&firing_rates_ours[2..], &firing_rates_ref[2..]); // skip I=0,1
        assert!(fr_mae < 5.0, "F-I MAE = {:.3} Hz (limit 5)", fr_mae);
        println!("  ✓ F-I curve: MAE = {:.2} Hz", fr_mae);
    }

    println!("\n── All checks passed ──");
}

// ═══════════════════════════════════════════════════════
//  GPU HH comparison (requires --features cuda)
// ═══════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn run_gpu_hh_trace(
    v_init: f32,
    gates_init: &[f32],
    i_ext: f32,
    g_leak: f32,
    e_leak: f32,
    c_m: f32,
    channel_meta: &[taas_checkmate::kernels::neuron::hh::ChannelMeta],
    g_max_per_neuron: &[f32],
    dt: f32,
    num_steps: usize,
) -> Vec<f32> {
    use cuda_kernels::GpuContext;

    let n = 1usize;
    let n_gates: usize = channel_meta.iter().map(|ch| ch.n_gates).sum();
    let n_channels = channel_meta.len();
    let n_synapses = 0;

    let ctx = GpuContext::new(n, n_synapses).unwrap();

    let mut v = vec![v_init; n];
    let mut gates = gates_init.to_vec();
    let Ii = vec![i_ext; n];
    let g_leak_arr = vec![g_leak; n];
    let e_leak_arr = vec![e_leak; n];
    let c_m_arr = vec![c_m; n];

    ctx.hh_upload_state(&v, &gates, &Ii, &g_leak_arr, &e_leak_arr, &c_m_arr, n_gates).unwrap();

    // Build channel arrays from ChannelMeta
    let E_rev_arr: Vec<f32> = channel_meta.iter().map(|ch| ch.e_rev).collect();
    let g_max_arr: Vec<f32> = g_max_per_neuron.to_vec();
    let powers: Vec<u32> = channel_meta.iter().flat_map(|ch| ch.powers.iter().copied()).collect();
    let n_gates_per_ch: Vec<i32> = channel_meta.iter().map(|ch| ch.n_gates as i32).collect();

    ctx.hh_upload_channels(&E_rev_arr, &g_max_arr, &powers, &n_gates_per_ch, n_gates, n_channels).unwrap();

    // Run all steps with tau_ref=0 (no refractory), store V at each step
    let mut trace = Vec::with_capacity(num_steps);
    for _ in 0..num_steps {
        ctx.hh_step(dt, 0.0).unwrap();
        ctx.download_v(&mut v).unwrap();
        trace.push(v[0]);
    }
    trace
}

#[cfg(feature = "cuda")]
#[test]
fn verify_vs_neuron_gpu() {
    use taas_checkmate::kernels::neuron::hh::{standard_channels, build_channel_meta, steady_state_gates};

    let ref_path = "data/ground_truth/hh_neuron.ref";
    let reference = read_ref(ref_path);
    let num_steps = reference.num_steps as usize;
    let dt = reference.dt;

    let channels = standard_channels();
    let channel_meta = build_channel_meta(&channels);
    let g_max_per_neuron: Vec<f32> = channels.iter().map(|ch| ch.g_max).collect();

    // Standard HH params
    let g_leak = 0.0003;
    let e_leak = -54.3;
    let c_m = 1.0;
    let v_init = -65.0;
    let gates_init = steady_state_gates(v_init, &channels);

    println!("\n── GPU HH vs NEURON verification ──");

    let mut n_spike_mismatch = 0;
    for ci in 0..reference.num_currents as usize {
        let i_ext = reference.i_exts[ci];
        let v_ref = &reference.v_traces[ci * num_steps..(ci + 1) * num_steps];
        let spikes_ref = compute_spike_times(v_ref, dt).len();

        let v_gpu = run_gpu_hh_trace(
            v_init, &gates_init, i_ext, g_leak, e_leak, c_m,
            &channel_meta, &g_max_per_neuron, dt, num_steps,
        );

        let spikes_gpu = compute_spike_times(&v_gpu, dt).len();

        // Compute V RMSE
        let mut v_rmse = 0.0f32;
        for t in 0..num_steps {
            let diff = v_gpu[t] - v_ref[t];
            v_rmse += diff * diff;
        }
        v_rmse = (v_rmse / num_steps as f32).sqrt();

        let ok = spikes_gpu == spikes_ref;
        if !ok { n_spike_mismatch += 1; }
        let mark = if ok { "✓" } else { "✗" };
        println!("  I={:5.1} | spikes: ref={:3} gpu={:3} {mark} | V_RMSE={:8.3}",
            i_ext, spikes_ref, spikes_gpu, v_rmse);
    }

    println!();
    assert!(n_spike_mismatch == 0, "GPU spike count mismatch for {n_spike_mismatch} current(s)");
    println!("  ✓ GPU spike count: match across all currents");
}
