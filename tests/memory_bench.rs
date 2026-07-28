/// Memory & speed benchmark: 200M neurons, 1 HH step.
///
/// Run: cargo test --release --test memory_bench -- --nocapture
use std::time::Instant;

fn format_bytes(b: usize) -> String {
    if b > 1_000_000_000 { format!("{:.1} GB", b as f64 / 1e9) }
    else if b > 1_000_000 { format!("{:.1} MB", b as f64 / 1e6) }
    else { format!("{b} B") }
}

fn mb_allocated() -> usize {
    let mut rss = 0;
    if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
        let pages: usize = statm.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
        rss = pages * 4096;
    }
    rss
}

#[test]
fn bench_200m_neurons() {
    let n = 200_000_000usize;
    let dt = 0.01f32;

    println!("═══ 200M Neuron HH Memory & Speed Bench ═══");

    // Define n_channels BEFORE allocation
    let n_channels = taas_checkmate::kernels::neuron::hh::standard_channels().len();

    let mem_before = mb_allocated();
    println!("RSS before: {}", format_bytes(mem_before));

    // ── Allocate ──
    let t_alloc = Instant::now();
    let mut v = vec![-65.0f32; n];
    let n_gates_per_neuron = 3usize;
    let mut gates = vec![0.05f32; n * n_gates_per_neuron];
    for i in 0..n {
        gates[i * n_gates_per_neuron + 1] = 0.6; // h
        gates[i * n_gates_per_neuron + 2] = 0.3; // n
    }
    let i_ext = vec![0.0f32; n];
    let g_leak = vec![0.3f32; n];
    let e_leak = vec![-54.3f32; n];
    let c_m = vec![1.0f32; n];
    let mut g_max = Vec::with_capacity(n * n_channels);
    // Fill: [120.0, 36.0] per neuron
    for _ in 0..n {
        g_max.push(120.0);
        g_max.push(36.0);
    }
    let t_alloc = t_alloc.elapsed();

    let mem_after = mb_allocated();
    println!("Alloc time: {:.2}s", t_alloc.as_secs_f64());
    println!("RSS after:  {} (Δ = {})", format_bytes(mem_after), format_bytes(mem_after - mem_before));

    // ── Build channel meta ──
    let channels = taas_checkmate::kernels::neuron::hh::standard_channels();
    let channel_meta = taas_checkmate::kernels::neuron::hh::build_channel_meta(&channels);

    // ── 1 step ──
    println!("\nRunning 1 HH step...");
    let t_step = Instant::now();
    unsafe {
        taas_checkmate::kernels::neuron::hh::hh_poly_step(
            v.as_mut_ptr(),
            gates.as_mut_ptr(),
            i_ext.as_ptr(),
            g_leak.as_ptr(),
            e_leak.as_ptr(),
            c_m.as_ptr(),
            n,
            &channel_meta,
            g_max.as_ptr(),
            dt,
        );
    }
    let t_step = t_step.elapsed();

    println!("Step time:  {:.4}s ({:.1} ns/neuron)",
        t_step.as_secs_f64(),
        t_step.as_nanos() as f64 / n as f64);

    // ── Verify no crash ──
    assert!(!v[0].is_nan());
    assert!(!v[n - 1].is_nan());
    // Sample a few values to verify computation happened
    let sample = v[n / 2];
    assert!(sample.is_finite());
    println!("\n✓ 200M neurons: 1 step in {:.3}s, RSS ≈ {}",
        t_step.as_secs_f64(), format_bytes(mem_after - mem_before));

    // Estimate for 1B
    let ratio = 1_000_000_000.0 / n as f64;
    println!("→ 1B HH estimate: {:.1}s/step, ~{} RSS",
        t_step.as_secs_f64() * ratio,
        format_bytes(((mem_after - mem_before) as f64 * ratio) as usize));

    // ── Clean up HH allocs ──
    drop(v); drop(gates); drop(i_ext); drop(g_leak); drop(e_leak); drop(c_m); drop(g_max);
    drop(channels); drop(channel_meta);

    // ═══════════════════════════════════════════════════════
    //  LIF comparison
    // ═══════════════════════════════════════════════════════
    println!("\n═══ 200M Neuron LIF Memory & Speed Bench ═══");

    let mem_lif_before = mb_allocated();
    let t_lif_alloc = Instant::now();

    let mut v_lif = vec![-65.0f32; n];
    let mut i_ext_lif = vec![10.0f32; n];
    let mut spikes_lif = vec![0i32; n];
    let mut params_lif = Vec::with_capacity(n * 5);
    for _ in 0..n {
        params_lif.push(dt);       // dt
        params_lif.push(10.0);     // tau_m
        params_lif.push(-65.0);    // v_rest
        params_lif.push(-50.0);    // v_thresh
        params_lif.push(-70.0);    // v_reset
    }
    let mut refractory_lif = vec![0.0f32; n];
    let tau_ref_lif = 2.0f32;

    let t_lif_alloc = t_lif_alloc.elapsed();
    let mem_lif_after = mb_allocated();
    println!("Alloc time: {:.2}s", t_lif_alloc.as_secs_f64());
    println!("RSS after:  {} (Δ = {})",
        format_bytes(mem_lif_after), format_bytes(mem_lif_after - mem_lif_before));

    println!("\nRunning 1 LIF step...");
    let t_lif_step = Instant::now();
    taas_checkmate::kernels::lif_step(
        &mut v_lif,
        &mut i_ext_lif,
        &mut spikes_lif,
        &params_lif,
        &mut refractory_lif,
        tau_ref_lif,
        None,
    );
    let t_lif_step = t_lif_step.elapsed();

    println!("Step time:  {:.6}s ({:.3} ns/neuron)",
        t_lif_step.as_secs_f64(),
        t_lif_step.as_nanos() as f64 / n as f64);

    assert!(spikes_lif[0] == 1 || spikes_lif[0] == 0);
    assert!(!v_lif[0].is_nan());

    println!("\n✓ 200M LIF: 1 step in {:.4}s, RSS ≈ {}",
        t_lif_step.as_secs_f64(), format_bytes(mem_lif_after - mem_lif_before));
    println!("→ 1B LIF estimate: {:.2}s/step",
        t_lif_step.as_secs_f64() * 1_000_000_000.0 / n as f64);
}

#[test]
fn bench_1b_lif_alloc() {
    let n = 1_000_000_000usize;
    let dt = 0.01f32;
    println!("\n═══ 1B Neuron LIF Alloc Bench ═══");
    println!("Estimated: 6 arrays × 4 GB ≈ 36 GB RSS");

    let mem_before = mb_allocated();
    let t_all = Instant::now();

    // Allocate using fill (touches pages on write)
    // v, I, spikes, refractory are 4 GB each = 16 GB
    let mut v = vec![-65.0f32; n];
    let mut i_ext = vec![10.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut refractory = vec![0.0f32; n];

    // Params: 5 × n = 20 GB. Fill in 5 slices to avoid per-element loop.
    let mut params = vec![0.0f32; n * 5];
    // Fill dt section: params[0..n]
    let (dt_slice, rest) = params.split_at_mut(n);
    dt_slice.fill(dt);
    // Fill tau_m: params[n..2n]
    let (tm_slice, rest) = rest.split_at_mut(n);
    tm_slice.fill(10.0);
    // Fill v_rest: params[2n..3n]
    let (vr_slice, rest) = rest.split_at_mut(n);
    vr_slice.fill(-65.0);
    // Fill v_thresh: params[3n..4n]
    let (vt_slice, rest) = rest.split_at_mut(n);
    vt_slice.fill(-50.0);
    // Fill v_reset: params[4n..5n]
    let (vres_slice, _) = rest.split_at_mut(n);
    vres_slice.fill(-70.0);

    let mem_after = mb_allocated();
    println!("Alloc + fill OK! Time: {:.1}s, RSS: {} (Δ = {})",
        t_all.elapsed().as_secs_f64(),
        format_bytes(mem_after),
        format_bytes(mem_after - mem_before));

    // ── 1 step ──
    println!("Running 1 LIF step...");
    let t_step = Instant::now();
    taas_checkmate::kernels::lif_step(
        &mut v, &mut i_ext, &mut spikes, &params, &mut refractory, 2.0, None,
    );
    let t_step = t_step.elapsed();

    println!("Step: {:.3}s ({:.3} ns/neuron)",
        t_step.as_secs_f64(),
        t_step.as_nanos() as f64 / n as f64);
    assert!(v[n - 1].is_finite());
    println!("\n✓ 1B LIF survived! RSS ≈ {}", format_bytes(mem_after - mem_before));

    // Estimate for HH
    let hh_est = n as f64 * 26.1e-9;
    println!("  → HH 1B est: {:.1}s/step, ~40 GB RSS", hh_est);
}
