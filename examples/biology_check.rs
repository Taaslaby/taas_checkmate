/// Biological plausibility diagnostics for connectome simulation.
///
/// Compares Hodgkin-Huxley (HH) vs LIF on real connectome data with receptor_types.
/// HH uses calibrated parameters: I_EXT=5.0 µA/cm², WEIGHT_SCALE=1e8, with
/// deterministic heterogeneity in g_leak, E_leak, C_m for diverse firing.
///
/// Usage:
///   cargo run --features connectome --example biology_check --release [taas_file]
///
/// If no taas_file is given, defaults to "princeton_dros.taas".
#[cfg(feature = "connectome")]
mod inner {
    use std::env;
    use std::fs::File;
    use std::time::Instant;
    use taas_connectome::{
        ConnectomeData, BrainState, NeuronModel, LifParams,
        sim_lif, sim_lif_synaptic, sim_hh, sim_hh_synaptic, SynapticKinetics,
    };
    use taas_core::neuron::hh::{standard_config, step_hh_lut};

    const N_SAMPLE: usize = 3;            // repetitions for timing
    const DT_MS: f32 = 0.5;               // 0.5ms step
    const N_STEPS: usize = 200;           // 100ms simulated @ 0.5ms
    const MIN_RATE_HZ: f32 = 0.5;
    const BURST_ISI_MS: f32 = 4.0;

    // ─── LIF parameters ────────────────────────────────────
    const LIF_TAU_M: f32 = 10.0;          // ms
    const LIF_V_REST: f32 = -65.0;        // mV
    const LIF_V_THRESH: f32 = -50.0;      // mV
    const LIF_V_RESET: f32 = -65.0;       // mV
    const LIF_TAU_REF: f32 = 2.0;         // ms
    const LIF_I_EXT: f32 = 16.0;          // nA — just above threshold (steady state = -49 mV)
    const LIF_WEIGHT_SCALE: f32 = 1e5;    // moderate — enough for recurrent modulation

    // ─── HH parameters (calibrated) ────────────────────────
    const HH_I_EXT: f32 = 5.0;            // µA/cm² — near poly HH threshold
    const HH_WEIGHT_SCALE: f32 = 1e8;     // physiological connectome strength for HH

    // ─── HH parameter heterogeneity ───────────────────────
    // Deterministic per-neuron variation (same neuron = same params every run).
    fn add_hh_heterogeneity(cfg: &mut taas_core::neuron::hh::HhLutConfig) {
        for i in 0..cfg.n {
            let seed = (i as f32 * 0.577) % 1.0;
            let seed2 = ((i + 1) as f32 * 0.577) % 1.0;
            let seed3 = ((i + 3) as f32 * 0.577) % 1.0;
            cfg.g_leak[i] = 0.15 + seed * 0.30;      // 0.15-0.45 mS/cm²
            cfg.E_leak[i] = -70.0 + seed2 * 10.0;     // -70 to -60 mV
            cfg.C_m[i] = 0.7 + seed3 * 0.6;            // 0.7-1.3 µF/cm²
        }
    }

    fn count_spikes(raster: &[i32], n: usize) -> Vec<u32> {
        let mut counts = vec![0u32; n];
        for chunk in raster.chunks(n) {
            for (i, &s) in chunk.iter().enumerate() {
                if s > 0 { counts[i] += 1; }
            }
        }
        counts
    }

    fn firing_stats(counts: &[u32], n_steps: usize, dt_ms: f32) -> (f64, f64, f64, f64, f64, f64, usize) {
        let n = counts.len();
        let total_time_s = n_steps as f64 * dt_ms as f64 / 1000.0;
        let rates: Vec<f32> = counts.iter().map(|&c| c as f32 / total_time_s as f32).collect();
        let mean_rate = rates.iter().sum::<f32>() as f64 / n as f64;
        let mut sorted = rates.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median_rate = sorted[n / 2] as f64;
        let n_active = rates.iter().filter(|&&r| r >= MIN_RATE_HZ).count();
        let active_rate = rates.iter().filter(|&&r| r >= MIN_RATE_HZ).sum::<f32>() as f64 / n_active.max(1) as f64;
        let var = rates.iter().map(|&r| (r as f64 - mean_rate).powi(2)).sum::<f64>() / n as f64;
        let cv = (var.sqrt() / mean_rate.max(1e-10));
        let max_rate = sorted.last().copied().unwrap_or(0.0) as f64;
        (mean_rate, median_rate, active_rate, max_rate, cv, n_active as f64 / n as f64 * 100.0, n_active)
    }

    fn burst_index(raster: &[i32], n: usize, dt_ms: f32) -> f64 {
        let max_gap = (BURST_ISI_MS / dt_ms).ceil() as usize;
        let n_steps = raster.len() / n;
        let mut burst_spikes = 0u64;
        let mut total_spikes = 0u64;
        for i in 0..n {
            let mut last_spike = n_steps + 1;
            for step in 0..n_steps {
                if raster[step * n + i] > 0 {
                    total_spikes += 1;
                    if step < last_spike + max_gap { burst_spikes += 1; }
                    last_spike = step;
                }
            }
        }
        if total_spikes == 0 { 0.0 } else { burst_spikes as f64 / total_spikes as f64 }
    }

    fn synchrony_index(raster: &[i32], n: usize) -> f64 {
        let n_steps = raster.len() / n;
        let pop_rates: Vec<f64> = (0..n_steps)
            .map(|step| {
                let base = step * n;
                raster[base..base + n].iter().filter(|&&s| s > 0).count() as f64
            })
            .collect();
        let mean_pop = pop_rates.iter().sum::<f64>() / n_steps as f64;
        if mean_pop < 0.001 * n as f64 { return 0.0; }
        let var = pop_rates.iter().map(|&r| (r - mean_pop).powi(2)).sum::<f64>() / n_steps as f64;
        (var - mean_pop) / mean_pop.max(1.0)
    }

    fn compute_lfp(raster: &[i32], n: usize) -> Vec<f64> {
        let n_steps = raster.len() / n;
        let alpha = 0.1;
        let mut lfp = Vec::with_capacity(n_steps);
        let mut smoothed = 0.0f64;
        for step in 0..n_steps {
            let base = step * n;
            let pop = raster[base..base + n].iter().filter(|&&s| s > 0).count() as f64 / n as f64;
            smoothed = alpha * pop + (1.0 - alpha) * smoothed;
            lfp.push(smoothed);
        }
        lfp
    }

    pub fn run() {
        let t0 = Instant::now();

        let args: Vec<String> = env::args().collect();
        let taas_path = if args.len() > 1 { args[1].clone() } else { "princeton_dros.taas".to_string() };

        println!();
        println!("╔═══ Biological Diagnostics ═══╗");
        println!();

        // ─── Load connectome ────────────────────────────────
        println!("1. Loading {} ...", taas_path);
        let mut f = File::open(&taas_path).expect(&format!("{} not found", taas_path));
        let mut conn = ConnectomeData::load_baked(&mut f).unwrap();
        drop(f);

        let n = conn.n_neurons;
        let n_syn = conn.n_synapses;
        let e_count = conn.nt_types.iter().filter(|&&t| t > 0.0).count();
        let i_count = conn.nt_types.iter().filter(|&&t| t < 0.0).count();

        // Build receptor_types if not present
        if conn.receptor_types.is_empty() {
            conn.build_receptor_types();
        }

        let nachr = conn.receptor_types.iter().filter(|&&r| r == 4).count();
        let glucl = conn.receptor_types.iter().filter(|&&r| r == 5).count();
        let machr = conn.receptor_types.iter().filter(|&&r| r == 6).count();
        let ampa = conn.receptor_types.iter().filter(|&&r| r == 0).count();
        let nmda = conn.receptor_types.iter().filter(|&&r| r == 1).count();
        let gaba_a = conn.receptor_types.iter().filter(|&&r| r == 2).count();
        let gaba_b = conn.receptor_types.iter().filter(|&&r| r == 3).count();
        let total_r = conn.receptor_types.len() as f64;

        println!("   ✓ {n} neurons, {n_syn} synapses");
        println!("   ✓ E/I: {e_count}/{i_count}  ({:.1}% I)", i_count as f64 / n as f64 * 100.0);
        println!("   ✓ nAChR: {:.1}%  GluCl: {:.1}%  AMPA: {:.1}%",
                 nachr as f64 / total_r * 100.0,
                 glucl as f64 / total_r * 100.0,
                 ampa as f64 / total_r * 100.0);
        println!("   ✓ GABA_A: {:.1}%  mAChR: {:.1}%  NMDA: {:.1}%  GABA_B: {:.1}%",
                 gaba_a as f64 / total_r * 100.0,
                 machr as f64 / total_r * 100.0,
                 nmda as f64 / total_r * 100.0,
                 gaba_b as f64 / total_r * 100.0);
        if glucl > 0 { println!("   ⭐ GluCl inhibitory: GLUT→Cl⁻ channel in insects!"); }

        println!();
        println!("2. Models:");
        println!("   LIF: I_EXT={LIF_I_EXT} nA, WEIGHT_SCALE={LIF_WEIGHT_SCALE:.0e}, tau_m={LIF_TAU_M}ms");
        println!("   HH:  I_EXT={HH_I_EXT} µA/cm², WEIGHT_SCALE={HH_WEIGHT_SCALE:.0e}");
        println!("       g_leak: 0.15-0.45 mS/cm², E_leak: -70 to -60 mV, C_m: 0.7-1.3 µF/cm²");
        println!("   Simulation: {:.1}s @ {DT_MS}ms dt ({N_STEPS} steps)",
                 N_STEPS as f64 * DT_MS as f64 / 1000.0);

        // ═════════════════════════════════════════════════════
        //  LIF simulations
        // ═════════════════════════════════════════════════════
        let mut conn_lif = conn.clone();
        println!("   Scaling LIF weights by {:.0e}...", LIF_WEIGHT_SCALE);
        for v in conn_lif.values.iter_mut() { *v *= LIF_WEIGHT_SCALE; }

        let lif_params = LifParams::new(
            &vec![LIF_TAU_M; n], &vec![LIF_V_REST; n], &vec![LIF_V_THRESH; n], &vec![LIF_V_RESET; n],
            DT_MS, LIF_TAU_REF,
        );
        let external_lif = vec![LIF_I_EXT; n];

        // ─── Instant-current LIF ────────────────────────────
        println!();
        println!("═══ LIF Instant-Current ═══");
        let mut t_lif = 0.0;
        let mut raster_lif = Vec::new();
        for _ in 0..N_SAMPLE {
            let mut state = BrainState::new(n, NeuronModel::LIF);
            let t1 = Instant::now();
            let r = sim_lif(&conn_lif, &mut state, &lif_params, &external_lif, N_STEPS);
            t_lif += t1.elapsed().as_secs_f64();
            if raster_lif.is_empty() { raster_lif = r; }
        }
        t_lif /= N_SAMPLE as f64;

        // ─── Conductance-based LIF ──────────────────────────
        println!();
        println!("═══ LIF Conductance-Based ═══");
        let mut t_syn = 0.0;
        let mut raster_syn = Vec::new();
        for _ in 0..N_SAMPLE {
            let mut state = BrainState::new(n, NeuronModel::LIF);
            let mut kinetics = SynapticKinetics::new(n);
            let t1 = Instant::now();
            let r = sim_lif_synaptic(&conn_lif, &mut state, &lif_params, &external_lif, N_STEPS, &mut kinetics);
            t_syn += t1.elapsed().as_secs_f64();
            if raster_syn.is_empty() { raster_syn = r; }
        }
        t_syn /= N_SAMPLE as f64;

        // ═════════════════════════════════════════════════════
        //  HH simulations
        // ═════════════════════════════════════════════════════
        let mut conn_hh = conn;
        println!();
        println!("   Scaling HH weights by {:.0e}...", HH_WEIGHT_SCALE);
        for v in conn_hh.values.iter_mut() { *v *= HH_WEIGHT_SCALE; }

        let external_hh = vec![HH_I_EXT; n];

        // ─── Instant-current HH ─────────────────────────────
        println!();
        println!("═══ HH Instant-Current ═══");
        let mut t_hh = 0.0;
        let mut raster_hh = Vec::new();
        for _ in 0..N_SAMPLE {
            let mut cfg = standard_config(n, DT_MS);
            add_hh_heterogeneity(&mut cfg);
            let t1 = Instant::now();
            let r = sim_hh(&conn_hh, &mut cfg, &external_hh, N_STEPS);
            t_hh += t1.elapsed().as_secs_f64();
            if raster_hh.is_empty() { raster_hh = r; }
        }
        t_hh /= N_SAMPLE as f64;

        // ─── Conductance-based HH ───────────────────────────
        println!();
        println!("═══ HH Conductance-Based ═══");
        let mut t_hh_syn = 0.0;
        let mut raster_hh_syn = Vec::new();
        for _ in 0..N_SAMPLE {
            let mut cfg = standard_config(n, DT_MS);
            add_hh_heterogeneity(&mut cfg);
            let mut kinetics = SynapticKinetics::new(n);
            let t1 = Instant::now();
            let r = sim_hh_synaptic(&conn_hh, &mut cfg, &external_hh, N_STEPS, &mut kinetics);
            t_hh_syn += t1.elapsed().as_secs_f64();
            if raster_hh_syn.is_empty() { raster_hh_syn = r; }
        }
        t_hh_syn /= N_SAMPLE as f64;

        let elapsed = t0.elapsed().as_secs_f64();

        // ═════════════════════════════════════════════════════
        //  ANALYSE RESULTS
        // ═════════════════════════════════════════════════════
        println!();
        println!("═══ RESULTS ══════════════════════════════════");

        let counts_lif = count_spikes(&raster_lif, n);
        let counts_syn = count_spikes(&raster_syn, n);
        let counts_hh = count_spikes(&raster_hh, n);
        let counts_hh_syn = count_spikes(&raster_hh_syn, n);

        let (m1, med1, act1, max1, cv1, pct1, nact1) = firing_stats(&counts_lif, N_STEPS, DT_MS);
        let (m2, med2, act2, max2, cv2, pct2, nact2) = firing_stats(&counts_syn, N_STEPS, DT_MS);
        let (m3, med3, act3, max3, cv3, pct3, nact3) = firing_stats(&counts_hh, N_STEPS, DT_MS);
        let (m4, med4, act4, max4, cv4, pct4, nact4) = firing_stats(&counts_hh_syn, N_STEPS, DT_MS);

        println!();
        println!("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┐");
        println!("│ Metric              │ LIF inst │ LIF cond │ HH inst  │ HH cond  │ Biological │");
        println!("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤");
        println!("│ Mean rate (Hz)      │ {:.2}   │ {:.2}   │ {:.2}   │ {:.2}   │ 0.5–5     │", m1, m2, m3, m4);
        println!("│ Median rate (Hz)    │ {:.2}   │ {:.2}   │ {:.2}   │ {:.2}   │ <1        │", med1, med2, med3, med4);
        println!("│ Active rate (Hz)    │ {:.2}   │ {:.2}   │ {:.2}   │ {:.2}   │ 2–10      │", act1, act2, act3, act4);
        println!("│ Max rate (Hz)       │ {:.1}   │ {:.1}   │ {:.1}   │ {:.1}   │ <200      │", max1, max2, max3, max4);
        println!("│ CV of rates         │ {:.2}    │ {:.2}    │ {:.2}    │ {:.2}    │ >0.5      │", cv1, cv2, cv3, cv4);
        println!("│ Active neurons (%)  │ {:.1}%   │ {:.1}%   │ {:.1}%   │ {:.1}%   │ 10–30%    │", pct1, pct2, pct3, pct4);
        println!("│ Active neurons (N)  │ {}    │ {}    │ {}  │ {}  │ —         │", nact1, nact2, nact3, nact4);
        println!("└─────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘");

        let burst_lif = burst_index(&raster_lif, n, DT_MS);
        let burst_syn = burst_index(&raster_syn, n, DT_MS);
        let burst_hh = burst_index(&raster_hh, n, DT_MS);
        let burst_hh_syn = burst_index(&raster_hh_syn, n, DT_MS);
        let sync_lif = synchrony_index(&raster_lif, n);
        let sync_syn = synchrony_index(&raster_syn, n);
        let sync_hh = synchrony_index(&raster_hh, n);
        let sync_hh_syn = synchrony_index(&raster_hh_syn, n);

        println!();
        println!("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┐");
        println!("│ Burst index         │ {:.4}  │ {:.4}  │ {:.4}  │ {:.4}  │ <0.3      │", burst_lif, burst_syn, burst_hh, burst_hh_syn);
        println!("│ Synchrony index     │ {:.4}  │ {:.4}  │ {:.4}  │ {:.4}  │ <0.1      │", sync_lif, sync_syn, sync_hh, sync_hh_syn);
        println!("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤");
        println!("│ Time/step (ms)      │ {:.3}   │ {:.3}   │ {:.3}   │ {:.3}   │ —         │",
                 t_lif / N_STEPS as f64 * 1000.0,
                 t_syn / N_STEPS as f64 * 1000.0,
                 t_hh / N_STEPS as f64 * 1000.0,
                 t_hh_syn / N_STEPS as f64 * 1000.0);
        println!("│ Throughput (M n·s/s)│ {:.1}    │ {:.1}    │ {:.1}    │ {:.1}    │ —         │",
                 n as f64 * N_STEPS as f64 / t_lif / 1_000_000.0,
                 n as f64 * N_STEPS as f64 / t_syn / 1_000_000.0,
                 n as f64 * N_STEPS as f64 / t_hh / 1_000_000.0,
                 n as f64 * N_STEPS as f64 / t_hh_syn / 1_000_000.0);
        println!("└─────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘");

        let lfp_lif = compute_lfp(&raster_lif, n);
        let lfp_syn = compute_lfp(&raster_syn, n);
        let lfp_hh = compute_lfp(&raster_hh, n);
        let lfp_hh_syn = compute_lfp(&raster_hh_syn, n);
        let lfp_mean_lif = lfp_lif.iter().sum::<f64>() / lfp_lif.len() as f64;
        let lfp_mean_syn = lfp_syn.iter().sum::<f64>() / lfp_syn.len() as f64;
        let lfp_mean_hh = lfp_hh.iter().sum::<f64>() / lfp_hh.len() as f64;
        let lfp_mean_hh_syn = lfp_hh_syn.iter().sum::<f64>() / lfp_hh_syn.len() as f64;

        println!();
        println!("┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┐");
        println!("│ LFP mean (frac.)    │ {:.6} │ {:.6} │ {:.6} │ {:.6} │ 0.001–0.1 │", lfp_mean_lif, lfp_mean_syn, lfp_mean_hh, lfp_mean_hh_syn);
        println!("└─────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘");

        // ─── HH validation ──────────────────────────────────
        println!();
        println!("═══ HH Single-Neuron Validation (I=15, dt=0.01) ═══");
        {
            let mut cfg = standard_config(1, 0.01);
            cfg.Ii[0] = 15.0;
            let mut sb = [0i32; 1];
            for _ in 0..2000 { step_hh_lut(&mut cfg, &mut sb); }
            let mut v_max = cfg.v[0];
            let mut v_min = cfg.v[0];
            for _ in 0..3000 {
                step_hh_lut(&mut cfg, &mut sb);
                let v = cfg.v[0];
                if v > v_max { v_max = v; }
                if v < v_min { v_min = v; }
            }
            println!("   v_max={v_max:.1} mV  v_min={v_min:.1} mV");
            println!("   {} Spike peak > 0 mV — NaV works", if v_max > 0.0 { "✅" } else { "⚠️" });
            println!("   {} AHP < -70 mV — KV works", if v_min < -70.0 { "✅" } else { "⚠️" });
        }

        // ─── Summary ────────────────────────────────────────
        println!();
        println!("╔═══ SUMMARY ═══════════════════════════════════╗");

        for (name, pct, cv_val, b, sync_val, max_r, mean_r) in [
            ("LIF inst", pct1, cv1, burst_lif, sync_lif, max1, m1),
            ("LIF cond", pct2, cv2, burst_syn, sync_syn, max2, m2),
            ("HH  inst", pct3, cv3, burst_hh, sync_hh, max3, m3),
            ("HH  cond", pct4, cv4, burst_hh_syn, sync_hh_syn, max4, m4),
        ] {
            let ok = |ok: bool, s: &str| if ok { format!("  ✅ {s}") } else { format!("  ⚠️  {s}") };
            println!();
            println!("  ── {name} ──");
            println!("{}", ok(pct >= 5.0 && pct <= 50.0, &format!("Active {pct:.1}% (5-50%)")));
            println!("{}", ok(cv_val > 0.5, &format!("CV {cv_val:.2} (>0.5)")));
            println!("{}", ok(b < 0.3, &format!("Burst {b:.4} (<0.3)")));
            println!("{}", ok(sync_val < 0.1, &format!("Sync {sync_val:.4} (<0.1)")));
            println!("{}", ok(max_r < 200.0, &format!("Max {max_r:.0} Hz (<200)")));
            println!("{}", ok(mean_r >= 0.1 && mean_r <= 10.0, &format!("Mean {mean_r:.2} Hz (0.1-10)")));
        }

        println!();
        println!("╚═══════════════════════════════════════════════╝");
        println!();
        println!("Total time: {:.1}s", elapsed);
    }
}

#[cfg(not(feature = "connectome"))]
mod inner {
    pub fn run() {
        println!("This example requires the `connectome` feature.");
        println!("  cargo run --features connectome --example biology_check --release");
    }
}

fn main() {
    inner::run();
}
