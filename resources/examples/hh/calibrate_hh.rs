/// HH calibration: grid search over I_EXT × WEIGHT_SCALE.
///
/// Includes deterministic heterogeneity in HH neuron parameters so the network can
/// produce asynchronous irregular firing instead of perfect synchrony.
///
/// Usage:
///   cargo run --features connectome --example calibrate_hh --release [taas_file] [I_EXT] [WEIGHT_SCALE]
///
/// With no extra args, runs grid search. With I_EXT and WEIGHT_SCALE, runs single combo.
#[cfg(feature = "connectome")]
mod inner {
    use std::env;
    use std::fs::File;
    use std::time::Instant;
    use taas_connectome::{sim_hh_synaptic, ConnectomeData, SynapticKinetics};
    use taas_core::neuron::hh::{standard_config, step_hh_lut};

    const N_STEPS: usize = 100; // 50ms @ 0.5ms
    const DT_MS: f32 = 0.5;
    const MIN_RATE_HZ: f32 = 0.5;

    fn add_heterogeneity_to_cfg(cfg: &mut taas_core::neuron::hh::HhLutConfig) {
        for i in 0..cfg.n {
            // Use index-based hash for reproducible variability
            let seed = (i as f32 * 0.577) % 1.0;
            let seed2 = ((i + 1) as f32 * 0.577) % 1.0;
            let seed3 = ((i + 3) as f32 * 0.577) % 1.0;

            // g_leak: 0.15-0.45 mS/cm² (nominal 0.3 ± 50%)
            cfg.g_leak[i] = 0.15 + seed * 0.30;

            // E_leak: -70 to -60 mV (nominal -65 ± 5 mV)
            cfg.E_leak[i] = -70.0 + seed2 * 10.0;

            // C_m: 0.7-1.3 µF/cm² (nominal 1.0 ± 30%)
            cfg.C_m[i] = 0.7 + seed3 * 0.6;

            // NOTE: Ii NOT modified here — it gets overwritten every step by sim_hh_synaptic!
            // Only parameters not touched by the sim loop (g_leak, E_leak, C_m) are varied.
        }
    }

    fn count_spikes(raster: &[i32], n: usize) -> Vec<u32> {
        let mut counts = vec![0u32; n];
        for chunk in raster.chunks(n) {
            for (i, &s) in chunk.iter().enumerate() {
                if s > 0 {
                    counts[i] += 1;
                }
            }
        }
        counts
    }

    fn firing_stats(counts: &[u32], n_steps: usize, dt_ms: f32) -> (f64, f64, f64, usize) {
        let n = counts.len();
        let total_time_s = n_steps as f64 * dt_ms as f64 / 1000.0;
        let rates: Vec<f32> = counts
            .iter()
            .map(|&c| c as f32 / total_time_s as f32)
            .collect();
        let mean_rate = rates.iter().sum::<f32>() as f64 / n as f64;
        let n_active = rates.iter().filter(|&&r| r >= MIN_RATE_HZ).count();
        let var = rates
            .iter()
            .map(|&r| (r as f64 - mean_rate).powi(2))
            .sum::<f64>()
            / n as f64;
        let cv = (var.sqrt() / mean_rate.max(1e-10));
        let max_rate = rates.iter().cloned().fold(0.0f32, f32::max) as f64;
        (mean_rate, max_rate, cv, n_active)
    }

    pub fn run() {
        let t0 = Instant::now();
        let args: Vec<String> = env::args().collect();
        let mut idx = 1;

        let taas_path = if args.len() > idx {
            args[idx].clone()
        } else {
            "princeton_dros.taas".to_string()
        };
        idx += 1;

        // Optional single-combo mode: I_EXT WEIGHT_SCALE
        let single_i_ext = if args.len() > idx {
            Some(args[idx].parse::<f32>().unwrap())
        } else {
            None
        };
        idx += 1;
        let single_scale = if args.len() > idx {
            Some(args[idx].parse::<f32>().unwrap())
        } else {
            None
        };

        // ─── Load connectome ────────────────────────────────
        println!("═══ HH Calibration ═══");
        println!();
        println!("Loading {} ...", taas_path);
        let mut f = File::open(&taas_path).expect(&format!("{} not found", taas_path));
        let mut conn = ConnectomeData::load_baked(&mut f).unwrap();
        drop(f);

        let n = conn.n_neurons;
        println!("   ✓ {} neurons, {} synapses", n, conn.n_synapses);
        if conn.receptor_types.is_empty() {
            conn.build_receptor_types();
        }

        // Report NT distribution
        let nachr = conn.receptor_types.iter().filter(|&&r| r == 4).count();
        let glucl = conn.receptor_types.iter().filter(|&&r| r == 5).count();
        let gaba_a = conn.receptor_types.iter().filter(|&&r| r == 2).count();
        println!("   ✓ nAChR={}  GluCl={}  GABA_A={}", nachr, glucl, gaba_a);

        // Single-combo mode
        if let (Some(i_ext), Some(scale)) = (single_i_ext, single_scale) {
            println!();
            println!("═══ Single Combo: I_EXT={i_ext}, WEIGHT_SCALE={scale:.0e} ═══");

            let mut conn_scaled = conn;
            for v in conn_scaled.values.iter_mut() {
                *v *= scale;
            }

            let mut cfg = standard_config(n, DT_MS);
            add_heterogeneity_to_cfg(&mut cfg);
            let mut kinetics = SynapticKinetics::new(n);
            let external_I = vec![i_ext; n];

            let t1 = Instant::now();
            let raster = sim_hh_synaptic(
                &conn_scaled,
                &mut cfg,
                &external_I,
                N_STEPS,
                &mut kinetics,
                1.0,
            );
            let sim_time = t1.elapsed().as_secs_f64();

            let counts = count_spikes(&raster, n);
            let (mean, max_rate, cv, n_active) = firing_stats(&counts, N_STEPS, DT_MS);
            let active_pct = n_active as f64 / n as f64 * 100.0;

            println!("   Mean rate: {mean:.2} Hz  Active: {active_pct:.1}%");
            println!("   Max rate: {max_rate:.1} Hz  CV: {cv:.3}");
            println!("   Time: {sim_time:.1}s");

            // Rate histogram (10 bins)
            let total_time_s = N_STEPS as f64 * DT_MS as f64 / 1000.0;
            let rates: Vec<f32> = counts
                .iter()
                .map(|&c| c as f32 / total_time_s as f32)
                .collect();
            let max_r = rates.iter().cloned().fold(0.0f32, f32::max).max(0.1);
            let bins = 10;
            let bin_w = max_r / bins as f32;
            let mut hist = vec![0usize; bins];
            for &r in &rates {
                let b = (r / bin_w).min((bins - 1) as f32) as usize;
                hist[b] += 1;
            }
            println!("   Rate histogram (max={max_r:.0} Hz, {bins} bins):");
            for (bi, &h) in hist.iter().enumerate() {
                let bar = "#".repeat(h / n.max(1) * 200 / bins.min(200));
                println!(
                    "   {:>5.0}-{:>5.0}: {:6} {}",
                    bi as f32 * bin_w,
                    (bi + 1) as f32 * bin_w,
                    h,
                    bar
                );
            }
            return;
        }

        // ─── Grid mode ──────────────────────────────────────
        let i_ext_vals: &[f32] = &[5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0, 12.0, 15.0];
        let scale_vals: &[f32] = &[1e6, 1e7, 1e8, 1e9];

        println!();
        println!(
            "Grid: {} I_EXT × {} WEIGHT_SCALE = {} combos (with heterogeneity)",
            i_ext_vals.len(),
            scale_vals.len(),
            i_ext_vals.len() * scale_vals.len()
        );
        println!("   g_leak: 0.15-0.45  E_leak: -70 to -60 mV  C_m: 0.7-1.3");
        println!();
        println!("┌──────────┬──────────────┬──────────┬──────────┬────────┬───────┐");
        println!("│ I_EXT     │ WEIGHT_SCALE │ Mean(Hz) │ Active%  │ Max Hz │ CV    │");
        println!("├──────────┼──────────────┼──────────┼──────────┼────────┼───────┤");

        let mut best_combo = (f64::MAX, 0.0f32, 0.0f32);
        let mut conn_scaled = conn.clone();

        for &i_ext in i_ext_vals {
            for &scale in scale_vals {
                let t1 = Instant::now();

                conn_scaled.values.copy_from_slice(&conn.values);
                for v in conn_scaled.values.iter_mut() {
                    *v *= scale;
                }

                let mut cfg = standard_config(n, DT_MS);
                add_heterogeneity_to_cfg(&mut cfg);
                let mut kinetics = SynapticKinetics::new(n);
                let external_I = vec![i_ext; n];
                let raster = sim_hh_synaptic(
                    &conn_scaled,
                    &mut cfg,
                    &external_I,
                    N_STEPS,
                    &mut kinetics,
                    1.0,
                );
                let counts = count_spikes(&raster, n);
                let (mean, max_rate, cv, n_active) = firing_stats(&counts, N_STEPS, DT_MS);
                let active_pct = n_active as f64 / n as f64 * 100.0;
                let elapsed = t1.elapsed().as_secs_f64();

                // Score: distance from ideal (1-5 Hz mean, 10-30% active, CV > 0.5)
                let mean_err = if mean >= 0.5 && mean <= 10.0 {
                    ((mean - 2.0).abs() / 2.0).min(5.0)
                } else {
                    5.0
                };
                let active_err = if active_pct >= 5.0 && active_pct <= 50.0 {
                    ((active_pct - 15.0).abs() / 15.0).min(5.0)
                } else {
                    5.0
                };
                let cv_score = if cv > 0.5 { 0.0 } else { 1.0 - cv / 0.5 };
                let score = mean_err + active_err + cv_score;

                if score < best_combo.0 && mean > 0.1 {
                    best_combo = (score, i_ext, scale);
                }

                let sd = format!("{:.0e}", scale);
                println!(
                    "│ {:>8.1} │ {:>12} │ {:>8.2} │ {:>8.2} │ {:>6.1} │ {:.3} │ {:.1}s│",
                    i_ext, sd, mean, active_pct, max_rate, cv, elapsed
                );
            }
        }

        println!("└──────────┴──────────────┴──────────┴──────────┴────────┴───────┘");

        // ─── Best combo detailed ────────────────────────────
        println!();
        println!("═══ Best Combo ═══");
        println!("   I_EXT = {:.1} µA/cm²", best_combo.1);
        println!("   WEIGHT_SCALE = {:.0e}", best_combo.2);
        let effective_mean = best_combo.2 * 1e-12;
        println!("   Effective mean weight = {:.4e} µA/cm²", effective_mean);

        // Run best combo and show rate histogram
        {
            let mut conn_scaled = conn.clone();
            for v in conn_scaled.values.iter_mut() {
                *v *= best_combo.2;
            }
            let mut cfg = standard_config(n, DT_MS);
            add_heterogeneity_to_cfg(&mut cfg);
            let mut kinetics = SynapticKinetics::new(n);
            let external_I = vec![best_combo.1; n];
            let raster = sim_hh_synaptic(
                &conn_scaled,
                &mut cfg,
                &external_I,
                N_STEPS,
                &mut kinetics,
                1.0,
            );
            let counts = count_spikes(&raster, n);
            let (mean, max_rate, cv, n_active) = firing_stats(&counts, N_STEPS, DT_MS);
            let active_pct = n_active as f64 / n as f64 * 100.0;

            // Rate histogram
            let total_time_s = N_STEPS as f64 * DT_MS as f64 / 1000.0;
            let rates: Vec<f32> = counts
                .iter()
                .map(|&c| c as f32 / total_time_s as f32)
                .collect();
            let max_r = rates.iter().cloned().fold(0.0f32, f32::max).max(0.1);
            let bins = 10;
            let bin_w = max_r / bins as f32;
            let mut hist = vec![0usize; bins];
            for &r in &rates {
                let b = (r / bin_w).min((bins - 1) as f32) as usize;
                hist[b] += 1;
            }
            println!();
            println!("   Best combo details:");
            println!("   Mean: {mean:.2} Hz  Active: {active_pct:.1}%  CV: {cv:.3}");
            println!(
                "   Rate histogram ({bins} bins, scale = 1 char ≈ {} neurons):",
                n / bins / 200
            );
            for (bi, &h) in hist.iter().enumerate() {
                let bar = "#".repeat(h.max(1) / (n / bins / 200).max(1));
                println!(
                    "   {:>5.0}-{:>5.0} Hz: {:6} {}",
                    bi as f32 * bin_w,
                    (bi + 1) as f32 * bin_w,
                    h,
                    bar
                );
            }
        }

        // ─── HH single-neuron verification ─────────────────
        println!();
        println!("═══ HH Single-Neuron Verification (I=15, dt=0.01) ═══");
        {
            let mut cfg = standard_config(1, 0.01);
            cfg.Ii[0] = 15.0;
            let mut sb = [0i32; 1];
            for _ in 0..2000 {
                step_hh_lut(&mut cfg, &mut sb, 6.3);

            }
            let mut v_max = cfg.v[0];
            let mut v_min = cfg.v[0];
            for _ in 0..3000 {
                step_hh_lut(&mut cfg, &mut sb, 6.3);

                let v = cfg.v[0];
                if v > v_max {
                    v_max = v;
                }
                if v < v_min {
                    v_min = v;
                }
            }
            println!("   v_max={v_max:.1} mV  v_min={v_min:.1} mV");
            let ok = if v_max > 0.0 { "✅" } else { "⚠️" };
            println!("   {ok} Spike peak > 0 mV");
            let ok = if v_min < -70.0 { "✅" } else { "⚠️" };
            println!("   {ok} AHP < -70 mV");
        }

        let total = t0.elapsed().as_secs_f64();
        println!();
        println!("Total: {total:.1}s");
    }
}

#[cfg(not(feature = "connectome"))]
mod inner {
    pub fn run() {
        println!("Requires --features connectome");
    }
}

fn main() {
    inner::run();
}
