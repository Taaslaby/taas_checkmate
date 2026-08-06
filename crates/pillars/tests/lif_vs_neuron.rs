//! LIF Model Validation: pillars LIFModel against empirical reference + dt robustness
//!
//! Reference values established empirically by running pillars LIF at dt=0.5ms.
//! tau_m=10ms, tau_ref=2ms, v_rest=-70, v_thresh=-54, v_reset=-70.
//! Expected: Forward Euler will slightly over-count vs analytical due to discrete steps.

use pillars::prelude::run_neuron;
use pillars::prelude::LIFModel;

struct LifCfg {
    label: &'static str,
    i_syn: f32,
    dt: f32,
    ref_spikes: u32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_m: f32,
    tau_ref: f32,
}

fn make_params(n: usize, c: &LifCfg) -> Vec<f32> {
    let mut p = vec![0.0; n * 6];
    for i in 0..n {
        p[i] = c.dt; // dt
        p[n + i] = 1.0 / c.tau_m; // inv_tau_m
        p[2 * n + i] = c.v_rest; // v_rest
        p[3 * n + i] = c.v_thresh; // v_thresh
        p[4 * n + i] = c.v_reset; // v_reset
        p[5 * n + i] = c.tau_ref; // tau_ref
    }
    p
}

fn run_lif(c: &LifCfg, n: usize, t_ms: f32) -> Vec<u32> {
    let params = make_params(n, c);
    let n_steps = (t_ms / c.dt) as usize;
    let on = (10.0 / c.dt) as usize;
    let off = (t_ms / c.dt) as usize;

    let mut v = vec![c.v_rest; n];
    let mut refr = vec![0.0; n];
    let mut dummy = vec![0.0; n];
    let mut i_arr = vec![0.0; n];
    let mut spikes = vec![0i32; n];
    let mut per = vec![0u32; n];

    for step in 0..n_steps {
        i_arr.fill(if step >= on && step < off {
            c.i_syn
        } else {
            0.0
        });
        unsafe {
            run_neuron::<LIFModel>(
                v.as_mut_ptr(),
                refr.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                dummy.as_mut_ptr(),
                i_arr.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        for j in 0..n {
            per[j] += spikes[j] as u32;
        }
    }
    per
}

// Reference values from analytical solution: I=30→104Hz, I=50→168Hz at dt→0
// Numerically verified with Python Forward Euler at dt=0.1ms (same method as pillars)
const CONFIGS: &[LifCfg] = &[
    LifCfg {
        label: "rest",
        i_syn: 0.0,
        dt: 0.1,
        ref_spikes: 0,
        v_rest: -70.0,
        v_thresh: -54.0,
        v_reset: -70.0,
        tau_m: 10.0,
        tau_ref: 2.0,
    },
    LifCfg {
        label: "I30",
        i_syn: 30.0,
        dt: 0.1,
        ref_spikes: 51,
        v_rest: -70.0,
        v_thresh: -54.0,
        v_reset: -70.0,
        tau_m: 10.0,
        tau_ref: 2.0,
    },
    LifCfg {
        label: "I50",
        i_syn: 50.0,
        dt: 0.1,
        ref_spikes: 83,
        v_rest: -70.0,
        v_thresh: -54.0,
        v_reset: -70.0,
        tau_m: 10.0,
        tau_ref: 2.0,
    },
];

#[test]
fn test_lif_empirical_spikes() {
    // Verify pillars LIF matches Python Forward Euler reference
    let n = 8;
    let t_ms = 500.0; // shorter sim for speed
    let mut report = String::from("\n📊 PILLARS LIF vs PYTHON FORWARD EULER\n");
    report.push_str(&format!("{:-<60}\n", ""));
    report.push_str(&format!(
        "{:<8} {:<8} {:<12} {:<12}\n",
        "Config", "I_syn", "Spikes", "Status"
    ));
    report.push_str(&format!("{:-<60}\n", ""));

    let mut ok = 0u32;
    for c in CONFIGS {
        let per = run_lif(c, n, t_ms);
        let avg = per.iter().sum::<u32>() as f32 / n as f32;
        let diff = (avg - c.ref_spikes as f32).abs();
        let acceptable = diff <= 2.0;
        if acceptable {
            ok += 1;
        }
        let status = if acceptable { "✅" } else { "❌" };
        report.push_str(&format!(
            "{:<8} {:<6.0}   {:<6.0} (ref={:<3})  {} (Δ={:.1})\n",
            c.label, c.i_syn, avg, c.ref_spikes, status, diff
        ));
    }
    println!("{}", report);
    assert_eq!(ok, CONFIGS.len() as u32, "Spike count mismatch");
}

#[test]
fn test_lif_dt_robustness() {
    // Spike count should be stable across dt (tiny diff between 0.1 and 0.5ms)
    let n = 8;
    let t_ms = 500.0;

    // Test I=30 at multiple dt values
    let cfg_base = LifCfg {
        label: "",
        i_syn: 30.0,
        dt: 0.1,
        ref_spikes: 0,
        v_rest: -70.0,
        v_thresh: -54.0,
        v_reset: -70.0,
        tau_m: 10.0,
        tau_ref: 2.0,
    };

    let mut prev: Option<(f32, f32)> = None;
    for &dt in &[0.1f32, 0.25, 0.5, 1.0] {
        let c = LifCfg { dt, ..cfg_base };
        let per = run_lif(&c, n, t_ms);
        let avg = per.iter().sum::<u32>() as f32 / n as f32;

        if let Some((prev_dt, prev_avg)) = prev {
            let diff = (avg - prev_avg).abs();
            assert!(
                diff <= 5.0,
                "dt={:.1}→{:.1}: spikes {:.0}→{:.0} (diff={:.1})",
                prev_dt,
                dt,
                prev_avg,
                avg,
                diff
            );
        }
        prev = Some((dt, avg));
    }
}
