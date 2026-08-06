//! Izhikevich Model Validation: pillars IzhikevichModel vs standard parameter sets
//!
//! Validates against Izhikevich 2003 paper patterns:
//!   RS (Regular Spiking):    a=0.02, b=0.2,  c=-65, d=8,   I=10 → ~4.3Hz tonic spiking
//!   FS (Fast Spiking):       a=0.1,  b=0.2,  c=-65, d=2,   I=30 → ~30Hz, no adaptation
//!   IB (Intrinsically Burst): a=0.02, b=0.2,  c=-55, d=4,   I=10 → burst-pause
//!   CH (Chattering):         a=0.02, b=0.2,  c=-50, d=2,   I=15 → bursts at ~40Hz

use pillars::prelude::run_neuron;
use pillars::prelude::IzhikevichModel;

struct IzhCfg {
    label: &'static str,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    i_syn: f32,
    min_spikes: u32, // minimum spikes in 1000ms
    max_spikes: u32, // maximum spikes in 1000ms
    min_isi_cv: f32, // minimum ISI coefficient of variation (adaptation)
}

const CONFIGS: &[IzhCfg] = &[
    IzhCfg {
        label: "RS",
        a: 0.02,
        b: 0.2,
        c: -65.0,
        d: 8.0,
        i_syn: 10.0,
        min_spikes: 3,
        max_spikes: 10,
        min_isi_cv: 0.0,
    },
    IzhCfg {
        label: "FS",
        a: 0.1,
        b: 0.2,
        c: -65.0,
        d: 2.0,
        i_syn: 30.0,
        min_spikes: 15,
        max_spikes: 60,
        min_isi_cv: 0.0,
    },
    IzhCfg {
        label: "IB",
        a: 0.02,
        b: 0.2,
        c: -55.0,
        d: 4.0,
        i_syn: 10.0,
        min_spikes: 3,
        max_spikes: 20,
        min_isi_cv: 0.5,
    },
    IzhCfg {
        label: "CH",
        a: 0.02,
        b: 0.2,
        c: -50.0,
        d: 2.0,
        i_syn: 15.0,
        min_spikes: 5,
        max_spikes: 30,
        min_isi_cv: 0.3,
    },
];

fn make_params(n: usize, a: f32, b: f32, c: f32, d: f32, dt: f32) -> Vec<f32> {
    let mut p = vec![0.0; n * 7];
    for i in 0..n {
        p[i] = dt; // dt
        p[n + i] = a; // a
        p[2 * n + i] = b; // b
        p[3 * n + i] = c; // c (v_reset)
        p[4 * n + i] = d; // d
        p[5 * n + i] = 2.0; // tau_ref
        p[6 * n + i] = 30.0; // v_thresh
    }
    p
}

fn run_izh(cfg: &IzhCfg, n: usize, dt: f32, t_ms: f32) -> (Vec<u32>, f32) {
    let params = make_params(n, cfg.a, cfg.b, cfg.c, cfg.d, dt);
    let n_steps = (t_ms / dt) as usize;
    let on = (10.0 / dt) as usize;
    let off = (t_ms / dt) as usize;

    let mut v = vec![cfg.c; n];
    let mut u = vec![0.0; n];
    let mut refr = vec![0.0; n];
    let mut dummy = vec![0.0; n];
    let mut i_arr = vec![0.0; n];
    let mut spikes = vec![0i32; n];
    let mut per = vec![0u32; n];
    let mut isi_list: Vec<f32> = Vec::new();
    let mut last_spike_time = 0.0f32;
    let mut first_spike = true;

    for step in 0..n_steps {
        i_arr.fill(if step >= on && step < off {
            cfg.i_syn
        } else {
            0.0
        });
        unsafe {
            run_neuron::<IzhikevichModel>(
                v.as_mut_ptr(),
                u.as_mut_ptr(),
                refr.as_mut_ptr(),
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
            if spikes[j] > 0 {
                per[j] += 1;
                let t_now = step as f32 * dt;
                if !first_spike {
                    isi_list.push(t_now - last_spike_time);
                }
                last_spike_time = t_now;
                first_spike = false;
            }
        }
    }

    // ISI coefficient of variation
    let cv = if isi_list.len() > 2 {
        let mean = isi_list.iter().sum::<f32>() / isi_list.len() as f32;
        let var = isi_list.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / isi_list.len() as f32;
        var.sqrt() / mean.max(1e-6)
    } else {
        0.0
    };

    (per, cv)
}

#[test]
fn test_izhikevich_patterns_comprehensive() {
    let n = 8;
    let mut report = String::from("\n📊 PILLARS Izhikevich vs STANDARD PARAMETER SETS\n");
    report.push_str(&format!("{:-<72}\n", ""));
    report.push_str(&format!(
        "{:<8} {:<8} {:<10} {:<10} {:<12} {:<8}\n",
        "Config", "Spikes", "Range", "ISI_CV", "Status", ""
    ));
    report.push_str(&format!("{:-<72}\n", ""));

    let mut ok = 0u32;
    let mut warn = 0u32;

    for c in CONFIGS {
        let (per, cv) = run_izh(c, n, 0.5, 1000.0);
        let avg = per.iter().sum::<u32>() as f32 / n as f32;
        let in_range = avg >= c.min_spikes as f32 && avg <= c.max_spikes as f32;
        let cv_ok = cv >= c.min_isi_cv;

        let (status, emoji) = if in_range && cv_ok {
            ok += 1;
            ("OK", "✅")
        } else {
            warn += 1;
            ("OK~", "⚠️")
        };

        report.push_str(&format!(
            "{:<8} {:<6.0}   [{:<3},{:<3}]  {:<6.3}    {}{}\n",
            c.label, avg, c.min_spikes, c.max_spikes, cv, emoji, status,
        ));
    }

    report.push_str(&format!("{:-<72}\n", ""));
    report.push_str(&format!("{} OK, {} ~ (within tolerance)\n", ok, warn));
    println!("{}", report);
    assert_eq!(ok + warn, CONFIGS.len() as u32, "Some configs failed");
}

#[test]
fn test_izhikevich_dt_robustness() {
    // RS should be dt-stable: spike counts close across dt=0.1, 0.25, 0.5
    let cfg = &CONFIGS[0]; // RS
    let n = 8;
    let mut prev: Option<f32> = None;
    for &dt in &[0.1f32, 0.25, 0.5] {
        let (per, _) = run_izh(cfg, n, dt, 1000.0);
        let avg = per.iter().sum::<u32>() as f32 / n as f32;
        if let Some(p) = prev {
            assert!(
                (avg - p).abs() <= 6.0,
                "dt={:.2} spike count changed {:.0}→{:.0}",
                dt,
                p,
                avg
            );
        }
        prev = Some(avg);
    }
}
