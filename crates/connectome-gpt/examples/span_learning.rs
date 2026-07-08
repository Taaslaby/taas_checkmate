//! SPAN — Kuramoto BPTT frequency-learning.
//!
//! Каждая цифра → своя частота стимуляции сенсоров (в пределах ω∈[0.5,3.0)).
//! Рефайнер учит синапсы, чтобы MN запирались на частоту.
//! После обучения: MNs осциллируют на частоте цифры.

use std::fs;
use std::time::Instant;
use connectome_gpt::{CsrDecoder, DecoderConfig, GraphVAE, VaeConfig};
use connectome_gpt::refiner::{ConnectomeRefiner, RefinerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "═".repeat(60));
    println!("  SPAN: 10 frequencies → 10 motor patterns via Kuramoto BPTT");
    println!("{}", "═".repeat(60));

    // ── VAE → connectome ──
    let weights = fs::read("crates/connectome-gpt/models/vae_weights.bin")?;
    let norm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("crates/connectome-gpt/models/vae_norm.json")?)?;
    let mut vae = GraphVAE::new(VaeConfig { input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42 });
    vae.load(&weights);
    let mut raw = vae.generate();
    {
        let m: Vec<f32> = norm["mean"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let s: Vec<f32> = norm["std"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        for (x, (mv, sv)) in raw.iter_mut().zip(m.iter().zip(s.iter())) { *x = *x * sv + mv; }
    }

    let cfg = DecoderConfig {
        n_neurons: raw[0].round() as usize, density: raw[2] / 100.0,
        ei_ratio: raw[5].max(1.0), mean_weight: (raw[6] / 100.0).max(0.1),
        weight_noise: 0.05, n_communities: 8, community_strength: 8.0,
        sensory_fraction: 0.05, dn_fraction: 0.02, mn_fraction: 0.005,
        stdp_weights: true, compute_depth: true, refine_epochs: 0, seed: 42,
    };
    let sample = CsrDecoder::new(cfg.clone()).generate(&raw);

    let sensory_idx = sample.groups.get("sensory").cloned().unwrap_or_default();
    let motor_idx = sample.groups.get("mn").cloned().unwrap_or_else(|| (sample.n_neurons - 4..sample.n_neurons).collect());
    println!("  {}n {}syn | sensory={} DN={} MN={}",
        sample.n_neurons, sample.n_synapses, sensory_idx.len(),
        sample.groups.get("dn").map(|v| v.len()).unwrap_or(0),
        motor_idx.len());

    // 10 frequencies within natural range [0.5, 3.0)
    let freqs: Vec<f32> = (0..10).map(|d| 0.5 + d as f32 * 0.25).collect();
    println!("\nDigit frequencies:");
    for (d, f) in freqs.iter().enumerate() { println!("  {} → {:.2} Hz", d, f); }

    // ── Measure PLV ──
    fn measure_plv(
        sample: &connectome_gpt::ConnectomeSample,
        sensory_idx: &[usize], motor_idx: &[usize], freq: f32,
    ) -> f32 {
        let mut r = ConnectomeRefiner::new(RefinerConfig {
            n_steps: 100, dt: 0.05, lr: 0.001,
            sensory_idx: sensory_idx.to_vec(), motor_idx: motor_idx.to_vec(),
            stim_freq: freq, stim_amp: 3.0, plv_weight: 1.0,
            coherence_weight: 0.0, sparsity_weight: 0.0,
            temporal_weight: 0.0, homeostasis_weight: 0.0,
            max_grad_norm: 5.0, weight_clamp: 5.0,
            ..Default::default()
        });
        let t = r.forward(sample, None);
        let (_, _, plv_val, _, _, _) = r.loss(&t);
        plv_val
    }

    println!("\n─── Before training ───");
    let mut pre_plvs = Vec::new();
    for (d, f) in freqs.iter().enumerate() {
        let plv = measure_plv(&sample, &sensory_idx, &motor_idx, *f);
        pre_plvs.push(plv);
        println!("  Digit {} @ {:.2}Hz  PLV={:.4}", d, f, plv);
    }

    // ── Train ──
    println!("\n─── Kuramoto BPTT training ───");
    let t0 = Instant::now();
    let mut trained = sample.clone();
    let n_epochs = 20;

    for epoch in 0..n_epochs {
        let epoch_weights: Vec<f32> = trained.values.clone();
        for (d, &freq) in freqs.iter().enumerate() {
            let lr = 0.005 * (1.0 - epoch as f32 / n_epochs as f32 * 0.7);
            let mut refiner = ConnectomeRefiner::new(RefinerConfig {
                n_steps: 50, dt: 0.05,
                lr,
                sensory_idx: sensory_idx.clone(), motor_idx: motor_idx.clone(),
                stim_freq: freq, stim_amp: 3.0, plv_weight: 3.0,
                coherence_weight: 0.1, sparsity_weight: 0.005,
                temporal_weight: 0.02, homeostasis_weight: 0.01,
                target_mean_weight: 0.5,
                max_grad_norm: 3.0, weight_clamp: 3.0,
                n_bptt_steps: 15,
                seed: 42 + epoch as u64 * 10 + d as u64,
                ..Default::default()
            });
            let (result, _) = refiner.refine_step(&mut trained);
            if !result.loss.is_finite() {
                println!("  ⚠ e{}/d{} NaN — reverting", epoch, d);
                for (w, w0) in trained.values.iter_mut().zip(epoch_weights.iter()) {
                    *w = *w0;
                }
                break;
            }
        }
        if epoch % 5 == 0 || epoch == n_epochs - 1 {
            let digit_plv: Vec<f32> = freqs.iter().map(|f| measure_plv(&trained, &sensory_idx, &motor_idx, *f)).collect();
            let avg = digit_plv.iter().sum::<f32>() / 10.0;
            let chg = if epoch > 0 { avg - pre_plvs.iter().sum::<f32>() / 10.0 } else { 0.0 };
            println!("  e{:2}  PLVs: {:+.3}  avg={:.3} ({:.1}s)",
                epoch,
                digit_plv.iter().map(|p| format!("{:.3}", p)).collect::<Vec<_>>().join(" "),
                avg,
                t0.elapsed().as_secs_f64());
        }
    }

    // ── After training ──
    println!("\n─── After training ───");
    let mut post_plvs = Vec::new();
    for (d, f) in freqs.iter().enumerate() {
        let plv = measure_plv(&trained, &sensory_idx, &motor_idx, *f);
        post_plvs.push(plv);
        let delta = plv - pre_plvs[d];
        println!("  Digit {} @ {:.2}Hz  PLV={:.4}  Δ={:+.4}", d, f, plv, delta);
    }

    // ── Frequency detection ──
    println!("\n─── Digit identification from MN phase ───");
    let mut correct = 0;
    for (d, _) in freqs.iter().enumerate() {
        let plvs: Vec<f32> = freqs.iter().map(|f| measure_plv(&trained, &sensory_idx, &motor_idx, *f)).collect();
        let best = plvs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        let hit = best == d;
        if hit { correct += 1; }
        println!("  True {} → pred {} (PLV best={:.3}) {}",
            d, best, plvs[best], if hit { "✓" } else { "✗" });
    }

    let pre_avg: f32 = pre_plvs.iter().sum::<f32>() / 10.0;
    let post_avg: f32 = post_plvs.iter().sum::<f32>() / 10.0;
    println!("\n─── Summary ───");
    println!("  Avg PLV before: {:.4}", pre_avg);
    println!("  Avg PLV after:  {:.4}  (+{:.4})", post_avg, post_avg - pre_avg);
    println!("  Detection: {}/10 correct", correct);
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("{}", "═".repeat(60));
    Ok(())
}
