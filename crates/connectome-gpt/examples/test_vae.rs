use std::fs;
use std::time::Instant;
use connectome_gpt::decoder::DecoderConfig;

fn unnormalize(f: &mut [f32], norm: &serde_json::Value) {
    let m: Vec<f32> = norm["mean"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let s: Vec<f32> = norm["std"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    for (x, (mv, sv)) in f.iter_mut().zip(m.iter().zip(s.iter())) { *x = *x * sv + mv; }
}

fn ei(s: &connectome_gpt::ConnectomeSample) -> f32 {
    let n_i = s.nt_types.iter().filter(|t| **t < 0.0).count();
    let n_e = s.nt_types.iter().filter(|t| **t > 0.0).count();
    if n_i > 0 { n_e as f32 / n_i as f32 } else { n_e as f32 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let weights = fs::read("crates/connectome-gpt/models/vae_weights.bin")?;
    let norm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("crates/connectome-gpt/models/vae_norm.json")?)?;

    // New deep VAE: 10→256→256→32
    let mut vae = connectome_gpt::GraphVAE::new(connectome_gpt::VaeConfig {
        input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42,
    });
    vae.load(&weights);

    let mut raw = vae.generate();
    unnormalize(&mut raw, &norm);
    println!("VAE features (subgraph scale): n={:.0} syn={:.0} dens={:.2}% deg={:.1} ei={:.1}",
        raw[0], raw[1], raw[2], raw[3], raw[5]);

    // ── 10K brain using subgraph-trained VAE ──
    // The VAE was trained on 800-neuron subgraphs (dens ~3.7%).
    // We keep style features (density, E/I, degree CV) and scale n_neurons via config.
    let t1 = Instant::now();
    let cfg = DecoderConfig {
        n_neurons: 10_000,
        density: raw[2] / 100.0,       // subgraph density ~4%
        ei_ratio: raw[5].max(1.0),
        mean_weight: raw[6] / 100.0,   // from subgraph
        weight_noise: 0.05,
        n_communities: 8,
        community_strength: 10.0,
        sensory_fraction: 0.06,
        dn_fraction: 0.02,
        mn_fraction: 0.005,
        stdp_weights: true,
        compute_depth: true,
        refine_epochs: 10,
        seed: 42,
    };
    println!("\n── Config ──");
    println!("10K neurons, {:.2}% density, ei={:.1}, {} communities, {} refine ep",
        cfg.density * 100.0, cfg.ei_ratio, cfg.n_communities, cfg.refine_epochs);

    let mut decoder = connectome_gpt::CsrDecoder::new(cfg);
    let brain = decoder.generate(&raw);
    let dt1 = t1.elapsed();

    let val = connectome_gpt::Validator::new(0.5, 50);
    let r = val.validate(&brain);

    println!("\n── Results ──");
    println!("Neurons:  {}", brain.n_neurons);
    println!("Synapses: {} ({:.1}/neu)", brain.n_synapses,
        brain.n_synapses as f32 / brain.n_neurons as f32);
    println!("Density:  {:.4}%", brain.n_synapses as f64 / (brain.n_neurons as f64).powi(2) * 100.0);
    println!("E/I:      {:.2}", ei(&brain));
    println!("Groups:   {:?}", brain.groups.keys()
        .map(|k| (k.as_str(), brain.groups[k].len())).collect::<Vec<_>>());
    println!("Score:    {:.1}/10", r.score);
    println!("Firing:   {:.1} Hz", r.mean_firing_rate);
    println!("Stable:   {}", r.stable);
    println!("Time:     {:.3}s", dt1.as_secs_f64());

    // ── 50K quick bench ──
    let t2 = Instant::now();
    let cfg2 = DecoderConfig {
        n_neurons: 50_000, density: raw[2] / 100.0, ei_ratio: raw[5].max(1.0),
        mean_weight: raw[6] / 100.0, n_communities: 32, refine_epochs: 0,
        seed: 43, ..Default::default()
    };
    let mut dec2 = connectome_gpt::CsrDecoder::new(cfg2);
    let brain2 = dec2.generate(&raw);
    let dt2 = t2.elapsed();
    println!("\n50K bench: {}n {}syn ({:.1}/n)  {:.3}s  {:.0}K syn/s",
        brain2.n_neurons, brain2.n_synapses,
        brain2.n_synapses as f32 / brain2.n_neurons as f32,
        dt2.as_secs_f64(),
        brain2.n_synapses as f64 / dt2.as_secs_f64() / 1_000.0);

    println!("\nTotal: {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}
