use std::time::Instant;

use connectome_gpt::{
    dataset::ConnectomeDataset,
    decoder::{CsrDecoder, DecoderConfig},
    graph_vae::{GraphVAE, VaeConfig},
    metrics::GraphMetrics,
    validate::Validator,
};

fn main() {
    macro_rules! bench {
        ($name:expr, $n:expr, $code:expr) => {
            let start = Instant::now();
            for _ in 0..$n { $code; }
            let elapsed = start.elapsed().as_secs_f64() / $n as f64;
            println!("  {:30} {:>8.1} µs  ({:>8.1} ms total)",
                $name, elapsed * 1_000_000.0, elapsed * 1_000.0 * $n as f64);
        };
    }

    println!("\n═══ connectome-gpt benchmarks ═══\n");

    // ─── 1. Small connectome (100 neurons) ───
    println!("--- Small (n=100) ---");
    let ds_100 = ConnectomeDataset::random(100, 0.05, 42);
    let sample_100 = &ds_100.samples[0];
    let metrics_100 = GraphMetrics::compute(sample_100);
    bench!("dataset random (100)", 100, {
        let _ = ConnectomeDataset::random(100, 0.05, 42);
    });
    bench!("graph metrics (100)", 1000, {
        let _ = GraphMetrics::compute(sample_100);
    });

    // ─── 2. Medium connectome (1K neurons) ───
    println!("\n--- Medium (n=1K) ---");
    let ds_1k = ConnectomeDataset::random(1000, 0.01, 42);
    let sample_1k = &ds_1k.samples[0];
    let metrics_1k = GraphMetrics::compute(sample_1k);
    bench!("dataset random (1K)", 10, {
        let _ = ConnectomeDataset::random(1000, 0.01, 42);
    });
    bench!("graph metrics (1K)", 100, {
        let _ = GraphMetrics::compute(sample_1k);
    });

    // ─── 3. VAE (forward only — training in Python) ───
    println!("\n--- VAE (inference only) ---");
    let ds_10k = ConnectomeDataset::random(10000, 0.005, 42);
    let features = ds_10k.samples[0].features.clone();
    let feat_d = features.len();
    let mut vae = GraphVAE::new(VaeConfig {
        input_dim: feat_d,
        hidden_dim: 64,
        latent_dim: 16,
        seed: 42,
    });
    bench!("VAE encode", 10000, {
        let _ = vae.encode(&features);
    });
    bench!("VAE forward deterministic", 10000, {
        let _ = vae.forward_deterministic(&features);
    });
    bench!("VAE generate (latent)", 10000, {
        let _ = vae.generate();
    });
    bench!("VAE save weights", 10000, {
        let _ = vae.save();
    });
    bench!("VAE interpolate", 10000, {
        let _ = vae.interpolate(&features, &features, 0.5);
    });

    // ─── 4. CSR decoder ───
    println!("\n--- CSR decoder ---");
    let mut decoder_100 = CsrDecoder::new(DecoderConfig {
        n_neurons: 100, density: 0.05, ..Default::default()
    });
    let mut decoder_1k = CsrDecoder::new(DecoderConfig {
        n_neurons: 1000, density: 0.01, ..Default::default()
    });
    let mut decoder_10k = CsrDecoder::new(DecoderConfig {
        n_neurons: 10000, density: 0.005, ..Default::default()
    });
    bench!("CSR gen (100)", 1000, {
        let _ = decoder_100.generate(&features);
    });
    bench!("CSR gen (1K)", 100, {
        let _ = decoder_1k.generate(&features);
    });
    bench!("CSR gen (10K)", 10, {
        let _ = decoder_10k.generate(&features);
    });
    bench!("CSR gen from VAE", 1000, {
        let _ = decoder_100.generate_from_vae(&mut vae);
    });

    // ─── 5. Validation simulation ───
    println!("\n--- Validation (Izhikevich sim) ---");
    let val = Validator::new(0.5, 50);
    let ds_val_100 = ConnectomeDataset::random(100, 0.05, 42);
    let ds_val_10 = ConnectomeDataset::random(10, 0.1, 42);
    bench!("validate (n=10)", 100, {
        let _ = val.validate(&ds_val_10.samples[0]);
    });
    bench!("validate (n=100)", 10, {
        let _ = val.validate(&ds_val_100.samples[0]);
    });

    // ─── 6. End-to-end: generate + validate ───
    println!("\n--- End-to-end ---");
    let mut decoder_e2e = CsrDecoder::new(DecoderConfig {
        n_neurons: 100, density: 0.05, ..Default::default()
    });
    let val_e2e = Validator::new(0.5, 50);
    bench!("gen + validate (100)", 10, {
        let sample = decoder_e2e.generate_from_vae(&mut vae);
        let _ = val_e2e.validate(&sample);
    });

    println!("\n═══ Done ═══");
}
