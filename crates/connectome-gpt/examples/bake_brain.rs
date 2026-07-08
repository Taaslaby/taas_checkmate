//! Bake VAE-generated connectome into a .taas file.
//!
//! Usage: cargo run --release --example bake_brain [output.taas]
//! Default output: crates/connectome-gpt/models/brain.taas

use std::fs;
use std::path::PathBuf;
use taas_connectome::{
    BrainIO, InputChannel, InputEncoding,
    OutputChannel, OutputReadout,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = std::env::args().nth(1).unwrap_or_else(||
        "crates/connectome-gpt/models/brain.taas".to_string()
    );

    println!("Baking connectome -> {}", out_path);

    // 1. Load VAE
    let weights = fs::read("crates/connectome-gpt/models/vae_weights.bin")?;
    let norm: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("crates/connectome-gpt/models/vae_norm.json")?)?;
    let mut vae = connectome_gpt::GraphVAE::new(connectome_gpt::VaeConfig {
        input_dim: 10, hidden_dim: 256, latent_dim: 32, seed: 42,
    });
    vae.load(&weights);
    let mut raw = vae.generate();
    {
        let m: Vec<f32> = norm["mean"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let s: Vec<f32> = norm["std"][0].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        for (x, (mv, sv)) in raw.iter_mut().zip(m.iter().zip(s.iter())) { *x = *x * sv + mv; }
    }

    // 2. Decode to connectome
    let cfg = connectome_gpt::decoder::DecoderConfig {
        n_neurons: raw[0].round() as usize, density: raw[2] / 100.0,
        ei_ratio: raw[5].max(1.0), mean_weight: (raw[6] / 100.0).max(0.1),
        weight_noise: 0.05, n_communities: raw[8].round() as usize,
        community_strength: 8.0, sensory_fraction: 0.05, dn_fraction: 0.02,
        mn_fraction: 0.005, stdp_weights: true, compute_depth: true,
        refine_epochs: 0, seed: 42,
    };
    let sample = connectome_gpt::CsrDecoder::new(cfg).generate(&raw);

    let groups: Vec<(&str, Vec<usize>)> = sample.groups.iter()
        .map(|(k, v)| (k.as_str(), v.clone())).collect();
    let conn = taas_connectome::ConnectomeData::new(
        sample.row_ptr, sample.col_indices,
        sample.values, sample.nt_types, groups,
    );

    println!("  {} neurons, {} synapses", conn.n_neurons, conn.n_synapses);
    for g in &conn.groups {
        println!("  group '{}': {} neurons", g.name, g.indices.len());
    }

    // 3. Build BrainIO (сенсоры → sensory, моторы ← mn)
    let sensory = conn.group("sensory").unwrap_or(&[]);
    let mn = conn.group("mn").unwrap_or(&[]);
    let n_mn = mn.len();
    let io = BrainIO {
        bias: 5.0,
        input_channels: vec![
            InputChannel {
                name: "state".into(),
                neuron_indices: sensory.to_vec(),
                scale: 20.0,
                encoding: InputEncoding::Direct,
            },
        ],
        output_channels: vec![
            OutputChannel {
                name: "action".into(),
                neuron_indices: mn.to_vec(),
                n_outputs: 2,
                readout: OutputReadout::Linear,
                weights: Some(vec![0.0f32; 2 * n_mn]),
            },
        ],
    };
    println!("  IO: {} input ch, {} output ch", io.input_channels.len(), io.output_channels.len());

    // 4. Save as .taas
    let mut writer = taas_format::TaasWriter::new();
    writer.add_metadata("source", "VAE FlyWire");
    writer.add_metadata("seed", "42");
    taas_format::save_connectome(&conn, &mut writer);
    taas_format::save_brain_io(&io, &mut writer);

    // Ensure parent dir exists
    if let Some(parent) = PathBuf::from(&out_path).parent() {
        fs::create_dir_all(parent)?;
    }
    writer.save_to_file(&out_path)?;
    println!("Done. {} bytes written.", fs::metadata(&out_path)?.len());
    Ok(())
}
