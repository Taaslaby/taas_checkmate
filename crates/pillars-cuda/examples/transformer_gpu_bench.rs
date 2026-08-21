//! TransformerModel GPU benchmark — persistent mode.
//!
//! Measures throughput of TransformerModel (25 state vars, softmax, STDP)
//! on GPU in persistent mode (state stays in VRAM).
//!
//! Run: cargo run -p pillars-cuda --example transformer_gpu_bench --release

use std::time::Instant;
use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, TransformerModel, TRANSFORMER_CUDA_SOURCE};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4096);

    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1000);

    let dt: f32 = 0.5;
    let ns = TransformerModel::N_STATE; // 25
    let np = TransformerModel::N_PARAMS; // 12

    println!("═══ TransformerModel {n} neurons × {steps} steps ═══");
    println!("  dt = {dt}, states = {ns}, params = {np}\n");

    // ── Setup: CPU population ─────────────────────────────────────
    let mut pop = NeuralPopulation::<TransformerModel>::new(n);
    // Init all wk/v to 1.0 (state slots 9..24)
    for s in 9..ns {
        pop.fill_state(s, 1.0);
    }
    // Params: dt, decay, gain, inv_tau_acc, thresh, k_scale, eta_k, eta_v, w_base, w_span, v_base, v_span
    let params: [f32; 12] = [dt, 0.95, 1.0, 0.5, 0.9, 0.5, 0.01, 0.01, 1.0, 2.0, 1.0, 2.0];
    for (i, &v) in params.iter().enumerate() {
        pop.fill_param(i, v);
    }

    let i_in: Vec<f32> = vec![1.0; n];

    // ── Snapshot for GPU ──────────────────────────────────────────
    let mut g_states = vec![0.0f32; ns * n];
    for c in 0..ns {
        g_states[c * n..(c + 1) * n].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np)
        .flat_map(|c| pop.param_slice(c).to_vec())
        .collect();
    let init_states = g_states.clone();

    // ── CPU SIMD baseline ─────────────────────────────────────────
    let mut cpu_total = 0i64;
    let mut spikes = vec![0i32; n];
    let t0 = Instant::now();
    for _ in 0..steps {
        pop.step(&i_in, &mut spikes);
        cpu_total += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;

    println!("  [CPU SIMD]    {:.4} ms/step  ({:.1} M neuron-step/s)  spikes={cpu_total}",
        cpu_ms, n as f64 / cpu_ms / 1e3);

    // ── GPU: persistent (state in VRAM, only spikes cross PCIe) ──
    let model = CompiledModel::new(
        TRANSFORMER_CUDA_SOURCE,
        "transformer_kernel",
        ns,
        np,
        n,
        n,
    ).expect("CUDA compile transformer kernel");

    model.upload(&init_states, &g_params, &i_in).expect("upload");
    let mut gpu_total = 0i64;
    let mut g_spikes = vec![0i32; n];
    let t1 = Instant::now();
    for _ in 0..steps {
        model.step_device(dt).expect("persistent step");
        model.download_spikes(&mut g_spikes).expect("download spikes");
        gpu_total += g_spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3 / steps as f64;

    println!("  [GPU persist] {:.4} ms/step  ({:.1} M neuron-step/s)  spikes={gpu_total}",
        gpu_ms, n as f64 / gpu_ms / 1e3);

    // ── GPU: persistent, spikes every 100 steps ───────────────────
    let model2 = CompiledModel::new(
        TRANSFORMER_CUDA_SOURCE,
        "transformer_kernel",
        ns,
        np,
        n,
        n,
    ).expect("CUDA compile");
    model2.upload(&init_states, &g_params, &i_in).expect("upload");
    let mut gpu_total2 = 0i64;
    let mut g_spikes2 = vec![0i32; n];
    let t2 = Instant::now();
    for s in 0..steps {
        model2.step_device(dt).expect("step");
        if s % 100 == 99 {
            model2.download_spikes(&mut g_spikes2).expect("download");
            gpu_total2 += g_spikes2.iter().map(|&s| s as i64).sum::<i64>();
        }
    }
    let batched_ms = t2.elapsed().as_secs_f64() * 1e3 / steps as f64;

    println!("  [GPU batch]   {:.4} ms/step  ({:.1} M neuron-step/s)  spikes={gpu_total2}",
        batched_ms, n as f64 / batched_ms / 1e3);

    // ── Comparison ────────────────────────────────────────────────
    println!();
    println!("  Speedup vs CPU SIMD:");
    println!("    GPU persistent  : {:.2}×", cpu_ms / gpu_ms);
    println!("    GPU batched     : {:.2}×", cpu_ms / batched_ms);

    // Spike parity
    println!();
    let parity_msg = if cpu_total == gpu_total {
        "CPU == GPU".to_string()
    } else {
        format!("CPU={cpu_total} GPU={gpu_total}")
    };
    println!("  Spike parity: {parity_msg}");

    // State parity
    model.download_states(&mut g_states).expect("download states");
    let mut max_bits = 0u32;
    for c in 0..ns {
        for i in 0..n {
            max_bits = max_bits.max(
                (pop.state_slice(c)[i].to_bits() ^ g_states[c * n + i].to_bits()).count_ones(),
            );
        }
    }
    println!("  Max differing bits (CPU vs GPU): {max_bits} (0 = bitwise identical)");

    // Throughput summary
    let gpu_ns_per_sec = n as f64 / batched_ms / 1e3;
    println!("\n  ═══ THROUGHPUT ═══");
    println!("  GPU batched: {gpu_ns_per_sec:.1} M neuron-steps/s");
    println!(
        "  ~tokens/sec (1 step/token): {:.0}",
        n as f64 / batched_ms * 1e3
    );
}
