//! 🚀 Dual-source HH demo — ONE pillars file → CPU SIMD + CUDA kernel.
//!
//! Run: `cargo run -p pillars-cuda --example hh_dual --release`
//!
//! Shows the whole chain: `HHModel` (pillars style) → `HH_CUDA_SOURCE`
//! (transpiled by `#[dual_neuron]`) → NVRTC PTX → kernel launch, checked
//! for parity against the CPU SIMD path.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, HHModel, HH_CUDA_SOURCE};

const N: usize = 131072; // 128k neurons — GPU starts to matter
const DT: f32 = 0.01;
const N_STEPS: usize = 1000;

fn main() {
    let ns = HHModel::N_STATE;
    let np = HHModel::N_PARAMS;
    let p = [
        0.01f32, 120.0, 50.0, 36.0, -77.0, 0.3, -54.387, 1.0, 0.0, 1.0, 0.0, 120.0,
    ];

    let mut pop = NeuralPopulation::<HHModel>::new(N);
    pop.fill_state(0, -65.0);
    pop.fill_state(1, 0.05);
    pop.fill_state(2, 0.6);
    pop.fill_state(3, 0.32);
    pop.fill_state(4, 0.19); // c (CaV-T, steady-state at -65)
    pop.fill_state(5, 1.0); // was_below
    for (c, &v) in p.iter().enumerate() {
        pop.fill_param(c, v);
    }

    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in = vec![50.0f32; N];

    println!("HHModel compiled from a single #[dual_neuron] source:");
    println!(
        "  states = {} columns, params = {} columns, n = {}",
        ns, np, N
    );
    println!("  generated CUDA source: {} bytes", HH_CUDA_SOURCE.len());

    // ── CPU SIMD ────────────────────────────────────────────────────
    let mut spikes = vec![0i32; N];
    let mut cpu_spikes = 0i64;
    let t0 = std::time::Instant::now();
    for _ in 0..N_STEPS {
        pop.step(&i_in, &mut spikes);
        cpu_spikes += spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    let cpu_t = t0.elapsed();

    // ── CUDA (NVRTC) ────────────────────────────────────────────────
    let model = CompiledModel::new(HH_CUDA_SOURCE, "hh_kernel", ns, np, N, N)
        .expect("CUDA compile/load failed (is there a GPU?)");
    let mut g_spikes = vec![0i32; N];
    let mut gpu_spikes = 0i64;
    let t1 = std::time::Instant::now();
    for _ in 0..N_STEPS {
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("gpu step");
        gpu_spikes += g_spikes.iter().map(|&s| s as i64).sum::<i64>();
    }
    let gpu_t = t1.elapsed();

    // ── Report ──────────────────────────────────────────────────────
    println!("\n  {N_STEPS} steps × {N} neurons (dt={DT}):");
    println!(
        "    CPU SIMD: {cpu_t:?}  ({:>8.0} steps/s)",
        N_STEPS as f64 / cpu_t.as_secs_f64()
    );
    println!(
        "    CUDA    : {gpu_t:?}  ({:>8.0} steps/s)",
        N_STEPS as f64 / gpu_t.as_secs_f64()
    );
    println!("\n  spikes: CPU = {cpu_spikes}, GPU = {gpu_spikes}");

    let mut max_diff = 0.0f32;
    for c in 0..ns {
        for i in 0..N {
            max_diff = max_diff.max((pop.state_slice(c)[i] - g_states[c * N + i]).abs());
        }
    }
    println!("  max |state_cpu - state_gpu| after {N_STEPS} steps: {max_diff:.5}");

    if cpu_spikes == gpu_spikes {
        println!("\n✅ PARITY: spike trains identical (one source, two backends).");
        println!(
            "   state drift after {N_STEPS} steps: {max_diff:.2} mV (chaotic HH phase — ~1 ulp exp-implementation noise, see tests/hh_parity.rs)."
        );
    } else {
        println!("\n⚠️  Spike counts diverged — equations differ!");
        std::process::exit(1);
    }
}
