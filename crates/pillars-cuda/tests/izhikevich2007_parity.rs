//! Parity test — the NeuroML `izhikevich2007Cell` zoo model transpiled in
//! `pillars` (`IZHIKEVICH_2007_CUDA_SOURCE`) vs the CPU SIMD path. Proves the
//! "parse neurons from NeuroML → SIMD *and* CUDA" architecture: one source of
//! truth, two backends, matching spike counts.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, Izhikevich2007Model, IZHIKEVICH_2007_CUDA_SOURCE};

const N: usize = 64; // multiple of 8 → capacity == n, stride == n
const DT: f32 = 1.0;
const N_STEPS: usize = 1000;
const I_INJ: f32 = 100.0;

fn fill_params() -> [f32; Izhikevich2007Model::N_PARAMS] {
    // Order matches #[param] fields: dt, cap, k, vr, vt, vpeak, a, b, c, d.
    let mut p = [0.0f32; Izhikevich2007Model::N_PARAMS];
    p[0] = DT;
    p[1] = 100.0; // C (pF)
    p[2] = 0.7; // k
    p[3] = -60.0; // vr
    p[4] = -40.0; // vt
    p[5] = 35.0; // vpeak
    p[6] = 0.03; // a
    p[7] = -2.0; // b
    p[8] = -50.0; // c (reset)
    p[9] = 100.0; // d
    p
}

#[test]
fn izhikevich2007_cpu_simd_vs_cuda_parity() {
    let params = fill_params();
    let ns = Izhikevich2007Model::N_STATE;
    let np = Izhikevich2007Model::N_PARAMS;

    // ── CPU side ───────────────────────────────────────────────────
    let mut pop = NeuralPopulation::<Izhikevich2007Model>::new(N);
    pop.fill_state(0, -60.0); // v
    pop.fill_state(1, 0.0); // u
    for (c, &v) in params.iter().enumerate() {
        pop.fill_param(c, v);
    }

    // ── GPU side: identical SoA snapshot ───────────────────────────
    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in = vec![I_INJ; N];

    let model = CompiledModel::new(
        IZHIKEVICH_2007_CUDA_SOURCE,
        "izhikevich2007_kernel",
        ns,
        np,
        N,
        N,
    )
    .expect("failed to compile the zoo-transpiled Izhikevich 2007 kernel (CUDA device required)");
    let mut g_spikes = vec![0i32; N];
    let mut spikes = vec![0i32; N];

    let mut cpu_counts = vec![0i64; N];
    let mut gpu_counts = vec![0i64; N];
    let mut max_bits = 0u32;
    for step in 0..N_STEPS {
        pop.step(&i_in, &mut spikes);
        for i in 0..N {
            cpu_counts[i] += spikes[i] as i64;
        }
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("gpu step");
        for i in 0..N {
            gpu_counts[i] += g_spikes[i] as i64;
        }
        for c in 0..ns {
            for i in 0..N {
                let d = (pop.state_slice(c)[i] - g_states[c * N + i]).abs();
                assert!(
                    d < 2e-3,
                    "parity broken at step {step} state[{c}][{i}]: CPU={} GPU={}",
                    pop.state_slice(c)[i],
                    g_states[c * N + i]
                );
                max_bits = max_bits.max(
                    (pop.state_slice(c)[i].to_bits() ^ g_states[c * N + i].to_bits()).count_ones(),
                );
            }
        }
    }

    for i in 0..N {
        assert_eq!(
            cpu_counts[i], gpu_counts[i],
            "neuron {i} spike count diverged: CPU={} GPU={}",
            cpu_counts[i], gpu_counts[i]
        );
    }
    let cpu_spikes: i64 = cpu_counts.iter().sum();
    assert!(
        cpu_spikes > 0,
        "Izhikevich 2007 must spike at I=100, got {cpu_spikes}"
    );
    eprintln!(
        "[parity] izhikevich2007: spikes={cpu_spikes} max differing bits per state float: {max_bits}"
    );
}

/// The zoo-transpiled source must be self-contained (kernel + step).
#[test]
fn izhikevich2007_cuda_source_is_self_contained() {
    assert!(IZHIKEVICH_2007_CUDA_SOURCE.contains("__global__ void izhikevich2007_kernel"));
    assert!(IZHIKEVICH_2007_CUDA_SOURCE.contains("izhikevich2007_step"));
}
