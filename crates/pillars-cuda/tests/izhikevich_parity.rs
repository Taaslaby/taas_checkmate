//! Parity test — zoo model transpiled IN pillars (`IZHIKEVICH_CUDA_SOURCE`)
//! vs the CPU SIMD path. Proves the "parse neurons from pillars → GPU"
//! architecture: pillars emits the CUDA string, pillars-cuda just compiles it.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, IzhikevichModel, IZHIKEVICH_CUDA_SOURCE};

const N: usize = 64; // multiple of 8 → capacity == n, stride == n
const DT: f32 = 0.5;
const N_STEPS: usize = 500;
const I_INJ: f32 = 10.0;

fn fill_params() -> [f32; IzhikevichModel::N_PARAMS] {
    // Order matches #[param] fields: dt, a, b, c, d, tau_ref, v_thresh.
    let mut p = [0.0f32; IzhikevichModel::N_PARAMS];
    p[0] = DT;
    p[1] = 0.02;
    p[2] = 0.2;
    p[3] = -65.0;
    p[4] = 8.0;
    p[5] = 2.0;
    p[6] = 30.0;
    p
}

#[test]
fn izhikevich_cpu_simd_vs_cuda_parity() {
    let params = fill_params();
    let ns = IzhikevichModel::N_STATE;
    let np = IzhikevichModel::N_PARAMS;

    // ── CPU side ───────────────────────────────────────────────────
    let mut pop = NeuralPopulation::<IzhikevichModel>::new(N);
    pop.fill_state(0, -65.0); // v
    pop.fill_state(1, 0.0); // u
    pop.fill_state(2, 0.0); // refr
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

    let model = CompiledModel::new(IZHIKEVICH_CUDA_SOURCE, "izhikevich_kernel", ns, np, N, N)
        .expect("failed to compile the zoo-transpiled Izhikevich kernel (CUDA device required)");
    let mut g_spikes = vec![0i32; N];
    let mut spikes = vec![0i32; N];

    // Izhikevich is pure arithmetic (Horner poly + cond()) — CPU and GPU
    // agree to ~1 ulp. Not bit-for-bit like LIF: cond().then().else_()
    // transpiles to a C++ ternary and the CPU SIMD path may contract
    // a*b+c into FMA, so a few mantissa bits can differ per state.
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
        "Izhikevich must spike at I=10, got {cpu_spikes}"
    );
    eprintln!(
        "[parity] izhikevich: spikes={cpu_spikes} max differing bits per state float: {max_bits}"
    );
}

/// The zoo-transpiled source must be self-contained (kernel + step).
#[test]
fn izhikevich_cuda_source_is_self_contained() {
    assert!(IZHIKEVICH_CUDA_SOURCE.contains("__global__ void izhikevich_kernel"));
    assert!(IZHIKEVICH_CUDA_SOURCE.contains("izhikevich_step"));
}
