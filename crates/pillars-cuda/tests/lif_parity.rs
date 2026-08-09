//! LIF parity — CPU SIMD vs NVRTC CUDA kernel.
//!
//! LIF is pure FMA arithmetic (no exp/log/sin), so the two backends must
//! agree **bit-for-bit** — a much stronger claim than HH's float-noise
//! parity. Half the population gets I=30 (spiking), half gets 0 (silent),
//! like the legacy `lif_cpu_vs_cuda` benchmark.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, LIFModel, LIF_CUDA_SOURCE};

const N: usize = 512; // multiple of 8 → stride == N
const DT: f32 = 0.1;
const N_STEPS: usize = 500;

#[test]
fn lif_cuda_source_contains_dt_bridge() {
    // `ctx.dt()` is the public authoring spelling; the transpiler must lower
    // it to the same runner-supplied DT used by the CUDA kernel.
    assert!(LIF_CUDA_SOURCE.contains("float DT"));
    assert!(LIF_CUDA_SOURCE.contains("* DT"));
    assert!(LIF_CUDA_SOURCE.contains("__global__ void lif_kernel"));
}

#[test]
fn lif_cpu_simd_vs_cuda_bitwise() {
    let ns = LIFModel::N_STATE; // 2: v, refr
    let np = LIFModel::N_PARAMS; // 6: dt, inv_tau_m, v_rest, v_thresh, v_reset, tau_ref

    let mut pop = NeuralPopulation::<LIFModel>::new(N);
    pop.fill_state(0, -70.0); // v
    pop.fill_state(1, 0.0); // refr
    pop.fill_param(0, DT);
    pop.fill_param(1, 0.1); // inv_tau_m (tau_m = 10)
    pop.fill_param(2, -70.0);
    pop.fill_param(3, -54.0);
    pop.fill_param(4, -70.0);
    pop.fill_param(5, 2.0); // tau_ref

    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 30.0 } else { 0.0 })
        .collect();

    let model = CompiledModel::new(LIF_CUDA_SOURCE, "lif_kernel", ns, np, N, N)
        .expect("failed to compile/load the dual_neuron LIF kernel (CUDA device required)");
    model.upload(&g_states, &g_params, &i_in).expect("upload");

    let mut spikes = vec![0i32; N];
    let mut g_spikes = vec![0i32; N];
    for _ in 0..N_STEPS {
        pop.step(&i_in, &mut spikes);
        model.step_device(DT).expect("gpu step");
        model
            .download_spikes(&mut g_spikes)
            .expect("download spikes");
        for i in 0..N {
            assert_eq!(
                spikes[i], g_spikes[i],
                "spike mismatch at step neuron {i}: CPU={} GPU={}",
                spikes[i], g_spikes[i]
            );
        }
    }

    // Bit-for-bit state check after the full run.
    model
        .download_states(&mut g_states)
        .expect("download states");
    for c in 0..ns {
        for i in 0..N {
            let cpu = pop.state_slice(c)[i];
            let gpu = g_states[c * N + i];
            assert_eq!(
                cpu.to_bits(),
                gpu.to_bits(),
                "state[{c}][{i}] differs: CPU={cpu} ({:#x}) GPU={gpu} ({:#x})",
                cpu.to_bits(),
                gpu.to_bits()
            );
        }
    }
}
