//! Parity test — THE point of `#[dual_neuron]`.
//!
//! One `HHModel` source is compiled to CPU SIMD (pillars `NeuralPopulation`)
//! and to a CUDA kernel (NVRTC). Both start from identical state/params and
//! run identical steps; the trajectories must match to float tolerance.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{CompiledModel, HHModel, HH_CUDA_SOURCE};

const N: usize = 64; // multiple of 8 → capacity == n, stride == n
const DT: f32 = 0.01;
const N_STEPS: usize = 1000;
// I=20 puts the first spike at ~step 135: phase 1 (100 steps) stays
// spike-free (smooth dynamics, tight parity), phase 2 (100..1000) catches
// the chaotic post-spike regime. With I=50 the first spike lands at ~79,
// i.e. inside phase 1, and phase 2 never sees a spike (counts stay 0).
const I_INJ: f32 = 20.0;

fn fill_params() -> [f32; HHModel::N_PARAMS] {
    // Order matches the #[param] fields of HHModel (pillars zoo, 6/12):
    // dt, g_Na, E_Na, g_K, E_K, g_Leak, E_Leak, q10, v_thresh, inv_cm,
    // g_CaT, E_CaT.
    let mut p = [0.0f32; HHModel::N_PARAMS];
    p[0] = DT;
    p[1] = 120.0;
    p[2] = 50.0;
    p[3] = 36.0;
    p[4] = -77.0;
    p[5] = 0.3;
    p[6] = -54.387;
    p[7] = 1.0; // Q10 at 6.3 °C
    p[8] = 0.0; // v_thresh
    p[9] = 1.0; // inv_cm
    p[10] = 0.0; // g_CaT — standard squid-axon HH (no CaT current)
    p[11] = 120.0; // E_CaT
    p
}

#[test]
fn hh_cpu_simd_vs_cuda_parity() {
    let params = fill_params();
    let ns = HHModel::N_STATE;
    let np = HHModel::N_PARAMS;

    // ── CPU side: pillars NeuralPopulation ────────────────────────
    let mut pop = NeuralPopulation::<HHModel>::new(N);
    pop.fill_state(0, -65.0); // v
    pop.fill_state(1, 0.05); // m
    pop.fill_state(2, 0.6); // h
    pop.fill_state(3, 0.32); // n
    pop.fill_state(4, 0.19); // c (CaV-T, steady-state at -65)
    pop.fill_state(5, 1.0); // was_below
    for (c, &v) in params.iter().enumerate() {
        pop.fill_param(c, v);
    }

    // ── GPU side: identical SoA snapshot ──────────────────────────
    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in = vec![I_INJ; N];

    let model = CompiledModel::new(HH_CUDA_SOURCE, "hh_kernel", ns, np, N, N)
        .expect("failed to compile/load the dual_neuron CUDA kernel (CUDA device required)");
    let mut g_spikes = vec![0i32; N];
    let mut spikes = vec![0i32; N];

    // ── Phase 1: 100 pre-spike steps — smooth dynamics, tight parity ──
    // (first HH spike at I=20 lands around step ~135, so this window is
    //  chaos-free and any divergence here means an equation mismatch.)
    // The tolerance covers the known drift: CPU fast_exp is a minimax poly
    // (~1e-5 err), GPU fast_exp maps to the SFU exp2f (MUFU.EX2, ~22-bit,
    // actually MORE accurate than the CPU poly). They differ by ~1 ulp per
    // call, which accumulates linearly. That is exp-implementation noise,
    // NOT an equation mismatch.
    for _ in 0..100 {
        pop.step(&i_in, &mut spikes);
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("gpu step");
        for c in 0..ns {
            for i in 0..N {
                let d = (pop.state_slice(c)[i] - g_states[c * N + i]).abs();
                assert!(
                    d < 2e-2,
                    "pre-spike parity broken at step state[{c}][{i}]: CPU={} GPU={}",
                    pop.state_slice(c)[i],
                    g_states[c * N + i]
                );
            }
        }
    } // ── Phase 2: remainder — per-neuron spike counts must match. ──
      // (Comparing the population-wide total could hide a phase swap where
      //  one neuron fires twice while a neighbour fires once.)
    let mut cpu_counts = vec![0i64; N];
    let mut gpu_counts = vec![0i64; N];
    for _ in 0..(N_STEPS - 100) {
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
    }
    let cpu_spikes: i64 = cpu_counts.iter().sum();
    let gpu_spikes: i64 = gpu_counts.iter().sum();
    // Self-guard against re-vacating the test: if a future param/I change
    // pushes the first spike back into phase 1 (as I=50 did — spikes at
    // ~79, counts stayed 0), this fails loudly instead of silently passing
    // with zero spikes on both sides.
    assert!(
        cpu_spikes > 0 && gpu_spikes > 0,
        "phase 2 must exercise spikes: CPU={cpu_spikes} GPU={gpu_spikes} — first spike likely landed inside phase 1"
    );
    for i in 0..N {
        assert_eq!(
            cpu_counts[i], gpu_counts[i],
            "neuron {i} spike count diverged: CPU={} GPU={}",
            cpu_counts[i], gpu_counts[i]
        );
    }
    assert_eq!(
        cpu_spikes, gpu_spikes,
        "population-wide spike totals diverged"
    );

    // After the first spike the system is chaotic: ~1 ulp exp2f-vs-poly
    // differences shift the spike phase by a few steps. Final states may
    // differ by a few mV while the *count* stays equal — that is float
    // drift, not an equation mismatch (Phase 1 already proved the
    // equations match).
    let mut max_diff = 0.0f32;
    for c in 0..ns {
        for i in 0..N {
            max_diff = max_diff.max((pop.state_slice(c)[i] - g_states[c * N + i]).abs());
        }
    }
    eprintln!("[parity] spike count equal ({cpu_spikes}); max |state_cpu - state_gpu| after {N_STEPS} steps: {max_diff:.5}");
}

/// Sanity: the transpiled C++ actually mentions the kernel we load.
#[test]
fn hh_cuda_source_is_self_contained() {
    assert!(HH_CUDA_SOURCE.contains("__global__ void hh_kernel"));
    assert!(HH_CUDA_SOURCE.contains("ohmic_gated"));
    assert!(HH_CUDA_SOURCE.contains("vtrap"));
}

/// SFU fast path: `fast_exp` must transpile to `exp2f(x * log2e)` (single
/// MUFU.EX2 instruction), vtrap divisions to `__fdividef` (MUFU.RCP), while
/// exact `exp()` stays `expf`. HH gates call fast_exp, so the generated
/// kernel should hit the hardware SFU, not the ~20-instr IEEE expf.
#[test]
fn hh_cuda_source_uses_sfu_fast_path() {
    // fast_exp → exp2f with the log2e fold
    assert!(
        HH_CUDA_SOURCE.contains("exp2f((") && HH_CUDA_SOURCE.contains("1.4426950408889634f"),
        "fast_exp must transpile to SFU exp2f(x * log2e)"
    );
    // vtrap divisions → MUFU.RCP
    assert!(
        HH_CUDA_SOURCE.contains("__fdividef"),
        "vtrap must use __fdividef (MUFU.RCP) for divisions"
    );
    // exact exp() stays on the IEEE path
    assert!(
        HH_CUDA_SOURCE.contains("expf("),
        "exact exp() must keep expf (IEEE ~1 ulp)"
    );
}

/// The transpiled zoo-HH source must compile through NVRTC (host-side — no GPU
/// needed). Catches transpiler C++ errors (bad inlining, redeclarations, …)
/// on machines without a device.
#[test]
fn hh_cuda_source_compiles_via_nvrtc() {
    pillars_cuda::engine::nvrtc_compile_check(HH_CUDA_SOURCE)
        .expect("HH_CUDA_SOURCE must produce valid PTX via NVRTC");
}
