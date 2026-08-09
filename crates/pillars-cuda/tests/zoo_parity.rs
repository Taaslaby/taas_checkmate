//! Zoo-wide parity — every `#[dual_neuron]` model transpiled IN pillars must
//! reproduce the CPU SIMD path on the GPU (NVRTC-compiled kernel).
//!
//! This is the end-to-end proof of the "parse neurons from pillars → GPU"
//! architecture at scale: 11 models, one generic harness, one GPU kernel each.
//! No model equations are duplicated in pillars-cuda — the CUDA C++ is the
//! `*_CUDA_SOURCE` consts emitted by the proc-macro in `pillars`.

use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars_cuda::{
    CompiledModel, ADEX_CUDA_SOURCE, ATLASOV_NEUROMOTOR_CUDA_SOURCE, FHN_CUDA_SOURCE,
    GLIF3_CUDA_SOURCE, GLIF4_CUDA_SOURCE, HAAP_CUDA_SOURCE, HR_CUDA_SOURCE,
    IZHIKEVICH_DUAL_CUDA_SOURCE, LIFAT_CUDA_SOURCE, MORRIS_LECAR_CUDA_SOURCE, RATE_CUDA_SOURCE,
    SAW_CUDA_SOURCE, VSN_CUDA_SOURCE, WC_CUDA_SOURCE,
};

const N: usize = 64; // multiple of 8 → capacity == n, stride == n

/// Run CPU SIMD vs CUDA for `steps`, asserting per-neuron spike-count equality
/// and per-state tolerance every step.
///
/// Returns `(total_spikes, max_differing_bits_per_state_float)`.
fn run_parity<M: Model>(
    kernel: &str,
    source: &str,
    init: &[(usize, f32)],
    params: &[f32],
    dt: f32,
    i_in: f32,
    steps: usize,
    tol: f32,
) -> (i64, u32) {
    let ns = M::N_STATE;
    let np = M::N_PARAMS;
    assert_eq!(np, params.len(), "{kernel}: param count mismatch");

    // ── CPU side ───────────────────────────────────────────────────
    let mut pop = NeuralPopulation::<M>::new(N);
    for &(c, v) in init {
        pop.fill_state(c, v);
    }
    for (c, &v) in params.iter().enumerate() {
        pop.fill_param(c, v);
    }

    // ── GPU side: identical SoA snapshot ───────────────────────────
    let mut g_states = vec![0.0f32; ns * N];
    for c in 0..ns {
        g_states[c * N..(c + 1) * N].copy_from_slice(pop.state_slice(c));
    }
    let g_params: Vec<f32> = (0..np).flat_map(|c| pop.param_slice(c).to_vec()).collect();
    let i_in_v = vec![i_in; N];

    let model = CompiledModel::new(source, kernel, ns, np, N, N)
        .expect("failed to compile zoo-transpiled kernel (CUDA device required)");
    let mut g_spikes = vec![0i32; N];
    let mut spikes = vec![0i32; N];
    let mut cpu_counts = vec![0i64; N];
    let mut gpu_counts = vec![0i64; N];
    let mut max_bits = 0u32;

    for step in 0..steps {
        pop.step(&i_in_v, &mut spikes);
        for i in 0..N {
            cpu_counts[i] += spikes[i] as i64;
        }
        model
            .step(&mut g_states, &g_params, &i_in_v, &mut g_spikes, dt)
            .expect("gpu step");
        for i in 0..N {
            gpu_counts[i] += g_spikes[i] as i64;
        }
        for c in 0..ns {
            for i in 0..N {
                let cpu = pop.state_slice(c)[i];
                let gpu = g_states[c * N + i];
                assert!(
                    (cpu - gpu).abs() < tol,
                    "{kernel} parity broken at step {step} state[{c}][{i}]: CPU={cpu} GPU={gpu}",
                );
                max_bits = max_bits.max((cpu.to_bits() ^ gpu.to_bits()).count_ones());
            }
        }
    }
    for i in 0..N {
        assert_eq!(
            cpu_counts[i], gpu_counts[i],
            "{kernel}: neuron {i} spike count diverged: CPU={} GPU={}",
            cpu_counts[i], gpu_counts[i]
        );
    }
    let total: i64 = cpu_counts.iter().sum();
    (total, max_bits)
}

// ═══════════════════════════════════════════════════════════════════
//  Spiking models
// ═══════════════════════════════════════════════════════════════════

#[test]
fn fhn_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::FHNModel>(
        "fhn_kernel",
        FHN_CUDA_SOURCE,
        &[(0, 0.0), (1, 0.0), (2, 1.0)], // v, w, was_below
        &[0.01, 0.1, 0.7, 0.8, 0.5],     // dt, inv_tau_w, a, b, v_thresh
        0.01,
        0.5,
        1000,
        2e-3,
    );
    assert!(spikes > 0, "fhn must spike at I=0.5, got {spikes}");
    eprintln!("[parity] fhn: spikes={spikes} max bits={bits}");
}

#[test]
fn adex_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::AdExModel>(
        "adex_kernel",
        ADEX_CUDA_SOURCE,
        &[(0, -65.0), (1, 0.0), (2, 0.0)], // v, w, refr
        &[
            0.05, 0.2, 0.01, -65.0, -50.0, -65.0, 2.0, 0.5, 0.5, 7.0, 2.0, 0.0,
        ], // dt, inv_tau_m, inv_tau_w, v_rest, v_thresh, v_reset, delta_t, inv_delta_t, a, b, tau_ref, v_peak
        0.05,
        15.0,
        1000,
        5e-3,
    );
    assert!(spikes > 0, "adex must spike at I=15, got {spikes}");
    eprintln!("[parity] adex: spikes={spikes} max bits={bits}");
}

#[test]
fn glif3_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::GLIF3Model>(
        "glif3_kernel",
        GLIF3_CUDA_SOURCE,
        &[(0, -65.0), (1, 0.0), (2, 0.0)], // v, g, refr
        &[1.0, 0.1, -65.0, -50.0, -65.0, 0.02, 0.01, -80.0, 2.0],
        1.0,
        30.0,
        500,
        2e-3,
    );
    assert!(spikes > 0, "glif3 must spike at I=30, got {spikes}");
    eprintln!("[parity] glif3: spikes={spikes} max bits={bits}");
}

#[test]
fn izhikevich_dual_parity() {
    // Izhikevich Dual Memory (Sun et al. 2026): fast Izh + slow state m.
    // Same params as `crates/kernels/tests/izhikevich_dual_parity.rs`.
    let (spikes, bits) = run_parity::<pillars_cuda::IzhikevichDualModel>(
        "izhikevichdual_kernel",
        IZHIKEVICH_DUAL_CUDA_SOURCE,
        &[(0, -65.0), (1, -13.0), (2, 0.0), (3, 0.0), (4, 0.5)], // v, u, refr, trace, m
        &[1.0, 0.02, 0.2, -65.0, 8.0, 3.0, 30.0, 0.5, 0.99, 5.0], // dt, a, b, c, d, tau_ref, v_thresh, trace_decay, tau_m, w_m
        1.0,
        10.0,
        500,
        1e-2, // nvcc FMA-contraction vs CPU non-fused accumulates through spikes
    );
    assert!(spikes > 0, "dual Izh must spike at I=10, got {spikes}");
    eprintln!("[parity] izhikevich_dual: spikes={spikes} max bits={bits}");
}

#[test]
fn lifat_parity() {
    // LIF-AT (GLIF2): adaptive threshold θ with exponential decay.
    // Same params as `crates/kernels/tests/lifat_parity.rs`.
    let (spikes, bits) = run_parity::<pillars_cuda::LIFAtModel>(
        "lifat_kernel",
        LIFAT_CUDA_SOURCE,
        &[(0, -70.0), (1, 10.0), (2, 0.0)], // v, theta (elevated), refr
        &[0.5, 0.1, -70.0, -54.0, -70.0, 0.05, 5.0, 3.0], // dt, inv_tau_m, v_rest, v_thresh, v_reset, inv_tau_theta, beta, tau_ref
        0.5,
        40.0,
        1000,
        1e-2, // exp vs expf: spike-timing can shift by one step (θ decays on spike)
    );
    assert!(spikes > 0, "lifat must spike at I=40, got {spikes}");
    eprintln!("[parity] lifat: spikes={spikes} max bits={bits}");
}

#[test]
fn glif4_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::GLIF4Model>(
        "glif4_kernel",
        GLIF4_CUDA_SOURCE,
        &[(0, -65.0), (1, 0.0), (2, 0.0), (3, 0.0)], // v, th, g, refr
        &[
            1.0,
            0.1,
            -65.0,
            -50.0,
            -65.0,
            1.0 / 30.0,
            0.01,
            -80.0,
            1.0 / 30.0,
            3.0,
            2.0,
        ],
        1.0,
        30.0,
        500,
        2e-3,
    );
    assert!(spikes > 0, "glif4 must spike at I=30, got {spikes}");
    eprintln!("[parity] glif4: spikes={spikes} max bits={bits}");
}

#[test]
fn haap_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::HAAPModel>(
        "haap_kernel",
        HAAP_CUDA_SOURCE,
        &[(0, -65.0), (1, 0.0), (2, 1.0)], // v, u, h
        &[
            0.05, 1.0, 0.1, -65.0, 0.5, -80.0, 0.3, -30.0, 0.01, 0.005, -35.0, 5.0, -75.0, 6.0,
            -40.0, -55.0, 0.02,
        ],
        0.05,
        15.0,
        1000,
        5e-3,
    );
    assert!(spikes > 0, "haap must spike at I=15, got {spikes}");
    eprintln!("[parity] haap: spikes={spikes} max bits={bits}");
}

#[test]
fn saw_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::SAWModel>(
        "saw_kernel",
        SAW_CUDA_SOURCE,
        &[(0, -65.0), (1, 0.0), (2, 0.0), (3, 0.0)], // v, u, tr, refr
        &[1.0, 0.02, 0.2, 0.01, 0.0, 2.0, 30.0, -65.0],
        1.0,
        15.0,
        500,
        1e-2, // fp noise accumulates through spike-triggered u/tr resets
    );
    assert!(spikes > 0, "saw must spike at I=15, got {spikes}");
    eprintln!("[parity] saw: spikes={spikes} max bits={bits}");
}

#[test]
fn vsn_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::VSNModel>(
        "vsn_kernel",
        VSN_CUDA_SOURCE,
        &[(0, -70.0), (1, 0.0), (2, 0.0)], // v, refr, surrogate
        &[0.5, 0.1, -70.0, -54.0, -70.0, 2.0, 0.9, 10.0],
        0.5,
        100.0,
        500,
        2e-3,
    );
    assert!(spikes > 0, "vsn must spike at I=100, got {spikes}");
    eprintln!("[parity] vsn: spikes={spikes} max bits={bits}");
}

#[test]
fn hr_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::HRModel>(
        "hr_kernel",
        HR_CUDA_SOURCE,
        &[(0, -1.6), (1, -12.0), (2, 0.0)], // x, y, z
        &[0.01, 1.0, 3.0, 1.0, 5.0, 0.006, 4.0, -1.6, 1.0],
        0.01,
        3.0,
        1000,
        2e-3,
    );
    assert!(spikes > 0, "hr must burst at I=3, got {spikes}");
    eprintln!("[parity] hr: spikes={spikes} max bits={bits}");
}

#[test]
fn morris_lecar_parity() {
    let (spikes, bits) = run_parity::<pillars_cuda::MorrisLecarModel>(
        "morrislecar_kernel",
        MORRIS_LECAR_CUDA_SOURCE,
        &[(0, -60.0), (1, 0.0), (2, 1.0)], // v, w, was_below
        &[
            1.0, 0.05, 2.0, -60.0, 4.4, 120.0, 8.0, -84.0, -1.2, 18.0, 2.0, 30.0, 0.04, 0.0,
        ],
        1.0,
        100.0,
        500,
        5e-3,
    );
    assert!(spikes > 0, "morris_lecar must spike at I=100, got {spikes}");
    eprintln!("[parity] morris_lecar: spikes={spikes} max bits={bits}");
}

// ═══════════════════════════════════════════════════════════════════
//  Non-spiking models (rate / muscle) — state parity only
// ═══════════════════════════════════════════════════════════════════

#[test]
fn wc_parity() {
    // Wilson-Cowan: rate model, no spikes. Feed I_ext = 0.5 to the
    // excitatory input; both populations must converge to the same (0,1).
    let (_, bits) = run_parity::<pillars_cuda::WCModel>(
        "wc_kernel",
        WC_CUDA_SOURCE,
        &[(0, 0.0), (1, 0.0)], // e, i
        &[0.1, 0.1, 0.1, 1.2, 0.4, 0.6, 0.8, 4.0],
        0.1,
        0.5,
        500,
        2e-3,
    );
    let mut pop = NeuralPopulation::<pillars_cuda::WCModel>::new(N);
    let params = [0.1, 0.1, 0.1, 1.2, 0.4, 0.6, 0.8, 4.0];
    for (c, &v) in params.iter().enumerate() {
        pop.fill_param(c, v);
    }
    let mut spikes = vec![0i32; N];
    let i_in = vec![0.5; N];
    for _ in 0..500 {
        pop.step(&i_in, &mut spikes);
    }
    for idx in 0..N {
        let e = pop.state_slice(0)[idx];
        let i = pop.state_slice(1)[idx];
        assert!(e > 0.0 && e < 1.0, "wc E[{idx}]={e} should be in (0,1)");
        assert!(i > 0.0 && i < 1.0, "wc I[{idx}]={i} should be in (0,1)");
    }
    eprintln!("[parity] wc: max bits={bits}");
}

#[test]
fn atlasov_parity() {
    // AtlasovNeuromotor: muscle, no spikes. i_in = 1.0 acts as the spike
    // mask that drives Ca²⁺ influx.
    let (spikes, bits) = run_parity::<pillars_cuda::AtlasovNeuromotorModel>(
        "atlasovneuromotor_kernel",
        ATLASOV_NEUROMOTOR_CUDA_SOURCE,
        &[(0, 0.0), (1, 0.0), (2, 0.0), (3, 1.0), (4, 0.0)], // ca, tension, reset_flag, length, velocity
        &[1.0, 0.1, 0.9, 0.5, 1.0, 0.2, 1.0, 1.0, 1.0],
        1.0,
        1.0,
        200,
        2e-3,
    );
    assert_eq!(spikes, 0, "atlasov is a muscle — must not spike");
    eprintln!("[parity] atlasov: max bits={bits}");
}

/// Every zoo-transpiled source must be self-contained (kernel + step fn).
#[test]
fn zoo_sources_are_self_contained() {
    let sources: &[(&str, &str, &str)] = &[
        ("fhn_kernel", "fhn_step", FHN_CUDA_SOURCE),
        ("adex_kernel", "adex_step", ADEX_CUDA_SOURCE),
        ("glif3_kernel", "glif3_step", GLIF3_CUDA_SOURCE),
        ("glif4_kernel", "glif4_step", GLIF4_CUDA_SOURCE),
        ("lifat_kernel", "lifat_step", LIFAT_CUDA_SOURCE),
        (
            "izhikevichdual_kernel",
            "izhikevichdual_step",
            IZHIKEVICH_DUAL_CUDA_SOURCE,
        ),
        ("haap_kernel", "haap_step", HAAP_CUDA_SOURCE),
        ("saw_kernel", "saw_step", SAW_CUDA_SOURCE),
        ("vsn_kernel", "vsn_step", VSN_CUDA_SOURCE),
        ("hr_kernel", "hr_step", HR_CUDA_SOURCE),
        (
            "morrislecar_kernel",
            "morrislecar_step",
            MORRIS_LECAR_CUDA_SOURCE,
        ),
        ("wc_kernel", "wc_step", WC_CUDA_SOURCE),
        (
            "atlasovneuromotor_kernel",
            "atlasovneuromotor_step",
            ATLASOV_NEUROMOTOR_CUDA_SOURCE,
        ),
        ("rate_kernel", "rate_step", RATE_CUDA_SOURCE),
    ];
    for (kernel, step, source) in sources {
        assert!(
            source.contains(&format!("__global__ void {kernel}")),
            "{kernel} missing"
        );
        assert!(source.contains(step), "{step} missing");
    }
}
