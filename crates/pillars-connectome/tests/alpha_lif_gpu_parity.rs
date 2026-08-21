//! GPU vs CPU parity for the alpha-LIF fly-brain worker (Shiu model).
//!
//! The neuron and Poisson schedule are **bit-identical** between CPU and GPU
//! (proven in `pillars-cuda`'s `zoo_parity::alpha_lif_parity`, `max bits=0`).
//! The recurrent scatter is a spike-driven **atomic** push (like PyTorch /
//! NEST / GeNN), so its summation order is not bit-reproducible — hence these
//! tests assert spike-count and membrane-potential *tolerance* parity, not
//! bit-parity. Requires the `cuda` feature + an NVRTC-capable GPU.
#![cfg(feature = "cuda")]

use pillars_connectome::{AlphaLifWorker, ConnectomeData, ShiuParams};
use pillars_cuda::{AlphaLifGpu, AlphaLifParams};

/// Deterministic xorshift64 random connectome: `n` neurons, `deg` outgoing
/// edges each, signed weights in ±`w_scale`. Independent of `rand` so the test
/// graph is byte-stable across runs.
fn random_connectome(n: usize, deg: usize, w_scale: f32, seed: u64) -> ConnectomeData {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut row_ptr = vec![0i32; n + 1];
    let mut col = Vec::with_capacity(n * deg);
    let mut val = Vec::with_capacity(n * deg);
    for pre in 0..n {
        for _ in 0..deg {
            let post = (next() % n as u64) as i32;
            let mag = w_scale * (0.5 + (next() % 1000) as f32 / 1000.0);
            let w = if (next() & 1) == 0 { mag } else { -mag };
            col.push(post);
            val.push(w);
        }
        row_ptr[pre + 1] = col.len() as i32;
    }
    ConnectomeData::new(row_ptr, col, val, vec![], vec![])
}

#[test]
fn alpha_lif_gpu_parity() {
    let n = 1024; // multiple of 8 → stride == n
    let conn = random_connectome(n, 12, 2.0, 0x5EED_CAFE);
    let grn: Vec<usize> = (0..8).collect();

    // GPU first (borrows the CSR slices), then move `conn` into the CPU worker.
    let mut gpu = AlphaLifGpu::new(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        AlphaLifParams::flywire(),
        &grn,
        200.0,
        68.75,
    )
    .expect("GPU worker construction");

    let mut cpu = AlphaLifWorker::new(conn, ShiuParams::flywire()).expect("CPU worker");
    cpu.set_sugar_grn(&grn);

    let gpu_out = gpu.run_seconds(0.1).expect("GPU run");
    let cpu_out = cpu.run_seconds(0.1);
    let cpu_v = cpu.v().to_vec();
    let gpu_v = gpu.download_v().expect("GPU v download");

    assert_eq!(gpu_v.len(), n);

    // Spike counts / activity: the atomic scatter can reorder ~1-ulp sums, so
    // allow a small delta (in practice identical for sparse simultaneous fan-in).
    let d_spikes = (cpu_out.total_spikes as i64 - gpu_out.total_spikes as i64).abs();
    assert!(
        d_spikes <= 4,
        "spike counts drifted too far: CPU={} GPU={}",
        cpu_out.total_spikes,
        gpu_out.total_spikes
    );
    let d_active = (cpu_out.active_neurons as i64 - gpu_out.active_neurons as i64).abs();
    assert!(
        d_active <= 4,
        "active neurons drifted: CPU={} GPU={}",
        cpu_out.active_neurons,
        gpu_out.active_neurons
    );

    // Membrane potentials within a tight tolerance (neuron bit-exact; the only
    // drift is ~1-ulp scatter reordering, which stays << 1e-3).
    let mut max_abs = 0.0f32;
    for i in 0..n {
        max_abs = max_abs.max((cpu_v[i] - gpu_v[i]).abs());
    }
    assert!(
        max_abs < 1e-3,
        "membrane potentials drifted: max|Δv|={max_abs}"
    );
    eprintln!(
        "[parity] alpha_lif_gpu: spikes={}/{} active={}/{} max|Δv|={max_abs:.2e}",
        cpu_out.total_spikes, gpu_out.total_spikes, cpu_out.active_neurons, gpu_out.active_neurons,
    );
}

/// Smaller smoke test: GPU worker runs at all and produces sane metrics.
#[test]
fn alpha_lif_gpu_runs() {
    let n = 256;
    let conn = random_connectome(n, 4, 1.0, 0xBEEF);
    let grn: Vec<usize> = (0..4).collect();
    let mut gpu = AlphaLifGpu::new(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        AlphaLifParams::flywire(),
        &grn,
        200.0,
        68.75,
    )
    .expect("GPU worker");
    let out = gpu.run_seconds(0.05).expect("GPU run");
    assert!(
        out.total_spikes > 0,
        "GPU worker should fire with Poisson input: {out:?}"
    );
    assert!(out.wall_seconds > 0.0);
}
