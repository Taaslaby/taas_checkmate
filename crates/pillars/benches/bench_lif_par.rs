//! 🔬 LIF Parallel Benchmark: pillars::zoo run_neuron_par vs run_neuron
//!
//! Сравнивает многопоточный `run_neuron_par::<LIFModel>` (Rayon) с однопоточным `run_neuron`.
//!
//! Запуск:
//!   cargo bench --package pillars --bench bench_lif_par
//!
//! Примечание: компилируется только с `features = ["rayon"]`.

#![cfg(feature = "rayon")]
#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars::prelude::*;

// ─── Константы ──────────────────────────────────────────

const N_NEURONS: usize = 138_000;
const N_STEPS: usize = 1000;
const DT: f32 = 0.1;
const INV_TAU_M: f32 = 0.1; // 1/10 ms⁻¹
const V_REST: f32 = -70.0;
const V_THRESH: f32 = -54.0;
const V_RESET: f32 = -70.0;
const TAU_REF: f32 = 2.0;

fn make_params(n: usize) -> Vec<f32> {
    // LIFModel layout: [0]dt, [1]inv_tau_m, [2]v_rest, [3]v_thresh,
    // [4]v_reset, [5]tau_ref — канонический zoo::LIFModel (из prelude).
    let mut p = vec![0.0f32; n * LIFModel::N_PARAMS];
    for i in 0..n {
        p[i] = DT;
        p[n + i] = INV_TAU_M;
        p[2 * n + i] = V_REST;
        p[3 * n + i] = V_THRESH;
        p[4 * n + i] = V_RESET;
        p[5 * n + i] = TAU_REF;
    }
    p
}

fn make_current(n: usize) -> Vec<f32> {
    let mut I = Vec::with_capacity(n);
    for i in 0..n {
        I.push(if i % 3 == 0 {
            40.0
        } else if i % 3 == 1 {
            10.0
        } else {
            0.0
        });
    }
    I
}

// ─── Sequential baseline ────────────────────────────────────

fn bench_seq(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![V_REST; n];
    let mut u = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];
    let params = make_params(n);
    let init_v = vec![V_REST; n];
    let init_u = vec![0.0f32; n];

    let mut group = c.benchmark_group("lif_par_138k");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("sequential", |b| {
        b.iter(|| {
            v.copy_from_slice(&init_v);
            u.copy_from_slice(&init_u);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron::<LIFModel>(
                        v.as_mut_ptr(),
                        u.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        Ii.as_ptr(),
                        spikes.as_mut_ptr(),
                        params.as_ptr(),
                        n,
                    );
                }
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

// ─── Parallel (Rayon) ───────────────────────────────────────

fn bench_par(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![V_REST; n];
    let mut u = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];
    let params = make_params(n);
    let init_v = vec![V_REST; n];
    let init_u = vec![0.0f32; n];

    let mut group = c.benchmark_group("lif_par_138k");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("parallel", |b| {
        b.iter(|| {
            v.copy_from_slice(&init_v);
            u.copy_from_slice(&init_u);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron_par::<LIFModel>(
                        v.as_mut_ptr(),
                        u.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        h.as_mut_ptr(),
                        Ii.as_ptr(),
                        spikes.as_mut_ptr(),
                        params.as_ptr(),
                        n,
                        256,
                    );
                }
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_seq, bench_par);
criterion_main!(benches);
