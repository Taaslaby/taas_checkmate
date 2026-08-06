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

// ─── LIFModel (pillars) ─────────────────────────────────

struct LIFModel;
impl Model for LIFModel {
    const N_PARAMS: usize = 6;
    const N_STATE: usize = 2;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let refr = ctx.s(1);
        let dv = (ctx.p(2) - v + ctx.i()) / ctx.p(1) * ctx.dt();
        let v_new = v + dv;
        let is_refr = refr.cmp_gt(S::zero());
        let v_dyn = S::blend(is_refr, v, v_new);
        let spike_raw = v_dyn.cmp_ge(ctx.p(3));
        let spike = S::blend(is_refr, S::splat(-1.0), spike_raw);
        ctx.spike(spike);
        ctx.set_s(0, S::blend(spike, ctx.p(4), v_dyn));
        let refr_dec = refr - ctx.dt();
        let refr_set = S::blend(spike, ctx.p(5), refr_dec);
        ctx.set_s(1, S::blend(refr_set.cmp_gt(S::zero()), refr_set, S::zero()));
    }
}

// ─── Константы ──────────────────────────────────────────

const N_NEURONS: usize = 138_000;
const N_STEPS: usize = 1000;
const DT: f32 = 0.1;
const TAU_M: f32 = 10.0;
const V_REST: f32 = -70.0;
const V_THRESH: f32 = -54.0;
const V_RESET: f32 = -70.0;
const TAU_REF: f32 = 2.0;

fn make_params(n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 6];
    for i in 0..n {
        p[i] = DT;
        p[n + i] = TAU_M;
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
