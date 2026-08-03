//! 🔬 LIF Benchmark Suite — pillars throughput & overhead
//!
//! ## Overhead probes
//! Сравнивает 3 уровня абстракции:
//!
//! | Уровень | Что меряет | SIMD? |
//! |---------|------------|:-----:|
//! | **raw_scalar** | Голый цикл `for i in 0..n { v += (V_rest - v + I) / tau * dt }` | ❌ |
//! | **pillars** | `run_neuron::<LIFModel>` через SimdLane + run_neuron | ✅ f32x8 |
//! | **pillars_pop** | `NeuralPopulation::<LIFModel>::step()` — безопасный API | ✅ f32x8 |
//!
//! Запуск:
//! ```bash
//! cargo bench --package pillars --bench bench_lif
//! ```
//!
//! Для assembly дампа:
//! ```bash
//! ./crates/pillars/tools/asm.sh LIFModel step
//! ```

#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars::prelude::*;

// ─── Константы ──────────────────────────────────────────────
const N_NEURONS: usize = 138_000;
const N_STEPS: usize = 1000;
const DT: f32 = 0.1;
const INV_TAU_M: f32 = 0.1;     // 1/10 ms⁻¹
const V_REST: f32 = -70.0;
const V_THRESH: f32 = -54.0;
const V_RESET: f32 = -70.0;
const TAU_REF: f32 = 2.0;

fn make_params(n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * LIFModel::N_PARAMS];
    for i in 0..n {
        p[i] = DT;                     // [0] dt
        p[n + i] = INV_TAU_M;        // [1] inv_tau_m
        p[2*n + i] = V_REST;           // [2] v_rest
        p[3*n + i] = V_THRESH;         // [3] v_thresh
        p[4*n + i] = V_RESET;          // [4] v_reset
        p[5*n + i] = TAU_REF;          // [5] tau_ref
    }
    p
}

fn make_current(n: usize) -> Vec<f32> {
    let mut I = Vec::with_capacity(n);
    for i in 0..n {
        I.push(if i % 3 == 0 { 40.0 } else if i % 3 == 1 { 10.0 } else { 0.0 });
    }
    I
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 0: Raw scalar — минимальный оверхед, никаких трейтов
// ═══════════════════════════════════════════════════════════════

fn bench_lif_raw_scalar(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![V_REST; n];
    let mut refr = vec![0.0f32; n];
    let Ii = make_current(n);

    let mut group = c.benchmark_group("lif_raw_scalar");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step1000", |b| {
        b.iter(|| {
            v.fill(V_REST);
            refr.fill(0.0);
            for _ in 0..N_STEPS {
                for i in 0..n {
                    let is_refr = refr[i] > 0.0;
                    let dv = (V_REST - v[i] + Ii[i]) * INV_TAU_M * DT;
                    let v_new = v[i] + dv;
                    let v_dyn = if is_refr { v[i] } else { v_new };
                    let spike = !is_refr && v_dyn >= V_THRESH;
                    v[i] = if spike { V_RESET } else { v_dyn };
                    let refr_dec = refr[i] - DT;
                    let refr_set = if spike { TAU_REF } else { refr_dec };
                    refr[i] = if refr_set > 0.0 { refr_set } else { 0.0 };
                }
            }
            black_box(&v);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 1: pillars::run_neuron — SimdLane + f32x8 через трейт
// ═══════════════════════════════════════════════════════════════

// Inline LIFModel (без import из zoo, для чистоты замера)
struct LIFModel;
impl Model for LIFModel {
    const N_PARAMS: usize = 6;
    const N_STATE: usize = 2;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let refr = ctx.s(1);
        // Forward Euler: dv = (V_rest - V + I) * inv_tau * dt
        let dv = (ctx.p(2) - v + ctx.i()) * ctx.p(1) * ctx.dt();
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

fn bench_lif_pillars(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![V_REST; n];
    let mut u = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];
    let params = make_params(n);

    let mut group = c.benchmark_group("lif_pillars_run_neuron");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step1000", |b| {
        b.iter(|| {
            v.fill(V_REST);
            u.fill(0.0);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron::<LIFModel>(
                        v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                        h.as_mut_ptr(), h.as_mut_ptr(), h.as_mut_ptr(),
                        Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
                    );
                }
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 2: NeuralPopulation — безопасный API, AlignedVec
// ═══════════════════════════════════════════════════════════════

fn bench_lif_population(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut pop = NeuralPopulation::<LIFModel>::new(n);
    pop.fill_param(0, DT);
    pop.fill_param(1, INV_TAU_M);
    pop.fill_param(2, V_REST);
    pop.fill_param(3, V_THRESH);
    pop.fill_param(4, V_RESET);
    pop.fill_param(5, TAU_REF);
    pop.fill_state(0, V_REST);

    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("lif_pillars_population");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step1000", |b| {
        b.iter(|| {
            pop.fill_state(0, V_REST);
            pop.fill_state(1, 0.0);
            for _ in 0..N_STEPS {
                pop.step(&Ii, &mut spikes);
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 3: n=8 microbenchmark — минимальный SIMD чанк
// ═══════════════════════════════════════════════════════════════

fn bench_lif_n8_all(c: &mut Criterion) {
    let n = 8;
    let mut v = vec![V_REST; n];
    let Ii = vec![30.0f32; n];

    // Raw scalar
    let mut refr = vec![0.0f32; n];
    let mut group = c.benchmark_group("lif_n8_raw");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("scalar", |b| {
        b.iter(|| {
            v.fill(V_REST);
            refr.fill(0.0);
            for i in 0..n {
                let is_refr = refr[i] > 0.0;
                let dv = (V_REST - v[i] + Ii[i]) * INV_TAU_M * DT;
                let v_new = v[i] + dv;
                let spike = !is_refr && v_new >= V_THRESH;
                v[i] = if spike { V_RESET } else { v_new };
            }
            black_box(&v);
        });
    });
    group.finish();

    // Pillars run_neuron
    let mut u = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];
    let params = make_params(n);
    let mut group = c.benchmark_group("lif_n8_pillars");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("run_neuron", |b| {
        b.iter(|| {
            v.fill(V_REST);
            unsafe {
                run_neuron::<LIFModel>(
                    v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                    h.as_mut_ptr(), h.as_mut_ptr(), h.as_mut_ptr(),
                    Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
                );
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lif_raw_scalar,
    bench_lif_pillars,
    bench_lif_population,
    bench_lif_n8_all,
);
criterion_main!(benches);
