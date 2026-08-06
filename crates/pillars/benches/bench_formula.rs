//! # 🔬 Formula DSL Benchmarks — SIMD Bridge to the Zoo
//!
//! Two questions answered here:
//!
//! 1. **Interpreter**: does `run_program` actually go fast on `f32x8`?
//!    (same LIF program, scalar `f32` vs 8-wide `f32x8`).
//! 2. **Bridge**: is `FormulaModel` population throughput in the same ballpark
//!    as the hand-written `LIFModel`? (The interpreter has a per-op match
//!    overhead by design; the goal is "same order of magnitude", not parity.)
//!
//! Run: `cargo bench -p pillars --bench bench_formula`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use wide::f32x8;

use pillars::api::formula::*;
use pillars::core::lane::SimdLane;
use pillars::prelude::*;

/// The canonical LIF program — parameterized over SoA columns
/// `[0]dt [1]inv_tau [2]v_rest [3]v_thresh [4]v_reset [5]tau_ref`.
fn lif_program() -> Vec<OpNode> {
    let t_clamped = TEMP0;
    vec![
        OpNode::new(OpKind::RefrClamp, t_clamped).with_in(&[S0, S1, P0 + 4]),
        OpNode::new(OpKind::Leak, S0).with_in(&[t_clamped, I_SYN, P0 + 1, P0 + 2]),
        OpNode::new(OpKind::FireReset, S0).with_in(&[S0, P0 + 3, P0 + 4]),
        OpNode::new(OpKind::RefrTimer, S1).with_in(&[S1, P0 + 5]),
    ]
}

fn fill_lif_params<M: Model>(pop: &mut NeuralPopulation<M>) {
    pop.fill_param(0, 0.5); // dt
    pop.fill_param(1, 0.1); // inv_tau_m
    pop.fill_param(2, -70.0); // v_rest
    pop.fill_param(3, -54.0); // v_thresh
    pop.fill_param(4, -70.0); // v_reset
    pop.fill_param(5, 2.0); // tau_ref
    pop.fill_state(0, -70.0); // v
}

fn bench_interpreter(c: &mut Criterion) {
    let program = lif_program();

    // ── Scalar: one register file, f32 lanes ──
    let mut regs = [0.0f32; N_REGS];
    for k in [1usize, 2, 3, 4, 5] {
        regs[P0 + k] = [0.1, -70.0, -54.0, -70.0, 2.0][k - 1];
    }
    regs[S0] = -70.0;
    let dt = 0.5f32;
    let i_syn = 30.0f32;

    let mut g = c.benchmark_group("formula_interpreter");
    g.throughput(Throughput::Elements(1));
    g.bench_function("scalar_f32", |b| {
        b.iter(|| black_box(run_program(&program, &mut regs, dt, i_syn)))
    });

    // ── SIMD: one register file, f32x8 lanes ──
    let mut regs8 = [f32x8::zero(); N_REGS];
    for k in [1usize, 2, 3, 4, 5] {
        regs8[P0 + k] = f32x8::splat([0.1, -70.0, -54.0, -70.0, 2.0][k - 1]);
    }
    regs8[S0] = f32x8::splat(-70.0);
    g.throughput(Throughput::Elements(8));
    g.bench_function("simd_f32x8", |b| {
        b.iter(|| {
            black_box(run_program(
                &program,
                &mut regs8,
                f32x8::splat(dt),
                f32x8::splat(i_syn),
            ))
        })
    });
    g.finish();
}

fn bench_bridge(c: &mut Criterion) {
    // Register the program once, then time the full population path.
    FormulaModel::<0>::register(lif_program()).expect("register");

    let n = 138_000usize;
    let i_in = vec![30.0f32; n];
    let mut spikes = vec![0i32; n];

    let mut g = c.benchmark_group("formula_population");
    g.throughput(Throughput::Elements(n as u64));

    // ── FormulaModel — program interpreter through the population ──
    let mut fpop = NeuralPopulation::<FormulaModel<0>>::new(n);
    fill_lif_params(&mut fpop);
    g.bench_function("formula_model_138k", |b| {
        b.iter(|| {
            fpop.step(&i_in, &mut spikes);
            black_box(&spikes);
        })
    });

    // ── Reference: hand-written zoo LIF ──
    let mut zpop = NeuralPopulation::<LIFModel>::new(n);
    fill_lif_params(&mut zpop);
    g.bench_function("zoo_lif_138k", |b| {
        b.iter(|| {
            zpop.step(&i_in, &mut spikes);
            black_box(&spikes);
        })
    });

    g.finish();
}

criterion_group!(benches, bench_interpreter, bench_bridge);
criterion_main!(benches);
