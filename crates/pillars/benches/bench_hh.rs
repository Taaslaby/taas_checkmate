//! 🔬 HH Benchmark Suite — Old compose_conductance! vs New ComputeNeuron
//!
//! Сравнивает 4 уровня абстракции:
//!
//! | Уровень | Что меряет | SIMD? | Абстракция |
//! |---------|------------|:-----:|:----------:|
//! | **raw_scalar** | Голый цикл `for i in 0..n` | ❌ | Нет |
//! | **pillars_old** | Inline SimdContext (compose_conductance! style) | ✅ f32x8 | SimdContext + ctx.s(0) |
//! | **pillars_new_run** | `run_neuron::<HHModel>` | ✅ f32x8 | ComputeNeuron bridge |
//! | **pillars_new_pop** | `NeuralPopulation::<HHModel>::step()` | ✅ f32x8 | ComputeNeuron + SoA |
//!
//! Запуск:
//! ```bash
//! cargo bench --package pillars --bench bench_hh
//! ```

#![allow(non_snake_case)]
#![allow(clippy::approx_constant)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars::ops::triangulated::{LutBuilder, LutConfig, TriangLUT};
use pillars::prelude::*;
use pillars::zoo::neurons::TriangulatedModel;
use std::sync::OnceLock;

static HH_LUT: OnceLock<TriangLUT> = OnceLock::new();

fn ensure_hh_lut() -> &'static TriangLUT {
    HH_LUT.get_or_init(|| {
        let params = make_params(1);
        let state_init = [-65.0, 0.05, 0.6, 0.32, 1.0];
        let config = LutConfig {
            dt: 0.01,
            input_modes: vec![(1.0, 50.0, 1)],
            v_min: -100.0,
            v_max: 60.0,
            n_steps: 2_000_000,
            seed: 42,
        };
        let lut = LutBuilder::new(config).build::<pillars::zoo::HHModel>(&params, &state_init);
        eprintln!("HH-LUT calibration complete (2M steps, I=50)");
        lut
    })
}

// ═══════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════

const N_NEURONS: usize = 138_000;
const N_STEPS: usize = 100;
const Q10: f32 = 1.0; // 3^((6.3-6.3)/10) = 3^0 = 1.0

// ═══════════════════════════════════════════════════════════════
//  Helpers (shared by all benchmarks)
// ═══════════════════════════════════════════════════════════════

fn make_params(n: usize) -> Vec<f32> {
    // HHModel::N_PARAMS = 12 (incl. CaV-T g_CaT / E_CaT columns).
    // ⚠️ Must match the model exactly — an undersized buffer made the
    // raw-runner benches read out of bounds and segfault at n=138k.
    let mut p = vec![0.0f32; n * pillars::zoo::HHModel::N_PARAMS];
    for i in 0..n {
        p[i] = 0.01; // dt
        p[n + i] = 120.0; // g_Na
        p[2 * n + i] = 50.0; // E_Na
        p[3 * n + i] = 36.0; // g_K
        p[4 * n + i] = -77.0; // E_K
        p[5 * n + i] = 0.3; // g_Leak
        p[6 * n + i] = -54.387; // E_Leak
        p[7 * n + i] = Q10; // q10
        p[8 * n + i] = 0.0; // v_thresh
        p[9 * n + i] = 1.0; // inv_cm
        p[10 * n + i] = 0.0; // g_CaT — 0 = standard squid-axon HH
        p[11 * n + i] = 120.0; // E_CaT
    }
    p
}

fn make_current(n: usize) -> Vec<f32> {
    let mut I = Vec::with_capacity(n);
    for i in 0..n {
        I.push(if i % 2 == 0 { 50.0 } else { 10.0 });
    }
    I
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 0: Raw scalar — минимальный overhead
// ═══════════════════════════════════════════════════════════════

/// Hodgkin-Huxley scalar step (no SIMD, no trait).
#[inline(never)]
fn hh_scalar_step(
    v: &mut f32,
    m: &mut f32,
    h: &mut f32,
    n: &mut f32,
    was_below: &mut f32,
    i_syn: f32,
    dt: f32,
    g_na: f32,
    e_na: f32,
    g_k: f32,
    e_k: f32,
    g_l: f32,
    e_l: f32,
    q10: f32,
    v_thresh: f32,
    inv_cm: f32,
) {
    let dt_q10 = dt * q10;

    // m-gate
    let vm40 = -40.0 - *v;
    let exp_m = (vm40 / 10.0).exp();
    let vtrap_m = if vm40.abs() > 1e-10 {
        vm40 / (exp_m - 1.0)
    } else {
        10.0
    };
    let alpha_m = 0.1 * vtrap_m;
    let beta_m = 4.0 * (-(*v + 65.0) / 18.0).exp();
    let sum_m = alpha_m + beta_m;
    let m_inf = alpha_m / sum_m;
    *m = m_inf + (*m - m_inf) * (-dt_q10 * sum_m).exp();

    // h-gate
    let alpha_h = 0.07 * (-(*v + 65.0) / 20.0).exp();
    let beta_h = 1.0 / (1.0 + (-(*v + 35.0) / 10.0).exp());
    let sum_h = alpha_h + beta_h;
    let h_inf = alpha_h / sum_h;
    *h = h_inf + (*h - h_inf) * (-dt_q10 * sum_h).exp();

    // n-gate
    let vn55 = -55.0 - *v;
    let exp_n = (vn55 / 10.0).exp();
    let vtrap_n = if vn55.abs() > 1e-10 {
        vn55 / (exp_n - 1.0)
    } else {
        10.0
    };
    let alpha_n = 0.01 * vtrap_n;
    let beta_n = 0.125 * (-(*v + 65.0) / 80.0).exp();
    let sum_n = alpha_n + beta_n;
    let n_inf = alpha_n / sum_n;
    *n = n_inf + (*n - n_inf) * (-dt_q10 * sum_n).exp();

    // Ionic currents
    let i_na = g_na * (*m).powi(3) * (*h) * (*v - e_na);
    let i_k = g_k * (*n).powi(4) * (*v - e_k);
    let i_leak = g_l * (*v - e_l);
    let dv = (i_syn - i_na - i_k - i_leak) * inv_cm * dt;
    *v += dv;

    // Edge-triggered spike
    let is_above = *v >= v_thresh;
    let spike = is_above && *was_below > 0.0;
    *was_below = if is_above { 0.0 } else { 1.0 };
    black_box(spike);
}

fn bench_hh_raw_scalar(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![-65.0; n];
    let mut m = vec![0.05; n];
    let mut h = vec![0.6; n];
    let mut k_g = vec![0.32; n];
    let mut flag = vec![1.0; n];
    let Ii = make_current(n);

    let mut group = c.benchmark_group("hh_raw_scalar");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            flag.fill(1.0);
            for _ in 0..N_STEPS {
                for i in 0..n {
                    hh_scalar_step(
                        &mut v[i],
                        &mut m[i],
                        &mut h[i],
                        &mut k_g[i],
                        &mut flag[i],
                        Ii[i],
                        0.01,
                        120.0,
                        50.0,
                        36.0,
                        -77.0,
                        0.3,
                        -54.387,
                        Q10,
                        0.0,
                        1.0,
                    );
                }
            }
            black_box(&v);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 1: Old compose_conductance! style (inline SimdContext)
// ═══════════════════════════════════════════════════════════════

struct OldHHModel;

impl Model for OldHHModel {
    const N_PARAMS: usize = 10;
    const N_STATE: usize = 5;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let m = ctx.s(1);
        let h = ctx.s(2);
        let n = ctx.s(3);
        let was_below = ctx.s(4);

        let dt_q10 = ctx.dt() * ctx.p(7); // q10

        // ─── Gates (same compute_gate_state as before) ───
        let m_new = old_gate(
            v, m, -40.0, 10.0, 0.1, 4.0, 65.0, 18.0, false, false, dt_q10,
        );
        let h_new = old_gate(v, h, -65.0, 20.0, 0.07, 1.0, 35.0, 10.0, true, true, dt_q10);
        let n_new = old_gate(
            v, n, -55.0, 10.0, 0.01, 0.125, 65.0, 80.0, false, false, dt_q10,
        );

        // ─── Ionic currents via loops (old style) ───
        let i_na = old_ohmic(ctx.p(1), v, ctx.p(2), m_new, 3, h_new, 1);
        let i_k = old_ohmic(ctx.p(3), v, ctx.p(4), n_new, 4, S::one(), 0);
        let i_leak = ctx.p(5) * (v - ctx.p(6));

        // ─── Membrane ───
        let dv = (ctx.i() - i_na - i_k - i_leak) * ctx.p(9) * ctx.dt();
        let v_new = v + dv;

        // ─── Edge spike ───
        let is_above = v_new.cmp_ge(ctx.p(8));
        let spike = is_above.and(was_below.cmp_gt(S::zero()));
        ctx.spike(spike);
        ctx.set_s(0, v_new);
        ctx.set_s(1, m_new);
        ctx.set_s(2, h_new);
        ctx.set_s(3, n_new);
        ctx.set_s(4, S::blend(is_above, S::zero(), S::one()));
    }
}

#[inline(always)]
fn old_gate<S: SimdLane>(
    v: S,
    gate: S,
    v_off: f32,
    scale: f32,
    alpha_c: f32,
    beta_c: f32,
    beta_volt: f32,
    beta_div: f32,
    alpha_exp: bool,
    beta_sigmoid: bool,
    dt_q10: S,
) -> S {
    let v_shift = S::splat(v_off) - v;
    let alpha = if alpha_exp {
        S::splat(alpha_c) * (v_shift / S::splat(scale)).exp()
    } else {
        S::splat(alpha_c) * S::vtrap(v_shift, S::splat(scale))
    };
    let exp_arg = -(v + S::splat(beta_volt)) / S::splat(beta_div);
    let beta = if beta_sigmoid {
        S::splat(beta_c) / (S::one() + exp_arg.exp())
    } else {
        S::splat(beta_c) * exp_arg.exp()
    };
    let sum = alpha + beta;
    let g_inf = alpha / sum;
    (-dt_q10 * sum).exp().mul_add(gate - g_inf, g_inf)
}

#[inline(always)]
fn old_ohmic<S: SimdLane>(g_max: S, v: S, e_rev: S, g1: S, p1: u32, g2: S, p2: u32) -> S {
    let mut g = g_max;
    for _ in 0..p1 {
        g = g * g1;
    }
    for _ in 0..p2 {
        g = g * g2;
    }
    g.mul_add(v, -g * e_rev)
}

fn bench_hh_old_pillars(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![-65.0; n];
    let mut m = vec![0.05; n];
    let mut h = vec![0.6; n];
    let mut k_g = vec![0.32; n];
    let mut flag = vec![1.0; n];
    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];
    let params = make_params(n);
    let mut dummy = vec![0.0; n];

    let mut group = c.benchmark_group("hh_pillars_old");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            flag.fill(1.0);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron::<OldHHModel>(
                        v.as_mut_ptr(),
                        m.as_mut_ptr(),
                        h.as_mut_ptr(),
                        k_g.as_mut_ptr(),
                        flag.as_mut_ptr(),
                        dummy.as_mut_ptr(),
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

// ═══════════════════════════════════════════════════════════════
//  LEVEL 2: New ComputeNeuron HHModel via run_neuron
// ═══════════════════════════════════════════════════════════════

fn bench_hh_new_run_neuron(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut v = vec![-65.0; n];
    let mut m = vec![0.05; n];
    let mut h = vec![0.6; n];
    let mut k_g = vec![0.32; n];
    let mut cat = vec![0.19; n]; // CaV-T gate (6th state)
    let mut flag = vec![1.0; n]; // was_below (7th state)
    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];
    let params = make_params(n);

    let mut group = c.benchmark_group("hh_pillars_new_run_neuron");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            cat.fill(0.19);
            flag.fill(1.0);
            for _ in 0..N_STEPS {
                unsafe {
                    // Pointer order: (v, m, h, n, c, was_below)
                    run_neuron::<pillars::zoo::HHModel>(
                        v.as_mut_ptr(),
                        m.as_mut_ptr(),
                        h.as_mut_ptr(),
                        k_g.as_mut_ptr(),
                        cat.as_mut_ptr(),
                        flag.as_mut_ptr(),
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

// ═══════════════════════════════════════════════════════════════
//  LEVEL 3: New ComputeNeuron via NeuralPopulation
// ═══════════════════════════════════════════════════════════════

fn bench_hh_new_population(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut pop = pillars::NeuralPopulation::<pillars::zoo::HHModel>::new(n);

    // Fill params across all neurons
    pop.fill_param(0, 0.01); // dt
    pop.fill_param(1, 120.0); // g_Na
    pop.fill_param(2, 50.0); // E_Na
    pop.fill_param(3, 36.0); // g_K
    pop.fill_param(4, -77.0); // E_K
    pop.fill_param(5, 0.3); // g_Leak
    pop.fill_param(6, -54.387); // E_Leak
    pop.fill_param(7, Q10); // q10
    pop.fill_param(8, 0.0); // v_thresh
    pop.fill_param(9, 1.0); // inv_cm

    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("hh_pillars_new_population");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            pop.fill_state(0, -65.0);
            pop.fill_state(1, 0.05);
            pop.fill_state(2, 0.6);
            pop.fill_state(3, 0.32);
            pop.fill_state(4, 1.0);
            for _ in 0..N_STEPS {
                pop.step(&Ii, &mut spikes);
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 4: TriangLUT (HH-calibrated)
// ═══════════════════════════════════════════════════════════════

fn bench_hh_triang_lut(c: &mut Criterion) {
    let lut = ensure_hh_lut();
    TriangulatedModel::set_lut(lut.clone());

    let n = N_NEURONS;
    let mut pop = pillars::NeuralPopulation::<TriangulatedModel>::new(n);
    pop.fill_param(0, 0.01); // dt
    pop.fill_param(1, 0.0); // v_thresh (HH uses 0.0)
    pop.fill_param(2, 0.5); // input_thresh
    pop.fill_state(0, -65.0); // v
    pop.fill_state(1, 0.0); // history

    let Ii = make_current(n);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("hh_triang_lut");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            pop.fill_state(0, -65.0);
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
//  LEVEL 5: Multithreaded — run_neuron (seq) vs run_neuron_par (Rayon)
// ═══════════════════════════════════════════════════════════════
//
// HHModel has exactly 6 states (v, m, h, n, c, was_below) → it fits the
// legacy `run_neuron_par` 6-pointer contract. This group shows real
// core-scaling on the same 138k workload.
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
fn bench_hh_par(c: &mut Criterion) {
    let n = N_NEURONS;
    let Ii = make_current(n);
    let params = make_params(n);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("hh_par_138k");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("sequential", |b| {
        let mut v = vec![-65.0; n];
        let mut m = vec![0.05; n];
        let mut h = vec![0.6; n];
        let mut k_g = vec![0.32; n];
        let mut cat = vec![0.19; n];
        let mut flag = vec![1.0; n];
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            cat.fill(0.19);
            flag.fill(1.0);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron::<pillars::zoo::HHModel>(
                        v.as_mut_ptr(),
                        m.as_mut_ptr(),
                        h.as_mut_ptr(),
                        k_g.as_mut_ptr(),
                        cat.as_mut_ptr(),
                        flag.as_mut_ptr(),
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

    group.bench_function("parallel", |b| {
        let mut v = vec![-65.0; n];
        let mut m = vec![0.05; n];
        let mut h = vec![0.6; n];
        let mut k_g = vec![0.32; n];
        let mut cat = vec![0.19; n];
        let mut flag = vec![1.0; n];
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            cat.fill(0.19);
            flag.fill(1.0);
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron_par::<pillars::zoo::HHModel>(
                        v.as_mut_ptr(),
                        m.as_mut_ptr(),
                        h.as_mut_ptr(),
                        k_g.as_mut_ptr(),
                        cat.as_mut_ptr(),
                        flag.as_mut_ptr(),
                        Ii.as_ptr(),
                        spikes.as_mut_ptr(),
                        params.as_ptr(),
                        n,
                        4096,
                    );
                }
            }
            black_box(&spikes);
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  Harness
// ═══════════════════════════════════════════════════════════════

fn bench_hh_n8(c: &mut Criterion) {
    let n = 8;
    let Ii = vec![30.0; n];
    let params = make_params(n);

    // Old — n8
    let mut v = vec![-65.0; n];
    let mut m = vec![0.05; n];
    let mut h = vec![0.6; n];
    let mut k_g = vec![0.32; n];
    let mut flag = vec![1.0; n];
    let mut spikes = vec![0i32; n];
    let mut dummy = vec![0.0; n];

    let mut group = c.benchmark_group("hh_n8_old");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("run_neuron", |b| {
        b.iter(|| {
            v.fill(-65.0);
            m.fill(0.05);
            h.fill(0.6);
            k_g.fill(0.32);
            flag.fill(1.0);
            unsafe {
                run_neuron::<OldHHModel>(
                    v.as_mut_ptr(),
                    m.as_mut_ptr(),
                    h.as_mut_ptr(),
                    k_g.as_mut_ptr(),
                    flag.as_mut_ptr(),
                    dummy.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            black_box(&spikes);
        });
    });
    group.finish();

    // New — n8
    let mut v2 = vec![-65.0; n];
    let mut m2 = vec![0.05; n];
    let mut h2 = vec![0.6; n];
    let mut k_g2 = vec![0.32; n];
    let mut cat2 = vec![0.19; n]; // CaV-T gate
    let mut flag2 = vec![1.0; n]; // was_below
    let mut group2 = c.benchmark_group("hh_n8_new");
    group2.throughput(Throughput::Elements(n as u64));
    group2.bench_function("run_neuron", |b| {
        b.iter(|| {
            v2.fill(-65.0);
            m2.fill(0.05);
            h2.fill(0.6);
            k_g2.fill(0.32);
            cat2.fill(0.19);
            flag2.fill(1.0);
            unsafe {
                // Pointer order: (v, m, h, n, c, was_below)
                run_neuron::<pillars::zoo::HHModel>(
                    v2.as_mut_ptr(),
                    m2.as_mut_ptr(),
                    h2.as_mut_ptr(),
                    k_g2.as_mut_ptr(),
                    cat2.as_mut_ptr(),
                    flag2.as_mut_ptr(),
                    Ii.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }
            black_box(&spikes);
        });
    });
    group2.finish();
}

criterion_group!(
    benches,
    bench_hh_raw_scalar,
    bench_hh_old_pillars,
    bench_hh_new_run_neuron,
    bench_hh_new_population,
    bench_hh_triang_lut,
    bench_hh_par,
    bench_hh_n8,
);
criterion_main!(benches);
