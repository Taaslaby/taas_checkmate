//! 🔬 GLIF4 & HAAP Benchmark — Old compose-style vs New ComputeNeuron
//!
//! Проверяет главный вопрос миграции: **zero-cost ли типизированный мост**?
//!
//! | Уровень | Что меряет | SIMD? | Абстракция |
//! |---------|------------|:-----:|:----------:|
//! | **old_compose** | Inline `Model::step` + `ctx.s(0)` / `ctx.p(4)` | ✅ f32x8 | SimdContext, как в compose_glif! |
//! | **new_run** | `run_neuron::<GLIF4Model>` | ✅ f32x8 | ComputeNeuron bridge (derive) |
//! | **new_pop** | `NeuralPopulation::<GLIF4Model>::step()` | ✅ f32x8 | ComputeNeuron + AlignedVec SoA |
//!
//! Если `new_run ≈ old_compose` — мост zero-cost, миграция не добавила
//! ни одного такта. Если `new_run` заметно медленнее — в bridge-коде
//! есть неинлайненная работа (from_ctx/to_ctx/StepContext).
//!
//! ## Известная несимметрия (учтена, влияет на ~1–2% в пользу new)
//! - Старый GLIF4 использует `ctx.dt()` (SIMD-load из SoA-памяти),
//!   новый — `ctx.dt_simd()` (broadcast скаляра из ctx).
//! - Старый HAAP использует скалярный хак `ctx.pf()` для сигмоид,
//!   новый — чистый SIMD `(-k·(v−v_half)).exp()`.
//!
//! Обе модели в обоих стилях считают через полный `exp()` — без fast_exp.
//!
//! `NeuralPopulation::step()` однопоточный — сравнение честное.
//!
//! ## ВАЖНО: `OldGLIF4Model` / `OldHAAPModel` — замороженные снапшоты
//! из git-истории (`src/model/models/*.rs`). При правке математики реальных
//! моделей эталоны здесь нужно обновлять синхронно, иначе бенч молча
//! перестанет быть репрезентативным.
//!
//! Запуск:
//! ```bash
//! cargo bench --package pillars --bench bench_glif4_haap
//! ```

#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars::prelude::*;
use pillars::zoo::{GLIF4ParamIdx, HAAParamIdx};

// ═══════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════

const N_NEURONS: usize = 138_000;
const N_STEPS: usize = 100;

// ═══════════════════════════════════════════════════════════════
//  Params (SoA layout, column-major)
// ═══════════════════════════════════════════════════════════════

fn make_glif4_params(n: usize) -> Vec<f32> {
    let inv_tau_ahp = 1.0 / 30.0;
    let inv_tau_theta = 1.0 / 30.0;
    let mut p = vec![0.0f32; n * GLIF4Model::N_PARAMS];
    for i in 0..n {
        p[GLIF4ParamIdx::DT as usize * n + i] = 1.0;
        p[GLIF4ParamIdx::G_Leak as usize * n + i] = 0.1;
        p[GLIF4ParamIdx::E_Leak as usize * n + i] = -65.0;
        p[GLIF4ParamIdx::V_Thresh_0 as usize * n + i] = -50.0;
        p[GLIF4ParamIdx::V_Reset as usize * n + i] = -65.0;
        p[GLIF4ParamIdx::Inv_Tau_AHP as usize * n + i] = inv_tau_ahp;
        p[GLIF4ParamIdx::Delta_G as usize * n + i] = 0.01;
        p[GLIF4ParamIdx::E_AHP as usize * n + i] = -80.0;
        p[GLIF4ParamIdx::Inv_Tau_Theta as usize * n + i] = inv_tau_theta;
        p[GLIF4ParamIdx::Delta_Theta as usize * n + i] = 3.0;
        p[GLIF4ParamIdx::Tau_Ref as usize * n + i] = 2.0;
    }
    p
}

fn make_haap_params(n: usize) -> Vec<f32> {
    let inv_tau_u = 1.0 / 100.0;
    let inv_tau_h = 1.0 / 200.0;
    let mut p = vec![0.0f32; n * HAAPModel::N_PARAMS];
    for i in 0..n {
        p[HAAParamIdx::DT as usize * n + i] = 0.05;
        p[HAAParamIdx::Inv_Cm as usize * n + i] = 1.0;
        p[HAAParamIdx::G_L as usize * n + i] = 0.1;
        p[HAAParamIdx::E_L as usize * n + i] = -65.0;
        p[HAAParamIdx::G_M as usize * n + i] = 0.5;
        p[HAAParamIdx::E_K as usize * n + i] = -80.0;
        p[HAAParamIdx::G_H as usize * n + i] = 0.3;
        p[HAAParamIdx::E_H as usize * n + i] = -30.0;
        p[HAAParamIdx::Inv_Tau_U as usize * n + i] = inv_tau_u;
        p[HAAParamIdx::Inv_Tau_H as usize * n + i] = inv_tau_h;
        p[HAAParamIdx::V_Half_U as usize * n + i] = -35.0;
        p[HAAParamIdx::K_U as usize * n + i] = 5.0;
        p[HAAParamIdx::V_Half_H as usize * n + i] = -75.0;
        p[HAAParamIdx::K_H as usize * n + i] = 6.0;
        p[HAAParamIdx::V_Thresh as usize * n + i] = -40.0;
        p[HAAParamIdx::V_Reset as usize * n + i] = -55.0;
        p[HAAParamIdx::Delta_U as usize * n + i] = 0.02;
    }
    p
}

fn make_current(n: usize, hi: f32, lo: f32) -> Vec<f32> {
    let mut I = Vec::with_capacity(n);
    for i in 0..n {
        I.push(if i % 2 == 0 { hi } else { lo });
    }
    I
}

// ═══════════════════════════════════════════════════════════════
//  LEVEL 1: OLD compose-style — inline Model::step + ctx.s()/ctx.p()
//  (восстановлено из git: crates/pillars/src/model/models/*.rs)
// ═══════════════════════════════════════════════════════════════

/// GLIF4 старым стилем: сырые индексы ctx.p(3), ctx.p(7), ...
struct OldGLIF4Model;

impl Model for OldGLIF4Model {
    const N_PARAMS: usize = 11;
    const N_STATE: usize = 4;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v_raw = ctx.s(0);
        let th = ctx.s(1);
        let g = ctx.s(2);
        let refr = ctx.s(3);

        let is_refr = refr.cmp_gt(S::zero()).cond();
        let v = is_refr.then(ctx.p(4)).else_(v_raw); // v_reset

        let v_th_eff = ctx.p(3) + th; // v_thresh_0 + θ
        let i_total = (ctx.i() + ctx.p(1) * ctx.p(2) + g * ctx.p(7)) - (ctx.p(1) + g) * v; // g_leak·E_leak + g·E_ahp − (g_leak+g)·v
        let v_new = v + i_total * ctx.dt();

        let theta_decay = th + ctx.dt() * (-th * ctx.p(8)); // inv_tau_theta
        let g_decay = g + ctx.dt() * (-g * ctx.p(5)); // inv_tau_ahp

        let spike = v_new.cmp_ge(v_th_eff);
        ctx.spike(spike);

        ctx.set_s(0, spike.cond().then(ctx.p(4)).else_(v_new));
        ctx.set_s(
            1,
            spike.cond().then(theta_decay + ctx.p(9)).else_(theta_decay),
        );
        ctx.set_s(2, spike.cond().then(g_decay + ctx.p(6)).else_(g_decay));

        let refr_dec = refr - ctx.dt();
        let refr_set = spike.cond().then(ctx.p(10)).else_(refr_dec);
        ctx.set_s(3, refr_set.max(S::zero()));
    }
}

/// HAAP старым стилем: сигмоиды через скалярный `ctx.pf()` хак.
struct OldHAAPModel;

impl Model for OldHAAPModel {
    const N_PARAMS: usize = 17;
    const N_STATE: usize = 3;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let u = ctx.s(1);
        let h = ctx.s(2);

        let i_leak = ctx.p(2) * (v - ctx.p(3));
        let i_m = ctx.p(4) * u * (v - ctx.p(5));
        let i_h = ctx.p(6) * h * (v - ctx.p(7));

        let i_total = i_leak + i_m + i_h;
        let dv = (ctx.i() - i_total) * ctx.p(1); // inv_cm
        let v_new = v + dv * ctx.dt();

        let u_inf = S::sigmoid(v_new, ctx.pf(10), ctx.pf(11)); // v_half_u, k_u
        let u_new = u + ctx.dt() * (u_inf - u) * ctx.p(8); // inv_tau_u

        let h_inf = S::sigmoid(ctx.p(12) - v_new, 0.0, ctx.pf(13)); // v_half_h, k_h
        let h_new = h + ctx.dt() * (h_inf - h) * ctx.p(9); // inv_tau_h

        let u_new = u_new.clamp01();
        let h_new = h_new.clamp01();

        let spike = v_new.cmp_ge(ctx.p(14)); // v_thresh
        ctx.spike(spike);

        ctx.set_s(0, spike.cond().then(ctx.p(15)).else_(v_new)); // v_reset
        let u_bump = (u_new + ctx.p(16)).clamp01(); // delta_u
        ctx.set_s(1, spike.cond().then(u_bump).else_(u_new));
        ctx.set_s(2, h_new);
    }
}

/// Общий прогон модели через run_neuron (n шагов, сброс состояния).
///
/// Состояния инициализируются из `init[0..M::N_STATE]`; хвостовые указатели
/// (s4/s5) передаются для полного N_STATE=6 интерфейса, но run_neuron читает
/// только первые `M::N_STATE` слотов — они не используются.
fn bench_model<M: Model>(
    c: &mut Criterion,
    group_name: &str,
    params: &[f32],
    i_in: &[f32],
    init: &[f32],
) {
    debug_assert!(
        init.len() >= M::N_STATE,
        "init must cover all {} states",
        M::N_STATE
    );
    let n = N_NEURONS;
    let mut v = vec![0.0f32; n];
    let mut s1 = vec![0.0f32; n];
    let mut s2 = vec![0.0f32; n];
    let mut s3 = vec![0.0f32; n];
    let mut s4 = vec![0.0f32; n];
    let mut s5 = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            v.fill(init[0]);
            if M::N_STATE > 1 {
                s1.fill(init[1]);
            }
            if M::N_STATE > 2 {
                s2.fill(init[2]);
            }
            if M::N_STATE > 3 {
                s3.fill(init[3]);
            }
            for _ in 0..N_STEPS {
                unsafe {
                    run_neuron::<M>(
                        v.as_mut_ptr(),
                        s1.as_mut_ptr(),
                        s2.as_mut_ptr(),
                        s3.as_mut_ptr(),
                        s4.as_mut_ptr(),
                        s5.as_mut_ptr(),
                        i_in.as_ptr(),
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

fn bench_glif4_old(c: &mut Criterion) {
    let params = make_glif4_params(N_NEURONS);
    let i_in = make_current(N_NEURONS, 30.0, 5.0);
    bench_model::<OldGLIF4Model>(
        c,
        "glif4_old_compose",
        &params,
        &i_in,
        &[-65.0, 0.0, 0.0, 0.0],
    );
}

fn bench_glif4_new_run(c: &mut Criterion) {
    let params = make_glif4_params(N_NEURONS);
    let i_in = make_current(N_NEURONS, 30.0, 5.0);
    bench_model::<GLIF4Model>(
        c,
        "glif4_new_run_neuron",
        &params,
        &i_in,
        &[-65.0, 0.0, 0.0, 0.0],
    );
}

fn bench_glif4_new_pop(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut pop = NeuralPopulation::<GLIF4Model>::new(n);
    let p1 = make_glif4_params(1);
    for (idx, &val) in p1.iter().enumerate().take(GLIF4Model::N_PARAMS) {
        pop.fill_param(idx, val);
    }
    let i_in = make_current(n, 30.0, 5.0);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("glif4_new_population");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            pop.fill_state(0, -65.0);
            pop.fill_state(1, 0.0);
            pop.fill_state(2, 0.0);
            pop.fill_state(3, 0.0);
            for _ in 0..N_STEPS {
                pop.step(&i_in, &mut spikes);
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

fn bench_haap_old(c: &mut Criterion) {
    let params = make_haap_params(N_NEURONS);
    let i_in = make_current(N_NEURONS, 15.0, -5.0);
    bench_model::<OldHAAPModel>(c, "haap_old_compose", &params, &i_in, &[-65.0, 0.0, 1.0]);
}

fn bench_haap_new_run(c: &mut Criterion) {
    let params = make_haap_params(N_NEURONS);
    let i_in = make_current(N_NEURONS, 15.0, -5.0);
    bench_model::<HAAPModel>(c, "haap_new_run_neuron", &params, &i_in, &[-65.0, 0.0, 1.0]);
}

fn bench_haap_new_pop(c: &mut Criterion) {
    let n = N_NEURONS;
    let mut pop = NeuralPopulation::<HAAPModel>::new(n);
    let p1 = make_haap_params(1);
    for (idx, &val) in p1.iter().enumerate().take(HAAPModel::N_PARAMS) {
        pop.fill_param(idx, val);
    }
    let i_in = make_current(n, 15.0, -5.0);
    let mut spikes = vec![0i32; n];

    let mut group = c.benchmark_group("haap_new_population");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("n138k_step100", |b| {
        b.iter(|| {
            pop.fill_state(0, -65.0);
            pop.fill_state(1, 0.0);
            pop.fill_state(2, 1.0);
            for _ in 0..N_STEPS {
                pop.step(&i_in, &mut spikes);
            }
            black_box(&spikes);
        });
    });
    group.finish();
}

// ═══════════════════════════════════════════════════════════════
//  Harness
// ═══════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_glif4_old,
    bench_glif4_new_run,
    bench_glif4_new_pop,
    bench_haap_old,
    bench_haap_new_run,
    bench_haap_new_pop,
);
criterion_main!(benches);
