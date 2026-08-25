//! # HH Benchmark — pillars-only (legacy `step_hh_lut` removed)
//!
//! После миграции `crates/connectome` полностью на `pillars::HHModel +
//! NeuralPopulation` старый движок удалён, поэтому бенчмарк измеряет два
//! pillars-режима на нейронах мухи:
//!
//! - `standard` — NaV + KV (g_CaT = 0): требует внешнего тока, чтобы спайковать.
//! - `pacemaking` — NaV + KV + CaV-T (g_CaT = 0.5): спайкует **спонтанно**.
//!
//! Сценарий «нейроны мухи»: FlyWire-подобный коннектом (кольцо + случайные
//! связи, ~38 синапсов/нейрон, 80/20 возбуждение/торможение, веса 0.1–2.0).

#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pillars_connectome::{
    hh_population_pacemaking, hh_population_standard, sim_hh_pillars, ConnectomeData,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

// ─── Fly-like connectome ─────────────────────────────────────

/// Кольцо + случайные дальние связи (~38 синапсов/нейрон), 80/20 exc/inh.
fn fly_connectome(n: usize, n_rand: usize) -> ConnectomeData {
    let mut rng = StdRng::seed_from_u64(42);
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut nt_types = Vec::new();
    let mut cnt = 0i32;

    for i in 0..n {
        // Кольцо: i → i+1 (возбуждающее)
        col_indices.push(((i + 1) % n) as i32);
        values.push(0.5);
        nt_types.push(1.0);
        cnt += 1;

        // Случайные связи: 70% дальние, 30% локальные
        for _ in 0..n_rand {
            let target = if rng.gen_bool(0.7) {
                (i + n / 4 + rng.gen_range(0..n / 2)) % n
            } else {
                (i + rng.gen_range(1..=32)) % n
            };
            col_indices.push(target as i32);
            values.push(rng.gen_range(0.1..2.0));
            // 80% возбуждающие, 20% тормозящие
            nt_types.push(if rng.gen_bool(0.8) { 1.0 } else { -1.0 });
            cnt += 1;
        }
        row_ptr[i + 1] = cnt;
    }

    ConnectomeData {
        n_neurons: n,
        n_synapses: values.len(),
        row_ptr,
        col_indices,
        values,
        nt_types,
        receptor_types: vec![],
        groups: vec![],
        phenotype: pillars_connectome::NtPhenotype::default(),
    }
}

// ─── Criterion benchmarks ────────────────────────────────────

fn bench_hh_standard(c: &mut Criterion) {
    let n = 1024;
    let dt = 0.1;
    let n_steps = 1000;

    let mut g = c.benchmark_group("hh_standard");
    g.throughput(Throughput::Elements((n * n_steps) as u64));

    g.bench_function("pillars_HHModel_standard", |b| {
        let conn = fly_connectome(n, 37);
        let mut pop = hh_population_standard(n, dt);
        let ext = vec![15.0f32; n];
        b.iter(|| {
            let buf = sim_hh_pillars(&conn, &mut pop, &ext, n_steps, 1.0);
            black_box(buf);
        });
    });

    g.finish();
}

fn bench_hh_pacemaking(c: &mut Criterion) {
    let n = 1024;
    let dt = 0.1;
    let n_steps = 1000;

    let mut g = c.benchmark_group("hh_pacemaking");
    g.throughput(Throughput::Elements((n * n_steps) as u64));

    g.bench_function("pillars_HHModel_pacemaking", |b| {
        let conn = fly_connectome(n, 37);
        let mut pop = hh_population_pacemaking(n, dt);
        let ext = vec![0.0f32; n]; // CaV-T drives the oscillation on its own
        b.iter(|| {
            let buf = sim_hh_pillars(&conn, &mut pop, &ext, n_steps, 1.0);
            black_box(buf);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_hh_standard, bench_hh_pacemaking);
criterion_main!(benches);
