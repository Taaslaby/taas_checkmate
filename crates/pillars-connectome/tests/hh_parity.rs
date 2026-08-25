//! # HH Integration Tests — pillars-only (legacy `step_hh_lut` removed)
//!
//! After migrating `crates/connectome` fully onto `pillars::HHModel +
//! NeuralPopulation` (standard AND pacemaking CaV-T), the old-vs-new parity
//! contract is superseded by direct pillars guarantees:
//!
//! 1. **SIMD/mixed consistency** — `run_neuron` n=8 (pure SIMD) must match
//!    n=9 (SIMD + scalar tail) neuron-for-neuron.
//! 2. **Pacemaking** — with `g_CaT = 0.5, E_CaT = +120` the neuron fires
//!    **spontaneously** with zero input (T-type Ca²⁺ drives the oscillation).
//! 3. **Standard silence** — with `g_CaT = 0` the squid-axon HH stays quiet
//!    at rest (the CaT gate is present but its current is zero).
//!
//! Run: `cargo test -p pillars-connectome --test hh_parity`

#![allow(non_snake_case)]

use pillars_connectome::{
    hh_population_pacemaking, hh_population_standard, sim_hh_pillars, ConnectomeData,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

// ─── Connectome builders ─────────────────────────────────────

/// Пустой коннектом (без синапсов) — изолированные нейроны.
fn empty_connectome(n: usize) -> ConnectomeData {
    ConnectomeData {
        n_neurons: n,
        n_synapses: 0,
        row_ptr: vec![0i32; n + 1],
        col_indices: vec![],
        values: vec![],
        nt_types: vec![],
        receptor_types: vec![],
        groups: vec![],
        phenotype: pillars_connectome::NtPhenotype::default(),
    }
}

/// FlyWire-подобный коннектом: кольцо + случайные связи, 80/20 exc/inh.
fn fly_connectome(n: usize, n_rand: usize) -> ConnectomeData {
    let mut rng = StdRng::seed_from_u64(7);
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut nt_types = Vec::new();
    let mut cnt = 0i32;

    for i in 0..n {
        col_indices.push(((i + 1) % n) as i32);
        values.push(0.5);
        nt_types.push(1.0);
        cnt += 1;
        for _ in 0..n_rand {
            let target = if rng.gen_bool(0.7) {
                (i + n / 4 + rng.gen_range(0..n / 2)) % n
            } else {
                (i + rng.gen_range(1..=32)) % n
            };
            col_indices.push(target as i32);
            values.push(rng.gen_range(0.1..2.0));
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

fn per_neuron_counts(spike_buffer: &[i32], n: usize, n_steps: usize) -> Vec<u32> {
    let mut counts = vec![0u32; n];
    for step in 0..n_steps {
        for i in 0..n {
            if spike_buffer[step * n + i] != 0 {
                counts[i] += 1;
            }
        }
    }
    counts
}

// ─── 1. Pacemaking: спонтанный огонь без входа ──────────────

/// CaV-T (T-type Ca²⁺) заставляет HH спайковать сам по себе: с g_CaT=0.5,
/// E_CaT=+120 и нулевым внешним током нейроны осциллируют. Это контракт
/// `BrainConfig::hh_pacemaking`.
#[test]
fn test_hh_pacemaking_fires_spontaneously() {
    for &(n, dt, steps) in &[
        (8usize, 0.1f32, 2000usize),
        (16, 0.05, 4000),
        (32, 0.1, 2000),
    ] {
        let conn = empty_connectome(n);
        let mut pop = hh_population_pacemaking(n, dt);
        let ext = vec![0.0f32; n]; // НОЛЬ внешнего тока!
        let buf = sim_hh_pillars(&conn, &mut pop, &ext, steps, 1.0);
        let total: u32 = buf.iter().map(|&s| s as u32).sum();
        println!("pacemaking n={n} dt={dt}: total={total} spikes (I=0)");
        assert!(
            total > 0,
            "pacemaking n={n}: must fire spontaneously with I=0, got {total}"
        );
    }
}

// ─── 2. Standard: тишина без входа, огонь с током ───────────

/// Стандартный HH (g_CaT=0) не спайкует без тока и спайкует при I=15.
#[test]
fn test_hh_standard_behavior() {
    let n = 16;
    let conn = empty_connectome(n);
    let dt = 0.1;

    // Silent at rest
    let mut pop0 = hh_population_standard(n, dt);
    let buf0 = sim_hh_pillars(&conn, &mut pop0, &vec![0.0f32; n], 2000, 1.0);
    let total0: u32 = buf0.iter().map(|&s| s as u32).sum();
    println!("standard I=0: total={total0} spikes");
    assert_eq!(total0, 0, "standard HH must be silent with I=0");

    // Fires with suprathreshold current
    let mut pop15 = hh_population_standard(n, dt);
    let buf15 = sim_hh_pillars(&conn, &mut pop15, &vec![15.0f32; n], 1000, 1.0);
    let total15: u32 = buf15.iter().map(|&s| s as u32).sum();
    println!("standard I=15: total={total15} spikes");
    assert!(total15 > 0, "standard HH must fire with I=15");
}

// ─── 3. SIMD/mixed consistency: n=8 vs n=9 ──────────────────

/// Каждый нейрон должен вести себя одинаково в чистом SIMD-чанке (n=8) и в
/// смешанном SIMD+scalar-tail (n=9).
///
/// Допуск |Δ| ≤ 2 спайка/нейрон: детектор `edge_spike` ловит АП на растущем
/// фронте, поэтому на границе окна фаза может дать ±1 спайк (документированный
/// артефакт — как в оригинальном парити-контракте old-vs-new).
#[test]
fn test_hh_simd_mixed_consistency() {
    for &(n_simd, n_mixed) in &[(8usize, 9usize), (16, 17)] {
        let dt = 0.1;
        let conn_s = empty_connectome(n_simd);
        let conn_m = empty_connectome(n_mixed);

        let mut pop_s = hh_population_standard(n_simd, dt);
        let mut pop_m = hh_population_standard(n_mixed, dt);

        let steps = 1000;
        let ext_s = vec![15.0f32; n_simd];
        let ext_m = vec![15.0f32; n_mixed];

        let buf_s = sim_hh_pillars(&conn_s, &mut pop_s, &ext_s, steps, 1.0);
        let buf_m = sim_hh_pillars(&conn_m, &mut pop_m, &ext_m, steps, 1.0);

        let counts_s = per_neuron_counts(&buf_s, n_simd, steps);
        let counts_m = per_neuron_counts(&buf_m, n_mixed, steps);

        let max_diff = counts_s
            .iter()
            .zip(&counts_m)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        println!(
            "SIMD/mixed n={n_simd}/vs{n_mixed}: SIMD={} mixed={} max|Δ|={max_diff}",
            counts_s.iter().sum::<u32>(),
            counts_m.iter().sum::<u32>()
        );
        assert!(
            max_diff <= 2,
            "SIMD/mixed n={n_simd}: max|Δ| = {max_diff} > 2 — движок разошёлся!"
        );
    }
}

// ─── 4. Fly-сеть: стабильность счёта (smoke) ────────────────

/// Fly-подобная сеть на pillars — счёт спайков стабилен и положителен.
#[test]
fn test_hh_fly_network_pillars_smoke() {
    let n = 128;
    let conn = fly_connectome(n, 37);
    let mut pop = hh_population_standard(n, 0.1);
    let ext = vec![15.0f32; n];
    let buf = sim_hh_pillars(&conn, &mut pop, &ext, 500, 1.0);
    let total: u32 = buf.iter().map(|&s| s as u32).sum();
    println!("fly n={n}: total={total} spikes");
    assert!(total > 0, "fly network must fire with I=15");
}
