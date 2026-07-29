//! RCM benchmark: сравнивает bandwidth и время CSR propagate до и после RCM.
//!
//! Синтетический коннектом: 100K нейронов, ~30 синапсов на нейрон.
//! Топология: кольцо (локальные связи) + случайные длинные связи (плохой ordering).
//!
//! Измерения:
//! - Bandwidth (max |i-j|) до/после RCM
//! - Mean bandwidth до/после RCM
//! - Время csr_propagate_from_queue (5% firing rate) до/после RCM

#![allow(non_snake_case)]
#![cfg(feature = "connectome")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, Rng, SeedableRng};

// ─── Синтетический коннектом ────────────────────────────

struct SynthConnectome {
    n: usize,
    row_ptr: Vec<i32>,
    col_indices: Vec<i32>,
    values: Vec<f32>,
    nt_types: Vec<f32>,
}

/// Генерирует синтетический CSR коннектом с плохим ordering.
///
/// Каждый нейрон i имеет:
/// - 1 кольцевое ребро: i → i+1 (локальное)
/// - ~10 случайных длинных рёбер (remote targets)
///
/// Из-за случайных целей bandwidth изначально большой (~n/2).
/// RCM должен уменьшить bandwidth в 10-50×.
fn generate_connectome(n: usize, n_random_syn: usize) -> SynthConnectome {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut syn_count = 0i32;

    for i in 0..n {
        // Кольцевое ребро: i → i+1 (с гарантированной локальностью)
        let ring_target = (i + 1) % n;
        col_indices.push(ring_target as i32);
        values.push(0.5);
        syn_count += 1;

        // Случайные "дальние" синапсы (создают плохой bandwidth)
        for _ in 0..n_random_syn {
            // С вероятностью 70% — действительно далеко, 30% — локально
            let target = if rng.random_bool(0.7) {
                // Дальний: на расстоянии n/4..3n/4
                let offset = n / 4 + rng.random_range(0..n / 2);
                (i + offset) % n
            } else {
                // Локальный: ±32
                let offset = rng.random_range(1..=32);
                (i + offset) % n
            };
            col_indices.push(target as i32);
            values.push(rng.random_range(0.1..2.0));
            syn_count += 1;
        }

        row_ptr[i + 1] = syn_count;
    }

    let n_syn = values.len();
    let nt_types = vec![1.0f32; n]; // все возбуждающие

    SynthConnectome {
        n,
        row_ptr,
        col_indices,
        values,
        nt_types,
    }
}

/// Перевести SynthConnectome в ConnectomeData (для RCM).
fn to_connectome_data(sc: &SynthConnectome) -> taas_connectome::ConnectomeData {
    taas_connectome::ConnectomeData {
        n_neurons: sc.n,
        n_synapses: sc.values.len(),
        row_ptr: sc.row_ptr.clone(),
        col_indices: sc.col_indices.clone(),
        values: sc.values.clone(),
        nt_types: sc.nt_types.clone(),
        groups: vec![],
    }
}

/// Запустить csr_propagate_from_queue для заданного числа спайков.
fn bench_propagate(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    n_neurons: usize,
    n_spikes: usize,
) -> f64 {
    use taas_connectome::{csr_propagate_from_queue, SpikeQueue};

    let mut queue = SpikeQueue::new(n_spikes.max(256));
    let mut I_buf = vec![0.0f32; n_neurons];

    // Push random spike IDs
    let mut rng = SmallRng::seed_from_u64(123);
    for _ in 0..n_spikes {
        queue.push(rng.random_range(0..n_neurons) as u32);
    }

    // Measure
    let t0 = std::time::Instant::now();
    let total = csr_propagate_from_queue(
        row_ptr,
        col_indices,
        values,
        nt_types,
        &mut queue,
        &mut I_buf,
        1.0,
    );
    let elapsed = t0.elapsed();

    // Prevent optimization: sum I_buf
    let _sum: f32 = I_buf.iter().sum::<f32>() + total as f32;
    black_box(_sum);

    elapsed.as_secs_f64()
}

// ═══════════════════════════════════════════════════════════
//  Criterion benchmarks
// ═══════════════════════════════════════════════════════════

fn bench_rcm_bandwidth(c: &mut Criterion) {
    let n = 100_000;
    let n_random_syn = 10; // ~11 syn/neuron total

    let sc = generate_connectome(n, n_random_syn);
    let conn = to_connectome_data(&sc);

    // Bandwidth до RCM
    let bw_before = taas_connectome::bandwidth(&conn.row_ptr, &conn.col_indices);
    let mean_bw_before = taas_connectome::mean_bandwidth(&conn.row_ptr, &conn.col_indices);

    // RCM (это preprocessing, измеряем отдельно)
    let t0 = std::time::Instant::now();
    let adj = taas_connectome::build_adjacency(&conn);
    let perm = taas_connectome::reverse_cuthill_mckee(&adj);
    let (new_rp, new_ci, _, _) = taas_connectome::apply_permutation_to_csr(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        &conn.nt_types,
        &perm,
    );
    let rcm_time = t0.elapsed();

    // Bandwidth после RCM
    let bw_after = taas_connectome::bandwidth(&new_rp, &new_ci);
    let mean_bw_after = taas_connectome::mean_bandwidth(&new_rp, &new_ci);

    println!(
        "\n═══ RCM Benchmark: {n} neurons, ~{} synapses/neuron ═══",
        (sc.values.len() + n - 1) / n
    );
    println!("  RCM preprocessing: {:.3}s", rcm_time.as_secs_f64());
    println!("  Synapses: {}", sc.values.len());
    println!();
    println!("  Bandwidth BEFORE: {bw_before:>10}");
    println!("  Bandwidth AFTER:  {bw_after:>10}");
    println!(
        "  Reduction:        {:.1}×",
        bw_before as f64 / bw_after.max(1) as f64
    );
    println!();
    println!("  Mean BW BEFORE:   {mean_bw_before:>10.2}");
    println!("  Mean BW AFTER:    {mean_bw_after:>10.2}");
    println!(
        "  Reduction:        {:.1}×",
        mean_bw_before / mean_bw_after.max(1e-10)
    );
}

fn bench_rcm_propagate(c: &mut Criterion) {
    let n = 100_000;
    let n_random_syn = 10;
    let sc = generate_connectome(n, n_random_syn);
    let conn = to_connectome_data(&sc);

    // RCM preprocessing (один раз)
    let adj = taas_connectome::build_adjacency(&conn);
    let perm = taas_connectome::reverse_cuthill_mckee(&adj);
    let (new_rp, new_ci, _, _) = taas_connectome::apply_permutation_to_csr(
        &conn.row_ptr,
        &conn.col_indices,
        &conn.values,
        &conn.nt_types,
        &perm,
    );

    // Firing rate: 5% = 5000 spikes
    let n_spikes = n / 20;

    let mut g = c.benchmark_group("csr_propagate_rcm");
    g.throughput(Throughput::Elements(n_spikes as u64));

    g.bench_function("before_rcm", |b| {
        b.iter(|| {
            let _ = bench_propagate(
                &conn.row_ptr,
                &conn.col_indices,
                &conn.values,
                &conn.nt_types,
                n,
                n_spikes,
            );
        })
    });

    g.bench_function("after_rcm", |b| {
        b.iter(|| {
            let _ = bench_propagate(
                &new_rp,
                &new_ci,
                &sc.values, // values unchanged by RCM (same values, just reordered)
                &sc.nt_types,
                n,
                n_spikes,
            );
        })
    });

    g.finish();

    // Print detailed results
    println!(
        "\n═══ CSR Propagate: {n} neurons, {n_spikes} spikes ({:.1}% firing) ═══",
        n_spikes as f64 / n as f64 * 100.0
    );
    println!("  Single-pass timings (not criterion):");

    // Warmup
    for _ in 0..3 {
        bench_propagate(
            &conn.row_ptr,
            &conn.col_indices,
            &conn.values,
            &conn.nt_types,
            n,
            n_spikes,
        );
        bench_propagate(&new_rp, &new_ci, &sc.values, &sc.nt_types, n, n_spikes);
    }

    // Timed runs
    let mut t_before = 0.0f64;
    let mut t_after = 0.0f64;
    let n_runs = 20;
    for _ in 0..n_runs {
        t_before += bench_propagate(
            &conn.row_ptr,
            &conn.col_indices,
            &conn.values,
            &conn.nt_types,
            n,
            n_spikes,
        );
        t_after += bench_propagate(&new_rp, &new_ci, &sc.values, &sc.nt_types, n, n_spikes);
    }
    t_before /= n_runs as f64;
    t_after /= n_runs as f64;

    println!("  BEFORE RCM: {:.3}ms", t_before * 1000.0);
    println!("  AFTER RCM:  {:.3}ms", t_after * 1000.0);
    if t_after > 0.0 {
        println!("  Speedup:    {:.1}×", t_before / t_after);
    }
}

criterion_group!(benches, bench_rcm_bandwidth, bench_rcm_propagate);
criterion_main!(benches);
