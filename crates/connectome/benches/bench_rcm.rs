//! RCM benchmark: bandwidth и время CSR propagate до/после RCM.
//!
//! Синтетический коннектом: 100K нейронов, ~11 синапсов/нейрон.
//! Топология: кольцо + случайные длинные связи (плохой ordering).

#![allow(non_snake_case)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use taas_connectome::{
    apply_permutation_to_csr, bandwidth, build_adjacency, csr_propagate_from_queue, mean_bandwidth,
    reverse_cuthill_mckee, ConnectomeData, SpikeQueue,
};

// ─── Синтетический коннектом ────────────────────────────

/// Сгенерировать CSR с 100K нейронов: кольцо + случайные длинные связи.
fn generate_connectome(n: usize, n_rand: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut cnt = 0i32;

    for i in 0..n {
        // Кольцо: i → i+1
        col_indices.push(((i + 1) % n) as i32);
        values.push(0.5);
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
            cnt += 1;
        }
        row_ptr[i + 1] = cnt;
    }

    let nt_types = vec![1.0f32; n];
    (row_ptr, col_indices, values, nt_types)
}

fn make_conn(rp: &[i32], ci: &[i32], vals: &[f32], nt: &[f32]) -> ConnectomeData {
    let n = rp.len() - 1;
    ConnectomeData {
        n_neurons: n,
        n_synapses: vals.len(),
        row_ptr: rp.to_vec(),
        col_indices: ci.to_vec(),
        values: vals.to_vec(),
        nt_types: nt.to_vec(),
        receptor_types: vec![],
        groups: vec![],
    }
}

/// Один прогон csr_propagate_from_queue: push n_spikes → propagate.
fn run_propagate(
    rp: &[i32],
    ci: &[i32],
    vals: &[f32],
    nt: &[f32],
    n: usize,
    n_spikes: usize,
) -> f64 {
    let mut q = SpikeQueue::new(n_spikes.max(256));
    let mut I_buf = vec![0.0f32; n];
    let mut rng = StdRng::seed_from_u64(123);
    for _ in 0..n_spikes {
        q.push(rng.gen_range(0..n) as u32);
    }
    let t0 = std::time::Instant::now();
    let total = csr_propagate_from_queue(rp, ci, vals, nt, &mut q, &mut I_buf, 1.0);
    let elapsed = t0.elapsed();
    // Prevent dead-code elimination
    let _drain: f32 = I_buf.iter().sum::<f32>() + total as f32;
    black_box(_drain);
    elapsed.as_secs_f64()
}

// ═══════════════════════════════════════════════════════════
//  Criterion benchmarks
// ═══════════════════════════════════════════════════════════

fn bench_rcm(c: &mut Criterion) {
    let n = 100_000;
    let n_rand = 10;

    // Generate
    let (rp, ci, vals, nt) = generate_connectome(n, n_rand);
    let conn = make_conn(&rp, &ci, &vals, &nt);
    let n_syn = vals.len();

    // Bandwidth до RCM
    let bw_before = bandwidth(&rp, &ci);
    let mbw_before = mean_bandwidth(&rp, &ci);

    // RCM preprocessing
    let t0 = std::time::Instant::now();
    let adj = build_adjacency(&conn);
    let perm = reverse_cuthill_mckee(&adj);
    let (new_rp, new_ci, _, _) = apply_permutation_to_csr(&rp, &ci, &vals, &nt, &perm);
    let rcm_time = t0.elapsed();

    // Bandwidth после RCM
    let bw_after = bandwidth(&new_rp, &new_ci);
    let mbw_after = mean_bandwidth(&new_rp, &new_ci);

    print!(
        "\n═══ RCM Benchmark: {n} neurons, ~{syn_avg} syn/neuron ═══\n",
        syn_avg = (n_syn + n / 2) / n
    );
    println!("  RCM time:      {:.3}s", rcm_time.as_secs_f64());
    println!("  Total syn:     {n_syn}");
    println!();
    println!("  BW BEFORE:     {bw_before:>10}");
    println!("  BW AFTER:      {bw_after:>10}");
    println!(
        "  BW reduction:  {:.1}×",
        bw_before as f64 / bw_after.max(1) as f64
    );
    println!("  Mean BW BEFORE: {mbw_before:>9.2}");
    println!("  Mean BW AFTER:  {mbw_after:>9.2}");
    println!(
        "  Mean reduction: {:.1}×",
        mbw_before / mbw_after.max(1e-10)
    );

    // ─── Propagate benchmark (criterion) ────────

    let p50 = n / 20; // 5% firing
    let p10 = n / 10; // 10% firing

    let mut g = c.benchmark_group("propagate_5pct");
    g.throughput(Throughput::Elements(p50 as u64));
    g.bench_function("before_rcm", |b| {
        b.iter(|| {
            run_propagate(&rp, &ci, &vals, &nt, n, p50);
        })
    });
    g.bench_function("after_rcm", |b| {
        b.iter(|| {
            run_propagate(&new_rp, &new_ci, &vals, &nt, n, p50);
        })
    });
    g.finish();

    let mut g = c.benchmark_group("propagate_10pct");
    g.throughput(Throughput::Elements(p10 as u64));
    g.bench_function("before_rcm", |b| {
        b.iter(|| {
            run_propagate(&rp, &ci, &vals, &nt, n, p10);
        })
    });
    g.bench_function("after_rcm", |b| {
        b.iter(|| {
            run_propagate(&new_rp, &new_ci, &vals, &nt, n, p10);
        })
    });
    g.finish();

    // ─── Wall-clock timings ────────────────────

    print!("\n─── Wall-clock ({n} neurons) ───\n");
    for (name, n_spikes) in [("5%", p50), ("10%", p10)] {
        // Warmup
        for _ in 0..3 {
            run_propagate(&rp, &ci, &vals, &nt, n, n_spikes);
            run_propagate(&new_rp, &new_ci, &vals, &nt, n, n_spikes);
        }
        let mut t_bef = 0.0f64;
        let mut t_aft = 0.0f64;
        let n_runs = 30;
        for _ in 0..n_runs {
            t_bef += run_propagate(&rp, &ci, &vals, &nt, n, n_spikes);
            t_aft += run_propagate(&new_rp, &new_ci, &vals, &nt, n, n_spikes);
        }
        t_bef /= n_runs as f64;
        t_aft /= n_runs as f64;
        let speedup = if t_aft > 0.0 { t_bef / t_aft } else { 0.0 };
        println!(
            "  {name:>3} spikes: before={:.3}ms  after={:.3}ms  speedup={:.1}×",
            t_bef * 1000.0,
            t_aft * 1000.0,
            speedup
        );
    }
    println!();
}

criterion_group!(benches, bench_rcm);
criterion_main!(benches);
