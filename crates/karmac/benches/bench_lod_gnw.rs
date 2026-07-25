//! Benchmarks: GNW forward (old alloc+rayon vs new zero-alloc) + CarmackBSP update_lod (old O(N×M) vs new octree).
//!
//! Usage:
//!   cargo bench -p karmac --bench bench_lod_gnw

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use karmac::gnw::GlobalWorkspace;
use karmac::lod::{CarmackBsp, LodLevel};

// ─── Helpers ───────────────────────────────────────────────────────────────

/// Uniform grid with stride `step` between adjacent points.
fn grid_positions(n: usize, step: f32) -> Vec<(f32, f32, f32)> {
    let side = (n as f32).cbrt().ceil() as usize;
    let mut pts = Vec::with_capacity(n);
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if pts.len() >= n {
                    break;
                }
                pts.push((i as f32 * step, j as f32 * step, k as f32 * step));
            }
        }
    }
    pts
}

/// A cluster of spikes near the origin — realistic activity wavefront.
fn local_spike_indices(n_neurons: usize, n_spikes: usize) -> Vec<usize> {
    (0..n_spikes.min(n_neurons)).collect() // first N neurons = near origin
}

// ─── GNW Bench: old (alloc+rayon) vs new (zero-alloc) ────────────────────

fn gnw_forward_old(gnw: &GlobalWorkspace, r: &mut [f32]) {
    let n = gnw.n_neurons();
    let ws = gnw.workspace_size();

    let mut gates = vec![0.0f32; ws];
    for i in 0..ws {
        let row_start = i * n;
        let row = gnw.gate_weights();
        gates[i] = row[row_start..row_start + n]
            .iter()
            .zip(r.iter())
            .map(|(w, x)| w * x)
            .sum();
    }

    let max_gate = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for g in gates.iter_mut() {
        *g = (*g - max_gate).exp();
        sum += *g;
    }
    let inv_sum = 1.0 / sum;
    for g in gates.iter_mut() {
        *g *= inv_sum;
    }

    // HEAP ALLOC every call
    let delta: Vec<f32> = (0..n)
        .map(|i| {
            let row = gnw.broadcast_weights();
            row[i * ws..(i + 1) * ws]
                .iter()
                .zip(gates.iter())
                .map(|(w, g)| w * g)
                .sum()
        })
        .collect();

    for (r_i, d_i) in r.iter_mut().zip(delta.iter()) {
        *r_i += d_i;
    }
}

fn bench_gnw(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnw_forward");

    for &n in &[128, 512, 2048] {
        let ws = (n / 16).max(8).min(128);
        let mut gnw = GlobalWorkspace::new(n, ws);
        let mut r = vec![0.1; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(format!("old_n{}", n), |b| {
            b.iter(|| {
                gnw_forward_old(&gnw, &mut r);
                black_box(&r);
            })
        });

        group.bench_function(format!("new_n{}", n), |b| {
            b.iter(|| {
                gnw.forward(&mut r);
                black_box(&r);
            })
        });
    }

    group.finish();
}

// ─── Carmack BSP Bench: old (O(N×M)) vs new (octree) ────────────────────

/// Old O(N×M) BSP update — пишет в bsp.lod чтобы избежать dead-code elimination.
fn bsp_old_update(bsp: &mut CarmackBsp, spike_indices: &[usize]) {
    let spike_positions: Vec<(f32, f32, f32)> = spike_indices
        .iter()
        .filter_map(|&idx| {
            let p = bsp.positions.get(idx)?;
            Some((p.x, p.y, p.z))
        })
        .collect();

    if spike_positions.is_empty() {
        return;
    }

    let dists = bsp.lod_distances;
    for i in 0..bsp.positions.len() {
        let p = &bsp.positions[i];
        let mut min_d2 = f32::MAX;
        for &(sx, sy, sz) in &spike_positions {
            let dx = p.x - sx;
            let dy = p.y - sy;
            let dz = p.z - sz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        if min_d2 != f32::MAX {
            let d = min_d2.sqrt();
            bsp.lod[i] = if d <= dists[0] {
                LodLevel::HhLut
            } else if d <= dists[1] {
                LodLevel::Izhikevich
            } else if d <= dists[2] {
                LodLevel::Lif
            } else {
                LodLevel::Tanh
            };
        }
    }
}

fn bench_bsp(c: &mut Criterion) {
    // Сценарий 1: малая карта (100×100×100), всплеск покрывает всё
    // Octree overhead > benefit, т.к. почти все нейроны в радиусе
    for &(n, label) in &[(100, "small"), (1000, "medium")] {
        let positions = grid_positions(n, 5.0); // stride 5 → bounding box ~40-80
        let n_spikes = (n / 20).max(3);
        let spikes = local_spike_indices(n, n_spikes);

        let mut old = CarmackBsp::new(&positions);
        let mut new = CarmackBsp::new(&positions);

        let mut group = c.benchmark_group(format!("bsp_{}", label));

        group.bench_function("old_O(N×M)", |b| {
            b.iter(|| {
                bsp_old_update(&mut old, &spikes);
                black_box(&old.lod);
            })
        });
        group.bench_function("new_octree", |b| {
            b.iter(|| {
                new.update_lod(&spikes);
                black_box(&new.lod);
            })
        });

        group.finish();
    }

    // Сценарий 2: большая карта (1000×1000×1000), локальный всплеск
    // Octree должен отсекать >99% нейронов
    for &(n, label) in &[(5000, "large_5k"), (20000, "large_20k")] {
        let positions = grid_positions(n, 50.0); // stride 50 → bounding box ~500-800
        let n_spikes = (n / 100).max(5); // fewer spikes, localized
        let spikes = local_spike_indices(n, n_spikes);

        let mut old = CarmackBsp::new(&positions);
        let mut new = CarmackBsp::new(&positions);

        let mut group = c.benchmark_group(format!("bsp_{}", label));

        group.bench_function("old_O(N×M)", |b| {
            b.iter(|| {
                bsp_old_update(&mut old, &spikes);
                black_box(&old.lod);
            })
        });
        group.bench_function("new_octree", |b| {
            b.iter(|| {
                new.update_lod(&spikes);
                black_box(&new.lod);
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_gnw, bench_bsp);
criterion_main!(benches);
