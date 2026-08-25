//! GPU/CPU diffusion parity tests (feature `gpu`).
#![cfg(feature = "gpu")]

use prometheus_grow::{Bounds, DiffusionBackend, GpuContext, Grid, Substance, Vec3, World};

fn diffuse_cpu_ref(
    grid: &Grid,
    field: &[f64],
    out: &mut [f64],
    dt: f64,
    diff: f64,
    deg: f64,
    lo: f64,
    hi: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx2 = grid.cell * grid.cell;
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let idx = grid.flat(i, j, k);
                let lap = (field[grid.flat(i - 1, j, k)]
                    + field[grid.flat(i + 1, j, k)]
                    + field[grid.flat(i, j - 1, k)]
                    + field[grid.flat(i, j + 1, k)]
                    + field[grid.flat(i, j, k - 1)]
                    + field[grid.flat(i, j, k + 1)]
                    - 6.0 * field[idx])
                    / dx2;
                out[idx] = (field[idx] + dt * diff * lap - dt * deg * field[idx]).clamp(lo, hi);
            }
        }
    }
    for idx in 0..field.len() {
        let k = idx % grid.nz;
        let j = (idx / grid.nz) % grid.ny;
        let i = idx / (grid.nz * grid.ny);
        if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
            out[idx] = field[idx];
        }
    }
}

async fn gpu_ctx() -> GpuContext {
    GpuContext::new()
        .await
        .expect("wgpu adapter/device required for parity test")
}

#[test]
fn gpu_diffusion_matches_cpu_within_f32_epsilon() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let ctx = gpu_ctx().await;
        let n = 12;
        let cell = 5.0;
        let grid = Grid::new(n, n, n, cell, Vec3::new(0.0, 0.0, 0.0));
        let vol = grid.volume();

        let mut field = vec![0.0f64; vol];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = (i as f64 - n as f64 / 2.0) * cell;
                    let y = (j as f64 - n as f64 / 2.0) * cell;
                    let z = (k as f64 - n as f64 / 2.0) * cell;
                    let r2 = x * x + y * y + z * z;
                    field[grid.flat(i, j, k)] = (-r2 / (2.0 * 15.0f64.powi(2))).exp();
                }
            }
        }

        let dt = 0.05;
        let diff = 2.0;
        let deg = 0.01;
        let (lo, hi) = (0.0, 5.0);

        let mut gpu_out = vec![0.0f64; vol];
        ctx.diffuse(&grid, &field, &mut gpu_out, dt, diff, deg, lo, hi);

        let mut cpu_out = vec![0.0f64; vol];
        diffuse_cpu_ref(&grid, &field, &mut cpu_out, dt, diff, deg, lo, hi);

        let mut max_rel = 0.0f64;
        let mut max_abs = 0.0f64;
        for i in 1..n - 1 {
            for j in 1..n - 1 {
                for k in 1..n - 1 {
                    let idx = grid.flat(i, j, k);
                    let a = gpu_out[idx];
                    let b = cpu_out[idx];
                    assert!(a.is_finite() && b.is_finite(), "non-finite at {idx}");
                    let abs_diff = (a - b).abs();
                    max_abs = max_abs.max(abs_diff);
                    if b.abs() > 1e-6 {
                        max_rel = max_rel.max(abs_diff / b.abs());
                    }
                }
            }
        }
        assert!(
            max_rel < 1e-3,
            "GPU/CPU relative error {max_rel:.2e} exceeds f32 envelope"
        );
        assert!(max_abs < 1e-4, "GPU/CPU absolute error {max_abs:.2e} too large");
    });
}

#[test]
fn gpu_preserves_boundary_cells_exactly() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let ctx = gpu_ctx().await;
        let n = 8;
        let grid = Grid::new(n, n, n, 1.0, Vec3::ZERO);
        let vol = grid.volume();
        let field: Vec<f64> = (0..vol).map(|i| (i % 7) as f64).collect();
        let mut out = vec![0.0f64; vol];
        ctx.diffuse(&grid, &field, &mut out, 0.1, 1.0, 0.0, 0.0, 100.0);
        for idx in 0..vol {
            let k = idx % grid.nz;
            let j = (idx / grid.nz) % grid.ny;
            let i = idx / (grid.nz * grid.ny);
            if i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1 {
                assert_eq!(
                    out[idx], field[idx],
                    "boundary cell {idx} was modified by GPU pass"
                );
            }
        }
    });
}

#[test]
fn gpu_mass_conservation_with_zero_degradation() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let ctx = gpu_ctx().await;
        let n = 10;
        let grid = Grid::new(n, n, n, 2.0, Vec3::ZERO);
        let vol = grid.volume();
        let mut field = vec![0.0f64; vol];
        for (i, v) in field.iter_mut().enumerate() {
            *v = ((i * 37) % 11) as f64 / 11.0;
        }
        let before: f64 = field.iter().sum();

        let mut out = vec![0.0f64; vol];
        ctx.diffuse(&grid, &field, &mut out, 0.01, 1.0, 0.0, 0.0, 100.0);
        let after: f64 = out.iter().sum();

        let drift = (after - before).abs() / before.max(1e-9);
        assert!(
            drift < 0.05,
            "mass drifted by {:.2}% — leak in stencil",
            drift * 100.0
        );
    });
}

#[test]
fn world_backend_gpu_agrees_with_cpu_backend() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let make_world = |backend: DiffusionBackend| {
            let grid = Grid::new(10, 10, 10, 3.0, Vec3::new(-15.0, -15.0, -15.0));
            let bounds = Bounds {
                min: Vec3::new(-15.0, -15.0, -15.0),
                max: Vec3::new(15.0, 15.0, 15.0),
            };
            let mut w = World::new(grid, 0.05, bounds, 42);
            w.add_substance(Substance::new("A", 1.0, 0.0, &w.grid, vec![]));
            w.set_diffusion_backend(backend);
            w.add_substance_at("A", Vec3::ZERO, 2.0);
            w
        };

        let mut cpu_w = make_world(DiffusionBackend::Cpu);
        let mut gpu_w = make_world(DiffusionBackend::Gpu(gpu_ctx().await));

        for _ in 0..20 {
            cpu_w.step();
            gpu_w.step();
        }

        let grid = cpu_w.grid;
        let mut max_diff = 0.0f64;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let p = Vec3::new(
                        grid.origin.x + (i as f64 + 0.5) * grid.cell,
                        grid.origin.y + (j as f64 + 0.5) * grid.cell,
                        grid.origin.z + (k as f64 + 0.5) * grid.cell,
                    );
                    let a = cpu_w.concentration("A", p);
                    let b = gpu_w.concentration("A", p);
                    max_diff = max_diff.max((a - b).abs());
                }
            }
        }
        assert!(
            max_diff < 1e-2,
            "World-level CPU/GPU fields diverge: max diff {max_diff:.2e}"
        );
    });
}
