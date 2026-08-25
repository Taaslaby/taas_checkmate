//! bench_diffusion.rs — CPU vs GPU diffusion benchmark
//!
//! ```bash
//! cargo run -p prometheus-grow --example bench_diffusion --features prometheus-grow/gpu
//! ```

use prometheus_grow::{DiffusionBackend, Grid, Vec3};
use std::time::Instant;

const NX: usize = 100;
const NY: usize = 100;
const NZ: usize = 100;
const CELL: f64 = 5.0;
const DT: f64 = 0.5;
const DIFF: f64 = 0.1;
const DEG: f64 = 0.01;
const LO: f64 = 0.0;
const HI: f64 = 100.0;
const STEPS: usize = 50;

fn main() {
    let grid = Grid::new(NX, NY, NZ, CELL, Vec3::ZERO);
    let vol = grid.volume();
    println!("┌──────────────────────────────────────────────┐");
    println!("│ Diffusion benchmark: CPU vs GPU (wgpu)        │");
    println!("├──────────────────────────────────────────────┤");
    println!(
        "│ Grid: {NX}×{NY}×{NZ} = {} cells ({:.1} MB)",
        vol,
        (vol * 8) as f64 / 1_048_576.0
    );
    println!("│ D={DIFF}, deg={DEG}, dt={DT}, steps={STEPS}");
    println!("└──────────────────────────────────────────────┘");

    let init_field = || {
        let mut f = vec![0.0f64; vol];
        let cx = Vec3::new(
            (NX / 2) as f64 * CELL,
            (NY / 2) as f64 * CELL,
            (NZ / 2) as f64 * CELL,
        );
        for i in 0..NX {
            for j in 0..NY {
                for k in 0..NZ {
                    let pos = Vec3::new(i as f64 * CELL, j as f64 * CELL, k as f64 * CELL);
                    let d2 =
                        (pos.x - cx.x).powi(2) + (pos.y - cx.y).powi(2) + (pos.z - cx.z).powi(2);
                    f[grid.flat(i, j, k)] = 10.0 * (-d2 / (2.0 * 100.0f64.powi(2))).exp();
                }
            }
        }
        f
    };

    // ── CPU ──
    let mut field = init_field();
    let cpu = DiffusionBackend::cpu();
    let mut c2 = vec![0.0f64; vol];
    let start = Instant::now();
    for _ in 0..STEPS {
        cpu.diffuse(&grid, &field, &mut c2, DT, DIFF, DEG, LO, HI);
        copy_boundary(&grid, &mut c2);
        std::mem::swap(&mut field, &mut c2);
    }
    let cpu_elapsed = start.elapsed();
    println!("\n─── CPU (rayon) ───");
    println!(
        "  {STEPS} steps: {:?} ({:.1} ms/step)",
        cpu_elapsed,
        cpu_elapsed.as_secs_f64() * 1000.0 / STEPS as f64
    );

    // ── GPU ──
    #[cfg(feature = "gpu")]
    {
        println!("\n─── GPU (wgpu) ───");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let gpu = rt.block_on(async { DiffusionBackend::new().await });
        let mut field = init_field();
        let mut c2 = vec![0.0f64; vol];
        let start = Instant::now();
        for _ in 0..STEPS {
            gpu.diffuse(&grid, &field, &mut c2, DT, DIFF, DEG, LO, HI);
            copy_boundary(&grid, &mut c2);
            std::mem::swap(&mut field, &mut c2);
        }
        let gpu_elapsed = start.elapsed();
        println!(
            "  {STEPS} steps: {:?} ({:.1} ms/step)",
            gpu_elapsed,
            gpu_elapsed.as_secs_f64() * 1000.0 / STEPS as f64
        );
        let speedup = cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64().max(1e-6);
        println!("\n  Speedup: {:.1}×", speedup);
        if speedup > 2.0 {
            println!("  GPU dominates!");
        } else if speedup > 1.0 {
            println!("  GPU is slightly faster.");
        } else {
            println!("  GPU overhead > gain on 100³.");
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("\n─── GPU (wgpu) ───");
        println!("  feature \"gpu\" is not enabled. Run with --features prometheus-grow/gpu");
    }
}

fn copy_boundary(grid: &Grid, c2: &mut [f64]) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1 {
                    let ii = if i == 0 {
                        1
                    } else if i == nx - 1 {
                        nx - 2
                    } else {
                        i
                    };
                    let jj = if j == 0 {
                        1
                    } else if j == ny - 1 {
                        ny - 2
                    } else {
                        j
                    };
                    let kk = if k == 0 {
                        1
                    } else if k == nz - 1 {
                        nz - 2
                    } else {
                        k
                    };
                    c2[grid.flat(i, j, k)] = c2[grid.flat(ii, jj, kk)];
                }
            }
        }
    }
}
