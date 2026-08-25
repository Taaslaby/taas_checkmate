//! bench_biodynamo_gpu.rs — CPU vs GPU: PROMETHEUS tissue benchmark
//!
//! The same simulation on CPU and GPU. Diffusion and total time are measured separately.
//! A 128³ grid — the GPU should give a real speedup.
//!
//! cargo run -p prometheus-grow --example bench_biodynamo_gpu --features prometheus-grow/gpu --release

use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, DiffusionBackend, Grid, MechanicsParams,
    Recognition, Secretor, SimpleCellCycle, Substance, Vec3, World,
};
use std::time::Instant;

const TARGET_CELLS: usize = 2048;
const STEPS: usize = 40;
const GRID_N: usize = 128;

fn build_world(grid: Grid, bounds: Bounds) -> World {
    let mut w = World::new(grid, 1.0, bounds, 42);
    w.max_cells = TARGET_CELLS;
    w.max_neurites = TARGET_CELLS;
    w.synapse_dist = 12.0;
    w.max_syn_per_cell = 8;
    w.synapse_tips_only = true;
    w.mechanics_params = MechanicsParams {
        adhesion_strength: 0.3,
        repulsion_strength: 1.0,
        adhesion_cutoff: 2.0,
        damping: 0.1,
        ..MechanicsParams::default()
    };

    let span = GRID_N as f64 * 20.0;
    let half = span / 2.0;
    w.add_substance(Substance::new(
        "netrin",
        0.1,
        0.01,
        &w.grid,
        vec![
            ArtificialGradient::Gaussian {
                axis: 0,
                max: 10.0,
                mean: half,
                sigma: half * 0.5,
            },
            ArtificialGradient::Gaussian {
                axis: 1,
                max: 10.0,
                mean: half,
                sigma: half * 0.5,
            },
            ArtificialGradient::Gaussian {
                axis: 2,
                max: 10.0,
                mean: half,
                sigma: half * 0.5,
            },
        ],
    ));

    let r_seed = half * 0.4;
    for i in 0..8 {
        let phi = i as f64 / 8.0 * std::f64::consts::TAU;
        let theta = (i as f64 + 0.5) / 8.0 * std::f64::consts::PI;
        let pos = Vec3::new(
            half + r_seed * theta.sin() * phi.cos(),
            half + r_seed * theta.sin() * phi.sin(),
            half + r_seed * theta.cos(),
        );
        let ct = if i < 4 {
            CellType::Excitatory
        } else {
            CellType::Inhibitory
        };
        let id = w.add_cell(pos, ct, 8.0);
        w.add_cell_module(id, Box::new(SimpleCellCycle::new(15.0, 10.0)));
        w.add_soma_module(id, Box::new(Secretor::new("netrin", 0.5)));
    }
    w
}

fn run_sim(w: &mut World, label: &str) -> (f64, f64, usize, usize) {
    println!("--- {} ---", label);
    w.mechanics_params.enabled = false;

    // Phase 1: division + diffusion
    let t0 = Instant::now();
    for t in 0..STEPS {
        w.step();
        if w.cells.len() >= TARGET_CELLS {
            for ci in 0..w.cells.len() {
                w.cells[ci].modules.clear();
            }
        }
        if (t + 1) % 10 == 0 {
            println!("  t={:3} cells={:5}", t + 1, w.cells.len());
        }
    }
    let div_time = t0.elapsed().as_secs_f64() * 1000.0;

    // Phase 2: mechanics + neurites + synapses
    w.mechanics_params.enabled = true;
    let nc = w.cells.len();
    for ci in 0..nc {
        let na = w.new_neurite(ci, true, 1.2);
        let mut chemo = ChemoAttraction::new("netrin");
        chemo.noise_weight = 0.08;
        chemo.gradient_weight = 1.0;
        chemo.stop_concentration = 7.0;
        w.add_neurite_module(ci, na, Box::new(chemo));
    }
    w.set_recognition(Recognition::all());

    let t1 = Instant::now();
    for t in 0..STEPS {
        w.step();
        if (t + 1) % 10 == 0 {
            let nn: usize = w.cells.iter().map(|c| c.neurites.len()).sum();
            println!(
                "  t={:3} neurites={:5} syn={:6}",
                STEPS + t + 1,
                nn,
                w.synapses.len()
            );
        }
    }
    let grow_time = t1.elapsed().as_secs_f64() * 1000.0;

    println!("  div={:.0}ms grow={:.0}ms", div_time, grow_time);
    (div_time, grow_time, w.cells.len(), w.synapses.len())
}

fn main() {
    let cell_size = 20.0;
    let span = cell_size * GRID_N as f64;
    let origin = Vec3::new(-span / 2.0, -span / 2.0, -span / 2.0);
    let grid = Grid::new(GRID_N, GRID_N, GRID_N, cell_size, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(span, span, span),
    };
    let vol = grid.volume();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  PROMETHEUS CPU vs GPU — tissue benchmark            ║");
    println!("╠════════════════════════════════════════════════════════╣");
    println!(
        "║  Grid: {}^3 = {} cells ({:.1} MB)    Target: {} cells ║",
        GRID_N,
        vol,
        (vol * 8) as f64 / 1_048_576.0,
        TARGET_CELLS
    );
    println!("╚════════════════════════════════════════════════════════╝");
    println!();

    // ── CPU ──
    let mut w_cpu = build_world(grid, bounds);
    w_cpu.set_diffusion_backend(DiffusionBackend::cpu());
    let start = Instant::now();
    let (cpu_div, cpu_grow, nc, ns) = run_sim(&mut w_cpu, "CPU (rayon)");
    let cpu_total = start.elapsed().as_secs_f64() * 1000.0;

    // ── GPU ──
    println!();
    let mut w_gpu = build_world(grid, bounds);

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let gpu_backend = rt.block_on(async { DiffusionBackend::new().await });
        w_gpu.set_diffusion_backend(gpu_backend);
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("feature 'gpu' not enabled — GPU run == CPU");
        w_gpu.set_diffusion_backend(DiffusionBackend::cpu());
    }

    let start = Instant::now();
    let (gpu_div, gpu_grow, _, _) = run_sim(&mut w_gpu, "GPU (wgpu)");
    let gpu_total = start.elapsed().as_secs_f64() * 1000.0;

    // ── Comparison ──
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  RESULTS: {} cells  {} synapses", nc, ns);
    println!("═══════════════════════════════════════════════════");
    println!("  Phase            CPU         GPU       Speedup");
    println!("  ─────────────────────────────────────────────");
    println!(
        "  Division+diff:   {:5.0} ms    {:5.0} ms    {:4.1}x",
        cpu_div,
        gpu_div,
        cpu_div / gpu_div.max(1e-6)
    );
    println!(
        "  Grow+synapses:   {:5.0} ms    {:5.0} ms    {:4.1}x",
        cpu_grow,
        gpu_grow,
        cpu_grow / gpu_grow.max(1e-6)
    );
    println!("  ─────────────────────────────────────────────");
    println!(
        "  Total:           {:5.0} ms    {:5.0} ms    {:4.1}x",
        cpu_total,
        gpu_total,
        cpu_total / gpu_total.max(1e-6)
    );
    println!(
        "  Per cell:        {:5.0} us    {:5.0} us",
        cpu_total * 1000.0 / nc.max(1) as f64,
        gpu_total * 1000.0 / nc.max(1) as f64,
    );

    let sp = cpu_total / gpu_total.max(1e-6);
    println!();
    if sp > 1.5 {
        println!("  GPU is {:.1}x faster!", sp);
    } else if sp > 0.9 {
        println!(
            "  GPU ~= CPU on {}^3 — diffusion is not the bottleneck at {} cells.",
            GRID_N, nc
        );
        println!("     Try 200^3 or enable TRP attention on the GPU.");
    } else {
        println!("  GPU is slower — overhead > gain at {}^3.", GRID_N);
    }
    println!("═══════════════════════════════════════════════════");
}
