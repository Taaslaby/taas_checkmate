//! bench_biodynamo.rs — BioDynaMo-style tissue growth benchmark
//!
//! 4 progenitors -> division -> ~1024 cells -> mechanics -> neurites -> synapses.
//! Each phase is timed. Compare with BioDynaMo benchmarks.
//!
//! cargo run -p prometheus-grow --example bench_biodynamo --release

use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Grid, MechanicsParams, Recognition,
    Secretor, SimpleCellCycle, Substance, Vec3, World,
};
use std::time::Instant;

const N_SEEDS: usize = 4;
const TARGET_CELLS: usize = 1024;
const STEPS_DIVISION: usize = 60;
const STEPS_MECHANICS: usize = 30;
const STEPS_GROWTH: usize = 30;
const GRID_N: usize = 64;

fn main() {
    let cell_size = 20.0;
    let span = cell_size * GRID_N as f64;
    let origin = Vec3::new(-span / 2.0, -span / 2.0, -span / 2.0);
    let grid = Grid::new(GRID_N, GRID_N, GRID_N, cell_size, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(span, span, span),
    };

    println!("===========================================================");
    println!("  PROMETHEUS vs BioDynaMo - tissue benchmark");
    println!("  Grid: {}^3   Target: {} cells", GRID_N, TARGET_CELLS);
    println!("===========================================================");
    println!();

    let mut w = World::new(grid, 1.0, bounds, 42);
    w.max_cells = TARGET_CELLS;
    w.max_neurites = TARGET_CELLS * 3;
    w.synapse_dist = 14.0;
    w.max_syn_per_cell = 12;
    w.synapse_tips_only = true;

    w.mechanics_params = MechanicsParams {
        adhesion_strength: 0.4,
        repulsion_strength: 1.5,
        adhesion_cutoff: 2.0,
        damping: 0.1,
        ..MechanicsParams::default()
    };

    // Morphogen attractant in center.
    let half = (GRID_N / 2) as f64 * cell_size;
    w.add_substance(Substance::new(
        "netrin",
        0.1,
        0.02,
        &w.grid,
        vec![
            ArtificialGradient::Gaussian {
                axis: 0,
                max: 10.0,
                mean: half,
                sigma: half * 0.6,
            },
            ArtificialGradient::Gaussian {
                axis: 1,
                max: 10.0,
                mean: half,
                sigma: half * 0.6,
            },
            ArtificialGradient::Gaussian {
                axis: 2,
                max: 10.0,
                mean: half,
                sigma: half * 0.6,
            },
        ],
    ));

    // 4 seed cells on a sphere.
    let r_seed = half * 0.5;
    for i in 0..N_SEEDS {
        let phi = i as f64 / N_SEEDS as f64 * std::f64::consts::TAU;
        let theta = (i as f64 + 0.5) / N_SEEDS as f64 * std::f64::consts::PI;
        let pos = Vec3::new(
            half + r_seed * theta.sin() * phi.cos(),
            half + r_seed * theta.sin() * phi.sin(),
            half + r_seed * theta.cos(),
        );
        let ct = if i < N_SEEDS / 2 {
            CellType::Excitatory
        } else {
            CellType::Inhibitory
        };
        let id = w.add_cell(pos, ct, 8.0);
        w.add_cell_module(id, Box::new(SimpleCellCycle::new(15.0, 10.0)));
        w.add_soma_module(id, Box::new(Secretor::new("netrin", 0.5)));
    }

    let wall = Instant::now();
    let mut phase_start = wall;

    // ---- Phase 1: Division ----
    println!("--- Phase 1: division ({} steps) ---", STEPS_DIVISION);
    w.mechanics_params.enabled = false;
    for t in 0..STEPS_DIVISION {
        w.step();
        if w.cells.len() >= TARGET_CELLS {
            for ci in 0..w.cells.len() {
                w.cells[ci].modules.clear();
            }
        }
        if (t + 1) % 15 == 0 {
            println!("  t={:3} cells={:5}", t + 1, w.cells.len());
        }
    }
    let div_time = phase_start.elapsed();
    let n_div = w.cells.len();
    println!("  done: {:?} {} cells", div_time, n_div);

    // ---- Phase 2: Mechanics + secretion ----
    println!();
    println!(
        "--- Phase 2: mechanics + secretion ({} steps) ---",
        STEPS_MECHANICS
    );
    w.mechanics_params.enabled = true;
    phase_start = Instant::now();
    for t in 0..STEPS_MECHANICS {
        w.step();
        if w.cells.len() >= TARGET_CELLS {
            for ci in 0..w.cells.len() {
                w.cells[ci].modules.clear();
            }
        }
        if (t + 1) % 10 == 0 {
            println!(
                "  t={:3} cells={} clusters={}",
                STEPS_DIVISION + t + 1,
                w.cells.len(),
                count_clusters(&w, 25.0)
            );
        }
    }
    let mech_time = phase_start.elapsed();
    let n_clusters = count_clusters(&w, 25.0);

    // ---- Phase 3: Neurite growth + synapses ----
    println!();
    println!(
        "--- Phase 3: neurites + synapses ({} steps) ---",
        STEPS_GROWTH
    );
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
    phase_start = Instant::now();
    for t in 0..STEPS_GROWTH {
        w.step();
        if (t + 1) % 10 == 0 {
            let nn: usize = w.cells.iter().map(|c| c.neurites.len()).sum();
            println!(
                "  t={:3} neurites={:5} synapses={:6}",
                STEPS_DIVISION + STEPS_MECHANICS + t + 1,
                nn,
                w.synapses.len()
            );
        }
    }
    let grow_time = phase_start.elapsed();

    // ---- Summary ----
    let total = wall.elapsed();
    let nn: usize = w.cells.iter().map(|c| c.neurites.len()).sum();
    let ncells = w.cells.len();
    let nsyns = w.synapses.len();
    let (ne, ni) = count_types(&w);

    println!();
    println!("===========================================================");
    println!("  RESULTS");
    println!("===========================================================");
    println!("  Cells:     {} ({}E + {}I)", ncells, ne, ni);
    println!("  Clusters:  {}", n_clusters);
    println!("  Neurites:  {}", nn);
    println!("  Synapses:  {}", nsyns);
    println!("  -------------------------------------------------------");
    println!(
        "  Phase 1 (division):      {:8.1} ms",
        div_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Phase 2 (mechanics):     {:8.1} ms",
        mech_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Phase 3 (neurites+syn):  {:8.1} ms",
        grow_time.as_secs_f64() * 1000.0
    );
    println!("  -------------------------------------------------------");
    println!(
        "  Total:                   {:8.1} ms",
        total.as_secs_f64() * 1000.0
    );
    println!(
        "  Per cell:                {:8.1} us",
        total.as_secs_f64() * 1_000_000.0 / ncells.max(1) as f64
    );
    println!();
    println!("  BioDynaMo reference (single-thread, 1000 cells):");
    println!("    Soma clustering: ~200-500 ms");
    println!("    Neurite growth:  ~500-2000 ms");
    println!(
        "    Our total:       ~{:.0} ms",
        total.as_secs_f64() * 1000.0
    );
    println!("===========================================================");
}

fn count_types(w: &World) -> (usize, usize) {
    w.cells.iter().fold((0, 0), |(e, i), c| match c.cell_type {
        CellType::Excitatory => (e + 1, i),
        CellType::Inhibitory => (e, i + 1),
        CellType::Glia => (e, i),
    })
}

fn count_clusters(w: &World, dist: f64) -> usize {
    let n = w.cells.len();
    if n == 0 {
        return 0;
    }
    let mut visited = vec![false; n];
    let mut clusters = 0;
    for i in 0..n {
        if visited[i] {
            continue;
        }
        clusters += 1;
        let mut stack = vec![i];
        visited[i] = true;
        while let Some(cur) = stack.pop() {
            for j in 0..n {
                if visited[j] {
                    continue;
                }
                if w.cells[cur].soma.pos.dist(w.cells[j].soma.pos) < dist {
                    visited[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    clusters
}
