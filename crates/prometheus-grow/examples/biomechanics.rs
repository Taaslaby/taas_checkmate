//! biomechanics.rs — self-organizing mini-brain via mechanics + division + growth.
//!
//! 4 progenitor cells (2 exc, 2 inh) → divide to ~32 cells
//! → adhesion pulls same-type cells into clusters
//! → repulsion pushes different-type clusters apart
//! → elastic keeps synapses on springs
//! → neurites grow between clusters.
//!
//! ```bash
//! cargo run -p prometheus-grow --example biomechanics
//! ```

use prometheus_grow::{
    Bounds, CellType, ChemoAttraction, Grid, MechanicsParams, Recognition, Secretor,
    SimpleCellCycle, Substance, Vec3, World,
};
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let _rng = StdRng::seed_from_u64(7);

    let origin = Vec3::new(-200.0, -200.0, -200.0);
    let cell = 8.0;
    let grid = Grid::new(50, 50, 50, cell, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(cell * 50.0, cell * 50.0, cell * 50.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    w.max_cells = 64;
    w.max_neurites = 2048;
    w.synapse_dist = 30.0;

    // ── Mechanics ──
    w.mechanics_params = MechanicsParams {
        adhesion_strength: 0.8,
        repulsion_strength: 3.0,
        elastic_strength: 0.5,
        adhesion_cutoff: 2.0,
        damping: 0.15,
        ..MechanicsParams::default()
    };

    // ── Morphogen: a common attractant in the center ──
    let attractant = Substance::new(
        "netrin",
        0.1,
        0.05,
        &w.grid,
        vec![
            prometheus_grow::ArtificialGradient::Gaussian {
                axis: 0,
                max: 8.0,
                mean: 0.0,
                sigma: 80.0,
            },
            prometheus_grow::ArtificialGradient::Gaussian {
                axis: 1,
                max: 8.0,
                mean: 0.0,
                sigma: 80.0,
            },
            prometheus_grow::ArtificialGradient::Gaussian {
                axis: 2,
                max: 8.0,
                mean: 0.0,
                sigma: 80.0,
            },
        ],
    );
    w.add_substance(attractant);

    // ── 4 progenitors in the corners of the cube ──
    let seeds = [
        (Vec3::new(-60.0, -60.0, -60.0), CellType::Excitatory, "E1"),
        (Vec3::new(60.0, 60.0, -60.0), CellType::Excitatory, "E2"),
        (Vec3::new(-60.0, 60.0, 60.0), CellType::Inhibitory, "I1"),
        (Vec3::new(60.0, -60.0, 60.0), CellType::Inhibitory, "I2"),
    ];

    for (pos, ct, tag) in &seeds {
        let id = w.add_cell(*pos, *ct, 6.0);
        w.cells[id].tag = tag.to_string();
        // Each progenitor divides.
        w.add_cell_module(id, Box::new(SimpleCellCycle::new(10.0, 8.0)));
        // Secretes netrin in small amounts (cluster auto-attraction).
        w.add_soma_module(id, Box::new(Secretor::new("netrin", 0.3)));
    }

    println!("┌────────────────────────────────────────────┐");
    println!("│ Biomechanics: self-organization of a mini-brain │");
    println!("└────────────────────────────────────────────┘");
    println!();
    println!("  Старт: 4 клетки (2E + 2I) по углам куба.");
    println!(
        "  Механика: adhesion={} repuls={} elastic={}",
        w.mechanics_params.adhesion_strength,
        w.mechanics_params.repulsion_strength,
        w.mechanics_params.elastic_strength,
    );

    // ── Phase 1: division + adhesion → clusters ──
    println!();
    println!("─── Фаза 1: деление + adhesion (40 шагов) ───");
    for t in 0..40 {
        w.step();
        if t % 10 == 0 || t == 39 {
            let (n_e, n_i) = count_types(&w);
            let clusters = count_clusters(&w, 20.0);
            println!(
                "  t={:3} | клеток: {} ({}E + {}I) | кластеров: {}",
                t,
                w.cells.len(),
                n_e,
                n_i,
                clusters
            );
        }
    }

    // ── Phase 2: neurites grow toward the center (netrin) ──
    println!();
    println!("─── Фаза 2: рост нейритов (30 шагов) ───");
    // One axon per cell, toward the center.
    let n_cells = w.cells.len();
    for ci in 0..n_cells {
        let na = w.new_neurite(ci, true, 1.5);
        let mut chemo = ChemoAttraction::new("netrin");
        chemo.noise_weight = 0.1;
        chemo.gradient_weight = 1.0;
        chemo.stop_concentration = 6.0;
        w.add_neurite_module(ci, na, Box::new(chemo));
    }

    for t in 0..30 {
        w.step();
        if t % 10 == 0 || t == 29 {
            let n_axons: usize = w
                .cells
                .iter()
                .map(|c| c.neurites.iter().filter(|n| n.is_axon).count())
                .sum();
            println!(
                "  t={:3} | нейритов: {} | синапсов: {}",
                40 + t,
                n_axons,
                w.synapses.len()
            );
        }
    }

    // ── Phase 3: synapses + elastic springs ──
    println!();
    println!("─── Фаза 3: синапсы + elastic (20 шагов) ───");
    w.set_recognition(Recognition::all());
    w.form_synapses(30.0);
    println!("  Синапсов после wiring: {}", w.synapses.len());

    for _t in 0..20 {
        w.step();
    }

    // ── Statistics ──
    println!();
    println!("═══════════════════════════════════════════");
    let (n_e, n_i) = count_types(&w);
    let e_clusters = count_clusters_filter(&w, 20.0, CellType::Excitatory);
    let i_clusters = count_clusters_filter(&w, 20.0, CellType::Inhibitory);
    println!("  Result:");
    println!("     Клеток:     {} ({}E + {}I)", w.cells.len(), n_e, n_i);
    println!("     Нейритов:   {}", w.total_neurites());
    println!("     Синапсов:   {}", w.synapses.len());
    println!(
        "     Кластеров:  {} (E:{} + I:{})",
        e_clusters + i_clusters,
        e_clusters,
        i_clusters
    );

    let (e_centroid, i_centroid) = type_centroids(&w);
    let dist = e_centroid.dist(i_centroid);
    println!("     E↔I центр:  {:.0} мкм", dist);

    // Cross-type vs same-type synapses.
    let (cross, same) = count_synapse_types(&w);
    println!("     Синапсы:    cross-type={} same-type={}", cross, same);
    println!("═══════════════════════════════════════════");
}

fn count_types(w: &World) -> (usize, usize) {
    w.cells.iter().fold((0, 0), |(e, i), c| match c.cell_type {
        CellType::Excitatory => (e + 1, i),
        CellType::Inhibitory => (e, i + 1),
        CellType::Glia => (e, i),
    })
}

/// Simple DBSCAN-like clustering by distance threshold.
fn count_clusters(w: &World, dist: f64) -> usize {
    let n = w.cells.len();
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

fn count_clusters_filter(w: &World, dist: f64, ct: CellType) -> usize {
    let indices: Vec<usize> = w
        .cells
        .iter()
        .enumerate()
        .filter(|(_, c)| c.cell_type == ct)
        .map(|(i, _)| i)
        .collect();
    let mut visited = vec![false; indices.len()];
    let mut clusters = 0;
    for ii in 0..indices.len() {
        if visited[ii] {
            continue;
        }
        clusters += 1;
        let mut stack = vec![ii];
        visited[ii] = true;
        while let Some(cur) = stack.pop() {
            for jj in 0..indices.len() {
                if visited[jj] {
                    continue;
                }
                if w.cells[indices[cur]]
                    .soma
                    .pos
                    .dist(w.cells[indices[jj]].soma.pos)
                    < dist
                {
                    visited[jj] = true;
                    stack.push(jj);
                }
            }
        }
    }
    clusters
}

fn type_centroids(w: &World) -> (Vec3, Vec3) {
    let (mut e_sum, mut e_n) = (Vec3::ZERO, 0usize);
    let (mut i_sum, mut i_n) = (Vec3::ZERO, 0usize);
    for c in &w.cells {
        match c.cell_type {
            CellType::Excitatory => {
                e_sum = e_sum + c.soma.pos;
                e_n += 1;
            }
            CellType::Inhibitory => {
                i_sum = i_sum + c.soma.pos;
                i_n += 1;
            }
            CellType::Glia => {}
        }
    }
    (
        if e_n > 0 {
            e_sum * (1.0 / e_n as f64)
        } else {
            Vec3::ZERO
        },
        if i_n > 0 {
            i_sum * (1.0 / i_n as f64)
        } else {
            Vec3::ZERO
        },
    )
}

fn count_synapse_types(w: &World) -> (usize, usize) {
    let types: Vec<CellType> = w.cells.iter().map(|c| c.cell_type).collect();
    w.synapses.iter().fold((0, 0), |(cross, same), s| {
        if types[s.pre] == types[s.post] {
            (cross, same + 1)
        } else {
            (cross + 1, same)
        }
    })
}
