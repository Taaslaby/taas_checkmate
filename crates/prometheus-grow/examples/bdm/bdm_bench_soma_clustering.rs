//! Benchmark mirroring BioDynaMo's `benchmark/soma_clustering_bm.cc`.
//!
//! Same setup as the BDM demo: 800³ closed space, two substances (D=0.05,
//! decay 0.001, 40³ grid), 32,000 cells (diameter 10) with `Secretion(1.0)`
//! + `Chemotaxis(0.75)`, 1000 steps. Prints wall time and cell-updates/sec so
//! it can be compared with BioDynaMo's published numbers.
//!
//! ```bash
//! cargo run -p prometheus-grow --release --example bdm_bench_soma_clustering [N]
//! ```

use std::time::Instant;

use prometheus_grow::{BoundaryCondition, CellModule, CellType, Substance, Vec3, World};

struct Secretion {
    substance: &'static str,
    rate: f64,
}
impl CellModule for Secretion {
    fn run(&mut self, world: &mut World, cell: usize) {
        let p = world.cells[cell].soma.pos;
        world.add_substance_at(self.substance, p, self.rate * world.dt);
    }
    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(Secretion { substance: self.substance, rate: self.rate })
    }
}

struct Chemotaxis {
    substance: &'static str,
    strength: f64,
}
impl CellModule for Chemotaxis {
    fn run(&mut self, world: &mut World, cell: usize) {
        let p = world.cells[cell].soma.pos;
        let g = world.gradient(self.substance, p);
        let mut step = g.scale(self.strength);
        let l = step.len();
        if l > 4.0 {
            step = step.scale(4.0 / l);
        }
        world.move_soma(cell, step);
    }
    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(Chemotaxis { substance: self.substance, strength: self.strength })
    }
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(32_000);
    let steps: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(1000);

    let side = 800.0;
    let dt = 0.01;
    let mut w = World::new(
        prometheus_grow::Grid::new(40, 40, 40, side / 40.0, Vec3::ZERO),
        dt,
        prometheus_grow::Bounds { min: Vec3::ZERO, max: Vec3::new(side, side, side) },
        4357,
    );
    w.max_cells = 1_000_000;
    for id in ["Substance_0", "Substance_1"] {
        w.add_substance(
            Substance::new(id, 0.05, 0.001, &w.grid, vec![])
                .with_boundary(BoundaryCondition::Neumann(0.0)),
        );
    }

    // CreateAgentsRandom(275, 250): cells in a 250³ box at the center.
    let half = 125.0;
    let mut placed = 0usize;
    while placed < n {
        let p = Vec3::new(
            400.0 + (w.rand() * 2.0 - 1.0) * half,
            400.0 + (w.rand() * 2.0 - 1.0) * half,
            400.0 + (w.rand() * 2.0 - 1.0) * half,
        );
        let c = w.add_cell(p, CellType::Excitatory, 5.0); // diameter 10
        w.add_cell_module(c, Box::new(Secretion { substance: "Substance_0", rate: 1.0 }));
        w.add_cell_module(c, Box::new(Chemotaxis { substance: "Substance_0", strength: 0.75 }));
        placed += 1;
    }

    println!("=== benchmark: soma_clustering (BioDynaMo setup) ===");
    println!("клеток {n}, шагов {steps}, мир {side}³, вещества: D=0.05, decay=0.001, сетка 40³, dt={dt}");
    let t0 = Instant::now();
    for _ in 0..steps {
        w.step();
    }
    let dt_sec = t0.elapsed().as_secs_f64();
    let updates = n as f64 * steps as f64;
    println!("время: {dt_sec:.2} с");
    println!("скорость: {:.0} клетко-шагов/с ({:.2} млн/с)", updates / dt_sec, updates / dt_sec / 1e6);
    println!("(одно ядро, debug vs release имеет значение — запускать с --release)");
    println!("клеток в конце: {}", w.cells.len());
}
