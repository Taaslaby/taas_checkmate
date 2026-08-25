//! BioDynaMo `demo/cell_division` — GrowthDivision behavior.
//!
//! A 4×4×4 grid of cells (spacing 20) with diameter 5..10 grows its volume by
//! 50 per step; once the diameter reaches 10 the cell divides (mother + one
//! daughter, the daughter inherits the behavior). 111 steps.
//!
//! Original: `forks/biodynamo-master/demo/cell_division/src/cell_division.h`
//!
//! ```bash
//! cargo run -p prometheus-grow --example bdm_01_cell_division
//! ```

use prometheus_grow::{CellModule, CellType, Vec3, World};

/// BDM `GrowthDivision(diameter_max, growth_ratio)`: grow the volume; divide
/// once the diameter exceeds `diameter_max`.
struct GrowthDivision {
    diameter_max: f64,
    growth_ratio: f64,
}

impl CellModule for GrowthDivision {
    fn run(&mut self, world: &mut World, cell: usize) {
        let diam = 2.0 * world.cells[cell].soma.radius;
        if diam >= self.diameter_max {
            world.request_division(cell);
        } else {
            // BDM GrowthDivision grows the volume by `growth_ratio` per step.
            world.grow_soma(cell, self.growth_ratio);
        }
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(GrowthDivision {
            diameter_max: self.diameter_max,
            growth_ratio: self.growth_ratio,
        })
    }
}

fn main() {
    // Grid3D(4, 20, construct): 4³ cells, spacing 20 → world 80³.
    let side = 80.0;
    let mut w = World::new(
        prometheus_grow::Grid::new(
            16,
            16,
            16,
            side / 16.0,
            Vec3::new(-side / 2.0, -side / 2.0, -side / 2.0),
        ),
        1.0,
        prometheus_grow::Bounds {
            min: Vec3::new(-side, -side, -side),
            max: Vec3::new(side, side, side),
        },
        7,
    );
    w.max_cells = 4096;

    let spacing = 20.0;
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let pos = Vec3::new(
                    i as f64 * spacing - 1.5 * spacing,
                    j as f64 * spacing - 1.5 * spacing,
                    k as f64 * spacing - 1.5 * spacing,
                );
                let c = w.add_cell(pos, CellType::Excitatory, 2.5); // diameter 5
                w.add_cell_module(
                    c,
                    Box::new(GrowthDivision {
                        diameter_max: 10.0,
                        growth_ratio: 50.0,
                    }),
                );
            }
        }
    }

    println!("=== BioDynaMo demo: cell_division (GrowthDivision) ===");
    println!(
        "старт: {} клеток, диаметр 5 → деление при 10, рост 50/шаг, 111 шагов",
        w.cells.len()
    );
    let mut last = w.cells.len();
    for step in 0..111 {
        w.step();
        if w.cells.len() != last && (step + 1) % 5 == 0 {
            println!(
                "  t={step:>3}: клеток {} (diam ~{:.1})",
                w.cells.len(),
                2.0 * w.cells[0].soma.radius
            );
            last = w.cells.len();
        }
    }
    let n = w.cells.len();
    println!("итого после 111 шагов: {n} клеток");
    // BDM asserts the simulation completes; growth must have happened
    // (64 seeds grow+divide for 111 steps — expect several hundred).
    assert!(n > 64, "клетки должны размножиться, got {n}");
    println!("cells grew and divided (GrowthDivision works)");
}
