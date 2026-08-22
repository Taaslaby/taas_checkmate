//! CX3D tutorial `DividingModule` / `DividingCell` (Zubler & Douglas 2009) —
//! Rust port.
//!
//! A cell grows its soma volume by 350/step; once the diameter exceeds 20 it
//! divides. The daughter inherits the `DividingModule` (`copy_on_divide`),
//! so the population doubles — a self-constructing lineage.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/DividingModule.java`

use prometheus_grow::{CellModule, World};

/// CX3D `DividingModule`: if the soma diameter is above 20, divide;
/// otherwise grow the volume by 300.
struct DividingModule;

impl CellModule for DividingModule {
    fn run(&mut self, world: &mut World, cell: usize) {
        let diam = 2.0 * world.cells[cell].soma.radius;
        if diam > 20.0 {
            world.request_division(cell);
        } else {
            world.grow_soma(cell, 300.0);
        }
    }

    fn copy_on_divide(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(DividingModule)
    }
}

fn main() {
    let mut w = World::new(
        prometheus_grow::Grid::new(8, 8, 8, 100.0, prometheus_grow::Vec3::new(-400.0, -400.0, -400.0)),
        1.0,
        prometheus_grow::Bounds {
            min: prometheus_grow::Vec3::new(-400.0, -400.0, -400.0),
            max: prometheus_grow::Vec3::new(400.0, 400.0, 400.0),
        },
        2,
    );
    w.max_cells = 512;
    let c = w.add_cell(prometheus_grow::Vec3::ZERO, prometheus_grow::CellType::Excitatory, 1.0);
    w.add_cell_module(c, Box::new(DividingModule));

    println!("=== CX3D tutorial: DividingModule ===");
    let mut last = 1usize;
    for step in 0..400 {
        w.step();
        if w.cells.len() != last {
            println!("  t={step:>3}: клеток {}", w.cells.len());
            last = w.cells.len();
        }
    }
    println!("итого клеток: {} (потолок {})", w.cells.len(), w.max_cells);
    assert!(w.cells.len() >= 32, "линия должна размножиться, got {}", w.cells.len());
}
