//! grow_arbor.rs — grow an arbor with the ported NETMORPH branching model
//! using balanced-forces branch angles (asymmetric, force-ratio based).
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_arbor
//! ```

use prometheus_grow::{
    BalancedForces, Bounds, CellType, Grid, VanPeltBranching, Vec3, World,
};

fn main() {
    let grid = Grid::new(8, 8, 8, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(800.0, 800.0, 800.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    let cell = w.add_cell(Vec3::new(400.0, 400.0, 400.0), CellType::Excitatory, 30.0);
    let n = w.new_neurite(cell, true, 1.0);
    w.extend_neurite(cell, n, Vec3::new(0.0, 0.0, 20.0)); // parent direction +Z

    let branch = VanPeltBranching::new(0.3, 50.0, 0.5, 0.0)
        .speed(2.0)
        .balanced_forces(BalancedForces::netmorph_defaults());
    w.add_neurite_module(cell, n, Box::new(branch));

    for _ in 0..200 {
        w.step();
    }

    let neurites = &w.cells[cell].neurites;
    println!("=== арбор: VanPeltBranching + balanced-forces ===");
    println!("терминальных сегментов: {}", neurites.len());
    println!("углы дочек к родителю (первые 12):");
    let mut shown = 0;
    for (i, neur) in neurites.iter().enumerate() {
        if let Some(p) = neur.parent {
            let pa = neurites[p].axis();
            let da = neur.axis();
            let ang = pa.dot(da).clamp(-1.0, 1.0).acos().to_degrees();
            println!("  ветвь {i:>3} (порядок {}) → {ang:5.1}°", neur.branch_order);
            shown += 1;
            if shown >= 12 {
                break;
            }
        }
    }
}
