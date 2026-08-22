//! grow_growth_loop.rs — the NETMORPH growth loop closed: a bifurcating
//! terminal splits its elongation speed between the daughters (ERI), gives
//! each an initial length proportional to its speed, and feeds those speeds
//! to balanced-forces so the faster daughter goes straighter.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_growth_loop
//! ```

use prometheus_grow::{BalancedForces, Bounds, CellType, Grid, Vec3, World};

fn main() {
    let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(400.0, 400.0, 400.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    let cell = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
    let n = w.new_neurite(cell, true, 1.0);
    w.extend_neurite(cell, n, Vec3::new(0.0, 0.0, 20.0));
    let axis = w.cells[cell].neurites[n].axis();
    w.cells[cell].neurites[n].elongation_speed = 2.0;

    let mut model = BalancedForces::netmorph_defaults();
    model.parallelogram_std = 0.0; // fixed 90° so only the force ratio matters
    w.bifurcate_neurite_growth(cell, n, 1.5, 3.0, 0.3, &model);

    println!("=== ERI + balanced-forces (родительская скорость 2.0) ===");
    let neurites = &w.cells[cell].neurites;
    for i in 1..3 {
        let v = neurites[i].elongation_speed;
        let l = neurites[i].points[0].dist(neurites[i].points[1]);
        let ang = axis.dot(neurites[i].axis()).clamp(-1.0, 1.0).acos().to_degrees();
        println!("  дочка {i}: speed={v:.2}  длина={l:.2}  угол={ang:.1}°");
    }
    println!("\n(быстрее дочка → длиннее старт и меньший угол от родителя)");
}
