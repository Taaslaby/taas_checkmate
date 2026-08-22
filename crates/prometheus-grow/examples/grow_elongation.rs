//! grow_elongation.rs — NETMORPH second-order terminal elongation: two growth
//! cones start equal and then diverge in speed via the second-order drift,
//! so each claims a different share of the arbor's elongation resource.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_elongation
//! ```

use prometheus_grow::{
    Bounds, CellType, Grid, SecondOrderElongation, Vec3, World,
};

fn total_len(neur: &prometheus_grow::Neurite) -> f64 {
    neur.points.windows(2).map(|w| w[0].dist(w[1])).sum()
}

fn main() {
    let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(400.0, 400.0, 400.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    let cell = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
    let n0 = w.new_neurite(cell, true, 1.0);
    let n1 = w.new_neurite(cell, true, 1.0);
    w.extend_neurite(cell, n0, Vec3::new(0.0, 0.0, 10.0));
    w.extend_neurite(cell, n1, Vec3::new(0.0, 0.0, 10.0));

    w.add_neurite_module(cell, n0, Box::new(SecondOrderElongation::new(1.0, 0.74, 5.0)));
    w.add_neurite_module(cell, n1, Box::new(SecondOrderElongation::new(1.0, 0.74, 5.0)));

    for _ in 0..100 {
        w.step();
    }

    println!("=== NETMORPH second-order elongation (два конуса) ===");
    println!(
        "скорости:  конус 0 = {:.3}   конус 1 = {:.3}",
        w.cells[cell].neurites[n0].elongation_speed,
        w.cells[cell].neurites[n1].elongation_speed
    );
    println!(
        "длины:     конус 0 = {:.2}   конус 1 = {:.2}",
        total_len(&w.cells[cell].neurites[n0]),
        total_len(&w.cells[cell].neurites[n1])
    );
    println!("\n(разные скорости → разные доли арборного ресурса)");
}
