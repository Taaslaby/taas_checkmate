//! grow_branch_angles.rs — demonstrate the NETMORPH balanced-forces branch
//! angle model: daughters leave the parent at angles set by their elongation
//! forces (fast daughter ≈ straight, slow daughter veers off).
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_branch_angles
//! ```

use prometheus_grow::{BalancedForces, Bounds, CellType, Grid, Vec3, World};

fn angle_between(a: Vec3, b: Vec3) -> f64 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

fn main() {
    // Fix the parallelogram angle at 90° so the force-ratio effect is clean
    // (the full model draws it from N(π/2, 0.5)).
    let mut model = BalancedForces::netmorph_defaults();
    model.parallelogram_std = 0.0;
    let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(400.0, 400.0, 400.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    let cell = w.add_cell(Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory, 30.0);
    let parent = w.new_neurite(cell, true, 1.0);
    // Give the parent a direction (+Z) so we can measure daughter deviations.
    w.extend_neurite(cell, parent, Vec3::new(0.0, 0.0, 50.0));
    let axis = w.cells[cell].neurites[parent].axis();

    println!("=== NETMORPH balanced-forces branch angles ===");
    for (f1, f2) in [(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)] {
        let before = w.cells[cell].neurites.len();
        w.bifurcate_neurite_balanced(cell, parent, 1.5, f1, f2, &model);
        let a1 = angle_between(axis, w.cells[cell].neurites[before].axis()).to_degrees();
        let a2 = angle_between(axis, w.cells[cell].neurites[before + 1].axis()).to_degrees();
        println!(
            "силы {f1}:{f2} → отклонения {a1:.0}° / {a2:.0}° (сумма {:.0}°)",
            a1 + a2
        );
    }
    println!("\n(равные силы — симметрично; доминирующая ветвь идёт прямее)");
}
