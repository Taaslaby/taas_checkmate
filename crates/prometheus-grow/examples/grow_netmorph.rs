//! grow_netmorph.rs — the whole NETMORPH growth loop in one module:
//! second-order elongation + van Pelt branching + ERI + balanced-forces.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_netmorph
//! ```

use prometheus_grow::{Bounds, CellType, Grid, NetmorphGrowth, Vec3, World};

fn main() {
    let grid = Grid::new(8, 8, 8, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(800.0, 800.0, 800.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    let cell = w.add_cell(Vec3::new(400.0, 400.0, 400.0), CellType::Excitatory, 30.0);
    let n = w.new_neurite(cell, true, 2.0);
    w.extend_neurite(cell, n, Vec3::new(0.0, 0.0, 20.0));

    let growth = NetmorphGrowth::new(1.0, 0.74, 5.0, 5.0, 20.0, 0.2, 0.2)
        .max_prob(0.15)
        .speed_constraint(3.0)
        .step(2.0)
        .eri_std(0.25);
    w.add_neurite_module(cell, n, Box::new(growth));

    for _ in 0..300 {
        w.step();
    }

    let neurites = &w.cells[cell].neurites;
    let terminals: Vec<usize> = neurites
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_terminal)
        .map(|(i, _)| i)
        .collect();
    println!("=== NetmorphGrowth: полный цикл роста ===");
    println!(
        "всего нейритов: {}   терминалов (конусов): {}",
        neurites.len(),
        terminals.len()
    );
    println!("скорости терминалов:");
    for &i in terminals.iter().take(8) {
        println!("  терминал {i:>2}  speed={:.3}", neurites[i].elongation_speed);
    }
    println!("углы дочек к родителю (первые 8):");
    let mut shown = 0;
    for (i, x) in neurites.iter().enumerate() {
        if let Some(p) = x.parent {
            let ang = neurites[p]
                .axis()
                .dot(x.axis())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            println!("  ветвь {i:>3} → {ang:5.1}°");
            shown += 1;
            if shown >= 8 {
                break;
            }
        }
    }
}
