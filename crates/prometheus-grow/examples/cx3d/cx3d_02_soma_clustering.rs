//! CX3D tutorial `SomaClustering` (Zubler & Douglas 2009) — Rust port.
//!
//! Two populations (60 "Yellow" + 60 "Violet") chemotax up the gradient of
//! the substance they themselves secrete — a classic self-organizing
//! aggregation. 400 inert nodes are passive obstacles.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/SomaClustering.java`

use prometheus_grow::{Bounds, CellType, Grid, Host, Module, Substance, Vec3, World};

/// CX3D `SomaClustering`: move up the gradient of `substance_id`, then
/// secrete 1000 units of it at the current position.
struct SomaClustering {
    substance: String,
}

impl Module for SomaClustering {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return };
        let speed = 100.0;
        let p = world.cells[cell].soma.pos;
        let grad = world.gradient(&self.substance, p);
        if grad.len() > 1e-9 {
            world.move_soma(cell, grad.normalized() * speed * world.dt);
        }
        world.add_substance_at(&self.substance, world.cells[cell].soma.pos, 1000.0);
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(SomaClustering { substance: self.substance.clone() })
    }
}

fn main() {
    let origin = Vec3::new(-900.0, -900.0, -900.0);
    let grid = Grid::new(19, 19, 19, 100.0, origin);
    let mut w = World::new(
        grid,
        0.05, // slow soma movement so the secreted field can form before cells hit walls
        Bounds { min: origin, max: origin + Vec3::new(1900.0, 1900.0, 1900.0) },
        1,
    );
    // Substance templates (CX3D: `new Substance("Yellow", 1000, 0.01)`).
    // D scaled by dx² (cell=100): the engine now normalizes the Laplacian by
    // dx², so D=1e7 keeps the effective exchange rate of the original 1000.
    w.add_substance(Substance::new("Yellow", 1e7, 0.01, &grid, vec![]));
    w.add_substance(Substance::new("Violet", 1e7, 0.01, &grid, vec![]));

    // 400 inert nodes (obstacles) — cells without modules.
    for _ in 0..400 {
        let p = w.random_vec(700.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    let mut yellow = Vec::new();
    let mut violet = Vec::new();
    for _ in 0..60 {
        let p = w.random_vec(50.0);
        let id = w.add_cell(p, CellType::Excitatory, 5.0);
        w.add_soma_module(id, Box::new(SomaClustering { substance: "Yellow".into() }));
        yellow.push(id);
    }
    for _ in 0..60 {
        let p = w.random_vec(50.0);
        let id = w.add_cell(p, CellType::Inhibitory, 5.0);
        w.add_soma_module(id, Box::new(SomaClustering { substance: "Violet".into() }));
        violet.push(id);
    }

    let mean_dist = |ids: &[usize]| -> f64 {
        let s: f64 = ids.iter().map(|&i| w.cells[i].soma.pos.len()).sum();
        s / ids.len() as f64
    };

    println!("=== CX3D tutorial: SomaClustering ===");
    println!("старт: жёлтые {:.0}, фиолетовые {:.0}", mean_dist(&yellow), mean_dist(&violet));
    for _ in 0..3000 {
        w.step();
    }
    let (d0, d1) = {
        let s0: f64 = yellow.iter().map(|&i| w.cells[i].soma.pos.len()).sum();
        let s1: f64 = violet.iter().map(|&i| w.cells[i].soma.pos.len()).sum();
        (s0 / yellow.len() as f64, s1 / violet.len() as f64)
    };
    println!("финал: жёлтые {d0:.0}, фиолетовые {d1:.0} (начало ≈ 34)");
    println!("среднее расстояние от центра должно СЖИМАТЬСЯ (кластеризация)");
    assert!(d0 < 30.0, "жёлтая популяция должна сгруппироваться, got {d0:.0}");
    assert!(d1 < 30.0, "фиолетовая популяция должна сгруппироваться, got {d1:.0}");
}
