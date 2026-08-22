//! CX3D tutorial `SomaRandomWalkModule` (Zubler & Douglas 2009) — Rust port.
//!
//! Five somas walk a correlated random walk: each step the direction is
//! perturbed by uniform noise and the soma is translated by `speed·dt`.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/SomaRandomWalkModule.java`

use prometheus_grow::{Bounds, CellType, Grid, Host, Module, Vec3, World};

/// CX3D `SomaRandomWalkModule`: persist the walking direction, jitter it each
/// step, and translate the soma (`super.cellElement.move(speed, direction)`).
struct SomaRandomWalk {
    direction: Option<Vec3>,
}

impl Module for SomaRandomWalk {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return };
        let speed = 50.0;
        let mut dir = self.direction.unwrap_or_else(|| world.random_vec(1.0).normalized());
        dir = (dir + world.random_vec(0.1)).normalized(); // deltaDirection
        self.direction = Some(dir);
        world.move_soma(cell, dir * speed * world.dt);
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(SomaRandomWalk { direction: None })
    }
}

fn main() {
    let mut w = World::new(
        Grid::new(8, 8, 8, 100.0, Vec3::new(-400.0, -400.0, -400.0)),
        1.0,
        Bounds { min: Vec3::new(-400.0, -400.0, -400.0), max: Vec3::new(400.0, 400.0, 400.0) },
        7,
    );
    let mut ids = Vec::new();
    for _ in 0..5 {
        let p = w.random_vec(40.0); // randomNoise(40, 3)
        let id = w.add_cell(p, CellType::Excitatory, 5.0);
        w.add_soma_module(id, Box::new(SomaRandomWalk { direction: None }));
        ids.push(id);
    }

    println!("=== CX3D tutorial: SomaRandomWalkModule ===");
    let start: Vec<Vec3> = ids.iter().map(|&i| w.cells[i].soma.pos).collect();
    for _ in 0..3000 {
        w.step();
    }
    for (i, &id) in ids.iter().enumerate() {
        let d = (w.cells[id].soma.pos - start[i]).len();
        println!("  soma {i}: {} → {}  (прошло {d:.0})", fmt(start[i]), fmt(w.cells[id].soma.pos));
    }
    let moved = ids
        .iter()
        .enumerate()
        .filter(|(i, &id)| (w.cells[id].soma.pos - start[*i]).len() > 100.0)
        .count();
    println!("сом, ушедших дальше 100: {moved}/5");
    assert!(moved >= 4, "сомы должны заметно ходить");
}

fn fmt(v: Vec3) -> String {
    format!("({:.0},{:.0},{:.0})", v.x, v.y, v.z)
}
