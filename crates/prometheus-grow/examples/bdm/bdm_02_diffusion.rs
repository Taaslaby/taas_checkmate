//! BioDynaMo `demo/diffusion` — Substance + Chemotaxis + Secretion.
//!
//! A "Kalium" substance diffuses (D=0.4, no decay, Neumann/closed boundary).
//! One cell at the center secretes 4 units/step; 8 cells on the grid corners
//! chemotax up the gradient (strength 0.5). 300 steps — the 8 cells should
//! drift toward the secreter and the concentration should peak at the center.
//!
//! Original: `forks/biodynamo-master/demo/diffusion/src/diffusion.h`
//!
//! ```bash
//! cargo run -p prometheus-grow --example bdm_02_diffusion
//! ```

use prometheus_grow::{CellModule, CellType, Vec3, World};

/// BDM `Chemotaxis(substance, strength)`: move the soma along the gradient
/// by `strength × gradient` each step (capped so weak fields drift slowly and
/// strong fields don't teleport).
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

/// BDM `Secretion(substance, rate)`: release `rate` units of substance into
/// the extracellular space every step.
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

fn main() {
    // BDM: space 0..100, substance grid resolution 25 → 25³ voxels.
    // dt=0.1 keeps the explicit scheme stable: D·dt/dx² = 25·0.1/16 ≈ 0.16 < 1/6.
    let side = 100.0;
    let dt = 0.1;
    let steps = 3000; // total simulated time = 300 (BDM's 300 steps)
    let mut w = World::new(
        prometheus_grow::Grid::new(25, 25, 25, side / 25.0, Vec3::ZERO),
        dt,
        prometheus_grow::Bounds {
            min: Vec3::ZERO,
            max: Vec3::new(side, side, side),
        },
        11,
    );
    // DefineSubstance(kKalium, "Kalium", D, 0, 25) + Neumann(0) closed boundary.
    let kalium = prometheus_grow::Substance::new("Kalium", 25.0, 0.0, &w.grid, vec![])
        .with_boundary(prometheus_grow::BoundaryCondition::Neumann(0.0));
    w.add_substance(kalium);

    // Grid3D(2, 100): 8 cells at the corners with Chemotaxis("Kalium", 0.5).
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let pos = Vec3::new(i as f64 * side, j as f64 * side, k as f64 * side);
                let c = w.add_cell(pos, CellType::Excitatory, 15.0); // diameter 30
                w.add_cell_module(c, Box::new(Chemotaxis { substance: "Kalium", strength: 0.5 }));
            }
        }
    }
    // Secreting cell at the center.
    let s = w.add_cell(Vec3::new(side / 2.0, side / 2.0, side / 2.0), CellType::Inhibitory, 15.0);
    w.add_cell_module(s, Box::new(Secretion { substance: "Kalium", rate: 40.0 }));

    let center = Vec3::new(side / 2.0, side / 2.0, side / 2.0);
    let dist0: f64 = (0..8).map(|i| (w.cells[i].soma.pos - center).len()).sum::<f64>() / 8.0;
    for step in 0..steps {
        w.step();
        if (step + 1) % 500 == 0 {
            let c = w.concentration("Kalium", center);
            println!("  t={step:>4} (sim {:.1}): концентрация в центре {c:.3}", (step as f64 + 1.0) * dt);
        }
    }

    let dist1: f64 = (0..8).map(|i| (w.cells[i].soma.pos - center).len()).sum::<f64>() / 8.0;
    let c_end = w.concentration("Kalium", center);
    let c_corner = w.concentration("Kalium", Vec3::new(0.0, 0.0, 0.0));
    println!("=== BioDynaMo demo: diffusion (Secretion + Chemotaxis) ===");
    println!("среднее расстояние хемотаксирующих клеток до центра: {dist0:.1} → {dist1:.1}");
    println!("концентрация Kalium: центр {c_end:.3}, угол {c_corner:.4}");
    assert!(dist1 < dist0, "клетки должны приблизиться к источнику: {dist0:.1} → {dist1:.1}");
    assert!(dist1 < dist0 * 0.98, "движение должно быть заметным: {dist0:.1} → {dist1:.1}");
    assert!(c_end > 0.5, "вещество должно накопиться у источника, got {c_end:.3}");
    assert!(c_end.is_finite() && c_corner.is_finite(), "NaN/бесконечность в поле");
    assert!(c_corner > 0.0, "фронт должен дойти до углов, got {c_corner:.5}");
    println!("✓ хемотаксис притянул клетки к секретору, диффузия стабильна и доходит до углов");
}
