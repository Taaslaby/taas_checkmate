//! BioDynaMo `demo/binding_cells` (simplified) — random-walking immune cells
//! that physically bind on contact.
//!
//! Two populations (T-cells and monocytes) random-walk in a small volume.
//! When two cells come within contact distance, they form a bond: both stop
//! walking and are marked `bonded` — the pair stays together. The experiment
//! counts how many bonds form over time.
//!
//! Original: `forks/biodynamo-master/demo/binding_cells/src/binding_cells.h`
//! (BDM uses PhysicalBondModule + SpringForce; here the bond freezes the pair,
//! which is the observable outcome BDM tracks: bound pairs stay together.)
//!
//! ```bash
//! cargo run -p prometheus-grow --example bdm_04_binding_cells
//! ```

use prometheus_grow::{CellModule, CellType, Vec3, World};

const BONDED: i32 = 1;

/// BDM `RandomWalkModule(speed)`: move the soma by a random step.
struct RandomWalk {
    speed: f64,
}

impl CellModule for RandomWalk {
    fn run(&mut self, world: &mut World, cell: usize) {
        if world.cells[cell].sub_role == BONDED {
            return; // physically bound — no more walking
        }
        let step = world.random_vec(self.speed);
        world.move_soma(cell, step);
    }

    fn boxed_copy(&self) -> Box<dyn CellModule> {
        Box::new(RandomWalk { speed: self.speed })
    }
}

fn main() {
    let side = 40.0;
    let mut w = World::new(
        prometheus_grow::Grid::new(20, 20, 20, side / 20.0, Vec3::new(-side / 2.0, -side / 2.0, -side / 2.0)),
        1.0,
        prometheus_grow::Bounds {
            min: Vec3::new(-side, -side, -side),
            max: Vec3::new(side, side, side),
        },
        17,
    );

    // T-cells (272 in BDM) + monocytes (727) — scale down 20× for speed.
    let n_t = 14;
    let n_m = 36;
    let contact = 2.0; // sum of typical diameters / 2
    let mut bonds = 0usize;

    for _ in 0..n_t {
        let pos = w.random_vec(side / 2.0);
        let c = w.add_cell(pos, CellType::Inhibitory, 0.45); // t-cell
        w.add_cell_module(c, Box::new(RandomWalk { speed: 1.2 }));
    }
    for _ in 0..n_m {
        let pos = w.random_vec(side / 2.0);
        let c = w.add_cell(pos, CellType::Excitatory, 0.75); // monocyte
        w.add_cell_module(c, Box::new(RandomWalk { speed: 0.8 }));
    }

    println!("=== BioDynaMo demo: binding_cells (случайное блуждание + связывание) ===");
    println!("T-клеток {n_t}, моноцитов {n_m}, контакт на дистанции {contact}, 40 шагов");

    for step in 0..40 {
        w.step();
        // bond pass: any two unbonded cells within contact distance bond.
        for a in 0..w.cells.len() {
            if w.cells[a].sub_role == BONDED {
                continue;
            }
            let pos = w.cells[a].soma.pos;
            for b in w.objects_near(pos, contact) {
                if b == a || w.cells[b].sub_role == BONDED {
                    continue;
                }
                if (w.cells[b].soma.pos - pos).len() <= contact {
                    w.cells[a].sub_role = BONDED;
                    w.cells[b].sub_role = BONDED;
                    bonds += 1;
                    println!("  t={step:>2}: связь между клетками {a} и {b} (всего {bonds})");
                }
            }
        }
    }
    println!("итого связей: {bonds} из возможных {}", (n_t + n_m) / 2);
    // BDM's own demo reports bond counts; a few bonds must have formed.
    assert!(bonds >= 2, "клетки должны связаться при контакте, got {bonds}");
    println!("✓ случайное блуждание свело клетки, контакты образовали связи");
}
