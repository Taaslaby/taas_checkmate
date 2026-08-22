//! CX3D tutorial `MembraneContact` (Zubler & Douglas 2009) — Rust port.
//!
//! 10 "red" somas carry a large membrane load of substance "A". 10 "violet"
//! somas walk randomly; the moment one touches a soma whose membrane
//! concentration of "A" exceeds 1, it turns (stops, flags itself) — the
//! CX3D contact-mediated signaling demo.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/MembraneContact.java`
//! Note: CX3D's `cleanAllLocalBiologyModules()` (stop all behaviour on
//! contact) is expressed here as a `contacted` membrane flag that the walk
//! module checks — same effect, explicit state.

use prometheus_grow::{Bounds, CellType, Grid, Host, Module, Vec3, World};

/// Soma random walk (port of `SomaRandomWalkModule`), stops once the soma
/// has been marked `contacted` (analog of `cleanAllLocalBiologyModules`).
struct Walk {
    direction: Option<Vec3>,
}

impl Module for Walk {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return };
        if world.membrane_concentration(cell, "contacted") > 0.5 {
            return; // stopped — the module list was "cleaned"
        }
        let speed = 50.0;
        let mut dir = self.direction.unwrap_or_else(|| world.random_vec(1.0).normalized());
        dir = (dir + world.random_vec(0.1)).normalized();
        self.direction = Some(dir);
        world.move_soma(cell, dir * speed * world.dt);
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(Walk { direction: None })
    }
}

/// CX3D `MembraneContact`: for every object in contact with my soma, if its
/// membrane "A" > 1 — mark myself contacted (CX3D: turn yellow + clean).
struct MembraneContact {
    contacted: bool,
}

impl Module for MembraneContact {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return };
        if self.contacted {
            return;
        }
        let p = world.cells[cell].soma.pos;
        let r = world.cells[cell].soma.radius;
        for other in world.objects_near(p, r * 3.0) {
            if other != cell && world.membrane_concentration(other, "A") > 1.0 {
                self.contacted = true;
                world.modify_membrane(cell, "contacted", 1.0);
                break;
            }
        }
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(MembraneContact { contacted: self.contacted })
    }
}

fn main() {
    let origin = Vec3::new(-400.0, -400.0, -400.0);
    let grid = Grid::new(11, 11, 11, 80.0, origin);
    let mut w = World::new(
        grid,
        1.0,
        Bounds { min: origin, max: origin + Vec3::new(880.0, 880.0, 880.0) },
        1,
    );

    // 10 red somas loaded with membrane "A" (CX3D: ×100000).
    let mut red = Vec::new();
    for _ in 0..10 {
        let p = w.random_vec(100.0);
        let id = w.add_cell(p, CellType::Excitatory, 10.0);
        w.modify_membrane(id, "A", 100000.0);
        red.push(id);
    }
    // 10 violet somas: random walk + membrane-contact detector.
    let mut violet = Vec::new();
    for _ in 0..10 {
        let p = w.random_vec(50.0);
        let id = w.add_cell(p, CellType::Inhibitory, 10.0);
        w.add_soma_module(id, Box::new(Walk { direction: None }));
        w.add_soma_module(id, Box::new(MembraneContact { contacted: false }));
        violet.push(id);
    }

    println!("=== CX3D tutorial: MembraneContact ===");
    let mut contacted = 0;
    for step in 0..4000 {
        w.step();
        contacted = violet
            .iter()
            .filter(|&&id| w.membrane_concentration(id, "contacted") > 0.5)
            .count();
        if contacted >= 5 {
            println!("  t={step}: {contacted}/10 фиолетовых коснулись красных");
            break;
        }
    }
    println!("фиолетовых, коснувшихся мембраны «A»: {contacted}/10");
    assert!(contacted >= 2, "контактная сигнализация должна сработать, got {contacted}");
    println!("координаты фиолетовых: {:?}", violet.iter().map(|&i| w.cells[i].soma.pos.z.round() as i32).collect::<Vec<_>>());
    let _ = red;
}
