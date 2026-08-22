//! CX3D tutorial `IntracellularDiffusion` (Zubler & Douglas 2009) — Rust port.
//!
//! The soma secretes "tubulin" into its intracellular pool; it diffuses along
//! the neurite tree (1D compartment diffusion, D = 1000, d = 0.01). The
//! growth cone reads the local tubulin concentration, moves with
//! `speed = c·5000` (capped at 100), consumes `c·100` per step, and may
//! bifurcate (p = 0.003). Growth is thus *fuel-limited*.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/IntracellularDiffusion.java`

use prometheus_grow::{Bounds, CellType, Grid, Host, Module, Vec3, World};

/// Soma module: `modifyIntracellularQuantity("tubulin", 60)` every step.
struct InternalSecretor;

impl Module for InternalSecretor {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Soma(cell) = host else { return };
        world.modify_intracellular(cell, "tubulin", 60.0);
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(InternalSecretor)
    }
}

/// Growth cone (CX3D `IntracellularDiffusion$GrowthCone`): speed from local
/// tubulin, consume as you move, bifurcate with fixed probability.
struct GrowthCone {
    previous_dir: Option<Vec3>,
}

impl Module for GrowthCone {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else { return };
        if !world.cells[cell].neurites[idx].is_terminal {
            return;
        }
        let concentration = world.neurite_intracellular(cell, idx, "tubulin");
        // A growth cone never moves backwards: speed ∈ [0, 100] (CX3D caps at 100).
        let speed = (concentration * 5000.0).clamp(0.0, 100.0); // speedFactor
        let prev = self.previous_dir.unwrap_or_else(|| world.cells[cell].neurites[idx].axis());
        let dir = (prev + world.random_vec(0.1)).normalized();
        self.previous_dir = Some(dir);
        world.extend_neurite(cell, idx, dir * speed * world.dt);
        world.modify_neurite_intracellular(cell, idx, "tubulin", -concentration.max(0.0) * 100.0);

        if world.rand() < 0.003 {
            world.bifurcate_neurite(cell, idx, std::f64::consts::PI / 3.0, 1.5);
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(GrowthCone { previous_dir: None })
    }
}

fn main() {
    let origin = Vec3::new(-1000.0, -1000.0, -1000.0);
    let grid = Grid::new(21, 21, 21, 100.0, origin);
    let mut w = World::new(
        grid,
        0.1, // CX3D SIMULATION_TIME_STEP ≈ 1/60 — stable explicit diffusion
        Bounds { min: origin, max: origin + Vec3::new(2100.0, 2100.0, 2100.0) },
        1,
    );
    w.max_neurites = 4000;
    // 18 inert nodes.
    for _ in 0..18 {
        let p = w.random_vec(500.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
    w.add_soma_module(c, Box::new(InternalSecretor));
    let n = w.new_neurite(c, true, 1.0);
    w.extend_neurite(c, n, Vec3::new(0.0, 0.0, 1.0));
    w.add_neurite_module(c, n, Box::new(GrowthCone { previous_dir: None }));

    let len = |w: &World, ni: usize| -> f64 {
        w.cells[c].neurites[ni].points.windows(2).map(|p| (p[1] - p[0]).len()).sum()
    };

    println!("=== CX3D tutorial: IntracellularDiffusion ===");
    let mut len_log = Vec::new();
    for step in 0..1500 {
        w.step();
        if step % 250 == 0 {
            len_log.push((step, len(&w, n)));
        }
    }
    let soma_tub = w.intracellular(c, "tubulin");
    let tip_tub = w.neurite_intracellular(c, n, "tubulin");
    println!("нейритов: {}  терминалов: {}", w.total_neurites(), w.cells[c].neurites.iter().filter(|x| x.is_terminal).count());
    println!("длина первичного нейрита: {:.0}", len(&w, n));
    println!("тубулин: сома {soma_tub:.1}, кончик {tip_tub:.3} (концентрация)");
    println!("рост длины: {:?}", len_log.iter().map(|(t, l)| format!("t{t}={l:.0}")).collect::<Vec<_>>().join(", "));
    println!("рост идёт за счёт внутриклеточного тубулина (fuel-limited growth)");
    assert!(len(&w, n) > 500.0, "конус должен вырасти, got {:.0}", len(&w, n));
    assert!(soma_tub > 0.0, "сома должна накопить тубулин");
    assert!(soma_tub.is_finite() && tip_tub.is_finite(), "нет NaN: {soma_tub} {tip_tub}");
}
