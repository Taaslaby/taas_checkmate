//! CX3D tutorial `RandomBranchingModule` (Zubler & Douglas 2009) — Rust port.
//!
//! One growth cone does a correlated random walk; each step it may bifurcate
//! (p = 0.005, daughters symmetric) or emit a side branch (p = 0.005).
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/RandomBranchingModule.java`

use prometheus_grow::{Bounds, CellType, Grid, Host, Module, Vec3, World};

/// CX3D `RandomBranchingModule`: jitter the direction, move the tip, then
/// randomly bifurcate or branch.
struct RandomBranching {
    direction: Option<Vec3>,
}

impl Module for RandomBranching {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else { return };
        if !world.cells[cell].neurites[idx].is_terminal {
            return; // deleted after bifurcation (CX3D)
        }
        let speed = 100.0;
        let mut dir = self.direction.unwrap_or_else(|| world.cells[cell].neurites[idx].axis());
        dir = (dir + world.random_vec(0.1)).normalized(); // deltaDirection
        self.direction = Some(dir);
        world.extend_neurite(cell, idx, dir * speed * world.dt);

        let p = 0.005;
        if world.rand() < p {
            world.bifurcate_neurite(cell, idx, std::f64::consts::PI / 3.0, 1.5);
        } else if world.rand() < p {
            world.branch_neurite(cell, idx); // side branch
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(RandomBranching { direction: None })
    }
}

fn main() {
    let origin = Vec3::new(-2000.0, -2000.0, -2000.0);
    let grid = Grid::new(25, 25, 25, 160.0, origin);
    let mut w = World::new(
        grid,
        1.0,
        Bounds { min: origin, max: origin + Vec3::new(4000.0, 4000.0, 4000.0) },
        7,
    );
    w.max_neurites = 4000;
    // 18 inert nodes (CX3D `randomNoise(1000, 3)` ×2 for our dt scale).
    for _ in 0..18 {
        let p = w.random_vec(2000.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    let p0 = w.random_vec(80.0);
    let c = w.add_cell(p0, CellType::Excitatory, 5.0);
    let n = w.new_neurite(c, true, 2.0);
    w.extend_neurite(c, n, Vec3::new(0.0, 0.0, 1.0));
    w.add_neurite_module(c, n, Box::new(RandomBranching { direction: None }));

    println!("=== CX3D tutorial: RandomBranchingModule ===");
    for _ in 0..600 {
        w.step();
    }
    let terminals: usize = w.cells[c].neurites.iter().filter(|x| x.is_terminal).count();
    println!("всего нейритов: {}  терминалов: {terminals}", w.total_neurites());
    println!("случайное блуждание + бифуркации/боковые ветви");
    assert!(w.total_neurites() > 3, "должно быть несколько ветвей");
}

fn _unused(_: &World) {}
