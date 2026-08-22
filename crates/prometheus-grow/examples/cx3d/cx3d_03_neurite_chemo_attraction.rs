//! CX3D tutorial `NeuriteChemoAttraction` (Zubler & Douglas 2009) — Rust port.
//!
//! One growth cone climbs a gaussian concentration of "A" (max 1.0 at
//! z = 4000, σ = 600 — tutorial values ×10 to match our dt = 1.0) and
//! branches with probability ∝ local concentration. Near the peak
//! (concentration > 0.3) the gradient is ignored and the cone wanders.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/NeuriteChemoAttraction.java`

use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, Grid, Host, Module, Substance, Vec3, World,
};

/// CX3D `NeuriteChemoAttraction`: steering = old direction (weight 1.0)
/// + gradient (0.2) + noise (0.6); branch with p = concentration·factor.
struct NeuriteChemoAttraction {
    substance: String,
    branching_factor: f64,
    direction: Option<Vec3>,
}

impl Module for NeuriteChemoAttraction {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else { return };
        // Deleted from the parent after bifurcation: only terminals grow.
        if !world.cells[cell].neurites[idx].is_terminal {
            return;
        }
        let tip = world.cells[cell].neurites[idx].tip();
        let concentration = world.concentration(&self.substance, tip);
        let mut grad = world.gradient(&self.substance, tip);
        if concentration > 0.3 {
            grad = Vec3::ZERO; // CX3D: near the peak the gradient is ignored
        }
        let old = self.direction.unwrap_or_else(|| world.cells[cell].neurites[idx].axis());
        let mut step_dir = old + grad.normalized() * 0.2 + world.random_vec(0.6);
        if step_dir.len() < 1e-9 {
            step_dir = old;
        }
        let speed = 100.0;
        world.extend_neurite(cell, idx, step_dir.normalized() * speed * world.dt);
        self.direction = Some((old * 5.0 + step_dir).normalized());

        if world.rand() < concentration * self.branching_factor {
            world.bifurcate_neurite(cell, idx, std::f64::consts::PI / 3.0, 1.5);
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(NeuriteChemoAttraction {
            substance: self.substance.clone(),
            branching_factor: self.branching_factor,
            direction: None,
        })
    }
}

fn main() {
    let origin = Vec3::new(-5000.0, -5000.0, -3000.0);
    let grid = Grid::new(41, 41, 33, 250.0, origin);
    let mut w = World::new(
        grid,
        0.1, // CX3D SIMULATION_TIME_STEP ≈ 1/60 — slow growth to the gradient scale
        Bounds { min: origin, max: origin + Vec3::new(10250.0, 10250.0, 8250.0) },
        0,
    );
    w.max_neurites = 4000;
    // Artificial gaussian gradient (CX3D `addArtificialGaussianConcentrationZ`,
    // max 1.0, mean 400, σ 160 — scaled ×10 for our dt).
    w.add_substance(Substance::new(
        "A",
        0.0,
        0.0,
        &grid,
        vec![ArtificialGradient::Gaussian { axis: 2, max: 1.0, mean: 4000.0, sigma: 600.0 }],
    ));
    // 10 inert nodes (CX3D `getPhysicalNodeInstance(randomNoise(500, 3))` ×10).
    for _ in 0..10 {
        let p = w.random_vec(5000.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    let c = w.add_cell(Vec3::ZERO, CellType::Excitatory, 5.0);
    let n = w.new_neurite(c, true, 2.0);
    w.add_neurite_module(c, n, Box::new(NeuriteChemoAttraction {
        substance: "A".into(),
        branching_factor: 0.02, // SmallNetwork value; default 0.005 is sparse
        direction: None,
    }));

    println!("=== CX3D tutorial: NeuriteChemoAttraction ===");
    println!("пик градиента «A»: z = 4000 (σ 600)");
    let mut max_z: f64 = 0.0;
    for _ in 0..800 {
        w.step();
        let z = w.cells[c].neurites[n].tip().z;
        max_z = max_z.max(z);
    }
    let tip = w.cells[c].neurites[n].tip();
    let terms = count_terminal(&w, c);
    println!("нейритов: {}  терминалов: {terms}", w.total_neurites());
    println!("макс. z: {max_z:.0}, кончик: ({:.0}, {:.0}, {:.0})", tip.x, tip.y, tip.z);
    println!("рост к пику z=4000, ветвление у пика");
    assert!(max_z > 3200.0, "конус должен подняться к пику, max z = {max_z:.0}");
    assert!(terms > 1, "должны появиться ветви");
}

fn count_terminal(w: &World, cell: usize) -> usize {
    w.cells[cell].neurites.iter().filter(|n| n.is_terminal).count()
}
