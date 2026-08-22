//! BioDynaMo `demo/pyramidal_cell` — a pyramidal neuron grown by gradient
//! guidance.
//!
//! Two static substance fields form Gaussian bands along Z (apical peak at
//! the top, basal at the bottom). The soma sprouts 1 apical + 3 basal
//! neurites; each growth cone elongates along a weighted blend of its old
//! direction, a random component and the local gradient, tapers its diameter
//! and stochastically branches. 500 steps — the arbor should grow upward in
//! the apical field and downward in the basal field.
//!
//! Original: `forks/biodynamo-master/demo/pyramidal_cell/src/pyramidal_cell.h`
//!
//! ```bash
//! cargo run -p prometheus-grow --example bdm_03_pyramidal_cell
//! ```

use prometheus_grow::{ArtificialGradient, Host, Module, Vec3, World};

/// BDM `ApicalDendriteGrowth` / `BasalDendriteGrowth` — one per neurite,
/// keeps the growth direction as state.
struct DendriteGrowth {
    substance: &'static str,
    gradient_weight: f64,
    randomness_weight: f64,
    old_direction_weight: f64,
    step_len: f64,
    taper: f64,
    branch_prob: f64,
    min_diameter: f64,
    dir: Vec3,
}

impl Module for DendriteGrowth {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else { return };
        let diam = world.cells[cell].neurites[idx].diameter;
        let terminal = world.cells[cell].neurites[idx].is_terminal;
        if diam <= self.min_diameter {
            return;
        }
        let p = world.cells[cell].neurites[idx].tip();
        // Keep the arbor inside the simulated volume (the soma sits at the
        // bottom, the apical band peaks at the top of the box).
        let inside = p.x >= 0.0 && p.x <= 300.0 && p.y >= 0.0 && p.y <= 300.0 && p.z >= 0.0 && p.z <= 300.0;
        if !inside {
            return;
        }
        let g = world.gradient(self.substance, p);
        let random_axis = world.random_vec(1.0);
        let new_dir = (self.dir.scale(self.old_direction_weight)
            + g.scale(self.gradient_weight)
            + random_axis.scale(self.randomness_weight))
            .normalized();
        self.dir = new_dir;
        // BDM elongates by `step_len` per simulation time unit; dt scales it
        // to a smooth per-step motion (BDM dt ≈ 0.01 → ~1 unit/step).
        world.extend_neurite(cell, idx, new_dir.scale(self.step_len * world.dt));
        world.cells[cell].neurites[idx].diameter -= self.taper * world.dt;

        // BDM branches only while the diameter is above the threshold — the
        // taper eventually stops the tree, so it doesn't explode exponentially.
        let branch = if terminal && diam > self.min_diameter {
            world.rand() < self.branch_prob
        } else {
            false
        };
        if branch {
            // The daughter is the last neurite of this cell; give it BDM's
            // daughter diameter (0.65). Its module copy inherits our current
            // `dir`, so it continues roughly along the parent's axis.
            world.branch_neurite(cell, idx);
            let daughter = world.cells[cell].neurites.len() - 1;
            if daughter != idx {
                world.cells[cell].neurites[daughter].diameter = 0.65;
            }
        }
    }

    fn copy_on_branch(&self) -> bool {
        true
    }

    fn boxed_copy(&self) -> Box<dyn Module> {
        Box::new(DendriteGrowth {
            substance: self.substance,
            gradient_weight: self.gradient_weight,
            randomness_weight: self.randomness_weight,
            old_direction_weight: self.old_direction_weight,
            step_len: self.step_len,
            taper: self.taper,
            branch_prob: self.branch_prob,
            min_diameter: self.min_diameter,
            dir: self.dir,
        })
    }
}

fn main() {
    let side = 300.0;
    let mut w = World::new(
        prometheus_grow::Grid::new(40, 40, 40, side / 40.0, Vec3::ZERO),
        0.01, // BDM dt: elongation 100/unit-time → ~1 unit/step
        prometheus_grow::Bounds {
            min: Vec3::ZERO,
            max: Vec3::new(side, side, side),
        },
        13,
    );
    w.max_neurites = 40_000;

    // CreateExtracellularSubstances: two static Gaussian bands along Z.
    let apical = prometheus_grow::Substance::new(
        "apical",
        0.0,
        0.0,
        &w.grid,
        vec![ArtificialGradient::Gaussian { axis: 2, max: 200.0, mean: side, sigma: 200.0 }],
    );
    let basal = prometheus_grow::Substance::new(
        "basal",
        0.0,
        0.0,
        &w.grid,
        vec![ArtificialGradient::Gaussian { axis: 2, max: 200.0, mean: 0.0, sigma: 200.0 }],
    );
    w.add_substance(apical);
    w.add_substance(basal);

    // AddInitialNeuron — soma placed at the vertical middle so both arbors
    // are visible: apical climbs to the top band, basal descends to the bottom
    // (BDM puts the soma at z=0, which clips half the basal arbor).
    let soma = w.add_cell(Vec3::new(150.0, 150.0, 150.0), prometheus_grow::CellType::Excitatory, 5.0);
    let mk = |substance: &'static str, gw: f64, rw: f64, ow: f64, len: f64, taper: f64, bp: f64, min_d: f64, dir: Vec3| {
        DendriteGrowth {
            substance,
            gradient_weight: gw,
            randomness_weight: rw,
            old_direction_weight: ow,
            step_len: len,
            taper,
            branch_prob: bp,
            min_diameter: min_d,
            dir,
        }
    };
    // BDM's own numbers; dt scaling makes the motion smooth (≈1 unit/step).
    // Taper 1.5e-3/unit-time: diameter 1.0 crosses the apical branching
    // threshold 0.55 at ~t 300, so the tree stops instead of exploding.
    let a = w.new_neurite(soma, false, 1.0);
    w.add_neurite_module(soma, a, Box::new(mk("apical", 0.06, 0.3, 4.0, 100.0, 1.5e-3, 0.03, 0.575, Vec3::new(0.0, 0.0, 1.0))));
    let b1 = w.new_neurite(soma, false, 1.0);
    w.add_neurite_module(soma, b1, Box::new(mk("basal", 0.03, 0.4, 6.0, 50.0, 1.5e-3, 0.006, 0.75, Vec3::new(0.0, 0.0, -1.0))));
    let b2 = w.new_neurite(soma, false, 1.0);
    w.add_neurite_module(soma, b2, Box::new(mk("basal", 0.03, 0.4, 6.0, 50.0, 1.5e-3, 0.006, 0.75, Vec3::new(0.0, 0.6, -0.8))));
    let b3 = w.new_neurite(soma, false, 1.0);
    w.add_neurite_module(soma, b3, Box::new(mk("basal", 0.03, 0.4, 6.0, 50.0, 1.5e-3, 0.006, 0.75, Vec3::new(0.3, -0.6, -0.8))));

    println!("=== BioDynaMo demo: pyramidal_cell (NeuriteElement по градиентам) ===");
    let mut last = 4usize;
    for step in 0..500 {
        w.step();
        let n = w.cells[soma].neurites.len();
        if n != last && (step + 1) % 25 == 0 {
            println!("  t={step:>3}: нейритов {n}");
            last = n;
        }
    }
    let neurites = w.cells[soma].neurites.len();
    // apical grows up (z+), basal grows down (z−): check mean tip z.
    let mut z_ap = 0.0f64;
    let mut z_ba = 0.0f64;
    let mut n_ap = 0usize;
    let mut n_ba = 0usize;
    // A neurite is apical iff it descends (via parents) from neurite 0.
    fn is_apical(neurites: &[prometheus_grow::Neurite], idx: usize) -> bool {
        let mut cur = idx;
        while let Some(p) = neurites[cur].parent {
            if p == 0 {
                return true;
            }
            cur = p;
        }
        cur == 0
    }
    for (i, nt) in w.cells[soma].neurites.iter().enumerate() {
        let z = nt.tip().z;
        if is_apical(&w.cells[soma].neurites, i) {
            z_ap += z;
            n_ap += 1;
        } else {
            z_ba += z;
            n_ba += 1;
        }
    }
    println!("нейритов: {neurites} (апикальных {n_ap}, базальных {n_ba})");
    println!("средний z кончиков: апикальные {:.1}, базальные {:.1} (сома z=150)", z_ap / n_ap.max(1) as f64, z_ba / n_ba.max(1) as f64);
    assert!(neurites > 8, "дерево должно разветвиться, got {neurites}");
    assert!(z_ap / n_ap.max(1) as f64 > 200.0, "апикальные должны расти к верхней полосе");
    assert!((z_ba / n_ba.max(1) as f64) < 100.0, "базальные должны расти к нижней полосе");
    println!("✓ пирамидальная клетка выросла по градиентам apical/basal");
}
