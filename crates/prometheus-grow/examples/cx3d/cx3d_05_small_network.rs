//! CX3D tutorial `SmallNetwork` (Zubler & Douglas 2009) — Rust port.
//!
//! 8 neurons (4 excitatory at z≈0, 4 inhibitory at z≈2000) each extend an
//! axon and a dendrite; both chemotax up the "L1" gradient (max 1.0 at
//! z = 4000) and branch with probability ∝ concentration. After the growth
//! phase, `ConnectionsMaker` places boutons/spines and synapses form on
//! contact. The grown scene is exported to the WebGL visualizer.
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/SmallNetwork.java`

use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, Grid, Host, Module, Substance, Vec3, World,
};

struct NeuriteChemoAttraction {
    substance: String,
    branching_factor: f64,
    direction: Option<Vec3>,
}

impl Module for NeuriteChemoAttraction {
    fn run(&mut self, world: &mut World, host: Host) {
        let Host::Neurite { cell, idx } = host else { return };
        if !world.cells[cell].neurites[idx].is_terminal {
            return;
        }
        let tip = world.cells[cell].neurites[idx].tip();
        let concentration = world.concentration(&self.substance, tip);
        let mut grad = world.gradient(&self.substance, tip);
        if concentration > 0.3 {
            grad = Vec3::ZERO;
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
    let origin = Vec3::new(-5000.0, -5000.0, -2500.0);
    let grid = Grid::new(41, 41, 33, 250.0, origin);
    let mut w = World::new(
        grid,
        0.2, // CX3D SIMULATION_TIME_STEP ≈ 1/60 — slow growth to the gradient scale
        Bounds { min: origin, max: origin + Vec3::new(10250.0, 10250.0, 8250.0) },
        0,
    );
    w.max_neurites = 8000;
    w.synapse_dist = 100.0;

    // L1 gradient (CX3D: gaussian along Z, max 1.0, mean 400, σ 60 ×10).
    w.add_substance(Substance::new(
        "L1",
        0.0,
        0.0,
        &grid,
        vec![ArtificialGradient::Gaussian { axis: 2, max: 1.0, mean: 4000.0, sigma: 600.0 }],
    ));
    // 10 inert nodes.
    for _ in 0..10 {
        let p = w.random_vec(5000.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    // 4 excitatory at z≈0, 4 inhibitory at z≈2000.
    for i in 0..8 {
        let (p, ty) = if i < 4 {
            (Vec3::new(w.random_vec(400.0).x, w.random_vec(400.0).y, 0.0), CellType::Excitatory)
        } else {
            (Vec3::new(w.random_vec(400.0).x, w.random_vec(400.0).y, 2000.0), CellType::Inhibitory)
        };
        let c = w.add_cell(p, ty, 5.0);
        let axon = w.new_neurite(c, true, 0.5);
        w.add_neurite_module(c, axon, Box::new(NeuriteChemoAttraction {
            substance: "L1".into(),
            branching_factor: 0.02,
            direction: None,
        }));
        let dend = w.new_neurite(c, false, 1.5);
        w.add_neurite_module(c, dend, Box::new(NeuriteChemoAttraction {
            substance: "L1".into(),
            branching_factor: 0.02,
            direction: None,
        }));
    }

    println!("=== CX3D tutorial: SmallNetwork ===");
    for _ in 0..300 {
        w.step();
    }
    println!("рост: {} нейритов, {} терминалов", w.total_neurites(), count_terminal(&w));

    // ConnectionsMaker: excrescences + synapses on contact.
    w.make_excrescences(300.0, 150.0);
    w.form_synapses_by_contact(150.0);

    let exc: usize = w.cells.iter().flat_map(|c| &c.neurites).map(|n| n.excrescences.len()).sum();
    let g = w.to_graph();
    let exc_n = g.edges.iter().filter(|e| e.kind == prometheus_grow::SynapseKind::Excitatory).count();
    let inh_n = g.edges.iter().filter(|e| e.kind == prometheus_grow::SynapseKind::Inhibitory).count();
    println!("синапсы: {} (возб. {exc_n}, торм. {inh_n}), бутоны/шипики: {exc}", g.edges.len());

    // Export to the WebGL visualizer (same pipeline as `grow_scene`).
    let json = prometheus_grow::exporter::scene_json(&w);
    let template = std::fs::read_to_string("visualizer/viewer.html").expect("viewer template");
    let html = template.replace("/*__SCENE_JSON__*/ null", &json);
    std::fs::write("visualizer/index.html", &html).expect("write index.html");
    println!("экспорт: visualizer/index.html ({:.0} KB)", html.len() as f64 / 1024.0);

    assert!(g.edges.len() > 0, "должны образоваться синапсы");
}

fn count_terminal(w: &World) -> usize {
    w.cells.iter().flat_map(|c| &c.neurites).filter(|n| n.is_terminal).count()
}
