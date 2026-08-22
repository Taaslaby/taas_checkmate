//! grow_timelapse.rs — BioDynaMo-style timelapse export (the `.pvd` idea).
//!
//! A small network (6 neurons, axon + dendrite per neuron) grows up an "L1"
//! gradient with concentration-dependent branching. Every `CAPTURE_EVERY`
//! steps the scene is captured as a *compact delta* (new neurites, appended
//! tip points, new excrescences/synapses — [`Timelapse`]), and the whole
//! movie is injected into the WebGL viewer (`visualizer/index.html`), which
//! replays growth on a timeline.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_timelapse
//! ```

use prometheus_grow::{
    exporter::Timelapse, ArtificialGradient, Bounds, CellType, Grid, Host, Module, Substance, Vec3,
    World,
};

const CAPTURE_EVERY: usize = 5;
const STEPS: usize = 100;

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
        0.2,
        Bounds { min: origin, max: origin + Vec3::new(10250.0, 10250.0, 8250.0) },
        3,
    );
    w.max_neurites = 6000;
    w.synapse_dist = 100.0;

    w.add_substance(Substance::new(
        "L1",
        0.0,
        0.0,
        &grid,
        vec![ArtificialGradient::Gaussian { axis: 2, max: 1.0, mean: 4000.0, sigma: 600.0 }],
    ));
    for _ in 0..6 {
        let p = w.random_vec(5000.0);
        w.add_cell(p, CellType::Glia, 5.0);
    }

    for i in 0..6 {
        let (p, ty) = if i < 3 {
            (Vec3::new(w.random_vec(400.0).x, w.random_vec(400.0).y, 0.0), CellType::Excitatory)
        } else {
            (Vec3::new(w.random_vec(400.0).x, w.random_vec(400.0).y, 2000.0), CellType::Inhibitory)
        };
        let c = w.add_cell(p, ty, 5.0);
        for (axon, diam) in [(true, 0.5), (false, 1.5)] {
            let n = w.new_neurite(c, axon, diam);
            w.add_neurite_module(c, n, Box::new(NeuriteChemoAttraction {
                substance: "L1".into(),
                branching_factor: 0.02,
                direction: None,
            }));
        }
    }

    // Timelapse: cells serialized once, then compact growth deltas.
    let mut tl = Timelapse::new(&w);
    tl.capture(&w); // frame 0: the initial neurites

    println!("=== grow_timelapse: рост + таймлайн ===");
    for step in 0..STEPS {
        w.step();
        if (step + 1) % CAPTURE_EVERY == 0 {
            tl.capture(&w);
        }
    }
    // Wiring finale: excrescences + contact synapses, then the final frame.
    w.make_excrescences(300.0, 150.0);
    w.form_synapses_by_contact(150.0);
    tl.capture(&w);

    let json = tl.finalize(&w);
    // WebGL viewer (timelapse) — embed the JSON in the HTML.
    let template = std::fs::read_to_string("visualizer/viewer.html").expect("viewer template");
    let html = template.replace("/*__SCENE_JSON__*/ null", &json);
    std::fs::write("visualizer/index.html", &html).expect("write index.html");
    // Native Rust viewer — plain JSON file.
    std::fs::write("visualizer/scene.json", &json).expect("write scene.json");

    let frames = json.matches("\"add_neurites\"").count();
    println!(
        "нейритов: {}  синапсов: {}  кадров: {frames}",
        w.total_neurites(),
        w.synapses.len()
    );
    println!("экспорт: visualizer/index.html ({:.0} KB) + scene.json ({:.0} KB)", html.len() as f64 / 1024.0, json.len() as f64 / 1024.0);
    assert!(w.synapses.len() > 0, "должны образоваться синапсы");
    assert!(frames > 5, "должно быть несколько кадров, got {frames}");
}
