//! grow_scene.rs — grow a small but pretty network and export it to the
//! WebGL visualizer (`visualizer/index.html`).
//!
//! 48 neurons in 6 type groups: each group's axons are steered by
//! `PointAttraction` toward its own projection region, dendrites grow a local
//! arbor via `NetmorphGrowth`, synapses form on bouton/spine contact.
//! The final geometry is exported as JSON and injected into the viewer
//! template (`visualizer/viewer.html` → `visualizer/index.html`).
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_scene
//! ```

use prometheus_grow::{
    BalancedForces, Bounds, CellType, Grid, NetmorphGrowth, PointAttraction, Recognition, Vec3,
    World,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const N_PER_GROUP: usize = 8;
const N_GROUPS: usize = 6;
const N: usize = N_PER_GROUP * N_GROUPS;
const STEPS: usize = 70;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // 6 type groups: 3 excitatory, 3 inhibitory.
    let types: Vec<CellType> = (0..N_GROUPS)
        .flat_map(|g| std::iter::repeat(if g < 3 { CellType::Excitatory } else { CellType::Inhibitory }).take(N_PER_GROUP))
        .collect();
    let tags: Vec<String> = (0..N_GROUPS)
        .flat_map(|g| std::iter::repeat(format!("T{g}")).take(N_PER_GROUP))
        .collect();

    // Somata in a ball around the origin.
    let mut pos: Vec<Vec3> = Vec::with_capacity(N);
    for _ in 0..N {
        let d = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
        let d = if d.len() < 1e-9 { Vec3::new(1.0, 0.0, 0.0) } else { d.normalized() };
        let r = 6000.0 * rng.gen::<f64>().cbrt();
        pos.push(d * r);
    }

    // One projection region per group, around a sphere at distance 18000.
    let targets: Vec<Vec3> = (0..N_GROUPS)
        .map(|g| {
            let a = g as f64 / N_GROUPS as f64 * std::f64::consts::TAU;
            let z = (g as f64 / (N_GROUPS - 1) as f64 - 0.5) * 2.0;
            Vec3::new(a.cos() * 16000.0, z * 14000.0, a.sin() * 16000.0)
        })
        .collect();

    // World: grid covering everything (somata + targets + grown axons).
    let origin = Vec3::new(-40000.0, -40000.0, -40000.0);
    let cell = 2000.0;
    let grid = Grid::new(40, 40, 40, cell, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(cell * 40.0, cell * 40.0, cell * 40.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 42);
    w.max_neurites = 200_000;

    for i in 0..N {
        let id = w.add_cell(pos[i], types[i], 340.0);
        debug_assert_eq!(id, i);
        w.cells[i].tag = tags[i].clone();

        // Axon: identity-directed guidance toward the group's projection region.
        let axon = w.new_neurite(id, true, 60.0);
        let guide = PointAttraction::new(targets[i / N_PER_GROUP], 500.0).stop_distance(1500.0);
        w.add_neurite_module(id, axon, Box::new(guide));

        // Dendrite: local arbor via NetmorphGrowth, seeded in a random direction.
        let dend = w.new_neurite(id, false, 90.0);
        let init = w.random_vec(120.0);
        w.extend_neurite(id, dend, init);
        w.add_neurite_module(
            id,
            dend,
            Box::new(
                NetmorphGrowth::new(300.0, 0.74, 5.0, 2.0, 50.0, 0.0, 0.0)
                    .max_prob(0.05)
                    .eri_std(0.1)
                    .step(100.0)
                    .branch(BalancedForces::netmorph_defaults()),
            ),
        );
    }

    for _ in 0..STEPS {
        w.step();
    }
    w.synapses.clear();

    // Wire by bouton/spine contact on the grown geometry.
    w.make_excrescences(1400.0, 1000.0);
    w.set_recognition(Recognition::all());
    w.form_synapses_by_contact(700.0);

    // Stats.
    let n_neurites: usize = w.cells.iter().map(|c| c.neurites.len()).sum();
    let n_segments: usize = w.cells.iter().flat_map(|c| &c.neurites).map(|n| n.points.len().saturating_sub(1)).sum();
    let n_exc: usize = w.cells.iter().flat_map(|c| &c.neurites).map(|n| n.excrescences.len()).sum();
    println!("=== grown scene ===");
    println!("клеток: {}  нейритов: {n_neurites}  сегментов: {n_segments}", w.cells.len());
    println!("бутонов/шипиков: {n_exc}  синапсов: {}", w.synapses.len());
    println!("экспорт в visualizer/index.html …");

    // Inject the scene into the viewer template.
    let json = prometheus_grow::exporter::scene_json(&w);
    let template = std::fs::read_to_string("visualizer/viewer.html").expect("viewer template");
    let html = template.replace("/*__SCENE_JSON__*/ null", &json);
    let bytes = html.len();
    std::fs::write("visualizer/index.html", &html).expect("write index.html");
    println!("готово: visualizer/index.html ({:.1} KB)", bytes as f64 / 1024.0);
}
