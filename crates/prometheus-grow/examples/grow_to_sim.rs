//! Bridge: grown connectome → `taas-checkmate` spike simulator.
//!
//! Grows a half-center pair from development rules (one GABA progenitor
//! divides, both cells extend axons that chase a shared attractant,
//! `form_synapses` wires them into a mutual-inhibition pair), then writes
//! the connectome in the `taas_checkmate.connectome_brain` input format:
//! CSR `row_ptr`/`col_indices`/`values` + `nt_types` (+1/−1) + exc/inh groups.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_to_sim
//! # → visualizer/connectome.json
//! ```
//!
//! Feed the file to the simulator from Python:
//! ```python
//! from taas_checkmate import connectome_brain, BrainRunner
//! import json
//! d = json.load(open("visualizer/connectome.json"))
//! b = BrainRunner(connectome_brain(d["row_ptr"], d["col_indices"], d["values"],
//!                                 d["nt_types"], d["groups"]))
//! spikes = b.run(1000)
//! ```

use prometheus_grow::{
    exporter::connectome_json, Bounds, CellType, ChemoAttraction, DivideOnce, Grid, Secretor,
    Substance, Vec3, World,
};
use std::io::Write;

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    // Shared attractant ("neuropil") both cells project into.
    world.add_substance(Substance::new("A", 2.5, 0.001, &world.grid, vec![]));

    // One GABA progenitor → 2 inhibitory cells.
    let c = world.add_cell(Vec3::ZERO, CellType::Inhibitory, 4.0);
    world.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
    world.add_soma_module(c, Box::new(Secretor::new("A", 0.05)));
    for _ in 0..30 {
        world.step();
        if world.cells.len() >= 2 {
            break;
        }
    }

    // Freeze the population, grow an axon from each cell toward the attractant.
    world.max_cells = world.cells.len();
    for ci in 0..world.cells.len() {
        let n = world.new_neurite(ci, true, 0.5);
        let mut chemo = ChemoAttraction::new("A");
        chemo.noise_weight = 0.1;
        world.add_neurite_module(ci, n, Box::new(chemo));
    }
    for _ in 0..120 {
        world.step();
    }

    // Structural check: the half-center motif must be there before exporting.
    let g = world.to_graph();
    let w01 = g.edges.iter().find(|e| e.pre == 0 && e.post == 1).map(|e| e.weight).unwrap_or(0.0);
    let w10 = g.edges.iter().find(|e| e.pre == 1 && e.post == 0).map(|e| e.weight).unwrap_or(0.0);
    let mutual = w01 > 0.0 && w10 > 0.0;
    println!(
        "grown graph: {} nodes, {} edges | mutual inhibition: {mutual} (w01={w01:.3}, w10={w10:.3})",
        g.nodes.len(),
        g.edges.len()
    );

    // Export in taas-checkmate ConnectomeData format.
    let json = connectome_json(&world);
    let path = "visualizer/connectome.json";
    let mut f = std::fs::File::create(path).expect("create connectome.json");
    f.write_all(json.as_bytes()).expect("write connectome.json");
    println!("wrote {path} ({} bytes)", json.len());

    if !mutual {
        println!("WARN: growth did not produce a mutual pair — simulator input still valid, just no loop");
    }
}
