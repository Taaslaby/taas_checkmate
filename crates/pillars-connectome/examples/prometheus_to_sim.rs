//! Native bridge: prometheus-grow → pillars-connectome (no JSON round-trip).
//!
//! Grows a half-center pair with **prometheus-grow** (the CX3D/BDM-class
//! growth engine merged into this workspace) and hands the grown connectome
//! straight to pillars-connectome's `BrainRunner` (pillars SoA + SIMD LIF).
//! This is the end-to-end pipeline:
//!
//! ```text
//! genome (modules) → cells divide / secrete morphogens
//!                        └─► growth cones follow gradients
//!                               └─► neurites meet → synapses → GrownGraph
//!                                      └─► ConnectomeData (CSR) → BrainRunner → spikes
//! ```
//!
//! ```bash
//! cargo run -p pillars-connectome --example prometheus_to_sim
//! ```

use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use prometheus_grow::{
    Bounds, CellType, ChemoAttraction, DivideOnce, Grid, Secretor, Substance, Vec3, World,
};

fn main() {
    // ── 1. Growth: half-center pair from development rules ──────────────
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

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

    // Freeze population, grow an axon from each cell toward the attractant.
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

    let g = world.to_graph();
    let w01 = g.edges.iter().find(|e| e.pre == 0 && e.post == 1).map(|e| e.weight).unwrap_or(0.0);
    let w10 = g.edges.iter().find(|e| e.pre == 1 && e.post == 0).map(|e| e.weight).unwrap_or(0.0);
    println!(
        "grown: {} neurons, {} synapses | mutual inhibition: {} (w01={w01:.2}, w10={w10:.2})",
        g.nodes.len(),
        g.edges.len(),
        w01 > 0.0 && w10 > 0.0
    );

    // ── 2. Bridge: GrownGraph → ConnectomeData (CSR) ────────────────────
    let n = g.nodes.len();
    let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n];
    for e in &g.edges {
        per[e.pre].push((e.post, e.weight as f32));
    }
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_indices = Vec::with_capacity(g.edges.len());
    let mut values = Vec::with_capacity(g.edges.len());
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = col_indices.len() as i32;
        for &(post, w) in list {
            col_indices.push(post as i32);
            values.push(w);
        }
    }
    row_ptr[n] = col_indices.len() as i32;
    let nt_types: Vec<f32> = g
        .nodes
        .iter()
        .map(|nd| match nd.cell_type {
            prometheus_grow::CellType::Inhibitory => -1.0,
            _ => 1.0,
        })
        .collect();

    let connectome = ConnectomeData::new(
        row_ptr,
        col_indices,
        values,
        nt_types,
        vec![("inh", vec![0, 1])],
    );
    println!("connectome: {} neurons, {} synapses", connectome.n_neurons, connectome.n_synapses);

    // ── 3. Simulate: pillars SoA + SIMD LIF-alpha ───────────────────────
    let mut runner = BrainRunner::new(connectome, BrainConfig::lif_alpha(0.5))
        .expect("BrainRunner init");
    let n_steps = 4000;
    let mut ext = vec![0.0f32; n];
    ext[0] = 1.0; // tonic drive to neuron 0
    let spikes = runner.run(n_steps, &ext);

    let mut n0 = 0usize;
    let mut n1 = 0usize;
    for t in 0..n_steps {
        if spikes[t * n] != 0 {
            n0 += 1;
        }
        if spikes[t * n + 1] != 0 {
            n1 += 1;
        }
    }
    println!(
        "sim: {} steps | spikes neuron0={n0}, neuron1={n1} | rate0={:.1} Hz",
        n_steps,
        n0 as f64 * 1000.0 / (n_steps as f64 * 0.5)
    );
    println!(
        "{}",
        if n0 > 0 {
            "OK: grown connectome drives spiking in the pillars SIMD engine"
        } else {
            "WARN: no spikes — try stronger drive"
        }
    );
}
