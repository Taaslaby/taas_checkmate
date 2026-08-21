//! Growth → pruning: the refinement story (CX3D `RetractNeurite`).
//!
//! Two growth cones from one cell chase the same attractant. One is capped at
//! `max_len_um` and stalls short of the target; the other reaches it and
//! stops inside the source. In the pruning phase a `RetractNeurite` module is
//! attached: the cone that failed to reach the target sits in low
//! concentration, retracts and is **pruned**; the successful connection
//! survives. This is exuberant growth followed by elimination of unsuccessful
//! connections — the developmental refinement seen in real nervous systems.
//!
//! ```bash
//! cargo run -p pillars-grow --example grow_prune
//! ```

use pillars_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Grid, RetractNeurite, Substance, Vec3,
    World,
};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    // Wide synapse radius so the successful cone's contact persists through
    // the pruning phase's small noise-driven drift.
    world.synapse_dist = 15.0;

    // Analytic attractant peaking at the target soma (0, 0, 30).
    world.add_substance(Substance::new(
        "ntf",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Gaussian {
            axis: 2,
            max: 1.0,
            mean: 30.0,
            sigma: 8.0,
        }],
    ));

    // Progenitor at the origin; target cell near the gradient peak.
    let p = world.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
    let target = world.add_cell(Vec3::new(0.0, 0.0, 30.0), CellType::Inhibitory, 4.0);

    // Two growth cones chasing the same cue:
    //  - axon_a: unlimited → reaches the source and stops (concentration > 0.3).
    //  - axon_b: capped at 5 µm → stalls far away in near-zero concentration.
    let a = world.new_neurite(p, true, 0.5);
    let mut chemo_a = ChemoAttraction::new("ntf");
    chemo_a.noise_weight = 0.05;
    world.add_neurite_module(p, a, Box::new(chemo_a));

    let b = world.new_neurite(p, true, 0.5);
    let mut chemo_b = ChemoAttraction::new("ntf");
    chemo_b.noise_weight = 0.05;
    chemo_b.max_len_um = Some(5.0);
    world.add_neurite_module(p, b, Box::new(chemo_b));

    println!("=== growth phase ===");
    for _ in 0..200 {
        world.step();
    }
    println!(
        "  neurites: {} (axon_a len {:.1} µm, axon_b len {:.1} µm)",
        world.total_neurites(),
        world.cells[p].neurites[a].length_um(),
        world.cells[p].neurites[b].length_um(),
    );
    assert_eq!(world.total_neurites(), 2);

    // Pruning: retract any cone whose tip sits below threshold concentration.
    world.add_neurite_module(p, b, Box::new(RetractNeurite::new("ntf", 0.25, 10.0)));

    println!("=== pruning phase ===");
    for _ in 0..40 {
        world.step();
    }
    let survivors = world.total_neurites();
    let tip_a = world.cells[p].neurites[a].tip();
    let dist_target = tip_a.dist(world.cells[target].soma.pos);
    println!("  survivors: {survivors} (axon_a tip {tip_a:?}, dist to target {dist_target:.1} µm)");
    assert_eq!(
        survivors, 1,
        "the capped cone must be pruned; axon_a survives"
    );
    assert!(
        dist_target < 25.0,
        "surviving cone must sit near the target (dist {dist_target:.1})"
    );

    let g = world.to_graph();
    println!(
        "grown graph: {} nodes, {} edges",
        g.nodes.len(),
        g.edges.len()
    );
    for e in &g.edges {
        println!(
            "  {} -> {}  kind={:?}  w={:.3}",
            e.pre, e.post, e.kind, e.weight
        );
    }
    let connected = g.edges.iter().any(|e| e.pre == p && e.post == target);
    println!(
        "  target connection: {}\n  → {}",
        if connected {
            "formed ✓"
        } else {
            "not formed"
        },
        if connected {
            "growth → pruning → refined network"
        } else {
            "axon_a is close; shrink synapse_dist or grow longer to touch"
        }
    );
}
