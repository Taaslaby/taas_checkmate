//! CX3D-style demo: one progenitor divides, then every cell grows an axon up a
//! morphogen gradient, and synapses form when tips meet other cells.
//!
//! ```bash
//! cargo run -p pillars-grow --example grow
//! ```

use pillars_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Grid, Secretor, SimpleCellCycle,
    Substance, Vec3, World,
};

fn main() {
    let grid = Grid::new(40, 40, 40, 5.0, Vec3::new(-100.0, -100.0, -100.0));
    let bounds = Bounds {
        min: Vec3::new(-100.0, -100.0, -100.0),
        max: Vec3::new(100.0, 100.0, 100.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    // Morphogen source: gaussian gradient along +z (the dpp/wg/hh analog).
    world.add_substance(Substance::new(
        "A",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Gaussian {
            axis: 2,
            max: 1.0,
            mean: 80.0,
            sigma: 40.0,
        }],
    ));

    // One progenitor at the origin, carrying the "genome": divide + secrete.
    let progenitor = world.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
    world.add_cell_module(progenitor, Box::new(SimpleCellCycle::new(300.0, 12.0)));
    world.add_soma_module(progenitor, Box::new(Secretor::new("A", 0.05)));

    println!("=== division phase ===");
    for t in 0..30 {
        world.step();
        if world.cells.len() >= 16 {
            break;
        }
        if t % 5 == 0 {
            println!("  t={t}: cells={}", world.cells.len());
        }
    }
    println!("  after division: cells={}", world.cells.len());

    // Freeze the population so the growth phase only extends neurites.
    world.max_cells = world.cells.len();

    // Growth phase: every cell extends an axon that chases the gradient.
    for ci in 0..world.cells.len() {
        let idx = world.new_neurite(ci, true, 0.5);
        world.add_neurite_module(
            ci,
            idx,
            Box::new(ChemoAttraction::new("A").branching(0.002)),
        );
    }

    println!("=== growth phase ===");
    for t in 0..300 {
        world.step();
        if t % 60 == 0 {
            println!(
                "  t={t}: cells={} synapses={}",
                world.cells.len(),
                world.synapses.len()
            );
        }
    }

    println!("\n=== final ===");
    println!(
        "time={:.0}  cells={}  synapses={}",
        world.time,
        world.cells.len(),
        world.synapses.len()
    );
    for c in &world.cells {
        let tips: Vec<Vec3> = c
            .neurites
            .iter()
            .filter(|n| n.is_axon)
            .map(|n| n.tip())
            .collect();
        let kind = match c.cell_type {
            CellType::Excitatory => "exc",
            CellType::Inhibitory => "inh",
        };
        println!(
            "  cell {} [{kind}] soma=({:5.0},{:5.0},{:5.0}) r={:.1} neurites={} tip_z={}",
            c.id,
            c.soma.pos.x,
            c.soma.pos.y,
            c.soma.pos.z,
            c.soma.radius,
            c.neurites.len(),
            tips.iter()
                .map(|t| format!("{:.0}", t.z))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
