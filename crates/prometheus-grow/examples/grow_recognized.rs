//! grow_recognized.rs — CX3D specific wiring via type recognition.
//!
//! A mixed population (2 inhibitory + 2 excitatory cells) is placed close
//! together. Without recognition every cell would synapse with every other.
//! With [`Recognition::same_type`] (the Dscam1/CadN analog) only compatible
//! types wire up: the GABA pair and the excitatory pair form reciprocal
//! connections, but there are no cross-type synapses.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_recognized
//! ```

use prometheus_grow::{Bounds, CellType, Grid, Recognition, Vec3, World};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    // Mixed population, all within synapse distance of each other.
    let g0 = world.add_cell(Vec3::new(0.0, 0.0, 0.0), CellType::Inhibitory, 4.0);
    let g1 = world.add_cell(Vec3::new(6.0, 0.0, 0.0), CellType::Inhibitory, 4.0);
    let e0 = world.add_cell(Vec3::new(0.0, 6.0, 0.0), CellType::Excitatory, 4.0);
    let e1 = world.add_cell(Vec3::new(6.0, 6.0, 0.0), CellType::Excitatory, 4.0);
    for c in [g0, g1, e0, e1] {
        world.new_neurite(c, true, 0.5);
    }

    // Dscam1 analog: a synapse only "like with like".
    world.set_recognition(Recognition::same_type());
    world.form_synapses(20.0);

    let g = world.to_graph();
    println!(
        "grown graph: {} nodes, {} edges  (recognition = same_type)",
        g.nodes.len(),
        g.edges.len()
    );
    for e in &g.edges {
        println!("  {} -> {}  {:?}  w={:.2}", e.pre, e.post, e.kind, e.weight);
    }

    // Node type lookup (id → CellType).
    let ty: std::collections::HashMap<usize, CellType> =
        g.nodes.iter().map(|n| (n.id, n.cell_type)).collect();

    let gaba_pair = g.edges.iter().any(|e| e.pre == g0 && e.post == g1)
        && g.edges.iter().any(|e| e.pre == g1 && e.post == g0);
    let exc_pair = g.edges.iter().any(|e| e.pre == e0 && e.post == e1)
        && g.edges.iter().any(|e| e.pre == e1 && e.post == e0);
    let cross = g.edges.iter().filter(|e| ty[&e.pre] != ty[&e.post]).count();

    println!("\n=== сверка ===");
    println!("  reciprocal GABA pair: {gaba_pair}");
    println!("  reciprocal exc  pair: {exc_pair}");
    println!("  cross-type edges:     {cross}");
    let verdict = if gaba_pair && exc_pair && cross == 0 {
        "specific wiring via recognition ✓"
    } else {
        "FAIL: recognition did not produce specific wiring"
    };
    println!("  → {verdict}");
}
