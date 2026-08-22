//! grow_synapse_model.rs — demonstrate the NETMORPH synapse-formation model.
//!
//! The port's key idea: the sign of a connection is a property of the
//! **pre × post** pair (which postsynaptic receptors form), not of the
//! presynaptic cell alone. `SynapseFormationModel::receptor_switch` shows the
//! fly GluCl story: one excitatory presynaptic type is excitatory onto
//! excitatory posts but inhibitory onto inhibitory posts.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_synapse_model
//! ```

use prometheus_grow::{
    Bounds, CellType, Grid, SynapseFormationModel, SynapseKind, Vec3, World,
};

fn world() -> World {
    let grid = Grid::new(4, 4, 4, 100.0, Vec3::ZERO);
    let bounds = Bounds {
        min: Vec3::ZERO,
        max: Vec3::new(400.0, 400.0, 400.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);
    // 0: excitatory pre (e.g. glutamate)         3: inhibitory pre (GABA)
    // 1: excitatory post                          2: inhibitory post
    for (i, (p, t)) in [
        (Vec3::new(100.0, 200.0, 200.0), CellType::Excitatory),
        (Vec3::new(200.0, 200.0, 200.0), CellType::Excitatory),
        (Vec3::new(200.0, 300.0, 200.0), CellType::Inhibitory),
        (Vec3::new(100.0, 300.0, 200.0), CellType::Inhibitory),
    ]
    .iter()
    .enumerate()
    {
        let id = w.add_cell(*p, *t, 30.0);
        debug_assert_eq!(id, i);
        w.new_neurite(id, true, 1.0); // axon tip at the soma
    }
    w
}

fn run(tag: &str, mut m: SynapseFormationModel) {
    // Scale the NETMORPH fibre thresholds (µm) to this demo's world units.
    m.max_fibre_distance = [[500.0; 3]; 3];
    let mut w = world();
    w.form_synapses_netmorph(&m);
    println!("{tag}");
    for s in &w.synapses {
        let kind = match s.kind {
            SynapseKind::Excitatory => "возб.",
            SynapseKind::Inhibitory => "торм.",
        };
        println!(
            "  {:?} → {:?} : {kind}",
            w.cells[s.pre].cell_type,
            w.cells[s.post].cell_type
        );
    }
    if w.synapses.is_empty() {
        println!("  (нет синапсов)");
    }
}

fn main() {
    println!("=== NETMORPH receptor-compatibility: sign = pre × post ===\n");
    run("receptor_switch (глутамат → AMPA/NMDA на одних, GluCl на других):", SynapseFormationModel::receptor_switch());
    println!();
    run("netmorph_defaults (знак = только пресинапс):", SynapseFormationModel::netmorph_defaults());
}
