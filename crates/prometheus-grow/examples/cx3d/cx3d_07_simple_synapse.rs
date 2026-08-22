//! CX3D tutorial `SimpleSynapse` (Zubler & Douglas 2009) — Rust port.
//!
//! An excitatory neuron at (0,0,-300) extends an axon upward; an inhibitory
//! neuron at (0,0,300) extends a dendrite downward. When the tips meet, a
//! bouton on the axon and a spine on the dendrite are placed; the synapse
//! forms on contact. The sign follows the presynaptic cell (excitatory).
//!
//! Original: `forks/cx3d/ini/cx3d/simulations/tutorial/SimpleSynapse.java`

use prometheus_grow::{Bounds, CellType, Grid, Vec3, World};

fn main() {
    let origin = Vec3::new(-100.0, -100.0, -100.0);
    let grid = Grid::new(11, 11, 33, 20.0, origin);
    let mut w = World::new(
        grid,
        1.0,
        Bounds { min: origin, max: origin + Vec3::new(220.0, 220.0, 660.0) },
        0,
    );
    w.synapse_dist = 5.0;

    let excit = w.add_cell(Vec3::new(0.0, 0.0, -300.0), CellType::Excitatory, 5.0);
    let inhib = w.add_cell(Vec3::new(0.0, 0.0, 300.0), CellType::Inhibitory, 5.0);
    let axon = w.new_neurite(excit, true, 0.5); // up
    let dend = w.new_neurite(inhib, false, 1.5); // down

    // Elongate the axon up and the dendrite down until the tips meet.
    println!("=== CX3D tutorial: SimpleSynapse ===");
    let mut steps = 0;
    loop {
        let az = w.cells[excit].neurites[axon].tip().z;
        let dz = w.cells[inhib].neurites[dend].tip().z;
        if az >= dz {
            break;
        }
        w.extend_neurite(excit, axon, Vec3::new(0.0, 0.0, 20.0));
        w.extend_neurite(inhib, dend, Vec3::new(0.0, 0.0, -20.0));
        w.step();
        steps += 1;
        assert!(steps < 1000, "конусы должны встретиться");
    }
    let az = w.cells[excit].neurites[axon].tip().z;
    let dz = w.cells[inhib].neurites[dend].tip().z;
    println!("кончики встретились: аксон z={az:.0}, дендрит z={dz:.0} (шагов {steps})");

    // Bouton + spine (CX3D `ConnectionsMaker`), synapse on contact.
    w.make_excrescences(20.0, 10.0);
    w.form_synapses_by_contact(15.0);

    let syn = w.synapses.clone();
    println!("синапсов: {} (из них exc→inh: {})", syn.len(), syn.iter().filter(|s| s.pre == excit && s.post == inhib).count());
    for s in &syn {
        println!("  пре {}({}) → пост {}({})  {:?}  w={:.2}", s.pre, ty(&w, s.pre), s.post, ty(&w, s.post), s.kind, s.weight);
    }
    assert!(!syn.is_empty(), "должен образоваться синапс");
    let e2i = syn.iter().find(|s| s.pre == excit && s.post == inhib);
    assert!(e2i.is_some(), "exc → inh синапс обязателен");
    assert_eq!(e2i.unwrap().kind, prometheus_grow::SynapseKind::Excitatory);
    println!("знак синапса = возбуждающий (от пресинаптической клетки) ✓");
}

fn ty(w: &World, id: usize) -> &'static str {
    match w.cells[id].cell_type {
        CellType::Excitatory => "exc",
        CellType::Inhibitory => "inh",
        CellType::Glia => "glia",
    }
}
