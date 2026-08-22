//! grow_contact.rs — CX3D synapse-by-contact (bouton ↔ spine).
//!
//! Two cells grow parallel neurites. A facing bouton (on the axon) and a facing
//! spine (on the dendrite) meet and synapse; a spine pointing away does not.
//! This is the `synapseBetweenExistingBS` mechanism ported from CX3D.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_contact
//! ```

use prometheus_grow::{
    Bounds, CellType, Excrescence, ExcrescenceKind, Grid, Vec3, World,
};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 0);

    // parallel neurites, 2 units apart
    let a = w.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0); // pre (axon)
    let b = w.add_cell(Vec3::new(0.0, 2.0, 0.0), CellType::Excitatory, 4.0); // post (dendrite)
    let na = w.new_neurite(a, true, 1.0);
    let nb = w.new_neurite(b, false, 1.0);
    for _ in 0..10 {
        w.extend_neurite(a, na, Vec3::new(1.0, 0.0, 0.0));
        w.extend_neurite(b, nb, Vec3::new(1.0, 0.0, 0.0));
    }

    // facing bouton + spine → synapse
    w.cells[a].neurites[na].excrescences.push(Excrescence {
        kind: ExcrescenceKind::Bouton,
        pos: Vec3::new(5.0, 0.0, 0.0),
        dir: Vec3::new(0.0, 1.0, 0.0),
        length: 2.0,
    });
    w.cells[b].neurites[nb].excrescences.push(Excrescence {
        kind: ExcrescenceKind::Spine,
        pos: Vec3::new(5.0, 2.0, 0.0),
        dir: Vec3::new(0.0, -1.0, 0.0),
        length: 2.0,
    });
    w.form_synapses_by_contact(0.5);
    let synapsed = w.to_graph().edges.iter().any(|e| e.pre == a && e.post == b);
    println!("параллельные нейриты + смотрящие навстречу → синапс: {synapsed}");

    // spine pointing away → no synapse
    w.synapses.clear();
    w.cells[b].neurites[nb].excrescences[0].dir = Vec3::new(0.0, 1.0, 0.0);
    w.form_synapses_by_contact(0.5);
    let synapsed2 = w.to_graph().edges.iter().any(|e| e.pre == a && e.post == b);
    println!("параллельные нейриты + шипик в сторону → синапс: {synapsed2}");

    println!(
        "\n→ контакт (d < len_bouton + len_spine + tol) + сонаправленность работают\n  \
         (порт CX3D synapseBetweenExistingBS: бутон смотрит на шипик, шипик — на бутон)"
    );
    assert!(synapsed && !synapsed2);
}
