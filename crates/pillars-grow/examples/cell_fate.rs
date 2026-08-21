//! Cell fate via a morphogen gradient — the dpp/wg/hh positional-information
//! mechanism, in the growth engine.
//!
//! A linear `dpp` gradient runs along z. Two progenitors (one per segment) each
//! divide once → 4 cells, and every cell "reads" the local dpp concentration to
//! decide its fate (`Differentiate`): `role = floor(dpp / bin_width)`.
//!
//! ```text
//!    dpp low ─────────────────────────────► dpp high
//!    segment 0 (z≈-20)          segment 1 (z≈+20)
//!      cell, cell                 cell, cell     ← each progenitor divided
//!      role=0                     role=1         ← fate from the gradient
//! ```
//!
//! ```bash
//! cargo run -p pillars-grow --example cell_fate
//! ```

use pillars_grow::{
    ArtificialGradient, Bounds, CellType, Differentiate, DivideOnce, Grid, Substance, Vec3, World,
};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    // dpp: linear gradient along z, 0 at z=-40, 2.0 at z=+40.
    world.add_substance(Substance::new(
        "dpp",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Linear {
            axis: 2,
            max: 2.0,
            peak: 40.0,
            zero: -40.0,
        }],
    ));

    // Two progenitors (one per future segment), each divides once.
    for z in [-20.0, 20.0] {
        let c = world.add_cell(Vec3::new(0.0, 0.0, z), CellType::Inhibitory, 4.0);
        world.add_cell_module(c, Box::new(DivideOnce::new(300.0, 12.0)));
        world.add_cell_module(c, Box::new(Differentiate::new("dpp", 1.0)));
    }

    for _ in 0..30 {
        world.step();
        if world.cells.len() >= 4 {
            break;
        }
    }

    println!("=== cell fate (dpp gradient → role) ===");
    let mut cells: Vec<_> = world.cells.iter().collect();
    cells.sort_by(|a, b| a.soma.pos.z.partial_cmp(&b.soma.pos.z).unwrap());
    for c in cells {
        let dpp = world.concentration("dpp", c.soma.pos);
        println!(
            "  cell {}  z={:+.1}  dpp={:.2}  → role={}",
            c.id, c.soma.pos.z, dpp, c.role
        );
    }

    // Segments: group cells by role.
    let mut roles: Vec<i32> = world.cells.iter().map(|c| c.role).collect();
    roles.sort();
    roles.dedup();
    println!("\nsegments (distinct roles): {roles:?}");
    println!("  → 2 сегмента: клетки сами «прочитали» свою позицию в градиенте dpp");
}
