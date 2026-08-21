//! Self-organization from the developmental literature:
//!
//! 1. **Delta–Notch lateral inhibition** (Collier, Monk, Maini & Lewis, 1996):
//!    a field of identical cells picks its own fates — cells that win the
//!    signalling race become primary (neurons) and suppress their neighbours,
//!    producing a regular, self-spaced pattern. No coordinates are imposed:
//!    the pattern *emerges* from the local coupling.
//! 2. **Neurotrophic competition** (Oppenheim, 1991; Barde, 1989): neurons
//!    compete for a target-derived trophic factor; those too far from the
//!    supply fall below their intake threshold and die.
//!
//! ```bash
//! cargo run -p pillars-grow --example grow_pattern
//! ```

use pillars_grow::{
    Bounds, CellType, Grid, LateralInhibition, Secretor, Substance, TrophicCompetition, Vec3, World,
};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    println!("=== 1. Delta–Notch lateral inhibition ===");
    let mut cells = Vec::new();
    for x in 0..5i32 {
        for y in 0..5i32 {
            let c = world.add_cell(
                Vec3::new(x as f64 * 12.0, y as f64 * 12.0, 0.0),
                CellType::Excitatory,
                3.0,
            );
            world.add_cell_module(c, Box::new(LateralInhibition::new(15.0)));
            cells.push(c);
        }
    }
    for _ in 0..300 {
        world.step();
    }
    let mut primaries = Vec::new();
    for y in 0..5i32 {
        let mut row = String::new();
        for x in 0..5i32 {
            let c = cells[(y * 5 + x) as usize];
            let ch = if world.cells[c].role == 1 { '#' } else { '.' };
            row.push(ch);
            if world.cells[c].role == 1 {
                primaries.push(c);
            }
        }
        println!("  {row}");
    }
    println!("  primaries: {} of 25", primaries.len());
    let mut min_gap = f64::INFINITY;
    for (i, &a) in primaries.iter().enumerate() {
        for &b in &primaries[i + 1..] {
            min_gap = min_gap.min(world.cells[a].soma.pos.dist(world.cells[b].soma.pos));
        }
    }
    println!("  min gap between primaries: {min_gap:.1} (neighbourhood radius 15)");
    let spaced = primaries.len() > 1 && min_gap > 13.0;
    println!(
        "  → {}",
        if spaced {
            "regular self-spaced pattern ✓ (lateral inhibition)"
        } else {
            "pattern not regular — check parameters"
        }
    );

    println!("\n=== 2. Neurotrophic competition ===");
    let mut world = World::new(grid, 1.0, bounds, 1);
    world.add_substance(Substance::new("ntf", 0.1, 0.001, &world.grid, vec![]));
    // Target secretes the trophic factor (the only supply).
    let target = world.add_cell(Vec3::ZERO, CellType::Inhibitory, 4.0);
    world.add_soma_module(target, Box::new(Secretor::new("ntf", 2.0)));
    // Consumers at increasing distance compete for the same supply.
    let dists = [8.0, 16.0, 24.0, 40.0];
    let mut consumers = Vec::new();
    for (k, d) in dists.iter().enumerate() {
        let c = world.add_cell(Vec3::new(0.0, 0.0, *d), CellType::Excitatory, 3.0);
        let mut tc = TrophicCompetition::new("ntf");
        tc.consume_rate = 0.05;
        tc.min_uptake = 0.02;
        tc.grace = 30.0;
        world.add_soma_module(c, Box::new(tc));
        consumers.push(c);
        let _ = k;
    }
    for _ in 0..300 {
        world.step();
    }
    for (k, &c) in consumers.iter().enumerate() {
        let conc = world.concentration("ntf", world.cells[c].soma.pos);
        println!(
            "  cell {k}: dist {:.0} · conc {conc:.3} · {}",
            dists[k],
            if world.cells[c].alive {
                "alive"
            } else {
                "dead"
            }
        );
    }
    let first_alive = world.cells[consumers[0]].alive;
    let last_dead = !world.cells[consumers[3]].alive;
    println!(
        "  → {}",
        if first_alive && last_dead {
            "near wins, far dies ✓ (competition for trophic factor)"
        } else {
            "check parameters — supply/demand not balanced"
        }
    );
}
