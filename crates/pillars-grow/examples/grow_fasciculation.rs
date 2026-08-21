//! Fasciculation: follower growth cones bundle onto a pioneer's tract
//! (CX3D `NeuriteFasciculation`).
//!
//! The pioneer chases a diagonal attractant with a turn-rate cap (smooth
//! curved tract). The follower starts beside the tract with **no chemo at
//! all** — only a [`Fasciculation`] module — yet reaches the target by riding
//! the pioneer's path instead of flying off on its own direction. This is how
//! real axon tracts bundle: later growth cones follow earlier ones.
//!
//! ```bash
//! cargo run -p pillars-grow --example grow_fasciculation
//! ```

use pillars_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Fasciculation, Grid, Substance, Vec3,
    World,
};

fn main() {
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);

    // Diagonal attractant toward (30, 0, 40): two axis-aligned Gaussians
    // summed under one id → the gradient points diagonally.
    world.add_substance(Substance::new(
        "ntf",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Gaussian {
            axis: 0,
            max: 1.0,
            mean: 30.0,
            sigma: 15.0,
        }],
    ));
    world.add_substance(Substance::new(
        "ntf",
        0.1,
        0.001,
        &world.grid,
        vec![ArtificialGradient::Gaussian {
            axis: 2,
            max: 1.0,
            mean: 40.0,
            sigma: 15.0,
        }],
    ));

    let target = Vec3::new(30.0, 0.0, 40.0);
    world.add_cell(target, CellType::Inhibitory, 4.0);
    let pioneer = world.add_cell(Vec3::ZERO, CellType::Excitatory, 4.0);
    let follower = world.add_cell(Vec3::new(0.0, 5.0, 0.0), CellType::Excitatory, 4.0);

    // Pioneer: chemo with a 15°/step turn cap → smooth curved tract. The
    // two Gaussians sum to ~2.0 at the target, so stop at 1.5 (near it).
    let pn = world.new_neurite(pioneer, true, 0.5);
    let mut chemo = ChemoAttraction::new("ntf");
    chemo.noise_weight = 0.02;
    chemo.turn_rate_deg = Some(15.0);
    chemo.stop_concentration = 1.5;
    world.add_neurite_module(pioneer, pn, Box::new(chemo));

    // Follower: no chemo — bundles onto the pioneer's tract, capped so it
    // stops at the target region instead of cruising past.
    let fn_ = world.new_neurite(follower, true, 0.5);
    let mut fas = Fasciculation::new(6.0);
    fas.noise_weight = 0.02;
    fas.max_len_um = Some(62.0);
    world.add_neurite_module(follower, fn_, Box::new(fas));

    println!("=== growth (pioneer chemo + follower fasciculation) ===");
    for _ in 0..250 {
        world.step();
    }

    let tip_p = world.cells[pioneer].neurites[pn].tip();
    let tip_f = world.cells[follower].neurites[fn_].tip();
    let len_p = world.cells[pioneer].neurites[pn].length_um();
    let len_f = world.cells[follower].neurites[fn_].length_um();
    println!(
        "pioneer  tip {tip_p:?}  len {len_p:.1} µm  dist to target {:.1}",
        tip_p.dist(target)
    );
    println!(
        "follower tip {tip_f:?}  len {len_f:.1} µm  dist to target {:.1}",
        tip_f.dist(target)
    );

    // The follower had no chemo — it can only reach the target via bundling.
    let reached = tip_f.dist(target) < 25.0;
    println!(
        "  → follower {}",
        if reached {
            "followed the tract to the target ✓ (bundling)"
        } else {
            "missed the target (radius too small?)"
        }
    );
    assert!(
        reached,
        "fasciculated follower must reach the target region, got dist {:.1}",
        tip_f.dist(target)
    );
}
