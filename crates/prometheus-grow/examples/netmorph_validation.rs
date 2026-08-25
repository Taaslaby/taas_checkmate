//! netmorph_validation.rs — grow a NETMORPH-style arbor and print the full
//! morphometric table from the statistics module, the way NETMORPH's
//! `Network_Generated_Statistics` reports arbors after development.
//!
//! Pipeline: NetmorphGrowth (second-order elongation + van Pelt branching +
//! ERI + balanced-forces angles) → bottom-up Rall diameters → morphometrics.
//!
//! ```bash
//! cargo run -p prometheus-grow --example netmorph_validation
//! ```

use prometheus_grow::{
    assign_rall_diameters, collect_cell_stats, Bounds, CellType, Grid, NetmorphGrowth,
    RallPowerLaw, Vec3, World,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn print_stats_row(name: &str, n: usize, mean: f64, std: f64, min: f64, max: f64) {
    println!("  {name:<34} {n:>5}  {mean:>9.3}  {std:>8.3}  {min:>8.3}  {max:>8.3}");
}

fn report(label: &str, s: &prometheus_grow::FiberStats) {
    println!("\n--- {label} ---");
    println!(
        "  {:<34} {:>5}  {:>9}  {:>8}  {:>8}  {:>8}",
        "metric", "n", "mean", "std", "min", "max"
    );
    if s.arbor_length.n > 0 {
        print_stats_row(
            "arbor length",
            s.arbor_length.n,
            s.arbor_length.mean(),
            s.arbor_length.std(),
            s.arbor_length.min(),
            s.arbor_length.max(),
        );
        print_stats_row(
            "terminal segments / arbor",
            s.terminal_segments.n,
            s.terminal_segments.mean(),
            s.terminal_segments.std(),
            s.terminal_segments.min(),
            s.terminal_segments.max(),
        );
    }
    for (name, st) in [
        ("segment length", &s.segment_len),
        ("terminal segment length", &s.term_seg_len),
        ("intermediate segment length", &s.interseg_len),
        ("length between bifurcations", &s.len_between_bifurcations),
        ("term len since bifurcation", &s.term_len_since_bifurcation),
        ("term len since soma", &s.term_len_since_soma),
        ("branch angle (rad)", &s.branch_angles),
        ("cart/fiber between bif", &s.cartratio_between_bifurcations),
        ("cart/fiber bif→term", &s.cartratio_bifurcation_to_term),
        ("cart/fiber soma→term", &s.cartratio_soma_to_term),
    ] {
        if st.n > 0 {
            print_stats_row(name, st.n, st.mean(), st.std(), st.min(), st.max());
        }
    }
}

fn main() {
    let grid = Grid::new(16, 16, 16, 50.0, Vec3::new(-400.0, -400.0, -400.0));
    let bounds = Bounds {
        min: Vec3::new(-400.0, -400.0, -400.0),
        max: Vec3::new(400.0, 400.0, 400.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 42);

    // One principal cell; one axon + one dendrite.
    let cell = w.add_cell(Vec3::ZERO, CellType::Excitatory, 10.0);
    let axon = w.new_neurite(cell, true, 2.0);
    let dend = w.new_neurite(cell, false, 3.0);
    w.extend_neurite(cell, axon, Vec3::new(0.0, 0.0, 15.0));
    w.extend_neurite(cell, dend, Vec3::new(0.0, 15.0, 0.0));

    let growth_axon = NetmorphGrowth::new(1.2, 0.74, 5.0, 5.0, 20.0, 0.25, 0.3)
        .max_prob(0.12)
        .speed_constraint(3.0)
        .step(2.0)
        .eri_std(0.3)
        .tension_full(2.0, 0.05, 0.0, 0.5); // tension + veer up to ~29°
    let growth_dend = NetmorphGrowth::new(0.8, 0.6, 6.0, 4.0, 25.0, 0.3, 0.4)
        .max_prob(0.1)
        .speed_constraint(2.5)
        .step(1.5)
        .eri_std(0.35)
        .tension_full(2.0, 0.08, 0.0, 0.6);
    w.add_neurite_module(cell, axon, Box::new(growth_axon));
    w.add_neurite_module(cell, dend, Box::new(growth_dend));

    let steps = 400;
    for _ in 0..steps {
        w.step();
    }

    // Bottom-up Rall diameters over the grown trees (NETMORPH ndm.rall).
    let mut rng = StdRng::seed_from_u64(7);
    assign_rall_diameters(&mut w.cells[cell], &RallPowerLaw::default(), &mut rng);

    let stats = collect_cell_stats(&w.cells[cell]);
    println!("=== NETMORPH validation run ===");
    println!("steps: {steps}");
    let neurites = &w.cells[cell].neurites;
    let terminals = neurites.iter().filter(|x| x.is_terminal).count();
    println!(
        "neurites: {}   terminals: {}   synapses: {}",
        neurites.len(),
        terminals,
        w.synapses.len()
    );

    let trunk_axon = neurites[axon].diameter;
    let trunk_dend = neurites[dend].diameter;
    let term_diams: Vec<f64> = neurites
        .iter()
        .filter(|x| x.is_terminal)
        .map(|x| x.diameter)
        .collect();
    let min_term = term_diams.iter().cloned().fold(f64::MAX, f64::min);
    println!("\n--- diameters (Rall bottom-up) ---");
    println!("  axon trunk:      {trunk_axon:.3}");
    println!("  dendrite trunk:  {trunk_dend:.3}");
    println!("  thinnest terminal: {min_term:.3}");

    report("AXON", &stats.axon);
    report("DENDRITE", &stats.dendrite);

    println!("\nValidation hints:");
    println!("  * branch angles should cluster near the balanced-forces mean (~90°)");
    println!("  * cart/fiber ratios < 1 everywhere; near 1 on trunks");
    println!("  * len between bifurcations decays with depth (van Pelt N^(-E))");
}
