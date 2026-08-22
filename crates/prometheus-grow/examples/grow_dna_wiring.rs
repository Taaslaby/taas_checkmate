//! Feed PROMETHEUS `genes/` (the DNA maps) into the growth engine.
//!
//! ```text
//! dna_nt_map.csv      Gad1  → sign -1 → CellType::Inhibitory (GABA)
//! dna_channel_map.csv Dscam1 → cell-recognition → same-type wiring rule
//!                     Ih/KCNQ → rebound / adaptation (physiology metadata)
//! ```
//!
//! The NT gene sets *who I am* (sign/type), the adhesion gene sets *who I wire
//! to* (recognition), the channel genes set *how I fire* (carried as metadata,
//! not topology — DNA does not encode synaptic strength). Then the engine grows
//! the skeleton and checks that the wiring matches the DNA.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_dna_wiring
//! ```

use std::fs;
use std::path::Path;

use prometheus_grow::{
    parse_channel_map, parse_dna_map, Bounds, CellType, DivideOnce, FlyGenome, Grid, Vec3, World,
};

fn find(candidates: &[&str]) -> String {
    candidates
        .iter()
        .find(|p| Path::new(p).exists())
        .unwrap_or(&candidates[0])
        .to_string()
}

fn main() {
    // 1. Load the two DNA maps from PROMETHEUS genes/.
    let nt_csv = find(&["../genes/dna_nt_map.csv", "../../genes/dna_nt_map.csv"]);
    let ch_csv = find(&["../genes/dna_channel_map.csv", "../../genes/dna_channel_map.csv"]);

    let nt_map = parse_dna_map(&fs::read_to_string(&nt_csv).expect("не читается dna_nt_map.csv"));
    let ch_map = parse_channel_map(&fs::read_to_string(&ch_csv).expect("не читается dna_channel_map.csv"));
    println!("=== ДНК-карты ===");
    println!("  NT-гены ({}):", nt_map.len());
    for g in &nt_map {
        println!(
            "    {} → {} → {}",
            g.gene,
            g.neurotransmitter,
            if g.sign < 0 { "тормозной (−)" } else { "возбуждающий (+)" }
        );
    }
    println!("  каналы/адгезия ({}):", ch_map.len());
    for c in &ch_map {
        println!("    {:<8} → {}", c.gene, c.param);
    }

    // 2. A half-center genome: Gad1 identity + Dscam1 adhesion + Ih/KCNQ channels.
    let genome = FlyGenome::half_center();
    let cell_type = genome.cell_type(&nt_map).expect("Gad1 нет в NT-карте");
    let nt = genome.neurotransmitter(&nt_map).unwrap_or_default();
    let rec = genome.recognition(&ch_map);
    let channels = genome.channels(&ch_map);
    println!("\ngenome: {} → {nt} → {cell_type:?}", genome.nt_gene);
    println!(
        "  adhesion: {:?} → same-type recognition ({} vs unrestricted)",
        genome.adhesion_gene,
        rec.allowed[1][1] && !rec.allowed[0][1]
    );
    for c in &channels {
        println!("  channel: {} → {} ({})", c.gene, c.param, c.family);
    }

    // 3. Grow: 1 Gad1 progenitor divides → 2 GABA cells; seed 1 excitatory (ChAT).
    let grid = Grid::new(32, 32, 32, 5.0, Vec3::new(-80.0, -80.0, -80.0));
    let bounds = Bounds {
        min: Vec3::new(-80.0, -80.0, -80.0),
        max: Vec3::new(80.0, 80.0, 80.0),
    };
    let mut world = World::new(grid, 1.0, bounds, 0);
    world.set_recognition(rec);

    let gaba0 = world.add_cell(Vec3::new(-6.0, 0.0, 0.0), CellType::Inhibitory, 4.0);
    world.add_cell_module(gaba0, Box::new(DivideOnce::new(300.0, 12.0)));
    world.add_cell(Vec3::new(6.0, 0.0, 0.0), CellType::Excitatory, 4.0);

    for _ in 0..40 {
        world.step();
        if world.cells.len() >= 3 {
            break;
        }
    }
    world.max_cells = world.cells.len();
    println!("\nпосле деления: {} клеток (Gad1 → 2×GABA + 1×ChAT)", world.cells.len());

    // Short straight axons, then wire by contact with the DNA-derived rule.
    for ci in 0..world.cells.len() {
        let n = world.new_neurite(ci, true, 0.5);
        for _ in 0..5 {
            world.extend_neurite(ci, n, Vec3::new(0.0, 0.0, 1.0));
        }
    }
    world.form_synapses(100.0);

    // 4. Check the grown skeleton against the DNA rule.
    let g = world.to_graph();
    println!("\ngrown graph: {} nodes, {} edges", g.nodes.len(), g.edges.len());
    for e in &g.edges {
        println!("  {} -> {}  kind={:?}  w={:.3}", e.pre, e.post, e.kind, e.weight);
    }

    let gaba: Vec<usize> = g
        .nodes
        .iter()
        .filter(|n| n.cell_type == CellType::Inhibitory)
        .map(|n| n.id)
        .collect();
    let excs: Vec<usize> = g
        .nodes
        .iter()
        .filter(|n| n.cell_type == CellType::Excitatory)
        .map(|n| n.id)
        .collect();

    let mutual = gaba.len() >= 2
        && g.edges.iter().any(|e| e.pre == gaba[0] && e.post == gaba[1])
        && g.edges.iter().any(|e| e.pre == gaba[1] && e.post == gaba[0]);
    let cross = g
        .edges
        .iter()
        .filter(|e| excs.contains(&e.pre) || excs.contains(&e.post))
        .count();

    println!("\nпроверка (DNA → skeleton):");
    println!("  GABA↔GABA взаимная пара: {mutual}");
    println!("  cross-type рёбер: {cross} (Dscam1 same-type → должно быть 0)");
    let verdict = mutual && cross == 0;
    println!(
        "  → {}",
        if verdict {
            "ДНК (Gad1 + Dscam1) → тип + проводка → скелет ✓"
        } else {
            "проводка НЕ соответствует ДНК ✗"
        }
    );
}
