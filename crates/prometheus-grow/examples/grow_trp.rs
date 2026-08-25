//! grow_trp.rs — TRP synaptogenesis: T·R^T × DistanceDecay → brain.
//!
//! Classic CX3D growth uses "distance + cell type".
//! TRP adds SEMANTICS: each cell has a Receptor ("who I am") and
//! a Tropism ("whom I seek"). A synapse forms by T·R match, decays
//! with distance and is pruned below the threshold.
//!
//! ```bash
//! cargo run -p prometheus-grow --example grow_trp
//! ```

use prometheus_grow::{Bounds, CellType, Grid, SynapseKind, Vec3, World};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   PROMETHEUS × TRP — Semantic Synapse Formation         ║");
    println!("║   T·R^T × DistanceDecay × Prune → brain                 ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let mut rng = StdRng::seed_from_u64(7);
    const N: usize = 200;

    // ── World ────────────────────────────────────────────────────────
    let origin = Vec3::new(-100.0, -100.0, -100.0);
    let cell = 20.0;
    let grid = Grid::new(10, 10, 10, cell, origin);
    let bounds = Bounds {
        min: origin,
        max: origin + Vec3::new(200.0, 200.0, 200.0),
    };
    let mut w = World::new(grid, 1.0, bounds, 7);

    // ── Cells in a sphere ────────────────────────────────────────────
    let mut positions = Vec::with_capacity(N);
    for i in 0..N {
        let d = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        let d = if d.len() < 1e-9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            d.normalized()
        };
        let r = 80.0 * rng.gen::<f64>().cbrt();
        let pos = d * r;
        positions.push(pos);
        // 30% inhibitory, 70% excitatory.
        let ct = if i % 3 == 0 {
            CellType::Inhibitory
        } else {
            CellType::Excitatory
        };
        w.add_cell(pos, ct, 3.0);
    }

    println!("  Cells: {N} ({} exc., {} inh.)\n", N - N / 3, N / 3);

    // ── TRP embeddings ───────────────────────────────────────────────
    w.init_trp_embeddings();
    println!(
        "  TRP embeddings: Receptor + Tropism ({}) per cell",
        prometheus_grow::TRP_DIM
    );

    // World scale: cells in an ~80 µm radius sphere → dist_sigma is scaled to match.
    w.trp_params.dist_sigma = 30.0;
    w.trp_params.max_dist = 40.0;
    w.trp_params.prune_threshold = 0.005;
    w.trp_params.top_k = 12;

    // ── TRP growth ───────────────────────────────────────────────────
    w.form_synapses_trp();
    println!("  Synapses formed via TRP: {}\n", w.synapses.len());

    // ── Statistics ───────────────────────────────────────────────────
    let (exc, inh) = w
        .synapses
        .iter()
        .fold((0usize, 0usize), |(e, i), s| match s.kind {
            SynapseKind::Excitatory => (e + 1, i),
            SynapseKind::Inhibitory => (e, i + 1),
        });
    println!("  Synapse signs:");
    println!("     Возбуждающих: {exc}");
    println!("     Тормозных:    {inh}\n");

    // Mean weight.
    let avg_w: f64 =
        w.synapses.iter().map(|s| s.weight).sum::<f64>() / w.synapses.len().max(1) as f64;
    println!("  Mean weight: {:.4}", avg_w);

    // Mean synapse distance.
    let avg_d: f64 = w
        .synapses
        .iter()
        .map(|s| {
            let p = positions[s.pre];
            let q = positions[s.post];
            (p - q).len()
        })
        .sum::<f64>()
        / w.synapses.len().max(1) as f64;
    println!("  Mean synapse distance: {:.2}", avg_d);

    // ── Pruning demo ─────────────────────────────────────────────────
    println!("\n  Pruning demo: shifting the Tropism of 20% of cells...");
    let nudge_count = N / 5;
    let before = w.synapses.len();

    // Shift the tropism (simulate "learning").
    let mut target = [0.0f32; prometheus_grow::TRP_DIM];
    for v in target.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }
    let mag: f32 = target.iter().map(|v| v * v).sum::<f32>().sqrt().max(0.001);
    for v in target.iter_mut() {
        *v /= mag;
    }

    for i in 0..nudge_count {
        if let Some(emb) = w.trp_embeddings.get_mut(&i) {
            emb.nudge_tropism(&target, 0.6);
        }
    }

    // 1. Pruning: recompute scores of existing synapses, drop weak ones.
    let pruned = w.prune_trp_synapses();
    let after_prune = w.synapses.len();

    // 2. Regrowth: new axons reach for new R-patterns.
    w.form_synapses_trp();
    let after_regrow = w.synapses.len();

    println!("     Синапсов до обучения:  {before}");
    println!("     Удалено прунингом:     {pruned}");
    println!("     Выжило:                {after_prune}");
    println!("     После перероста:       {after_regrow}");
    println!(
        "     Δ итого:               {}\n",
        after_regrow as isize - before as isize
    );

    println!("  TRP in PROMETHEUS works.");
    println!("     Синапсы растут по семантике (T·R), а не только по расстоянию.");
    println!("     Прунинг отмирает связи при изменении Tropism, перерост — новые.");
}
