//! # From genome to gamma — CLI-отчёт над общим `grow_engine`
//!
//! Флагманская демонстрация обоих движков: сеть из 200 нейронов **вырастает**
//! из правил развития (диффузия + градиенты + контактное замыкание) на
//! `prometheus-grow`, затем её динамика считается на SIMD-движке pillars
//! (LIF-α, модель Shiu et al. 2024).
//!
//! **Вопрос:** достаточно ли только правил развития (без рукописных весов и
//! таблиц геометрии) для нетривиальной динамики? Или любой случайный граф с
//! той же степенью даст то же самое?
//!
//! **Контроль:** перемешиваем мишени синапсов (степенная последовательность
//! сохранена, геометрия уничтожена). Сравниваем пространственную
//! синхронизацию (корреляция спайков vs. расстояние между сомами) и ритм.
//!
//! Вся логика живёт в [`pillars_connectome::grow_engine`] — этот пример
//! только печатает отчёт. Тот же код использует GUI-режим «Развитие»
//! в connscope.
//!
//! ```bash
//! cargo run -p pillars-connectome --example genome_to_gamma --release
//! ```

use pillars_connectome::grow_engine::{run_experiment, GrowConfig};

fn main() {
    let cfg = GrowConfig::default();
    let r = run_experiment(&cfg);
    let n = cfg.n();
    let n_syn = r.circuit.edges.len();

    println!("═══ FROM GENOME TO GAMMA ═══════════════════════════════════");
    println!("development rules → structure → function, end to end\n");

    let n_e = cfg.n_e;
    let n_i = cfg.n_i;
    println!("[1] development  (prometheus-grow, {} steps)", cfg.grow_steps);
    println!(
        "    {n} neurons ({n_e} E / {n_i} I) → {n_syn} synapses in {:.2}s",
        r.circuit.growth_s
    );
    println!(
        "    clustering C = {:.3} | E-driven {:.0}% | I-driven {:.0}%",
        r.cc_grown,
        r.circuit
            .edges
            .iter()
            .filter(|e| r.circuit.nt[e.0] > 0.0)
            .count() as f64
            * 100.0
            / n_syn as f64,
        100.0
            - r.circuit
                .edges
                .iter()
                .filter(|e| r.circuit.nt[e.0] > 0.0)
                .count() as f64
                * 100.0
                / n_syn as f64
    );

    println!(
        "[2] control      shuffled targets: C = {:.3} (same degree sequence)",
        r.cc_shuf
    );

    let cell_steps = n as f64 * cfg.sim_steps as f64;
    println!("\n[3] simulation   (pillars, SIMD LIF-α, {} steps × {n} neurons)", cfg.sim_steps);
    println!(
        "    grown: {:.3}s | shuffled: {:.3}s | {:.0} M cell-steps/s (grown)",
        r.sim_s,
        r.sim_s_shuf,
        cell_steps / r.sim_s / 1e6
    );

    let mean_g = r.rates_g.iter().sum::<f32>() / n as f32;
    let mean_s = r.rates_s.iter().sum::<f32>() / n as f32;
    let rate_e: f32 = r.rates_g[..cfg.n_e].iter().sum::<f32>() / cfg.n_e as f32;
    let rate_i: f32 = r.rates_g[cfg.n_e..].iter().sum::<f32>() / cfg.n_i as f32;

    println!("\n[4] population dynamics (binned 4 ms)");
    println!("    ┌───────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("    │           │  rate Hz │ dom. freq│ peak/bg  │  osc     │");
    println!("    ├───────────┼──────────┼──────────┼──────────┼──────────┤");
    println!(
        "    │ grown     │ {:>8.1} │ {:>8.1} │ {:>8.2} │ {:>8.3} │",
        mean_g, r.fg, r.cg, r.og
    );
    println!(
        "    │ shuffled  │ {:>8.1} │ {:>8.1} │ {:>8.2} │ {:>8.3} │",
        mean_s, r.fs, r.cs, r.os
    );
    println!("    └───────────┴──────────┴──────────┴──────────┴──────────┘");
    println!(
        "    E rate {rate_e:.1} Hz, I rate {rate_i:.1} Hz (E/I balance {:.2})",
        rate_e / rate_i.max(0.1)
    );

    println!("\n[5] spatial synchrony — spike corr. vs. soma distance (50 ms bins)");
    println!("    ┌───────────┬──────────┬──────────┬──────────┐");
    println!("    │ distance  │  grown   │ shuffled │  ratio   │");
    println!("    ├───────────┼──────────┼──────────┼──────────┤");
    for b in 0..r.scg.len() {
        let (d, cgv) = r.scg[b];
        let (_, csv) = r.scs[b];
        println!(
            "    │ {:>7.0} µm │ {:>8.3} │ {:>8.3} │ {:>8.2} │",
            d,
            cgv,
            csv,
            cgv / csv.max(1e-6)
        );
    }
    println!("    └───────────┴──────────┴──────────┴──────────┘");

    let (d0, c0) = r.scg[0];
    let (d1, c1) = *r.scg.last().unwrap();
    let (_, s0) = r.scs[0];
    let (_, s1) = *r.scs.last().unwrap();
    let decay_g = c0 - c1;
    let decay_s = s0 - s1;
    println!("\n[6] verdict");
    if decay_g > decay_s + 0.03 && c0 > s0 {
        println!(
            "    ✅ Development rules built spatially coherent assemblies: neurons within\n    \
             {d0:.0} µm co-fire at r={c0:.2} (vs r={s0:.2} shuffled), while {d1:.0} µm apart\n    \
             the correlation drops to r={c1:.2} — a gradient of {decay_g:.2}. The shuffled\n    \
             control has no spatial structure (gradient {decay_s:.2})."
        );
        if (20.0..=80.0).contains(&r.fg) {
            println!(
                "    Bonus: the population oscillates — a sharp beta/gamma rhythm: {:.0} Hz, with a spectral\n    \
                 peak {:.1}× above background).",
                r.fg, r.cg
            );
        }
        println!("    Geometry matters: destroy it (same degrees!) and co-firing collapses.");
    } else {
        println!(
            "    ⚠️  weak spatial contrast (grown {c0:.2}→{c1:.2}, shuffled {s0:.2}→{s1:.2}). \
             Tune drive/weights."
        );
    }
}
