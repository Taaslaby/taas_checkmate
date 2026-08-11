//! Native runner for the Rust-owned Hide & Seek arena (Mission 03).
//!
//! 2 hiders vs 4 seekers on a large procedural map with box barricades.
//! Line-of-sight vision (walls/boxes block sight) + shared pack intel.
//! No expert-pipeline shaping — the reward is emergent (survival, fleeing,
//! chasing, barricading, tagging).
//!
//! Run:
//! ```text
//! cargo run --example hide_seek_arena --features karmac
//! ```

use taas_checkmate::karmac::HideSeekArena;

fn main() {
    let mut arena = HideSeekArena::new(0x5EED);
    println!("pillars_hide_seek source=rust_core");
    println!(
        "mission=hide_seek: 2 hiders vs 4 seekers, {}x{}, {} boxes, {} shelters, emergent R-STDP",
        arena.snapshot().width,
        arena.snapshot().height,
        arena.snapshot().boxes.len(),
        arena.snapshot().shelters.len()
    );

    let mut printed_events = 0;
    for _ in 0..320 {
        let snapshot = arena.step();
        for event in snapshot.events.iter().skip(printed_events) {
            println!("event step={} {event}", snapshot.step);
        }
        printed_events = snapshot.events.len();
        if snapshot.finished {
            break;
        }
    }
    let snapshot = arena.snapshot();
    println!(
        "SNAPSHOT step={} finished={} survival={:.1} tags={} walls={} boxes={}",
        snapshot.step,
        snapshot.finished,
        snapshot.survival,
        snapshot.tags,
        snapshot.walls.len(),
        snapshot.boxes.len()
    );

    // Training: all six brains learn via reward-modulated STDP.
    println!("\n== training 6 episodes ==");
    arena.train(6, 250);
    let t = arena.training();
    println!(
        "survival: {:?}",
        t.survival
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>()
    );
    println!("tags:     {:?}", t.tags);
    println!(
        "rewards:  {:?}",
        t.rewards
            .iter()
            .map(|r| format!("{r:.2}"))
            .collect::<Vec<_>>()
    );

    let final_snapshot = arena.run_for_demo(300);
    println!(
        "trained_run step={} finished={} tags={} survival={:.1}",
        final_snapshot.step, final_snapshot.finished, final_snapshot.tags, final_snapshot.survival
    );
}
