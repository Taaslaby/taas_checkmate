//! Native runner for the Rust-owned Cognitive Arena.
//!
//! The simulation lives in `taas_checkmate::karmac::CognitiveArena`; this
//! example intentionally contains no second implementation.
//!
//! Run:
//! ```text
//! cargo run --example cognitive_arena --features karmac
//! ```

use taas_checkmate::karmac::CognitiveArena;

fn main() {
    let mut arena = CognitiveArena::new();
    println!("pillars_cognitive_arena source=rust_core");
    println!("mission=two_key_vault: spiking GNW brains + dichromatic keys + bidirectional semantic messages + synchronized opening");
    let mut printed_events = 0;
    for _ in 0..500 {
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
        "SNAPSHOT {}",
        serde_json::to_string(&snapshot).expect("snapshot should serialize")
    );

    // Training: the perturbed brains learn via reward-modulated STDP.
    println!("\n== training 8 episodes ==");
    arena.train(8, 800);
    let training = arena.snapshot().training;
    println!(
        "rewards: {:?}",
        training
            .rewards
            .iter()
            .map(|r| format!("{r:.2}"))
            .collect::<Vec<_>>()
    );
    println!("steps:   {:?}", training.steps);
    let trained = arena.run(800);
    println!(
        "trained_run finished={} step={}",
        trained.finished, trained.step
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_runner_uses_rust_core() {
        let mut arena = CognitiveArena::new();
        let snapshot = arena.run(800);
        assert!(snapshot.finished);
        assert_eq!(snapshot.messages_sent, 2);
        assert_eq!(snapshot.scout.carrying, "red");
        assert_eq!(snapshot.carrier.carrying, "blue");
    }
}
