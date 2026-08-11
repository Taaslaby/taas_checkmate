//! JavaScript boundary for the Rust-owned Cognitive Arena missions.
//!
//! JS receives snapshots; it never owns or reimplements simulation state.

use wasm_bindgen::prelude::*;

use crate::karmac::{CognitiveArena, HideSeekArena, PlatformerArena};

#[wasm_bindgen]
pub struct CognitiveArenaWasm {
    inner: CognitiveArena,
}

#[wasm_bindgen]
impl CognitiveArenaWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: CognitiveArena::new(),
        }
    }

    /// Advance exactly one Rust simulation tick and return the serialized snapshot.
    pub fn step(&mut self) -> Result<String, JsValue> {
        self.inner.step();
        self.snapshot()
    }

    /// Run at most `steps` Rust ticks and return the final snapshot.
    pub fn run(&mut self, steps: usize) -> Result<String, JsValue> {
        self.inner.run(steps);
        self.snapshot()
    }

    /// Train the spiking brains over `episodes` episodes of at most
    /// `max_steps` each (reward-modulated STDP; world resets between
    /// episodes, weights persist) and return the final snapshot with the
    /// learning curve.
    pub fn train(&mut self, episodes: usize, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train(episodes, max_steps);
        self.snapshot()
    }

    /// One training episode (continuous learning): first call perturbs the
    /// brains, later calls keep the learned weights. Call repeatedly to train
    /// indefinitely.
    pub fn train_one(&mut self, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train_one(max_steps);
        self.snapshot()
    }

    /// Start a fresh round with the trained brains preserved (auto-repeat /
    /// play-after-training). Returns the reset snapshot.
    pub fn next_round(&mut self) -> Result<String, JsValue> {
        self.inner.reset_episode();
        self.snapshot()
    }

    /// Reset the Rust simulation to its deterministic initial state.
    pub fn reset(&mut self) -> Result<String, JsValue> {
        self.inner.reset();
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<String, JsValue> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsValue::from_str(&format!("snapshot serialization failed: {error}")))
    }

    pub fn finished(&self) -> bool {
        self.inner.finished()
    }
}

impl Default for CognitiveArenaWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Mission 03: Hide & Seek — 2 hiders vs 3 seekers on a large procedural
/// map with box barricades. Same Rust-owned contract as Mission 02.
#[wasm_bindgen]
pub struct HideSeekWasm {
    inner: HideSeekArena,
}

#[wasm_bindgen]
impl HideSeekWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: HideSeekArena::new(0x5EED),
        }
    }

    pub fn step(&mut self) -> Result<String, JsValue> {
        self.inner.step();
        self.snapshot()
    }

    pub fn run(&mut self, steps: usize) -> Result<String, JsValue> {
        for _ in 0..steps {
            if self.inner.finished() {
                break;
            }
            self.inner.step();
        }
        self.snapshot()
    }

    /// Train all five brains over `episodes` episodes (world resets between
    /// episodes, weights persist) and return the learning curve snapshot.
    pub fn train(&mut self, episodes: usize, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train(episodes, max_steps);
        self.snapshot()
    }

    /// One training episode (continuous learning); weights persist across
    /// calls. Call repeatedly to train indefinitely.
    pub fn train_one(&mut self, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train_one(max_steps);
        self.snapshot()
    }

    /// Start a fresh round with the trained brains preserved (auto-repeat /
    /// play-after-training). Advances the round counter so every repeat gets
    /// fresh spawn layouts. Returns the reset snapshot.
    pub fn next_round(&mut self) -> Result<String, JsValue> {
        self.inner.next_round();
        self.snapshot()
    }

    pub fn reset(&mut self) -> Result<String, JsValue> {
        self.inner.reset();
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<String, JsValue> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsValue::from_str(&format!("snapshot serialization failed: {error}")))
    }

    pub fn finished(&self) -> bool {
        self.inner.finished()
    }
}

impl Default for HideSeekWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Mission 04: Neon Run — a Rust-owned 2D platformer controlled by a
/// spiking brain. The browser receives world geometry, physics telemetry and
/// brain activity; it never implements the game rules.
#[wasm_bindgen]
pub struct PlatformerWasm {
    inner: PlatformerArena,
}

#[wasm_bindgen]
impl PlatformerWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: PlatformerArena::default(),
        }
    }

    pub fn step(&mut self) -> Result<String, JsValue> {
        self.inner.step();
        self.snapshot()
    }

    pub fn run(&mut self, steps: usize) -> Result<String, JsValue> {
        for _ in 0..steps {
            if self.inner.finished() || self.inner.dead() {
                break;
            }
            self.inner.step();
        }
        self.snapshot()
    }

    pub fn train(&mut self, episodes: usize, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train(episodes, max_steps);
        self.snapshot()
    }

    pub fn train_one(&mut self, max_steps: usize) -> Result<String, JsValue> {
        self.inner.train_one(max_steps);
        self.snapshot()
    }

    pub fn next_round(&mut self) -> Result<String, JsValue> {
        self.inner.next_round();
        self.snapshot()
    }

    pub fn replay_best(&mut self) -> Result<String, JsValue> {
        self.inner.replay_best();
        self.snapshot()
    }

    pub fn evaluate(&mut self, episodes: usize, max_steps: usize) -> Result<String, JsValue> {
        self.inner.evaluate(episodes, max_steps);
        self.snapshot()
    }

    pub fn reset(&mut self) -> Result<String, JsValue> {
        self.inner.reset();
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<String, JsValue> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsValue::from_str(&format!("snapshot serialization failed: {error}")))
    }

    pub fn finished(&self) -> bool {
        self.inner.finished()
    }
}

impl Default for PlatformerWasm {
    fn default() -> Self {
        Self::new()
    }
}
