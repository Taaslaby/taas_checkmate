//! # Neuromotor Kernel — Discrete Ca²⁺ & Active Reset Muscle Engine
//!
//! Канон pillars::api: step-функция и stateful-классы перенесены
//! из core. Модель — `run_neuron::<NeuromotorModel>` (single source
//! of truth).
//!
//! Models:
//! - Spike-triggered Calcium Injection upon motor neuron spikes
//! - Exponential pump clearance & active mechanical reset kinetics
//! - Cooperative saturation muscle tension (Hill-type curve)
//! - Agonist-Antagonist Tug-of-War (Joint net torque & dynamic stiffness)
//!
//! Author / Concept: Anton Atlasov (Атласов Антон)
//!
//! маленькие мускулы из маленьких клеток строят маленьких человечков,
//! которые потом будут строить человечков побольше — и так до бесконечности
//!
//! ## Module boundaries (anti-monolith)
//!
//! | Concern | Module |
//! |---------|--------|
//! | SIMD/serial kernel ([`step_neuromotor`]) | [`kernel`](self) |
//! | Biomechanics/fiber/leg configs | [`config`](self) |
//! | Object-oriented runtime ([`NeuromotorState`]) | [`state`](self) |

mod config;
mod kernel;
mod state;
#[cfg(test)]
mod tests;

pub use config::{BiomechanicsProfile, DrosophilaLegConfig, FiberPoolConfig};
pub use kernel::step_neuromotor;
pub use state::{NeuromotorState, NeuromotorStateBuilder};
