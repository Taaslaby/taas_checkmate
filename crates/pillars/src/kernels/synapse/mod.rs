//! # ⚡ Synapse Microkernels — Conductance-Based Kinetics (7 Receptors)
//!
//! Port of the legacy `crates/kernels/src/neuron/synapse.rs` — replaces
//! instant current injection with realistic conductance dynamics.
//!
//! ## Kinetics
//!
//! ```text
//!   g(t+dt) = g(t) · exp(-dt/τ)         ← exponential decay
//!   g(t)   += w · spike(t)              ← pre-spike jump
//!   I_syn   = Σ g_r · (E_rev_r − V)     ← ohmic current (NEURON convention)
//!             └─ NMDA: g · B(V)         ← Mg²⁺ voltage block
//! ```
//!
//! ## Drosophila Receptor Types
//!
//! | Code | Receptor | τ_decay | E_rev | Ion | Effect | NT |
//! |------|----------|---------|-------|-----|--------|-----|
//! | 0 | `AMPA` | 2 ms | 0 mV | Na⁺ | Fast excitatory (mammalian fallback) | Glu |
//! | 1 | `NMDA` | 50 ms | 0 mV | Ca²⁺ | Slow excitatory, Mg²⁺ voltage block | Glu |
//! | 2 | `GABA_A` | 10 ms | -65 mV | Cl⁻ | Shunting inhibition (insect) | GABA |
//! | 3 | `GABA_B` | 200 ms | -90 mV | G-protein | Slow inhibition (GPCR) | GABA |
//! | 4 | `nAChR` | 3 ms | 0 mV | Na⁺/K⁺ | Fast excitatory (main insect exc) | ACh |
//! | 5 | `GluCl` | 5 ms | -70 mV | Cl⁻ | Inhibitory Cl⁻ channel | Glutamate |
//! | 6 | `mAChR` | 150 ms | -90 mV | GPCR | Slow modulatory (DA/SER/OCT) | DA/5-HT/OA |
//!
//! ## Lifecycle (per simulation step)
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────┐
//!   │  1. decay_all(factors)    g *= exp(-dt/τ)           │
//!   │  2. jump(neuron, type, w) g += w · pre_spike        │
//!   │  3. compute_currents(V)   I = Σ g·(E_rev − V)       │
//!   └──────────────────────────────────────────────────────┘
//!                                     │
//!                                     ∨
//!                          i_in[] → NeuralPopulation::step()
//! ```
//!
//! ## NMDA Mg²⁺ Block
//!
//! `B(V) = 1 / (1 + 0.33·exp(-0.06·V))`
//!
//! At rest (-65 mV): B ≈ 0.02 — channel is blocked.
//! At spike peak (+30 mV): B ≈ 0.95 — channel unblocks.
//!
//! ## Usage
//!
//! ```
//! use pillars::kernels::synapse::{SynapticKinetics, DecayFactors, ReceptorType};
//!
//! // 1. Allocate conductance buffers for 8 neurons
//! let mut kin = SynapticKinetics::new(8);
//!
//! // 2. Precompute decay factors (rebuild only when dt changes)
//! let factors = DecayFactors::new(0.1);  // dt = 0.1 ms
//!
//! // per-step loop:
//! //   3. Decay all conductances: g *= exp(-dt/τ)
//! //   4. Jump on pre-spike: g_post += w * spike
//! //   5. Compute synaptic current contribution
//! kin.decay_all(&factors);                   // SIMD: 7× f32x8 decay
//! kin.jump(0, ReceptorType::nAChR, 0.5);     // pre-spike → ACh conductance ↑
//! kin.jump(1, ReceptorType::GABA_A, 0.3);    // pre-spike → GABA conductance ↑
//!
//! let mut i_syn = vec![0.0f32; 8];
//! let v_memb = vec![-65.0f32; 8];
//! kin.compute_currents(&v_memb, &mut i_syn); // I_out += Σ g·(E_rev − V)
//! // i_syn[0] = 0.5 * (0 - (-65)) = 32.5 pA (depol via nAChR)
//! // i_syn[1] = 0.3 * (-65 - (-65)) = 0 pA (GABA_A shunt at rest)
//! ```
//!
//! ## References
//!
//! - Dayan & Abbott (2001) *Theoretical Neuroscience*, MIT Press.
//! - Wilson et al. (2018) "Drosophila olfactory bulb — fast ACh excitation".

mod kinetics;
mod receptor;
mod stp;
#[cfg(test)]
mod tests;

pub use kinetics::*;
pub use receptor::*;
pub use stp::*;
