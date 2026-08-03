//! Core neural computation: spiking neuron models, Kuramoto oscillators, and NN layers.
//!
//! - [`kuramoto`] — Kuramoto and T3 oscillator dynamics with sparse/dense CSR propagation.
//! - [`neuron`] — Izhikevich, LIF, AdEx, sawtooth neuron models.

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]

pub mod kuramoto;
pub mod neuron;
pub mod sparse;

// Поглощённые крейты (были отдельными, свернуты для уменьшения воркспейса):
pub mod cpe; // ChaosPatternEngine — dopamine-STDP на Kuramoto
pub mod iit; // Integrated Information Theory (Φ)

pub use taas_kernels as kernels;
