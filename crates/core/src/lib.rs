//! Core neural computation: spiking neuron models, Kuramoto oscillators, and NN layers.
//!
//! - [`kuramoto`] — Kuramoto and T3 oscillator dynamics with sparse/dense CSR propagation.
//! - [`neuron`] — Izhikevich, LIF, AdEx, sawtooth neuron models.

pub mod neuron;
pub mod kuramoto;
pub mod sparse;
pub mod koit;

pub use taas_kernels as kernels;
