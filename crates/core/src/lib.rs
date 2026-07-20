//! Core neural computation: spiking neuron models, Kuramoto oscillators, and NN layers.
//!
//! - [`kuramoto`] — Kuramoto and T3 oscillator dynamics with sparse/dense CSR propagation.
//! - [`neuron`] — Izhikevich, LIF, AdEx, sawtooth neuron models.

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod neuron;
pub mod kuramoto;
pub mod sparse;
pub mod koit;

pub use taas_kernels as kernels;
