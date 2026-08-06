#![allow(non_snake_case)] // R_a/R_m/C_m, V_soma, dV_dt — физическая нотация

//! # Cable Dendrite Microkernels — Compartmentalized Dendrites
//!
//! Each neuron has `k` compartments connected axially (`R_a`). All
//! compartments have membrane resistance (`R_m`) to the soma.
//!
//! ## Cable equation (Forward Euler)
//! ```text
//! C_m·dV_k/dt = (V_{k−1} − V_k)/R_a   ← axial current from previous
//!              + (V_{k+1} − V_k)/R_a   ← axial current to next
//!              + (V_soma − V_k)/R_m    ← electrotonic current to soma
//!              + Σ g_r[k]·(E_rev_r − V_k)  ← local synaptic current
//!
//! I_dend += (V_k − V_soma)/R_m          ← total current into soma
//! ```
//!
//! ## Layout (column-major, stride = n_padded)
//! ```text
//! buf[comp * n_padded + nrn] = compartment comp, neuron nrn
//! ```
//! This lets us load 8 neurons of one compartment with a single `f32x8`.
//!
//! ## Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`config`] | [`DendriteConfig`], [`DendriteDecayFactors`], [`ActiveDendriteChannels`] |
//! | [`cable`] | [`CableDendriteState`] — cable_step, conductance setters, getters |

pub mod cable;
pub mod config;

pub use cable::{CableDendriteState, DendriteState};
pub use config::{ActiveDendriteChannels, DendriteConfig, DendriteDecayFactors};
