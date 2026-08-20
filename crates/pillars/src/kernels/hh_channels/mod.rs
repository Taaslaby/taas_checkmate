//! # HH Channel Definitions + Polynomial Engine — standard squid-axon & pacemaking (CaV-T)
//!
//! Port of the **entire** legacy `crates/kernels/src/neuron/hh.rs` (волна 8,
//! транш 1e). Two layers:
//!
//! 1. **Channel specs** — what a conductance-based model is made of: the channel
//!    list, per-gate rate functions α(V)/β(V) and steady-state helpers.
//! 2. **Polynomial engine** — `hh_poly_step` / `fused_hh_poly_step`: batch
//!    `f32x8` SIMD + scalar-tail integrators with Q10 temperature correction,
//!    bit-identical to the deleted legacy kernels (SIMD explicit-Euler,
//!    scalar implicit-Euler divergence preserved; fused-scalar clamps v to
//!    [-100, 50], fused-SIMD clamps only Ii to ±100 — as in legacy).
//!
//! The zoo `HHModel` (`run_neuron`) remains the canonical integrator for the
//! standard/pacemaking layouts; this engine is the generic-arity fallback and
//! the reference implementation for parity tests.
//!
//! # Channel sets
//!
//! - [`standard_channels`] — NaV (m³h) + KV (n⁴): the classic Hodgkin-Huxley
//!   squid axon. Requires external drive to spike.
//! - [`pacemaking_channels`] — standard + **CaV-T (T-type calcium)**: a single
//!   low-threshold activation gate `c`. The T-type current depolarizes the
//!   membrane past NaV threshold on its own → the neuron fires **spontaneously**
//!   without any input current (the basis of rhythmic VNC/CPG activity).
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`spec`] | Channel specification types: `ClosedForm`, `GateKinetics`, `GateSpec`, `ChannelSpec`, canonical rate fns |
//! | [`markov`] | Markov kinetic-scheme gates (`gateKS`/`gateFractional`): `MarkovGateSpec`, master-equation integrator |
//! | [`engine`] | Polynomial HH engine: `hh_poly_step` / `fused_hh_poly_step` (SIMD + scalar tail), `ChannelMeta` |
//! | [`tests`] | Scalar reference integrator + parity/validation tests (test-only) |

mod engine;
mod markov;
mod spec;
#[cfg(test)]
mod tests;

pub use engine::*;
pub use markov::*;
pub use spec::*;
