//! # d_λ discretization + passive cable solver (Фаза 5d) — MorphoIR → cable
//!
//! Bridges real morphology ([`MorphoIR`](super::parse::MorphoIR), from NeuroML
//! `<morphology>` or SWC) to a flat compartment tree and runs the **passive**
//! cable equation over it (implicit Euler via the Hines algorithm).
//!
//! ## d_λ rule
//!
//! Each segment is split into `n = max(1, ⌈L/(d_λ·λ)⌉)` compartments of length
//! `L/n` and the segment's average diameter, where `λ = √(d·Rm/4·Ra)` is the DC
//! space constant. A segment's first compartment connects to the parent
//! segment's last compartment; the rest chain linearly. The root (soma) is one
//! compartment — a zero-length single-point soma gets effective length = its
//! diameter (cylinder convention, matching NEURON's `L = diam` for SWC soma).
//!
//! When a segment feeds **more than one** child, a zero-volume **branch node**
//! is inserted at its distal end (`r_axial=0`, `area=cm=g_leak=0`); the children
//! connect to it. This keeps the parent's distal half-resistance a single shared
//! resistor (NEURON's branch-point convention) instead of duplicating it per
//! child — duplicating would over-couple the daughters (a real ~1% error).
//!
//! ## Unit contract (NEURON convention, same as `kernels::dendrite::cable`)
//!
//! * length / diameter — **µm**; axial resistivity `Ra` — **Ω·cm**;
//! * specific membrane capacitance `cm` — **µF/cm²**; specific membrane
//!   resistance `Rm` — **Ω·cm²**;
//! * derived per compartment: area `A = π·d·L` (µm²), capacitance
//!   `C = cm·A` in **nF**, leak conductance `g_leak = A/Rm` in **µS**, axial
//!   resistance `R = 4·Ra·L/(π·d²)` in **MΩ**.
//!
//! The solver uses V in **mV**, t in **ms**, I in **nA** — the set where
//! `C[nF]·dV/dt[ms] = g[µS]·(E−V)[mV] = (ΔV)[mV]/R[MΩ] = I[nA]` are all
//! dimensionally identical, so the membrane time constant is simply
//! `τ = C[nF]/g[µS]` (ms).
//!
//! ## Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`geometry`] | `CableParams`, `Compartment`, `CompartmentTree`, passive leak helpers + physics constants |
//! | [`segmentation`] | the `d_λ` `discretize` pass: `MorphoIR` → `CompartmentTree` |
//! | [`cable`] | passive `CableSolver` (implicit Euler / Hines) |
//! | [`channels`] | active HH + Markov + stochastic (binomial) channels |
//! | [`nernst`] | Nernst/GHK Ca²⁺ channels, `CaPool`, flux scales |
//! | [`active`] | `ActiveCableSolver` (semi-implicit cable + active conductances) |
//! | [`population`] | `CablePopulation` — the `Network` bridge |

mod active;
mod cable;
mod channels;
mod geometry;
mod nernst;
mod population;
mod segmentation;
#[cfg(test)]
mod tests;

pub(crate) use active::*;
pub(crate) use cable::*;
pub(crate) use channels::*;
pub(crate) use geometry::*;
pub(crate) use nernst::*;
pub(crate) use population::*;
pub(crate) use segmentation::*;
