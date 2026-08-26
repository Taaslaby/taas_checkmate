//! -------------------------------------------------------------------------
//! Filename:    src/api/net/mod.rs
//! Class:       net
//! Purpose:     Network wiring — spike routing, topology, synapse specs
//! Description: The substrate that `Network` (the orchestrator) drives:
//!              CSR topology, type-erased populations, delay ring buffers,
//!              event-driven push, and synapse kinetics/plasticity profiles.
//!
//! Modules:
//!   graph       — SparseGraph (CSR adjacency)
//!   population  — Population trait (type erasure) + PopulationId
//!   delay       — DelayBuffer (ring buffer of synaptic currents)
//!   compact     — spike compaction (SIMD move_mask + tzcnt)
//!   push        — event-driven scatter (seq + rayon-par)
//!   synapse     — PlasticityProfile + SynapseKinetics
//!   network     — Network (multi-population orchestrator)
//!
//! Re-exported at `crate::api::{graph, population, delay, compact, push,
//! synapse, network}` so external callers keep their historical paths.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

pub mod compact;
pub mod delay;
pub mod graph;
pub mod network;
pub mod population;
pub mod push;
pub mod synapse;
