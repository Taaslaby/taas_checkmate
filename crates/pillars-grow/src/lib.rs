//! # pillars-grow — developmental growth engine
//!
//! Rust port of the **CX3D** instruction-language core (Zubler & Douglas, 2009).
//! CX3D grows neural networks from a *genome* = a set of small rules
//! (`LocalBiologyModule` / `CellModule`) attached to cells, rather than from
//! explicit weights.
//!
//! ```text
//! genome (modules) ──► cells divide / migrate / secrete morphogens
//!                          │
//!                          └─► growth cones follow morphogen gradients
//!                                 │
//!                                 └─► neurites meet → synapses → network
//! ```
//!
//! ## Layers (mirroring CX3D)
//!
//! | this crate | CX3D | role |
//! |---|---|---|
//! | [`Module`] | `LocalBiologyModule` | an instruction living inside a cell element |
//! | [`CellModule`] | `CellModule` (`SimpleCellCycle`) | a cell-level instruction (division) |
//! | [`Substance`] | `Substance` | extracellular morphogen (diffusion + gradients) |
//! | [`World`] | `ECM` + `Scheduler` | the simulation: cells, substances, time |
//!
//! The "genome" is the set of modules you attach to the first progenitor cell.
//! At division/branching each module is copied into the daughter element when
//! its `copy_on_*` flags allow it — exactly the self-construction mechanism of
//! CX3D's *An instruction language for self-construction* (Zubler et al., 2011).

mod math;
mod module;
mod substance;
mod cell;
mod modules;
mod genome;
mod dna;
mod world;

pub use math::Vec3;
pub use module::{CellModule, Host, Module};
pub use substance::{ArtificialGradient, Grid, Substance};
pub use cell::{Cell, CellType, Neurite, Soma};
pub use modules::{ChemoAttraction, Differentiate, DivideOnce, Secretor, SimpleCellCycle};
pub use genome::{parse_dna_map, DnaGene, FlyGenome};
pub use dna::{extract_gene, parse_fasta, parse_gene_coords, read_maybe_gzip, revcomp};
pub use world::{Bounds, GrownEdge, GrownGraph, GrownNode, Synapse, SynapseKind, World};
