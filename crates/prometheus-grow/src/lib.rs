//! # prometheus-grow — developmental growth engine
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

#[path = "engine/branch_model.rs"]
pub mod branch_model;
#[path = "entity/cell.rs"]
mod cell;
#[path = "engine/diameter_model.rs"]
pub mod diameter_model;
#[path = "engine/direction_model.rs"]
pub mod direction_model;
#[path = "genetics/dna.rs"]
mod dna;
#[path = "entity/entity.rs"]
pub mod entity;
#[path = "engine/exporter.rs"]
pub mod exporter;
#[path = "genetics/genome.rs"]
mod genome;
#[path = "gpu/mod.rs"]
pub mod gpu;
#[path = "entity/handle.rs"]
pub mod handle;
#[path = "engine/math.rs"]
mod math;
#[path = "engine/mechanics.rs"]
pub mod mechanics;
#[path = "engine/metamorphosis.rs"]
pub mod metamorphosis;
#[path = "engine/module.rs"]
mod module;
#[path = "engine/modules.rs"]
mod modules;
#[path = "engine/operation.rs"]
pub mod operation;
#[path = "engine/param.rs"]
pub mod param;
#[path = "genetics/physiology.rs"]
pub mod physiology;
#[path = "engine/statistics.rs"]
pub mod statistics;
#[path = "engine/substance.rs"]
mod substance;
#[path = "engine/synapse_model.rs"]
pub mod synapse_model;
#[path = "engine/trp.rs"]
pub mod trp;
#[path = "engine/turning_model.rs"]
pub mod turning_model;
#[path = "engine/uniform_grid.rs"]
pub mod uniform_grid;
#[path = "engine/world.rs"]
mod world;

pub use branch_model::BalancedForces;
pub use cell::{Cell, CellType, Excrescence, ExcrescenceKind, Neurite, Soma};
pub use diameter_model::{assign_rall_diameters, assign_top_down, RallPowerLaw};
pub use direction_model::{RadialSteering, TensionSteering};
pub use dna::{extract_gene, parse_fasta, parse_gene_coords, read_maybe_gzip, revcomp};
pub use entity::{Entity, EntityKind};
pub use genome::{parse_channel_map, parse_dna_map, ChannelGene, DnaGene, FlyGenome};
#[cfg(feature = "gpu")]
pub use gpu::trp_backend::TrpPipeline;
#[cfg(feature = "gpu")]
pub use gpu::GpuContext;
pub use gpu::{interior_index, DiffusionBackend};
pub use math::Vec3;
pub use mechanics::{compute_mechanics, step_mechanics, CellVelocity, MechanicsParams};
pub use metamorphosis::Metamorphosis;
pub use module::{CellModule, DeferredOp, Host, Module, ParCellModule, ParWorldView};
pub use modules::{
    Astrocyte, BranchOnDiameter, CellApoptosis, ChemoAttraction, ChemoRepulsion, Differentiate,
    DivideOnce, Fasciculation, LateralInhibition, NetmorphGrowth, PointAttraction, RetractNeurite,
    SecondOrderElongation, Secretor, SimpleCellCycle, SomaMigration, TrophicCompetition,
    VanPeltBranching, VanPeltElongation,
};
pub use physiology::{gene_haap, nt_gene};
pub use statistics::{collect_cell_stats, ArborStats, FiberStats, SampleStats};
pub use substance::{ArtificialGradient, BoundaryCondition, Grid, Substance};
pub use synapse_model::{Receptor, SynapseFormationModel};
pub use trp::{form_synapses_trp, init_embeddings, TrpEmbedding, TrpParams, TRP_DIM};
pub use turning_model::TurnModel;
pub use uniform_grid::UniformGrid;
pub use world::{
    Bounds, GrownEdge, GrownGraph, GrownNode, Recognition, Synapse, SynapseKind, TagRecognition,
    World,
};
