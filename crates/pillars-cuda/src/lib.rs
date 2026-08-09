//! # 🚀 pillars-cuda — one pillars source → CPU SIMD + CUDA
//!
//! NVRTC engine that runs the `#[dual_neuron]` kernels:
//!
//! ```rust,ignore
//! #[dual_neuron]            // ← proc-macro from pillars-macros
//! pub mod hh {
//!     #[derive(NeuronModel)]
//!     pub struct HHModel { #[state] v, #[param] g_Na, … }
//!
//!     impl ComputeNeuron for HHModel {
//!         fn step<S: SimdLane>(…) { … }   // THE single source of truth
//!     }
//! }
//! ```
//!
//! - The Rust code is untouched → runs on CPU SIMD via `NeuralPopulation`.
//! - The macro also emits `HH_CUDA_SOURCE` — a CUDA C++ kernel string.
//! - [`CompiledModel::new`] feeds it to **NVRTC** at runtime → PTX → kernel.
//!
//! No duplicated equations, no AOT `.cu` files, no codegen build steps.
//!
//! All neuron models live **in `pillars::zoo::neurons`** (single source of
//! truth) — every model is `#[dual_neuron]` there and exports its
//! `*_CUDA_SOURCE`. This crate re-exports them and adds the NVRTC driver.

pub mod engine;
pub mod ffi;

/// AOT GPU microkernels + persistent device context (moved verbatim from the
/// legacy `cuda-kernels` crate, Волна 13). One `.cu` per microkernel in
/// `kernels/`, FFI wrappers here. Exposed upstream as `cuda`:
/// `pub use pillars_cuda::gpu as cuda;`.
pub mod gpu;

pub use engine::CompiledModel;

// ── Zoo models transpiled IN pillars (no copy here — single source) ──
pub use pillars::zoo::neurons::{
    AdExModel, AtlasovNeuromotorModel, FHNModel, GLIF3Model, GLIF4Model, HAAPModel, HHModel,
    HRModel, IzhikevichDualModel, IzhikevichModel, LIFAtModel, LIFModel, MorrisLecarModel,
    RateModel, SAWModel, VSNModel, WCModel,
};
pub use pillars::zoo::neurons::{
    ADEX_CUDA_SOURCE, ATLASOV_NEUROMOTOR_CUDA_SOURCE, FHN_CUDA_SOURCE, GLIF3_CUDA_SOURCE,
    GLIF4_CUDA_SOURCE, HAAP_CUDA_SOURCE, HH_CUDA_SOURCE, HR_CUDA_SOURCE, IZHIKEVICH_CUDA_SOURCE,
    IZHIKEVICH_DUAL_CUDA_SOURCE, LIFAT_CUDA_SOURCE, LIF_CUDA_SOURCE, MORRIS_LECAR_CUDA_SOURCE,
    RATE_CUDA_SOURCE, SAW_CUDA_SOURCE, VSN_CUDA_SOURCE, WC_CUDA_SOURCE,
};
