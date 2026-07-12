//! Spiking RL, T3-SPIKE GRAD BPTT, optimizers.

pub mod rl;

#[cfg(feature = "t3")]
pub mod spike;

#[cfg(feature = "t3")]
pub mod spike_grad;

#[cfg(all(feature = "t3", feature = "cuda"))]
pub mod spike_cuda;

#[cfg(feature = "connectome")]
pub mod connectome_chase;

pub use taas_kernels as kernels;
pub use taas_core as core;
#[cfg(feature = "t3")]
pub use taas_t3 as t3;
#[cfg(feature = "cuda")]
pub use cuda_kernels as cuda;
