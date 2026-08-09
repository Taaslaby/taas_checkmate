#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(deprecated)]

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

pub use pillars::api::neurons as neuron;
pub use pillars::kernels; // Фаза 11: kernels-крейт удалён — прямой pillars-канон
#[cfg(feature = "cuda")]
pub use pillars_cuda::gpu as cuda;
#[cfg(feature = "t3")]
pub use taas_t3 as t3;
