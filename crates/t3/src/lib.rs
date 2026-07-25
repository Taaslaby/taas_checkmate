#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_without_is_empty)]

pub use taas_kernels as kernels;

#[cfg(feature = "cuda")]
pub use cuda_kernels as cuda;

pub mod attention_coupling;
pub mod backend;
pub mod phase_gate;
pub mod cochlea;
pub mod pst;
pub mod trajectory;
pub mod ridge;
pub mod decoders;
pub mod oscillator;
pub mod pst_control;
pub mod spiker;
#[cfg(feature = "cuda")]
pub mod backend_cuda;

pub const PI: f32 = std::f32::consts::PI;
pub const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

pub use backend::{T3Backend, Avx2Backend, ScalarBackend};

#[cfg(feature = "cuda")]
pub use backend_cuda::CudaBackend;

pub use oscillator::TriOscillator;

pub use phase_gate::{phase_gate, gate_coupling};

pub use cochlea::{Biquad, Cochlea};

pub use pst::{pst_features, pst_linear_features, pst_linear_features_into, pst_linear_dim, pst_fast, pst_fast_dim};

pub use trajectory::TrajectoryBuffer;

pub use decoders::{
    Decoder,
    NlmsDecoder, KalmanDecoder,
    InverseKuramotoDecoder, StdpDecoder, PstDecoder,
    HybridDecoder,
    nlms_decode, nlms_update,
    kalman_decode,
    inverse_kuramoto_decode,
    stdp_decode,
};

pub use ridge::{RidgeSolver, solve_cg, compute_pearson, normalize_features};

pub use pst_control::{PstControl, PstControlConfig};

pub use spiker::{
    Spiker, SpikeTrace, HologramEncoder, SpikeStats,
    NormalizedSpiker, HybridSpiker,
};
