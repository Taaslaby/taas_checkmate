#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_without_is_empty)]

pub use pillars::kernels; //: kernels-крейт удалён — прямой pillars-канон

#[cfg(feature = "cuda")]
pub use pillars_cuda::gpu as cuda;

pub mod attention_coupling;
pub mod backend;
#[cfg(feature = "cuda")]
pub mod backend_cuda;
pub mod cochlea;
pub mod decoders;
pub mod oscillator;
pub mod phase_gate;
pub mod pst;
pub mod pst_control;
pub mod ridge;
pub mod spiker;
pub mod torus;
pub mod trajectory;

pub const PI: f32 = std::f32::consts::PI;
pub const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

pub use backend::{Avx2Backend, ScalarBackend, T3Backend};

#[cfg(feature = "cuda")]
pub use backend_cuda::CudaBackend;

pub use oscillator::TriOscillator;

pub use phase_gate::{gate_coupling, phase_gate};

pub use cochlea::{Biquad, Cochlea};

pub use pst::{
    pst_fast, pst_fast_dim, pst_features, pst_linear_dim, pst_linear_features,
    pst_linear_features_into,
};

pub use trajectory::TrajectoryBuffer;

pub use decoders::{
    inverse_kuramoto_decode, kalman_decode, nlms_decode, nlms_update, stdp_decode, Decoder,
    HybridDecoder, InverseKuramotoDecoder, KalmanDecoder, NlmsDecoder, PstDecoder, StdpDecoder,
};

pub use ridge::{compute_pearson, normalize_features, solve_cg, RidgeSolver};

pub use pst_control::{PstControl, PstControlConfig};

pub use spiker::{HologramEncoder, HybridSpiker, NormalizedSpiker, SpikeStats, SpikeTrace, Spiker};
