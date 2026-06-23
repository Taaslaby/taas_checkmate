//! # T3 (Temporal Triplet) engine — super-modular architecture.
//!
//! 3-population Kuramoto oscillator system with Stuart-Landau amplitudes,
//! PST features, Phase Gate, Cochlea filter bank, and pluggable decoders.
//!
//! ## Модули
//!
//! | Модуль | Назначение |
//! |--------|-----------|
//! | [`backend`] | `T3Backend` trait + AVX2/CUDA бэкенды |
//! | [`oscillator`] | `TriOscillator<B: T3Backend>` — generic 3-population |
//! | [`phase_gate`] | Sigmoid phase gate для cross-coupling |
//! | [`cochlea`] | Biquad filter bank + envelope follower |
//! | [`pst`] | PST features (pairwise O(N²) + linear O(N)) |
//! | [`trajectory`] | Ring buffer для L-frame траекторий |
//! | [`decoders`] | `Decoder` trait + NLMS/Kalman/Inverse/STDP/PST |
//! | [`ridge`] | RidgeSolver + CG + Pearson + normalization |
//!
//! ## Пример
//!
//! ```rust
//! use taas_checkmate::t3::{TriOscillator, Avx2Backend, Decoder, NlmsDecoder};
//!
//! let mut tri = TriOscillator::new(128, 0.8, 1.2, Avx2Backend);
//! tri.step(0.001);
//!
//! let (ph1, ph2, ph3) = tri.phases();
//! let mut dec = NlmsDecoder::new(128, 0.01, 1e-8);
//! dec.train(ph1, 0.5);
//! ```

#![allow(non_snake_case)]

pub mod backend;
pub mod phase_gate;
pub mod cochlea;
pub mod pst;
pub mod trajectory;
pub mod ridge;
pub mod decoders;
pub mod oscillator;
pub mod spike;  // T2-SPIGRAD — differentiable T3-SPIKE encoder

// Re-export spike module items
pub use spike::{
    T3SpikeTrajectory,
    t3spike_forward,
    t3spike_forward_with_backend,
    t3spike_backward,
};

#[cfg(feature = "cuda")]
pub mod backend_cuda;

#[cfg(feature = "cuda")]
pub mod spike_cuda;  // GPU-accelerated T2-SPIGRAD backward (K1)

// ─── Constants ───────────────────────────────────────────

pub const PI: f32 = std::f32::consts::PI;
pub const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

// ─── Re-exports ──────────────────────────────────────────

pub use backend::{T3Backend, Avx2Backend, ScalarBackend};

#[cfg(feature = "cuda")]
pub use backend_cuda::CudaBackend;

pub use oscillator::TriOscillator;

pub use phase_gate::{phase_gate, gate_coupling};

pub use cochlea::{Biquad, Cochlea};

pub use pst::{pst_features, pst_linear_features, pst_linear_dim};

pub use trajectory::TrajectoryBuffer;

pub use decoders::{
    Decoder,
    NlmsDecoder, KalmanDecoder,
    InverseKuramotoDecoder, StdpDecoder, PstDecoder,
    HybridDecoder,
};

pub use ridge::{RidgeSolver, solve_cg, compute_pearson, normalize_features};
