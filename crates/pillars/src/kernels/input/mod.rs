//! Input Encoding — Poisson, BSA, TTFS, Phase
//!
//! Converts analog signals into spike trains for SNN input encoding.
//!
//! ## Encoder types
//!
//! | Encoder | Mechanism | Output | Use |
//! |---------|----------|--------|-----|
//! | [`PoissonGenerator`] | P(spike) = rate·dt, LCG RNG | Poisson spike train | Rate-coded input |
//! | [`bsa_encode`] | `|signal − recon| > θ → emit + add kernel·sign(err)` | Spike train + reconstruction | Bens Spiker Algorithm |
//! | [`phase_encode`] | `phase += signal·gain, spike if cos(phase) > θ` | Phase-dependent spikes | Temporal encoding |
//! | [`TtfsEncoder`] | `delay = (1−value)·max_delay` | Single spike at t = delay | Time-to-First-Spike |
//! | [`PwmEncoder`] | `sensor > (t/period + shift) mod 1 → spike_current` | PWM duty-cycle encoding | Analog→spike rate |
//! | [`LinearDecoder`] | `out = W·states + bias` | Linear readout (kernel: `signal::linear_decoder`) | Decode populations |
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | `poisson` | `PoissonGenerator` |
//! | `bsa` | `bsa_encode`, `bsa_kernel_hamming` |
//! | `phase` | `phase_encode` |
//! | `pwm` | `PwmEncoder` |
//! | `linear` | `LinearDecoder` |
//! | `ttfs` | `TtfsEncoder` |
//! | `tests` | unit tests (test-only) |

mod bsa;
mod linear;
mod phase;
mod poisson;
mod pwm;
#[cfg(test)]
mod tests;
mod ttfs;

pub use bsa::*;
pub use linear::*;
pub use phase::*;
pub use poisson::*;
pub use pwm::*;
pub use ttfs::*;
