//! # Spike Statistics Microkernels
//!
//! Firing rates, interspike intervals (ISI), coefficient of variation (CV),
//! population rate decoding and the [`SpikeRaster`] event collector.
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`decode`] | `population_rate_decode` |
//! | [`rates`] | `firing_rates`, `mean_firing_rate` |
//! | [`isi`] | `compute_isi`, `compute_cvs` |
//! | [`raster`] | `SpikeRaster`, `spike_raster_to_csv` |
//! | [`tests`] | unit tests (test-only) |

mod decode;
mod isi;
mod raster;
mod rates;
#[cfg(test)]
mod tests;

pub use decode::*;
pub use isi::*;
pub use raster::*;
pub use rates::*;
