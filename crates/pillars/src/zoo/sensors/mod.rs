//! # 📡 Sensors — Bio-Inspired Sensory Neuron Models
//!
//! The **input layer** of the brain: biological transducers that convert
//! physical signals (stretch, pressure, light, sound) into spike trains
//! the network can consume.
//!
//! ## Design
//!
//! Each sensor is a **stateless or near-stateless transform** — feed it a
//! physical stimulus, get spikes out. Sensors live at the edge of the
//! network: they read the world, not the connectome.
//!
//! ## Models
//!
//! | Model | Stimulus | Biology |
//! |-------|----------|---------|
//! | [`chordotonal`] | mechanical stretch | Drosophila scolopidial mechanoreceptors (SIMD-accelerated) |

pub mod chordotonal;
pub use chordotonal::chordotonal_step;
