//! Leaky Integrate-and-Fire (current-based propagation).

use pillars_connectome::{BrainConfig, NeuronConfig};

use super::NeuronModel;

/// LIF — τ_m=20мс, v_rest=-70, v_thresh=-58, v_reset=-70, τ_ref=2мс.
pub struct Lif;

impl NeuronModel for Lif {
    fn short_name(&self) -> &'static str {
        "LIF"
    }
    fn name(&self) -> &'static str {
        "LIF"
    }
    fn config(&self, dt_ms: f32) -> NeuronConfig {
        BrainConfig::lif(dt_ms)
    }
}
