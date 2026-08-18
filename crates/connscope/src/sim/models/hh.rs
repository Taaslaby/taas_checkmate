//! Hodgkin-Huxley (NaV + KV channels).

use pillars_connectome::{BrainConfig, NeuronConfig};

use super::NeuronModel;

/// Hodgkin-Huxley — стандартные каналы NaV + KV, edge-triggered без v_reset.
pub struct Hh;

impl NeuronModel for Hh {
    fn short_name(&self) -> &'static str {
        "HH"
    }
    fn name(&self) -> &'static str {
        "HH"
    }
    fn config(&self, dt_ms: f32) -> NeuronConfig {
        BrainConfig::hh(dt_ms)
    }
}
