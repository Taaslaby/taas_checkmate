//! Izhikevich (regular spiking).

use pillars_connectome::{BrainConfig, NeuronConfig};

use super::NeuronModel;

/// Izhikevich — a=0.02, b=0.2, c=-65, d=8 (regular spiking).
pub struct Izh;

impl NeuronModel for Izh {
    fn short_name(&self) -> &'static str {
        "Izh"
    }
    fn name(&self) -> &'static str {
        "Izhikevich"
    }
    fn config(&self, dt_ms: f32) -> NeuronConfig {
        BrainConfig::izh(dt_ms)
    }
}
