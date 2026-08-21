//! HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting).

use pillars_connectome::{BrainConfig, BrainRunner, NeuronConfig};

use crate::core::pharma::Target;

use super::NeuronModel;

/// HAAP — M-current (`u`) адаптация + h-current (`h`) rebound: частота не
/// разгоняется, как у HH. Колонка h-тока настраивается слайдером `g_h`.
pub struct Haap;

impl NeuronModel for Haap {
    fn short_name(&self) -> &'static str {
        "HAAP"
    }
    fn name(&self) -> &'static str {
        "HAAP"
    }
    fn config(&self, dt_ms: f32) -> NeuronConfig {
        BrainConfig::haap(dt_ms)
    }
    fn tune(&self, runner: &mut BrainRunner, n: usize, gh: f32) {
        runner.set_haap_param_column(6, &vec![gh; n]);
    }
    fn has_h_current(&self) -> bool {
        true
    }
    fn has_channel(&self, target: Target) -> bool {
        matches!(target, Target::M | Target::Hcn | Target::Leak)
    }
}
