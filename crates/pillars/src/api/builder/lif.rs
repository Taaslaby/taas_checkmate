//! Builder module.

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::zoo::neurons::LIFModel;

/// Builder for Leaky Integrate-and-Fire neuron populations.
///
/// # Example
/// ```rust,ignore
/// let pop = LIFBuilder::new(1000)
///     .dt(0.1)
///     .tau_m(10.0)
///     .v_thresh(-54.0)
///     .build();
/// ```
pub struct LIFBuilder {
    n: usize,
    dt: f32,
    tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_ref: f32,
    v_init: f32,
}

impl LIFBuilder {
    /// Create a new builder for `n` neurons with default parameters.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            dt: 0.1,
            tau_m: 10.0,
            v_rest: -70.0,
            v_thresh: -54.0,
            v_reset: -70.0,
            tau_ref: 2.0,
            v_init: -70.0,
        }
    }

    /// Set time step (ms). Default: 0.1 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set membrane time constant (ms). Default: 10.0 ms.
    pub fn tau_m(mut self, tau_m: f32) -> Self {
        self.tau_m = tau_m;
        self
    }

    /// Set resting potential (mV). Default: -70.0 mV.
    pub fn v_rest(mut self, v_rest: f32) -> Self {
        self.v_rest = v_rest;
        self
    }

    /// Set spike threshold (mV). Default: -54.0 mV.
    pub fn v_thresh(mut self, v_thresh: f32) -> Self {
        self.v_thresh = v_thresh;
        self
    }

    /// Set reset potential (mV). Default: -70.0 mV.
    pub fn v_reset(mut self, v_reset: f32) -> Self {
        self.v_reset = v_reset;
        self
    }

    /// Set refractory period (ms). Default: 2.0 ms.
    pub fn tau_ref(mut self, tau_ref: f32) -> Self {
        self.tau_ref = tau_ref;
        self
    }

    /// Set initial membrane potential (mV). Default: -70.0 mV.
    pub fn v_init(mut self, v: f32) -> Self {
        self.v_init = v;
        self
    }

    /// Build the population.
    pub fn build(self) -> NeuralPopulation<LIFModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<LIFModel> {
        let mut pop = NeuralPopulation::<LIFModel>::new(self.n);

        // Parameters
        pop.fill_param(0, self.dt);
        pop.fill_param(1, 1.0 / self.tau_m); // inv_tau_m
        pop.fill_param(2, self.v_rest);
        pop.fill_param(3, self.v_thresh);
        pop.fill_param(4, self.v_reset);
        pop.fill_param(5, self.tau_ref);

        // Initial state
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, 0.0); // refr

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<LIFModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("tau_m", self.tau_m),
            ("v_rest", self.v_rest),
            ("v_thresh", self.v_thresh),
            ("v_reset", self.v_reset),
            ("tau_ref", self.tau_ref),
            ("v_init", self.v_init),
        ])?;
        positive("dt", self.dt)?;
        positive("tau_m", self.tau_m)?;
        positive("tau_ref", self.tau_ref)?;
        Ok(self.build_unchecked())
    }
}

// ═══════════════════════════════════════════════════════════════
//  AdEx Builder
// ═══════════════════════════════════════════════════════════════

impl ModelBuilder for LIFBuilder {
    type Model = LIFModel;
    fn build(self) -> NeuralPopulation<LIFModel> {
        LIFBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<LIFModel>, BuildError> {
        LIFBuilder::try_build(self)
    }
}
