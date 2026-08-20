//! Builder module.

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::zoo::neurons::AdExModel;

/// Builder for Adaptive Exponential Integrate-and-Fire neuron populations.
///
/// # Example
/// ```rust,ignore
/// let pop = AdExBuilder::new(1000)
///     .dt(0.05)
///     .tau_m(5.0)
///     .delta_t(2.0)
///     .build();
/// ```
pub struct AdExBuilder {
    n: usize,
    dt: f32,
    tau_m: f32,
    tau_w: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    delta_t: f32,
    a: f32,
    b: f32,
    tau_ref: f32,
    v_peak: f32,
    v_init: f32,
    w_init: f32,
}

impl AdExBuilder {
    /// Create a new builder for `n` neurons with default parameters.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            dt: 0.05,
            tau_m: 5.0,
            tau_w: 100.0,
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            delta_t: 2.0,
            a: 0.5,
            b: 7.0,
            tau_ref: 2.0,
            v_peak: 0.0,
            v_init: -65.0,
            w_init: 0.0,
        }
    }

    /// Set time step (ms). Default: 0.05 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set membrane time constant (ms). Default: 5.0 ms.
    pub fn tau_m(mut self, tau_m: f32) -> Self {
        self.tau_m = tau_m;
        self
    }

    /// Set adaptation time constant (ms). Default: 100.0 ms.
    pub fn tau_w(mut self, tau_w: f32) -> Self {
        self.tau_w = tau_w;
        self
    }

    /// Set resting potential (mV). Default: -65.0 mV.
    pub fn v_rest(mut self, v_rest: f32) -> Self {
        self.v_rest = v_rest;
        self
    }

    /// Set spike threshold (mV). Default: -50.0 mV.
    pub fn v_thresh(mut self, v_thresh: f32) -> Self {
        self.v_thresh = v_thresh;
        self
    }

    /// Set reset potential (mV). Default: -65.0 mV.
    pub fn v_reset(mut self, v_reset: f32) -> Self {
        self.v_reset = v_reset;
        self
    }

    /// Set exponential slope factor (mV). Default: 2.0 mV.
    pub fn delta_t(mut self, delta_t: f32) -> Self {
        self.delta_t = delta_t;
        self
    }

    /// Set subthreshold adaptation (nS). Default: 0.5 nS.
    pub fn a(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    /// Set spike-triggered adaptation (nA). Default: 7.0 nA.
    pub fn b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Set refractory period (ms). Default: 2.0 ms.
    pub fn tau_ref(mut self, tau_ref: f32) -> Self {
        self.tau_ref = tau_ref;
        self
    }

    /// Set spike peak threshold (mV). Default: 0.0 mV.
    pub fn v_peak(mut self, v_peak: f32) -> Self {
        self.v_peak = v_peak;
        self
    }

    /// Set initial membrane potential (mV). Default: -65.0 mV.
    pub fn v_init(mut self, v: f32) -> Self {
        self.v_init = v;
        self
    }

    /// Set initial adaptation variable. Default: 0.0.
    pub fn w_init(mut self, w: f32) -> Self {
        self.w_init = w;
        self
    }

    /// Build the population.
    pub fn build(self) -> NeuralPopulation<AdExModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<AdExModel> {
        let mut pop = NeuralPopulation::<AdExModel>::new(self.n);

        // Parameters
        pop.fill_param(0, self.dt);
        pop.fill_param(1, 1.0 / self.tau_m);
        pop.fill_param(2, 1.0 / self.tau_w);
        pop.fill_param(3, self.v_rest);
        pop.fill_param(4, self.v_thresh);
        pop.fill_param(5, self.v_reset);
        pop.fill_param(6, self.delta_t);
        pop.fill_param(7, 1.0 / self.delta_t); // inv_delta_t
        pop.fill_param(8, self.a);
        pop.fill_param(9, self.b);
        pop.fill_param(10, self.tau_ref);
        pop.fill_param(11, self.v_peak);

        // Initial state
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, self.w_init);
        pop.fill_state(2, 0.0); // refr

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<AdExModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("tau_m", self.tau_m),
            ("tau_w", self.tau_w),
            ("v_rest", self.v_rest),
            ("v_thresh", self.v_thresh),
            ("v_reset", self.v_reset),
            ("delta_t", self.delta_t),
            ("a", self.a),
            ("b", self.b),
            ("tau_ref", self.tau_ref),
            ("v_peak", self.v_peak),
            ("v_init", self.v_init),
            ("w_init", self.w_init),
        ])?;
        positive("dt", self.dt)?;
        positive("tau_m", self.tau_m)?;
        positive("tau_w", self.tau_w)?;
        positive("delta_t", self.delta_t)?;
        positive("tau_ref", self.tau_ref)?;
        Ok(self.build_unchecked())
    }
}

impl ModelBuilder for AdExBuilder {
    type Model = AdExModel;
    fn build(self) -> NeuralPopulation<AdExModel> {
        AdExBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<AdExModel>, BuildError> {
        AdExBuilder::try_build(self)
    }
}

// ═══════════════════════════════════════════════════════════════
//  Hodgkin-Huxley (HH) Builder
// ═══════════════════════════════════════════════════════════════
