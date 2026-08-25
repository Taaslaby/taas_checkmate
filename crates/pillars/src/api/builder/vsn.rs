//! -------------------------------------------------------------------------
//! Filename:    src/builder/vsn.rs
//! Class:       builder
//! Purpose:
//! Description:
//!
//! билдер апи для vsn
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::zoo::neurons::VSNModel;

/// Builder for Variable Spiking Neuron (VSN) populations with surrogate gradient state.
pub struct VSNBuilder {
    n: usize,
    dt: f32,
    inv_tau_m: f32,
    v_rest: f32,
    v_thresh: f32,
    v_reset: f32,
    tau_ref: f32,
    beta: f32,
    surrogate_gain: f32,
    v_init: f32,
}

impl VSNBuilder {
    /// Create a new builder for `n` VSN neurons with default parameters.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            dt: 0.5,
            inv_tau_m: 1.0 / 10.0,
            v_rest: -70.0,
            v_thresh: -54.0,
            v_reset: -70.0,
            tau_ref: 2.0,
            beta: 0.9,
            surrogate_gain: 10.0,
            v_init: -70.0,
        }
    }

    /// Set time step (ms). Default: 0.5 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set membrane time constant (ms). Automatically converts to `1/τ_m`.
    pub fn tau_m(mut self, tau_m: f32) -> Self {
        self.inv_tau_m = 1.0 / tau_m;
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

    /// Set leakage mixing factor (0..1). Default: 0.9.
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set surrogate gradient gain. Default: 10.0.
    pub fn surrogate_gain(mut self, surrogate_gain: f32) -> Self {
        self.surrogate_gain = surrogate_gain;
        self
    }

    /// Set initial membrane potential (mV). Default: -70.0 mV.
    pub fn v_init(mut self, v: f32) -> Self {
        self.v_init = v;
        self
    }

    /// Build the VSN population.
    pub fn build(self) -> NeuralPopulation<VSNModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<VSNModel> {
        let mut pop = NeuralPopulation::<VSNModel>::new(self.n);

        // Parameters (8)
        pop.fill_param(0, self.dt);
        pop.fill_param(1, self.inv_tau_m);
        pop.fill_param(2, self.v_rest);
        pop.fill_param(3, self.v_thresh);
        pop.fill_param(4, self.v_reset);
        pop.fill_param(5, self.tau_ref);
        pop.fill_param(6, self.beta);
        pop.fill_param(7, self.surrogate_gain);

        // Initial states (3): v, refr, surrogate
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, 0.0); // refr
        pop.fill_state(2, 0.0); // surrogate

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<VSNModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("inv_tau_m", self.inv_tau_m),
            ("v_rest", self.v_rest),
            ("v_thresh", self.v_thresh),
            ("v_reset", self.v_reset),
            ("tau_ref", self.tau_ref),
            ("beta", self.beta),
            ("surrogate_gain", self.surrogate_gain),
            ("v_init", self.v_init),
        ])?;
        positive("dt", self.dt)?;
        positive("inv_tau_m", self.inv_tau_m)?;
        positive("tau_ref", self.tau_ref)?;
        if !(0.0..=1.0).contains(&self.beta) {
            return Err(BuildError::InvalidValue {
                field: "beta",
                value: self.beta,
                requirement: "finite and in [0, 1]",
            });
        }
        positive("surrogate_gain", self.surrogate_gain)?;
        Ok(self.build_unchecked())
    }
}

impl ModelBuilder for VSNBuilder {
    type Model = VSNModel;
    fn build(self) -> NeuralPopulation<VSNModel> {
        VSNBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<VSNModel>, BuildError> {
        VSNBuilder::try_build(self)
    }
}
