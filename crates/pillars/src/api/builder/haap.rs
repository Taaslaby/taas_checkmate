//! -------------------------------------------------------------------------
//! Filename:    src/builder/haap.rs
//! Class:       builder
//! Purpose:
//! Description:
//!
//! билдер апи для хаап
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::zoo::neurons::HAAPModel;

/// Builder for Hyperpolarization-Activated Adaptive Pulse-Bursting (HAAP) populations.
pub struct HAAPBuilder {
    n: usize,
    dt: f32,
    inv_cm: f32,
    g_l: f32,
    e_l: f32,
    g_m: f32,
    e_k: f32,
    g_h: f32,
    e_h: f32,
    tau_u: f32,
    tau_h: f32,
    v_half_u: f32,
    k_u: f32,
    v_half_h: f32,
    k_h: f32,
    v_thresh: f32,
    v_reset: f32,
    delta_u: f32,
    v_init: f32,
}

impl HAAPBuilder {
    /// Create a new builder for `n` HAAP neurons with default parameters.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            dt: 0.1,
            inv_cm: 1.0,
            g_l: 0.1,
            e_l: -70.0,
            g_m: 1.5,
            e_k: -90.0,
            g_h: 1.0,
            e_h: -30.0,
            tau_u: 100.0,
            tau_h: 200.0,
            v_half_u: -35.0,
            k_u: 5.0,
            v_half_h: -75.0,
            k_h: 6.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            delta_u: 0.05,
            v_init: -70.0,
        }
    }

    /// Set time step (ms). Default: 0.1 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set membrane capacitance (µF/cm²).
    pub fn cm(mut self, cm: f32) -> Self {
        self.inv_cm = 1.0 / cm;
        self
    }

    /// Set M-current conductance (mS/cm²).
    pub fn g_m(mut self, g_m: f32) -> Self {
        self.g_m = g_m;
        self
    }

    /// Set h-current conductance (mS/cm²).
    pub fn g_h(mut self, g_h: f32) -> Self {
        self.g_h = g_h;
        self
    }

    /// Set M-current adaptation time constant (ms).
    pub fn tau_u(mut self, tau_u: f32) -> Self {
        self.tau_u = tau_u;
        self
    }

    /// Set h-current activation time constant (ms).
    pub fn tau_h(mut self, tau_h: f32) -> Self {
        self.tau_h = tau_h;
        self
    }

    /// Set spike threshold (mV).
    pub fn v_thresh(mut self, v_thresh: f32) -> Self {
        self.v_thresh = v_thresh;
        self
    }

    /// Set reset potential (mV).
    pub fn v_reset(mut self, v_reset: f32) -> Self {
        self.v_reset = v_reset;
        self
    }

    /// Build the HAAP population.
    pub fn build(self) -> NeuralPopulation<HAAPModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<HAAPModel> {
        let mut pop = NeuralPopulation::<HAAPModel>::new(self.n);

        // Parameters (17)
        pop.fill_param(0, self.dt);
        pop.fill_param(1, self.inv_cm);
        pop.fill_param(2, self.g_l);
        pop.fill_param(3, self.e_l);
        pop.fill_param(4, self.g_m);
        pop.fill_param(5, self.e_k);
        pop.fill_param(6, self.g_h);
        pop.fill_param(7, self.e_h);
        pop.fill_param(8, 1.0 / self.tau_u);
        pop.fill_param(9, 1.0 / self.tau_h);
        pop.fill_param(10, self.v_half_u);
        pop.fill_param(11, self.k_u);
        pop.fill_param(12, self.v_half_h);
        pop.fill_param(13, self.k_h);
        pop.fill_param(14, self.v_thresh);
        pop.fill_param(15, self.v_reset);
        pop.fill_param(16, self.delta_u);

        // States (3). The gates start at their Boltzmann steady state at
        // `v_init` (the same equations the step function integrates toward),
        // so an unstimulated population sits at its resting equilibrium
        // instead of spending a long (tau_h ~ 200 ms) initialization transient.
        // Args are clamped to ±50 like the step function's safety net.
        let u_arg = (-(self.v_init - self.v_half_u) / self.k_u).clamp(-50.0, 50.0);
        let u0 = 1.0 / (1.0 + u_arg.exp());
        let h_arg = ((self.v_init - self.v_half_h) / self.k_h).clamp(-50.0, 50.0);
        let h0 = 1.0 / (1.0 + h_arg.exp());

        pop.fill_state(0, self.v_init);
        pop.fill_state(1, u0); // u — M-current (opens on depolarization)
        pop.fill_state(2, h0); // h — HCN (opens on hyperpolarization)

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<HAAPModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("inv_cm", self.inv_cm),
            ("g_l", self.g_l),
            ("e_l", self.e_l),
            ("g_m", self.g_m),
            ("e_k", self.e_k),
            ("g_h", self.g_h),
            ("e_h", self.e_h),
            ("tau_u", self.tau_u),
            ("tau_h", self.tau_h),
            ("v_half_u", self.v_half_u),
            ("k_u", self.k_u),
            ("v_half_h", self.v_half_h),
            ("k_h", self.k_h),
            ("v_thresh", self.v_thresh),
            ("v_reset", self.v_reset),
            ("delta_u", self.delta_u),
            ("v_init", self.v_init),
        ])?;
        positive("dt", self.dt)?;
        positive("inv_cm", self.inv_cm)?;
        positive("tau_u", self.tau_u)?;
        positive("tau_h", self.tau_h)?;
        positive("k_u", self.k_u)?;
        positive("k_h", self.k_h)?;
        for (field, value) in [("g_l", self.g_l), ("g_m", self.g_m), ("g_h", self.g_h)] {
            if value < 0.0 {
                return Err(BuildError::InvalidValue {
                    field,
                    value,
                    requirement: "finite and >= 0",
                });
            }
        }
        Ok(self.build_unchecked())
    }
}

impl ModelBuilder for HAAPBuilder {
    type Model = HAAPModel;
    fn build(self) -> NeuralPopulation<HAAPModel> {
        HAAPBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<HAAPModel>, BuildError> {
        HAAPBuilder::try_build(self)
    }
}
