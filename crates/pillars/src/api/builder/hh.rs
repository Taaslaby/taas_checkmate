//! -------------------------------------------------------------------------
//! Filename:    src/builder/hh.rs
//! Class:       builder
//! Purpose:
//! Description:
//!
//! билдер апи для ходжина
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::q10_from_celsius_s;
use crate::zoo::neurons::HHModel;

// ═══════════════════════════════════════════════════════════════

/// Pre-configured Hodgkin-Huxley neuron presets.
#[derive(Copy, Clone, Debug)]

pub enum HHPreset {
    /// Classic Squid Axon (6.3°C): g_Na=120, g_K=36, g_L=0.3, E_Na=50, E_K=-77, E_L=-54.3
    SquidAxon,
    /// Warm Mammalian Cortical (37.0°C) with accelerated Q10 kinetics
    Cortical37C,
    /// Thalamic Pacemaking with active CaV-T current (g_CaT=0.5, E_CaT=120.0)
    ThalamicPacemaking,
}

/// Builder for Hodgkin-Huxley continuous biophysical neuron populations.
///
/// # Example
/// ```rust,ignore
/// let pop = HHBuilder::new(500)
///     .dt(0.1)
///     .preset(HHPreset::Cortical37C)
///     .build();
/// ```
pub struct HHBuilder {
    n: usize,
    dt: f32,
    celsius: f32,
    g_na: f32,
    e_na: f32,
    g_k: f32,
    e_k: f32,
    g_leak: f32,
    e_leak: f32,
    v_thresh: f32,
    inv_cm: f32,
    g_cat: f32,
    e_cat: f32,
    v_init: f32,
}

impl HHBuilder {
    /// Create a new builder for `n` neurons with default SquidAxon parameters.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            dt: 0.1,
            celsius: 6.3,
            g_na: 120.0,
            e_na: 50.0,
            g_k: 36.0,
            e_k: -77.0,
            g_leak: 0.3,
            e_leak: -54.3,
            v_thresh: 0.0,
            inv_cm: 1.0,
            g_cat: 0.0,
            e_cat: 120.0,
            v_init: -65.0,
        }
    }

    /// Set time step (ms). Default: 0.1 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Apply a pre-configured biological preset.
    pub fn preset(mut self, preset: HHPreset) -> Self {
        match preset {
            HHPreset::SquidAxon => {
                self.celsius = 6.3;
                self.g_cat = 0.0;
            }
            HHPreset::Cortical37C => {
                self.celsius = 37.0;
                self.g_cat = 0.0;
            }
            HHPreset::ThalamicPacemaking => {
                self.celsius = 37.0;
                self.g_cat = 0.5;
                self.e_cat = 120.0;
            }
        }
        self
    }

    /// Set temperature in Celsius. Default: 6.3°C.
    pub fn celsius(mut self, celsius: f32) -> Self {
        self.celsius = celsius;
        self
    }

    /// Set Na⁺ max conductance (mS/cm²). Default: 120.0.
    pub fn g_na(mut self, g_na: f32) -> Self {
        self.g_na = g_na;
        self
    }

    /// Set Na⁺ reversal potential (mV). Default: 50.0.
    pub fn e_na(mut self, e_na: f32) -> Self {
        self.e_na = e_na;
        self
    }

    /// Set K⁺ max conductance (mS/cm²). Default: 36.0.
    pub fn g_k(mut self, g_k: f32) -> Self {
        self.g_k = g_k;
        self
    }

    /// Set K⁺ reversal potential (mV). Default: -77.0.
    pub fn e_k(mut self, e_k: f32) -> Self {
        self.e_k = e_k;
        self
    }

    /// Set Leak conductance (mS/cm²). Default: 0.3.
    pub fn g_leak(mut self, g_leak: f32) -> Self {
        self.g_leak = g_leak;
        self
    }

    /// Set Leak reversal potential (mV). Default: -54.3.
    pub fn e_leak(mut self, e_leak: f32) -> Self {
        self.e_leak = e_leak;
        self
    }

    /// Set CaV-T T-type calcium max conductance (mS/cm²). Default: 0.0.
    pub fn g_cat(mut self, g_cat: f32) -> Self {
        self.g_cat = g_cat;
        self
    }

    /// Set CaV-T reversal potential (mV). Default: 120.0.
    pub fn e_cat(mut self, e_cat: f32) -> Self {
        self.e_cat = e_cat;
        self
    }

    /// Set spike detection threshold (mV). Default: 0.0 mV.
    pub fn v_thresh(mut self, v_thresh: f32) -> Self {
        self.v_thresh = v_thresh;
        self
    }

    /// Set membrane capacitance (µF/cm²). Inverse is stored internally.
    pub fn cm(mut self, cm: f32) -> Self {
        self.inv_cm = 1.0 / cm;
        self
    }

    /// Set initial membrane potential (mV). Default: -65.0 mV.
    pub fn v_init(mut self, v: f32) -> Self {
        self.v_init = v;
        self
    }

    /// Build the HH population.
    pub fn build(self) -> NeuralPopulation<HHModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<HHModel> {
        let mut pop = NeuralPopulation::<HHModel>::new(self.n);
        let q10 = q10_from_celsius_s(self.celsius);

        // Parameters (12)
        pop.fill_param(0, self.dt);
        pop.fill_param(1, self.g_na);
        pop.fill_param(2, self.e_na);
        pop.fill_param(3, self.g_k);
        pop.fill_param(4, self.e_k);
        pop.fill_param(5, self.g_leak);
        pop.fill_param(6, self.e_leak);
        pop.fill_param(7, q10);
        pop.fill_param(8, self.v_thresh);
        pop.fill_param(9, self.inv_cm);
        pop.fill_param(10, self.g_cat);
        pop.fill_param(11, self.e_cat);

        // Initial states (6): v, m, h, n, c, was_below
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, 0.0529); // m_inf at resting -65mV
        pop.fill_state(2, 0.5961); // h_inf at resting -65mV
        pop.fill_state(3, 0.3177); // n_inf at resting -65mV
        pop.fill_state(4, 0.0); // c_init
        pop.fill_state(5, 1.0); // was_below = true

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<HHModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("celsius", self.celsius),
            ("g_na", self.g_na),
            ("e_na", self.e_na),
            ("g_k", self.g_k),
            ("e_k", self.e_k),
            ("g_leak", self.g_leak),
            ("e_leak", self.e_leak),
            ("v_thresh", self.v_thresh),
            ("inv_cm", self.inv_cm),
            ("g_cat", self.g_cat),
            ("e_cat", self.e_cat),
            ("v_init", self.v_init),
        ])?;
        positive("dt", self.dt)?;
        positive("inv_cm", self.inv_cm)?;
        let q10 = q10_from_celsius_s(self.celsius);
        if !q10.is_finite() {
            return Err(BuildError::InvalidValue {
                field: "q10",
                value: q10,
                requirement: "finite",
            });
        }
        for (field, value) in [
            ("g_na", self.g_na),
            ("g_k", self.g_k),
            ("g_leak", self.g_leak),
            ("g_cat", self.g_cat),
        ] {
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

impl ModelBuilder for HHBuilder {
    type Model = HHModel;
    fn build(self) -> NeuralPopulation<HHModel> {
        HHBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<HHModel>, BuildError> {
        HHBuilder::try_build(self)
    }
}
