//! # Builder API — Type-Safe Model Construction
//!
//! Eliminates magic parameter indices and provides named, validated configuration.
//!
//! ## Example
//! ```rust,ignore
//! use pillars::prelude::*;
//!
//! // Old way (error-prone):
//! let mut pop = NeuralPopulation::<IzhikevichModel>::new(1000);
//! pop.fill_param(0, 0.5);   // what is 0?
//! pop.fill_param(1, 0.02);  // what is 1?
//! // ...
//!
//! // New way (type-safe):
//! let pop = IzhikevichBuilder::new(1000)
//!     .dt(0.5)
//!     .pattern(IzhPattern::RegularSpiking)
//!     .build();
//! ```

use crate::core::population::NeuralPopulation;
use crate::q10_from_celsius_s;
use crate::zoo::neurons::{AdExModel, HAAPModel, HHModel, IzhikevichModel, LIFModel, VSNModel};
use crate::zoo::Model;

/// A validated population-construction failure.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// A value is not finite or violates a named positive/range constraint.
    InvalidValue {
        field: &'static str,
        value: f32,
        requirement: &'static str,
    },
    /// A required universal-builder slot was not configured.
    MissingValue { field: &'static str },
    /// A universal-builder index is outside the model layout.
    IndexOutOfRange {
        kind: &'static str,
        index: usize,
        limit: usize,
    },
    /// A universal-builder slot was assigned more than once.
    DuplicateIndex { kind: &'static str, index: usize },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue {
                field,
                value,
                requirement,
            } => write!(f, "builder: {field}={value} must satisfy {requirement}"),
            Self::MissingValue { field } => {
                write!(f, "builder: required value `{field}` is missing")
            }
            Self::IndexOutOfRange { kind, index, limit } => write!(
                f,
                "builder: {kind} index {index} out of range (model has {limit})"
            ),
            Self::DuplicateIndex { kind, index } => {
                write!(f, "builder: duplicate {kind} index {index}")
            }
        }
    }
}

impl std::error::Error for BuildError {}

fn finite(field: &'static str, value: f32) -> Result<(), BuildError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(BuildError::InvalidValue {
            field,
            value,
            requirement: "finite",
        })
    }
}

fn positive(field: &'static str, value: f32) -> Result<(), BuildError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(BuildError::InvalidValue {
            field,
            value,
            requirement: "finite and > 0",
        })
    }
}

fn finite_fields(fields: &[(&'static str, f32)]) -> Result<(), BuildError> {
    for &(field, value) in fields {
        finite(field, value)?;
    }
    Ok(())
}

fn validate_specs(
    specs: &[(usize, f32)],
    limit: usize,
    kind: &'static str,
) -> Result<(), BuildError> {
    let mut seen = std::collections::HashSet::new();
    for &(index, value) in specs {
        if index >= limit {
            return Err(BuildError::IndexOutOfRange { kind, index, limit });
        }
        if !seen.insert(index) {
            return Err(BuildError::DuplicateIndex { kind, index });
        }
        finite("universal value", value)?;
    }
    Ok(())
}

/// Unified builder interface.
///
/// `try_build()` is the safe path for configuration that comes from users or
/// files. `build()` remains the concise compatibility path and panics with the
/// same [`BuildError`] message when validation fails. The default `try_build`
/// keeps external custom implementations source-compatible; built-in builders
/// override it with complete validation.
pub trait ModelBuilder {
    /// The neuron model created by this builder.
    type Model: crate::zoo::Model;

    /// Build the configured neural population.
    fn build(self) -> NeuralPopulation<Self::Model>;

    /// Validate configuration before allocating the population.
    fn try_build(self) -> Result<NeuralPopulation<Self::Model>, BuildError>
    where
        Self: Sized,
    {
        // External custom builders keep source compatibility. Built-in builders
        // override this default with full validation; custom implementations can
        // opt into stronger checks by overriding it.
        Ok(self.build())
    }
}

// ═══════════════════════════════════════════════════════════════
//  Izhikevich Builder
// ═══════════════════════════════════════════════════════════════

/// Pre-configured Izhikevich neuron patterns from the 2003 paper.
#[derive(Copy, Clone, Debug)]
pub enum IzhPattern {
    /// Regular Spiking (RS): a=0.02, b=0.2, c=-65, d=8
    RegularSpiking,
    /// Intrinsically Bursting (IB): a=0.02, b=0.2, c=-55, d=4
    IntrinsicallyBursting,
    /// Chattering (CH): a=0.02, b=0.2, c=-50, d=2
    Chattering,
    /// Fast Spiking (FS): a=0.1, b=0.2, c=-65, d=2
    FastSpiking,
    /// Low-Threshold Spiking (LTS): a=0.02, b=0.25, c=-65, d=2
    LowThresholdSpiking,
    /// Thalamo-Cortical (TC): a=0.02, b=0.25, c=-65, d=0.05
    ThalamoCortical,
    /// Resonator (RZ): a=0.1, b=0.26, c=-65, d=2
    Resonator,
}

impl IzhPattern {
    /// Returns (a, b, c, d) parameters for this pattern.
    pub fn params(self) -> (f32, f32, f32, f32) {
        match self {
            IzhPattern::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            IzhPattern::IntrinsicallyBursting => (0.02, 0.2, -55.0, 4.0),
            IzhPattern::Chattering => (0.02, 0.2, -50.0, 2.0),
            IzhPattern::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            IzhPattern::LowThresholdSpiking => (0.02, 0.25, -65.0, 2.0),
            IzhPattern::ThalamoCortical => (0.02, 0.25, -65.0, 0.05),
            IzhPattern::Resonator => (0.1, 0.26, -65.0, 2.0),
        }
    }
}

/// Builder for Izhikevich neuron populations.
///
/// # Example
/// ```rust,ignore
/// let pop = IzhikevichBuilder::new(1000)
///     .dt(0.5)
///     .pattern(IzhPattern::RegularSpiking)
///     .v_thresh(30.0)
///     .build();
/// ```
pub struct IzhikevichBuilder {
    n: usize,
    dt: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tau_ref: f32,
    v_thresh: f32,
    v_init: f32,
    u_init: f32,
}

impl IzhikevichBuilder {
    /// Create a new builder for `n` neurons with default RS parameters.
    pub fn new(n: usize) -> Self {
        let (a, b, c, d) = IzhPattern::RegularSpiking.params();
        Self {
            n,
            dt: 0.5,
            a,
            b,
            c,
            d,
            tau_ref: 2.0,
            v_thresh: 30.0,
            v_init: -65.0,
            u_init: -13.0,
        }
    }

    /// Set time step (ms). Default: 0.5 ms.
    pub fn dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set neuron pattern (RS, FS, IB, etc.). Overrides a, b, c, d.
    pub fn pattern(mut self, pattern: IzhPattern) -> Self {
        let (a, b, c, d) = pattern.params();
        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
        self
    }

    /// Set recovery time scale. Default: from pattern.
    pub fn a(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    /// Set recovery sensitivity. Default: from pattern.
    pub fn b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Set reset voltage (mV). Default: from pattern.
    pub fn c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Set recovery bump on spike. Default: from pattern.
    pub fn d(mut self, d: f32) -> Self {
        self.d = d;
        self
    }

    /// Set refractory period (ms). Default: 2.0 ms.
    pub fn tau_ref(mut self, tau_ref: f32) -> Self {
        self.tau_ref = tau_ref;
        self
    }

    /// Set spike threshold (mV). Default: 30.0 mV.
    pub fn v_thresh(mut self, v_thresh: f32) -> Self {
        self.v_thresh = v_thresh;
        self
    }

    /// Set initial membrane potential (mV). Default: -65.0 mV.
    pub fn v_init(mut self, v: f32) -> Self {
        self.v_init = v;
        self
    }

    /// Set initial recovery variable. Default: -13.0.
    pub fn u_init(mut self, u: f32) -> Self {
        self.u_init = u;
        self
    }

    /// Build the population.
    pub fn build(self) -> NeuralPopulation<IzhikevichModel> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }

    fn build_unchecked(self) -> NeuralPopulation<IzhikevichModel> {
        let mut pop = NeuralPopulation::<IzhikevichModel>::new(self.n);

        // Parameters (column-major SoA)
        pop.fill_param(0, self.dt);
        pop.fill_param(1, self.a);
        pop.fill_param(2, self.b);
        pop.fill_param(3, self.c);
        pop.fill_param(4, self.d);
        pop.fill_param(5, self.tau_ref);
        pop.fill_param(6, self.v_thresh);

        // Initial state
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, self.u_init);
        pop.fill_state(2, 0.0); // refr

        pop
    }

    /// Fallible construction for user-provided configuration.
    pub fn try_build(self) -> Result<NeuralPopulation<IzhikevichModel>, BuildError> {
        finite_fields(&[
            ("dt", self.dt),
            ("a", self.a),
            ("b", self.b),
            ("c", self.c),
            ("d", self.d),
            ("tau_ref", self.tau_ref),
            ("v_thresh", self.v_thresh),
            ("v_init", self.v_init),
            ("u_init", self.u_init),
        ])?;
        positive("dt", self.dt)?;
        positive("tau_ref", self.tau_ref)?;
        Ok(self.build_unchecked())
    }
}

// ═══════════════════════════════════════════════════════════════
//  LIF Builder
// ═══════════════════════════════════════════════════════════════

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

impl ModelBuilder for IzhikevichBuilder {
    type Model = IzhikevichModel;
    fn build(self) -> NeuralPopulation<IzhikevichModel> {
        IzhikevichBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<IzhikevichModel>, BuildError> {
        IzhikevichBuilder::try_build(self)
    }
}

impl ModelBuilder for LIFBuilder {
    type Model = LIFModel;
    fn build(self) -> NeuralPopulation<LIFModel> {
        LIFBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<LIFModel>, BuildError> {
        LIFBuilder::try_build(self)
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

// ═══════════════════════════════════════════════════════════════
//  HAAP Builder
// ═══════════════════════════════════════════════════════════════

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

        // States (3)
        pop.fill_state(0, self.v_init);
        pop.fill_state(1, 0.0); // u
        pop.fill_state(2, 0.0); // h

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

// ═══════════════════════════════════════════════════════════════
//  Universal Population Builder
// ═══════════════════════════════════════════════════════════════

/// Universal fluent builder for any neuron model `M: Model`.
///
/// Allows setting parameters and initial states by index or enum without needing
/// a model-specific builder.
///
/// # Example
/// ```rust,ignore
/// use pillars::prelude::*;
///
/// let pop = UniversalBuilder::<VSNModel>::new(1000)
///     .param(VSNParamIdx::DT, 0.5)
///     .param(VSNParamIdx::Inv_Tau_M, 0.1)
///     .param(VSNParamIdx::V_Rest, -70.0)
///     .param(VSNParamIdx::V_Thresh, -54.0)
///     .param(VSNParamIdx::V_Reset, -70.0)
///     .param(VSNParamIdx::Tau_Ref, 2.0)
///     .param(VSNParamIdx::Beta, 0.9)
///     .param(VSNParamIdx::Surrogate_Gain, 10.0)
///     .state(0, -70.0)
///     .build();
/// ```
pub struct UniversalBuilder<M: Model> {
    n: usize,
    params: Vec<(usize, f32)>,
    states: Vec<(usize, f32)>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Model> UniversalBuilder<M> {
    /// Create a new universal builder for `n` neurons of model `M`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            params: Vec::new(),
            states: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set parameter by index or enum (`idx.into()`).
    pub fn param(mut self, idx: impl Into<usize>, val: f32) -> Self {
        self.params.push((idx.into(), val));
        self
    }

    /// Set initial state by index or enum (`idx.into()`).
    pub fn state(mut self, idx: impl Into<usize>, val: f32) -> Self {
        self.states.push((idx.into(), val));
        self
    }

    /// Build the population after validating indices, duplicates and values.
    pub fn try_build(self) -> Result<NeuralPopulation<M>, BuildError> {
        validate_specs(&self.params, M::N_PARAMS, "param")?;
        validate_specs(&self.states, M::N_STATE, "state")?;
        let dt = self
            .params
            .iter()
            .find(|&&(index, _)| index == 0)
            .map(|&(_, value)| value)
            .ok_or(BuildError::MissingValue {
                field: "dt (param 0)",
            })?;
        positive("dt", dt)?;

        let mut pop = NeuralPopulation::<M>::new(self.n);
        for (idx, val) in self.params {
            pop.fill_param(idx, val);
        }
        for (idx, val) in self.states {
            pop.fill_state(idx, val);
        }
        Ok(pop)
    }

    /// Compatibility convenience wrapper for callers that want panic-on-error.
    pub fn build(self) -> NeuralPopulation<M> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<M: Model> ModelBuilder for UniversalBuilder<M> {
    type Model = M;
    fn build(self) -> NeuralPopulation<M> {
        UniversalBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<M>, BuildError> {
        UniversalBuilder::try_build(self)
    }
}

// ═══════════════════════════════════════════════════════════════
//  VSN Builder
// ═══════════════════════════════════════════════════════════════

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich_builder_rs() {
        let pop = IzhikevichBuilder::new(100)
            .dt(0.5)
            .pattern(IzhPattern::RegularSpiking)
            .build();
        assert_eq!(pop.len(), 100);
    }

    #[test]
    fn test_izhikevich_builder_fs() {
        let pop = IzhikevichBuilder::new(50)
            .pattern(IzhPattern::FastSpiking)
            .v_thresh(25.0)
            .build();
        assert_eq!(pop.len(), 50);
    }

    #[test]
    fn test_lif_builder() {
        let pop = LIFBuilder::new(200)
            .dt(0.1)
            .tau_m(15.0)
            .v_thresh(-50.0)
            .build();
        assert_eq!(pop.len(), 200);
    }

    #[test]
    fn test_adex_builder() {
        let pop = AdExBuilder::new(150).dt(0.05).delta_t(3.0).build();
        assert_eq!(pop.len(), 150);
    }

    #[test]
    fn test_hh_builder_preset() {
        let pop = HHBuilder::new(300)
            .dt(0.1)
            .preset(HHPreset::Cortical37C)
            .build();
        assert_eq!(pop.len(), 300);
    }

    #[test]
    fn test_haap_builder() {
        let pop = HAAPBuilder::new(80).dt(0.1).tau_u(80.0).build();
        assert_eq!(pop.len(), 80);
    }

    #[test]
    fn test_generic_model_builder_trait() {
        fn build_any<B: ModelBuilder>(builder: B) -> usize {
            builder.build().len()
        }

        assert_eq!(build_any(LIFBuilder::new(10)), 10);
        assert_eq!(build_any(HHBuilder::new(20)), 20);
        assert_eq!(build_any(HAAPBuilder::new(30)), 30);
        assert_eq!(build_any(VSNBuilder::new(40)), 40);
    }

    #[test]
    fn test_vsn_builder() {
        let pop = VSNBuilder::new(120)
            .dt(0.5)
            .beta(0.95)
            .surrogate_gain(15.0)
            .build();
        assert_eq!(pop.len(), 120);
    }

    #[test]
    fn test_validated_builders_reject_invalid_values() {
        assert!(matches!(
            LIFBuilder::new(8).dt(0.0).try_build(),
            Err(BuildError::InvalidValue { field: "dt", .. })
        ));
        assert!(matches!(
            AdExBuilder::new(8).tau_m(f32::NAN).try_build(),
            Err(BuildError::InvalidValue { field: "tau_m", .. })
        ));
        assert!(matches!(
            VSNBuilder::new(8).beta(1.1).try_build(),
            Err(BuildError::InvalidValue { field: "beta", .. })
        ));
        assert!(matches!(
            HAAPBuilder::new(8).g_m(-1.0).try_build(),
            Err(BuildError::InvalidValue { field: "g_m", .. })
        ));
    }

    #[test]
    fn test_universal_builder_validation() {
        let missing_dt = UniversalBuilder::<VSNModel>::new(4)
            .state(0usize, -70.0)
            .try_build();
        assert!(matches!(
            missing_dt,
            Err(BuildError::MissingValue {
                field: "dt (param 0)"
            })
        ));

        let duplicate = UniversalBuilder::<VSNModel>::new(4)
            .param(0usize, 0.5)
            .param(0usize, 0.5)
            .try_build();
        assert!(matches!(
            duplicate,
            Err(BuildError::DuplicateIndex {
                kind: "param",
                index: 0
            })
        ));

        let out_of_range = UniversalBuilder::<VSNModel>::new(4)
            .param(0usize, 0.5)
            .param(VSNModel::N_PARAMS, 1.0)
            .try_build();
        assert!(matches!(
            out_of_range,
            Err(BuildError::IndexOutOfRange { kind: "param", .. })
        ));
    }

    #[test]
    fn test_universal_builder() {
        use crate::zoo::VSNParamIdx;

        let pop = UniversalBuilder::<VSNModel>::new(50)
            .param(VSNParamIdx::DT, 0.5)
            .param(VSNParamIdx::Inv_Tau_M, 0.1)
            .param(VSNParamIdx::V_Rest, -70.0)
            .param(VSNParamIdx::V_Thresh, -54.0)
            .param(VSNParamIdx::V_Reset, -70.0)
            .param(VSNParamIdx::Tau_Ref, 2.0)
            .param(VSNParamIdx::Beta, 0.9)
            .param(VSNParamIdx::Surrogate_Gain, 10.0)
            .state(0usize, -70.0)
            .build();
        assert_eq!(pop.len(), 50);
    }
}
