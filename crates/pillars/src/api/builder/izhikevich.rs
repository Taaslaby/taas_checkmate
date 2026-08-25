//! -------------------------------------------------------------------------
//! Filename:    src/builder/izhekevich.rs
//! Class:       builder
//! Purpose:
//! Description:
//!
//! билдер апи для ижекевича
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use super::{finite_fields, positive, BuildError, ModelBuilder, NeuralPopulation};
use crate::zoo::neurons::IzhikevichModel;

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

impl ModelBuilder for IzhikevichBuilder {
    type Model = IzhikevichModel;
    fn build(self) -> NeuralPopulation<IzhikevichModel> {
        IzhikevichBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<IzhikevichModel>, BuildError> {
        IzhikevichBuilder::try_build(self)
    }
}
