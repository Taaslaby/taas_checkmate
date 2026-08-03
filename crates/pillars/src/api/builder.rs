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
use crate::zoo::neurons::{IzhikevichModel, LIFModel, AdExModel};

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
        let pop = AdExBuilder::new(150)
            .dt(0.05)
            .delta_t(3.0)
            .build();
        assert_eq!(pop.len(), 150);
    }
}
