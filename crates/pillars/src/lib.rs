//! # Pillars — Zero-Cost SIMD Primitives for Neural Dynamics
//!
//! Pillars provides the foundational building blocks for high-performance
//! neural simulation. Every primitive works in **two modes simultaneously**:
//!
//! | Mode | Width | Target | Hardware |
//! |------|:-----:|--------|----------|
//! | **SIMD** | 8 neurons | `f32x8` | AVX2 / NEON / WASM SIMD128 |
//! | **Scalar** | 1 neuron | `f32` | Any CPU (scalar tail for non-aligned counts) |
//!
//! ## Quick Start
//!
//! ```rust
//! use pillars::prelude::*;
//!
//! // 1. Pick a model and configure it
//! let n = 8;
//! let dt = 0.5;
//! let mut v = vec![-70.0f32; n];
//! let mut refr = vec![0.0f32; n];
//! let mut dummy = vec![0.0f32; n];
//! let i_syn = vec![30.0f32; n];
//! let mut spikes = vec![0i32; n];
//!
//! let mut params = vec![0.0f32; n * LIFModel::N_PARAMS];
//! for i in 0..n {
//!     params[i]           = dt;        // dt (ms)
//!     params[n + i]       = 1.0 / 10.0; // inv_tau_m
//!     params[2*n + i]     = -70.0;     // v_rest
//!     params[3*n + i]     = -54.0;     // v_thresh
//!     params[4*n + i]     = -70.0;     // v_reset
//!     params[5*n + i]     = 2.0;       // tau_ref
//! }
//!
//! // 2. Step 8 neurons in one SIMD call
//! unsafe {
//!     run_neuron::<LIFModel>(
//!         v.as_mut_ptr(), refr.as_mut_ptr(), dummy.as_mut_ptr(),
//!         dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
//!         i_syn.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
//!     );
//! }
//!
//! // 3. Results: SIMD path processes 8 neurons at once
//! assert!(v[0] > -70.0, "Membrane depolarised: {:.1} mV", v[0]);
//! ```
//!
//! ## Available Models
//!
//! | Model | Type | States | Description |
//! |-------|------|:------:|-------------|
//! | `LIFModel` | Point neuron | 2 | Leaky Integrate-and-Fire |
//! | `HHModel` | Conductance | 5 | Hodgkin-Huxley (NaV+KV) |
//! | `IzhikevichModel` | Point neuron | 3 | Izhikevich (RS/FS/IB) |
//! | `AdExModel` | Point neuron | 3 | Adaptive Exponential |
//! | `FHNModel` | Point neuron | 3 | FitzHugh-Nagumo |
//! | `HAAPModel` | Conductance | 3 | VNC M-current + HCN |
//! | `HRModel` | Point neuron | 3 | Hindmarsh-Rose bursting |
//! | `WCModel` | Rate model | 2 | Wilson-Cowan (E/I) |
//! | `MorrisLecarModel` | Conductance | 2 | Ca²⁺ + K⁺ |
//! | `GLIF3Model` | GLIF | 3 | LIF + after-spike current |
//! | `GLIF4Model` | GLIF | 4 | GLIF3 + threshold adapt |
//! | `SAWModel` | Point neuron | 4 | ψ-wave (Izhikevich-like) |
//! | `VSNModel` | Point neuron | 2 | Variable Spiking (trainable Θ+β) |
//! | `AstrocyteModel` | Glia | 4 | Hidden-state EAAT+K⁺+volume TX |
//! | `CaPlasticityModel` | Synapse | 3 | Calcium-based triplet STDP |
//! | `AtlasovNeuromotorModel` | Muscle | 5 | Ca²⁺-tension dynamics |
//! | `Exp2SynModel` | Synapse | 2 | Dual-exponential synapse |
//!
//! ## SimdLane — Full Math Toolbox
//!
//! The [`SimdLane`](model/lane/trait.SimdLane.html) trait provides 35+ operations,
//! each a single instruction on AVX2 and a scalar operation on f32:
//!
//! - **Trig**: sin, cos, tan, asin, acos, atan, atan2
//! - **Exp/Log**: exp, ln, log2, log10, powf, sqrt, cbrt, hypot
//! - **Arith**: abs, signum, mul_add (FMA), recip
//! - **Round**: round, floor, ceil, trunc, fract
//! - **Bio**: sigmoid, gate_integrate, vtrap, ohmic

use wide::f32x8;

pub mod arch;
pub mod buf;
pub mod math;
pub mod model;
pub mod prelude;
#[cfg(feature = "vm")]
pub mod vm;

/// Legacy TNMT bytecode VM — gated behind `vm` feature.
/// Contains ~60 opcodes for ion channels, plasticity, learning.
/// Most files are untested — use `cargo test -p pillars --features vm` to include them.
#[cfg(not(feature = "vm"))]
#[doc(hidden)]
pub mod vm {}

// All public API is re-exported through `prelude`.
// Use `use pillars::prelude::*;` in your code.

// ═══════════════════════════════════════════════════════════
//  Temperature (Q10) — NEURON convention
// ═══════════════════════════════════════════════════════════

/// Q10 temperature factor: `3^((T - 6.3) / 10)`.
/// At 6.3°C → 1.0, at 37°C → 29.
#[inline(always)] pub fn q10_from_celsius_s(celsius: f32) -> f32 {
    (3.0_f32).powf((celsius - 6.3) / 10.0)
}

/// SIMD broadcast of Q10.
#[inline(always)] pub fn q10_from_celsius(celsius: f32) -> f32x8 {
    f32x8::splat(q10_from_celsius_s(celsius))
}

/// τ_eff = τ / Q10 — kinetics accelerate at higher temperature.
#[inline(always)] pub fn scale_tau_q10_s(tau: f32, celsius: f32) -> f32 {
    tau / q10_from_celsius_s(celsius)
}

/// SIMD broadcast of scaled tau.
#[inline(always)] pub fn scale_tau_q10(tau: f32, celsius: f32) -> f32x8 {
    f32x8::splat(scale_tau_q10_s(tau, celsius))
}

// ═══════════════════════════════════════════════════════════
//  Drosophila E_rev Constants
// ═══════════════════════════════════════════════════════════

/// Chloride channels (GluCl, GABA_A): E_Cl = -70 mV.
pub const E_CL: f32 = -70.0;

/// Sodium channels (AMPA, nAChR): E_Na = 0 mV.
pub const E_NA: f32 = 0.0;

/// Potassium channels (delayed rectifier): E_K = -80 mV.
pub const E_K: f32 = -80.0;

/// Calcium channels (NMDA): E_Ca = 0 mV.
pub const E_CA: f32 = 0.0;

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use wide::f32x8;
    use crate::prelude::*;

    #[test]
    fn test_sigmoid_simd() {
        let v = f32x8::from([-65.0, -50.0, -35.0, -20.0, 0.0, 20.0, 40.0, 60.0]);
        let s = sigmoid(v, -35.0, 5.0);
        let a = s.to_array();
        assert!((a[2] - 0.5).abs() < 0.01, "sigmoid at Vh={}", a[2]);
        assert!(a[0] < 0.01, "sigmoid at -65 ≈ 0: {}", a[0]);
        assert!((a[7] - 1.0).abs() < 0.01, "sigmoid at 60 ≈ 1: {}", a[7]);
    }

    #[test]
    fn test_q10() {
        assert!((q10_from_celsius_s(6.3) - 1.0).abs() < 0.001);
        assert!((q10_from_celsius_s(37.0) - 29.0).abs() < 1.0);
    }

    #[test]
    fn test_erev_invariants() {
        assert!(E_CL < 0.0, "Cl⁻ must be hyperpolarising");
        assert!((E_CL - (-70.0)).abs() < 1.0);
    }
}
