//! #  Ops — Biological Primitives (The LEGO Bricks)
//!
//! Each op is a **pure function** `fn op<S: SimdLane>(arg1: S, arg2: S) -> S`.
//! Same code compiles to both SIMD and scalar. No state. No memory access.
//! Just math.
//!
//! ## Categories
//!
//! | Module | What | Example |
//! |--------|------|---------|
//! | `membrane` | Leak integration | `leak_integrate(v, inv_tau, v_rest) → v + (v_rest - v)·inv_tau·dt` |
//! | `ohmic` | Ohm's law | `ohmic(g, e, v) → g·(v - e)` |
//! | `spike` | Spike detection | `fire_reset(v, thresh, reset) → (spike, v_reset)` |
//! | `refractory` | Refractory | `refractory_clamp(v, refr, v_reset)` |
//! | `gate` | Gate kinetics | `gate_integrate(g, g_inf, inv_tau, dt)` |
//! | `adaptation` | Adaptation | `m_current(v, v_half, k, g_max, e_k)` |
//! | `calcium` | Ca²⁺ dynamics | `calcium_step(ca, spike, alpha, inv_tau, dt)` |
//! | `synapse` | Synaptic current | `dual_exp_step(g1, g2, spike, ...)` |
//! | `polynomial` | Horner forms | `v * (0.04 * v + 5.0) + 140` (Izhikevich poly) |
//! | `triangulated` | LUT inference | `quantize_v() + lookup()` |
//! | `channels` | Ion channels | `kir`, `hcn`, `gap`, `glucl`, `km`, `leak` |
//! | `physics` | Biophysics | `diffusion`, `field`, `force`, `heating`, `metabolism` |
//! | `neurochemistry` | Chemical signaling | `gpcr`, `receptors`, `signaling` |
//! | `plasticity` | Synaptic plasticity | `stdp`, `stp`, `homeostatic`, `trace` |
//! | `learning` | Learning rules | `hebbian`, `sgd`, `surrogate` |
//!
//! ## Pattern
//!
//! ```text
//! pub fn op_name<S: SimdLane>(input: S, param: S) -> S {
//!     input * param + S::one()   // ← works for f32x8 AND f32
//! }
//! ```
//!
//! Ops NEVER access memory directly. They transform `S` values.
//! Memory access is the job of `core/`.

pub mod adaptation;
pub mod calcium;
pub mod channels;
pub mod gate;
pub mod learning;
pub mod membrane;
pub mod neurochemistry;
pub mod ohmic;
pub mod physics;
pub mod plasticity;
pub mod polynomial;
pub mod refractory;
pub mod spike;
pub mod synapse;
pub mod triangulated;

pub use adaptation::*;
pub use calcium::*;
pub use channels::*;
pub use gate::*;
pub use learning::*;
pub use membrane::*;
pub use neurochemistry::*;
pub use ohmic::*;
pub use physics::*;
pub use plasticity::*;
pub use polynomial::*;
pub use refractory::*;
pub use spike::*;
pub use synapse::*;
