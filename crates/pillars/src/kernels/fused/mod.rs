//! # 🧮 Fused layer kernels — spike propagation + neuron dynamics, one SIMD pass
//!
//! Canonical port of the legacy `kernels::neuron::fused` batch kernels
//!/3 консолидации — single source of truth в pillars):
//!
//! | Function | Dynamics | Used by |
//! |----------|----------|---------|
//! | [`fused_hidden_izhikevich_step`] | Izhikevich (single-step Euler) + weight accumulation | RL hidden layer, `neuron_tests` |
//! | [`fused_output_lif_step`] | LIF (Exponential Euler) + weight accumulation | RL output readout |
//! | [`poisson_fused_hidden_step`] | Poisson RNG → weights → Izhikevich | RL hidden layer (fused encoder) |
//!
//! ## Philosophy
//!
//! - **Safe API, SIMD internals** — flat `&[f32]` slices in, `f32x8` lanes
//!   (AVX2/NEON) with a scalar tail out.
//! - **Bit-identical to legacy** — arithmetic uses explicit mul+add (two
//!   roundings), НЕ `mul_add` (под `target-cpu=native` + FMA fused-FMA
//!   расходится на 1 ULP от скалярного эталона); полином
//!   [`fast_exp_poly_simd`](crate::math::fast_exp::fast_exp_poly_simd)
//!   для LIF decay, тот же LCG Poisson RNG (`RNG_SCALE_24`). Guarded by
//!   `crates/kernels/tests/fused_parity.rs`.
//! - **Zero allocation hot path** — the caller-supplied `active_buf` is the
//!   only scratch; no per-step `Vec` creation.
//!
//! # ⚠️ RNG semantics — НЕ упрощать до `input::PoissonGenerator`
//!
//! `poisson_fused_hidden_step` продвигает LCG **всегда** (даже при rate=0),
//! повторяя легаси-кернел и `train/rl.rs` траектории. `input::PoissonGenerator::`
//! step` продвигает RNG **только при p > 0** — траектории разойдутся, если
//! заменить fused-путь на `PoissonGenerator` + `fused_hidden_izhikevich_step`.
//!
//! # Weight layout
//!
//! Weights are **column-major**: `weights[i * n_post + j]` — row `i`
//! (pre-neuron), column `j` (post-neuron). Post-size must be padded to ×8.
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | `izhikevich` | `fused_hidden_izhikevich_step` |
//! | `output_lif` | `fused_output_lif_step` |
//! | `poisson` | `poisson_fused_hidden_step` |
//! | `tests` | unit tests (test-only) |

mod izhikevich;
mod output_lif;
mod poisson;
#[cfg(test)]
mod tests;

pub use izhikevich::*;
pub use output_lif::*;
pub use poisson::*;
