//! # 🎓 E-Prop microkernels — eligibility-trace SNN training, one SIMD pass
//!
//! Canonical port of the legacy `kernels::neuron::{lif,izhikevich,adex,saw}`
//! e-prop batch kernels (волна 5 консолидации — single source of truth в
//! pillars):
//!
//! | Function | Dynamics | Used by |
//! |----------|----------|---------|
//! | [`eprop_lif_step`] | LIF (Forward Euler) + eligibility, refractory-freeze | `core::neuron::lif` (Python), `neuron_tests` |
//! | [`eprop_izhikevich_step`] | Izhikevich (single-step Euler) + eligibility | `core::neuron::izhikevich` (Python) |
//! | [`eprop_izhikevich_step_multi`] | dense weights + per-synapse eligibility | `core::neuron::izhikevich` (Python) |
//! | [`eprop_adex_step_multi`] | AdEx (Euler, exp-clamped) + per-synapse eligibility | `core::neuron::adex` (Python) |
//! | [`eprop_saw_step`] | SAW dynamics + eligibility | `pillars::zoo::neurons::SAWModel` (канон) |
//!
//! ## Philosophy
//!
//! - **Safe API, SIMD internals** — flat `&[f32]`/`&[i32]` slices in, `f32x8`
//!   lanes (AVX2/NEON) with a scalar tail out.
//! - **Bit-identical to legacy** — arithmetic uses explicit mul+add (two
//!   roundings), НЕ `mul_add`: под `target-cpu=native` с FMA `wide::mul_add`
//!   становится одноокруглённым fused-FMA и расходится на 1 ULP от скалярного
//!   эталона. Та же формула, что у легаси; полином
//!   [`fast_exp_poly_simd`](crate::math::fast_exp::fast_exp_poly_simd)
//!   для τ_e decay и AdEx exp-term, те же NaN-spike semantics. Guarded by
//!   `crates/kernels/tests/eprop_parity.rs`.
//! - **Zero allocation** — dense `e[n_pre × n]` is written in place.
//!
//! # Semantics kept from legacy (НЕ упрощать)
//!
//! - `eprop_lif_step`/`eprop_saw_step` clamp I to ±100; `eprop_adex_step_multi`
//!   clamps I to ±100 **and** the exp argument to ±10 with a `v > v_th − 20`
//!   mask (exp term zeroed below).
//! - `eprop_adex_step_multi` treats NaN v as a spike (`v.cmp_ne(v)` OR-ed into
//!   the spike mask).
//! - `eprop_lif_step` freezes eligibility/psi during refractory and zeroes psi
//!   there; the surrogate `diff` is floored at `1e-6` (lif only).
//! - `eprop_izhikevich_step_multi` uses the 10× weight convention (matches
//!   `fused_hidden_izhikevich_step`); eligibility decays only when `z_pre = 0`.
//!
//! # Module split
//!
//! | File | Contains |
//! |------|----------|
//! | [`lif`] | `eprop_lif_step` |
//! | [`izhikevich`] | `eprop_izhikevich_step` (single-input) |
//! | [`izhikevich_multi`] | `eprop_izhikevich_step_multi` (dense, per-synapse eligibility) |
//! | [`adex`] | `eprop_adex_step_multi` |
//! | [`saw`] | `eprop_saw_step` |
//! | [`tests`] | unit tests (test-only) |

use wide::f32x8;

use crate::core::util::mask_to_spikes;

#[inline(always)]
fn count_spikes(sm: f32x8) -> usize {
    mask_to_spikes(sm).iter().sum::<i32>() as usize
}

mod adex;
mod izhikevich;
mod izhikevich_multi;
mod lif;
mod saw;
#[cfg(test)]
mod tests;

pub use adex::*;
pub use izhikevich::*;
pub use izhikevich_multi::*;
pub use lif::*;
pub use saw::*;
