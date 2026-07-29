//! # Pillars VM — TNMT Bio Primitives
//!
//! Виртуальная машина с регистровой архитектурой для нейросимуляции.
//! 16 регистров `[f32x8; 16]`, ~60 опкодов.
//!
//! ## Модули
//!
//! | Модуль | Содержит |
//! |--------|----------|
//! | `bio` | 25+ ионных каналов (KV1, KV3, KV4, CAV1, CAV3, HCN, KIR, KNA, KM, Gap, Leak) |
//! | `math` | add, sub, mul, fast_exp, utils (vtrap) |
//! | `physics` | ohm, capacitor, force, diffusion, field, chaos |
//! | `plasticity` | STDP, STP, homeo, trace |
//! | `neurochemistry` | NMDA, GABA_A, nAChR, GluCl, pharmacology, PK, signaling |
//! | `neurons` | integrate (LIF), dyn2d (Izhikevich) |
//! | `learning` | surrogate gradient, hebbian, optimizer |
//!
//! ## Использование
//!
//! ```rust
//! use pillars::vm::bio::op_bio_leak;
//! let mut regs = [wide::f32x8::splat(0.0); 16];
//! regs[0] = wide::f32x8::splat(-65.0); // V
//! op_bio_leak(&mut regs, 1, 0, 0.1, -65.0); // I_leak at reg1
//! ```

// TNMT VM uses non-snake_case bio names (g_M, E_K, V_half) and takes many arguments
#![allow(non_snake_case, clippy::too_many_arguments)]

pub mod bio;
pub mod math;
pub mod physics;
pub mod plasticity;
pub mod neurochemistry;
pub mod neurons;
pub mod learning;

