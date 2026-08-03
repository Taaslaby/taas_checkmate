//! # taas-rule — программные нейроны: интерпретируемая VM правил
//!
//! Каждый нейрон несёт компактную программу (до [`MAX_INSTRUCTIONS`]
//! инструкций), исполняемую интерпретатором за каждый такт симуляции.
//! Программа оперирует аккумулятором, переменными нейрона и дендрита,
//! химическими концентрациями тела и случайностью — и полностью определяет
//! поведение нейрона: пороги, утечку, потенциал покоя, постоянство, шум,
//! спайковый выход и параметры обучения.
//!
//! ## Архитектура
//!
//! ```text
//! opcode ──► vm ──► neuron / dendrite / tract / lobe
//!   │          │
//!   │          └─► vars (NeuronVars / DendriteVars)
//!   └── Rule (до 16 инструкций, геном-программируемый)
//! ```
//!
//! | Модуль | Роль |
//! |--------|------|
//! | [`opcode`] | каталог инструкций + сборка программ |
//! | [`vm`]     | интерпретатор (аккумулятор + tendRate) |
//! | [`vars`]   | 8 переменных нейрона и 8 переменных дендрита |
//! | [`neuron`] | нейрон: переменные + программа + состояние VM |
//! | [`dendrite`] | дендрит: ST/LT/STRENGTH |
//! | [`tract`]  | пучок дендритов + обучение (reward/punishment) |
//! | [`lobe`]   | решётка нейронов: проведение, WTA, обучение |
//!
//! ## Ключевые идеи
//!
//! **Тендинг** — экспоненциальная релаксация `v → цель` со скоростью
//! `tendRate`: дискретный аналог `dv = (x − v)/τ`. Это базовый механизм
//! адаптации: состояния тянутся к целям, веса — к значениям памяти.
//!
//! **Двухконтурные веса**: каждый дендрит несёт краткосрочный `ST` (быстро
//! учится) и долгосрочный `LT` (медленно консолидирует `ST → LT`). Без
//! подкрепления `ST` тает, `LT` удерживает след — два временны́х масштаба
//! памяти.
//!
//! **Reward-химия**: обучение гейтируется концентрациями химических веществ
//! тела (награда/наказание), адресуемых по id нейрона.
//!
//! ## Пример
//!
//! ```rust
//! use taas_rule::*;
//! use taas_rule::vm::run_program;
//!
//! // Программа: порог входа → утечка → покой → постоянство → спайк.
//! let rule = Rule::from_ops(vec![
//!     OpCode::NominalThreshold(Operand::Value(0.05)),
//!     OpCode::LeakageRate(Operand::Value(0.2)),
//!     OpCode::RestState(Operand::Value(0.0)),
//!     OpCode::Persistence(Operand::Value(0.4)),
//!     OpCode::Fire(Operand::Value(0.6)),
//!     OpCode::Stop,
//! ]);
//!
//! let mut vars = NeuronVars::default();
//! let mut fired = false;
//! for _ in 0..10 {
//!     vars.input = 1.0; // сильный синаптический вход (каждый такт)
//!     let out = run_program(&rule, &mut vars, None, &[], 0, 0, None, &mut Rng::new(42));
//!     fired |= out.fired; // STATE накапливается через Persistence
//! }
//! assert!(fired);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

pub mod dendrite;
pub mod lobe;
pub mod neuron;
pub mod opcode;
pub mod rng;
pub mod tract;
pub mod vars;
pub mod vm;

pub use dendrite::Dendrite;
pub use lobe::Lobe;
pub use neuron::Neuron;
pub use opcode::{Operand, OpCode, Rule, Target, MAX_INSTRUCTIONS};
pub use rng::Rng;
pub use tract::{Learning, Tract, TractConfig};
pub use vars::{bound_m11, DendriteVar, DendriteVars, NeuronVar, NeuronVars};
pub use vm::{RuleVm, VmContext, VmOutcome};
