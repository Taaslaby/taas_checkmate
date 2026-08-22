//! # NeuroML Validation Suite
//!
//! Сверка NeuroML-моделей pillars с эталонными данными NEURON/Brian2/OSB.
//!
//! ## Структура
//!
//! ```text
//! validate/
//!   ├── reference.rs   // парсер эталонных трасс (NEURON JSON: {name, t, v})
//!   ├── metrics.rs     // max |ΔV|, RMSE, подсчёт спайков
//!   ├── benchmarks.rs  // HH-аксон, AdEx Figure 4, Poirazi CA1
//!   └── mod.rs         // оркестрация: run_validation_suite()
//! ```
//!
//! ## Запуск
//!
//! ```bash
//! cargo test -p pillars --lib api::validate
//! cargo test -p pillars --lib api::validate -- poirazi   # конкретный бенчмарк
//! ```

pub mod benchmarks;
pub mod metrics;
pub mod reference;

use std::fmt;
use std::path::PathBuf;

use crate::api::neuroml::NeuromlError;

// ─────────────────────────────────────────────────────────────────────────────
//  Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Ошибка загрузки эталона или модели при валидации.
#[derive(Debug)]
pub enum ValidationError {
    /// Не удалось прочитать файл эталонной трассы.
    ReadReference(PathBuf, std::io::Error),
    /// NeuroML-файл не загрузился.
    LoadNeuroML(PathBuf, NeuromlError),
    /// Длина симулированной трассы не совпала с эталонной.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadReference(path, err) => {
                write!(f, "failed to read reference file at {path:?}: {err}")
            }
            Self::LoadNeuroML(path, err) => {
                write!(f, "failed to load NeuroML file at {path:?}: {err}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "mismatched dimensions: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadReference(_, err) => Some(err),
            Self::LoadNeuroML(_, err) => Some(err),
            Self::DimensionMismatch { .. } => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Results
// ─────────────────────────────────────────────────────────────────────────────

/// Результат одного валидационного теста.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Имя модели/теста.
    pub name: &'static str,
    /// Прошла ли валидация.
    pub passed: bool,
    /// Максимальная абсолютная ошибка (мВ) против эталона.
    pub max_error_mv: f32,
    /// RMSE (мВ) против эталона.
    pub rmse_mv: f32,
    /// Использованный допуск (мВ).
    pub tolerance_mv: f32,
    /// Причина провала, если есть.
    pub failure_reason: Option<String>,
}

impl ValidationResult {
    /// Успешный результат с посчитанными метриками.
    pub fn passed(name: &'static str, max_error_mv: f32, rmse_mv: f32) -> Self {
        Self {
            name,
            passed: true,
            max_error_mv,
            rmse_mv,
            tolerance_mv: 0.0,
            failure_reason: None,
        }
    }

    /// Проваленный результат с причиной.
    pub fn failed(name: &'static str, reason: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            max_error_mv: f32::NAN,
            rmse_mv: f32::NAN,
            tolerance_mv: 0.0,
            failure_reason: Some(reason.into()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Suite
// ─────────────────────────────────────────────────────────────────────────────

/// Прогнать весь набор быстрых валидаций.
///
/// Тяжёлые проверки (полная Poirazi, минуты в debug) в набор не входят —
/// вызываются отдельно из [`benchmarks`].
pub fn run_validation_suite() -> Vec<ValidationResult> {
    vec![
        benchmarks::hh_classic(),
        benchmarks::adex_regular_spiking(),
        benchmarks::poirazi_ca1_soma(),
    ]
}

/// Сверить симулированную трассу с эталоном по мВ-допуску.
///
/// Трассы должны быть одной длины (шаг эталона = шагу симуляции);
/// иначе — [`ValidationError::DimensionMismatch`].
pub fn validate_trace(
    name: &'static str,
    sim_v_mv: &[f32],
    reference: &reference::ReferenceTrace,
    tolerance_mv: f32,
) -> Result<ValidationResult, ValidationError> {
    if sim_v_mv.len() != reference.v_mv.len() {
        return Err(ValidationError::DimensionMismatch {
            expected: reference.v_mv.len(),
            actual: sim_v_mv.len(),
        });
    }
    let m = metrics::compare(sim_v_mv, &reference.v_mv);
    if m.max_abs_error <= tolerance_mv {
        Ok(ValidationResult::passed(
            name,
            m.max_abs_error,
            m.rmse,
        ))
    } else {
        Ok(ValidationResult {
            name,
            passed: false,
            max_error_mv: m.max_abs_error,
            rmse_mv: m.rmse,
            tolerance_mv,
            failure_reason: Some(format!(
                "max error {:.4} mV exceeded tolerance {:.4} mV",
                m.max_abs_error, tolerance_mv
            )),
        })
    }
}
