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

#[cfg(test)]
use crate::zoo::neurons::{LIFModel, VSNModel};
#[cfg(test)]
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

mod adex;
mod haap;
mod hh;
mod izhikevich;
mod lif;
mod universal;
mod vsn;

pub use adex::AdExBuilder;
pub use haap::HAAPBuilder;
pub use hh::{HHBuilder, HHPreset};
pub use izhikevich::{IzhPattern, IzhikevichBuilder};
pub use lif::LIFBuilder;
pub use universal::UniversalBuilder;
pub use vsn::VSNBuilder;

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
