//! Builder module.

use super::{BuildError, ModelBuilder, positive, validate_specs};
use crate::core::population::NeuralPopulation;
use crate::zoo::Model;

/// Universal fluent builder for any neuron model `M: Model`.
///
/// Allows setting parameters and initial states by index or enum without needing
/// a model-specific builder.
///
/// # Example
/// ```rust,ignore
/// use pillars::prelude::*;
///
/// let pop = UniversalBuilder::<VSNModel>::new(1000)
///     .param(VSNParamIdx::DT, 0.5)
///     .param(VSNParamIdx::Inv_Tau_M, 0.1)
///     .param(VSNParamIdx::V_Rest, -70.0)
///     .param(VSNParamIdx::V_Thresh, -54.0)
///     .param(VSNParamIdx::V_Reset, -70.0)
///     .param(VSNParamIdx::Tau_Ref, 2.0)
///     .param(VSNParamIdx::Beta, 0.9)
///     .param(VSNParamIdx::Surrogate_Gain, 10.0)
///     .state(0, -70.0)
///     .build();
/// ```
pub struct UniversalBuilder<M: Model> {
    n: usize,
    params: Vec<(usize, f32)>,
    states: Vec<(usize, f32)>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Model> UniversalBuilder<M> {
    /// Create a new universal builder for `n` neurons of model `M`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            params: Vec::new(),
            states: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set parameter by index or enum (`idx.into()`).
    pub fn param(mut self, idx: impl Into<usize>, val: f32) -> Self {
        self.params.push((idx.into(), val));
        self
    }

    /// Set initial state by index or enum (`idx.into()`).
    pub fn state(mut self, idx: impl Into<usize>, val: f32) -> Self {
        self.states.push((idx.into(), val));
        self
    }

    /// Build the population after validating indices, duplicates and values.
    pub fn try_build(self) -> Result<NeuralPopulation<M>, BuildError> {
        validate_specs(&self.params, M::N_PARAMS, "param")?;
        validate_specs(&self.states, M::N_STATE, "state")?;
        let dt = self
            .params
            .iter()
            .find(|&&(index, _)| index == 0)
            .map(|&(_, value)| value)
            .ok_or(BuildError::MissingValue {
                field: "dt (param 0)",
            })?;
        positive("dt", dt)?;

        let mut pop = NeuralPopulation::<M>::new(self.n);
        for (idx, val) in self.params {
            pop.fill_param(idx, val);
        }
        for (idx, val) in self.states {
            pop.fill_state(idx, val);
        }
        Ok(pop)
    }

    /// Compatibility convenience wrapper for callers that want panic-on-error.
    pub fn build(self) -> NeuralPopulation<M> {
        self.try_build().unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<M: Model> ModelBuilder for UniversalBuilder<M> {
    type Model = M;
    fn build(self) -> NeuralPopulation<M> {
        UniversalBuilder::build(self)
    }
    fn try_build(self) -> Result<NeuralPopulation<M>, BuildError> {
        UniversalBuilder::try_build(self)
    }
}

// ═══════════════════════════════════════════════════════════════
//  VSN Builder
// ═══════════════════════════════════════════════════════════════

