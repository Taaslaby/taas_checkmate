//! # Population — Type Erasure for Heterogeneous Networks
//!
//! === BOUNDARY: Safe Rust ===
//! The [`Population`] trait hides the concrete model type (`LIF`, `HH`, …)
//! behind an object-safe interface so a single [`Network`](crate::api::Network) can drive
//! heterogeneous populations through one `Vec<Box<dyn Population>>`.
//!
//! [`PopulationId`] is the opaque handle the network hands out when a
//! population is added — it is just the index into that boxed vector.
//!
//! ## Why separate from `Network`?
//!
//! Keeps the type-erasure contract (this file) independent of the
//! orchestration concerns (routing, delays, threads — see [`Network`](crate::api::Network)):
//! both sides evolve without touching each other.

use std::any::Any;

use crate::api::NeuralPopulation;
use crate::zoo::Model;

/// Opaque handle returned by [`Network::add_population`](crate::api::Network::add_population).
pub type PopulationId = usize;

/// Abstract interface for a neuron population.
///
/// Enables heterogeneous populations (LIF + HH + Izhikevich) in one
/// [`Network`](crate::api::Network).
pub trait Population: 'static {
    /// Advance one time step.
    ///
    /// `i_in` — input currents, length >= `len()`.
    /// `spikes_out` — output spikes (0 or 1), length >= `len()`.
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]);

    /// Number of neurons in this population.
    fn len(&self) -> usize;

    /// Returns `true` if the population has no neurons.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Read-only access to a state column.
    fn state_slice(&self, col: usize) -> &[f32];

    /// Read-only access to a parameter column.
    fn param_slice(&self, col: usize) -> &[f32];

    /// Mutable access to a parameter column.
    fn param_slice_mut(&mut self, col: usize) -> &mut [f32];

    /// Fill a parameter column with a value (all neurons).
    fn fill_param(&mut self, col: usize, value: f32);

    /// Fill a state column with a value (all neurons).
    fn fill_state(&mut self, col: usize, value: f32);/// Erase to `dyn Any` for safe typed downcasting.
///
/// Required (not defaulted) so the trait stays object-safe and callable
/// through `&mut dyn Population`. Implementors must erase `self`; without
/// it, [`Network::pop_typed_mut`](crate::api::Network::pop_typed_mut) silently returns `None`.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<M: Model + 'static> Population for NeuralPopulation<M> {
    #[inline(always)]
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        self.step(i_in, spikes_out)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn state_slice(&self, col: usize) -> &[f32] {
        self.state_slice(col)
    }

    #[inline(always)]
    fn param_slice(&self, col: usize) -> &[f32] {
        self.param_slice(col)
    }

    #[inline(always)]
    fn param_slice_mut(&mut self, col: usize) -> &mut [f32] {
        self.param_slice_mut(col)
    }

    #[inline(always)]
    fn fill_param(&mut self, col: usize, value: f32) {
        self.fill_param(col, value)
    }

    #[inline(always)]
    fn fill_state(&mut self, col: usize, value: f32) {
        self.fill_state(col, value)
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::neurons::LIFModel;

    #[test]
    fn test_population_trait_object() {
        // Verify that NeuralPopulation<LIFModel> implements Population
        let pop = NeuralPopulation::<LIFModel>::new(8);
        let mut boxed: Box<dyn Population> = Box::new(pop);
        assert_eq!(boxed.len(), 8);
        assert!(!boxed.is_empty());

        let mut spikes = vec![0i32; 8];
        let i_in = vec![0.0; 8];
        boxed.step(&i_in, &mut spikes);
        // Must not crash
        assert_eq!(boxed.state_slice(0).len(), 8);
    }

    #[test]
    fn test_population_empty() {
        let pop = NeuralPopulation::<LIFModel>::new(0);
        let boxed: Box<dyn Population> = Box::new(pop);
        assert_eq!(boxed.len(), 0);
        assert!(boxed.is_empty());
    }
}
