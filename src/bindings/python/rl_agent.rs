//! PyO3 bindings for Actor-Critic RL agent — dopamine-STDP.

#![allow(non_snake_case)]

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::training::rl;

/// Actor-Critic RL Agent с дофамин-модулированным STDP (3-factor plasticity).
///
/// Использование:
///     agent = ActorCriticAgent(n_input=4, n_hidden=256, n_actions=2)
///     action, value = agent.act(state)
///     td_error = agent.learn(reward, next_value, done)
///
/// Геттеры/сеттеры: gamma, lr_actor, lr_critic, tau_trace, tau_e, a_ltp, a_ltd
#[pyclass(name = "ActorCriticAgent")]
pub struct PyActorCriticAgent {
    pub inner: rl::ActorCriticAgent,
}

#[pymethods]
impl PyActorCriticAgent {
    #[new]
    #[pyo3(signature = (n_input, n_hidden, n_actions, n_steps=30, dt=1.0, seed=42))]
    fn new(n_input: usize, n_hidden: usize, n_actions: usize, n_steps: usize, dt: f32, seed: u32) -> Self {
        Self { inner: rl::ActorCriticAgent::new(n_input, n_hidden, n_actions, n_steps, dt, seed) }
    }

    fn act(&mut self, state: Bound<'_, PyArray1<f32>>) -> PyResult<(usize, f32)> {
        let state = state.readonly();
        Ok(self.inner.act(state.as_slice()?))
    }

    fn learn(&mut self, reward: f32, next_value: f32, done: bool) -> f32 {
        self.inner.learn(reward, next_value, done)
    }

    fn reset_episode(&mut self) { self.inner.reset_episode(); }

    // ─── Properties (вместо copy_*) ──────────────────

    /// Input→hidden веса как np.ndarray.
    #[getter]
    fn weights_ih(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.w_ih.clone()).into()
    }

    /// Hidden→output веса как np.ndarray.
    #[getter]
    fn weights_ho(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.w_ho.clone()).into()
    }

    /// Input→critic веса как np.ndarray.
    #[getter]
    fn weights_ic(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.w_ic.clone()).into()
    }

    /// Input→hidden eligibility как np.ndarray.
    #[getter]
    fn elig_ih(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.elig_ih.clone()).into()
    }

    /// Hidden→output eligibility как np.ndarray.
    #[getter]
    fn elig_ho(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.elig_ho.clone()).into()
    }

    /// Input→critic eligibility как np.ndarray.
    #[getter]
    fn elig_ic(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.elig_ic.clone()).into()
    }

    // ─── Getters ─────────────────────────────────────

    #[getter] fn gamma(&self) -> f32 { self.inner.gamma }
    #[getter] fn lr_actor(&self) -> f32 { self.inner.lr_actor }
    #[getter] fn lr_critic(&self) -> f32 { self.inner.lr_critic }
    #[getter] fn tau_trace(&self) -> f32 { self.inner.tau_trace }
    #[getter] fn tau_e(&self) -> f32 { self.inner.tau_e }
    #[getter] fn a_ltp(&self) -> f32 { self.inner.a_ltp }
    #[getter] fn a_ltd(&self) -> f32 { self.inner.a_ltd }
    #[getter] fn w_min(&self) -> f32 { self.inner.w_min }
    #[getter] fn w_max(&self) -> f32 { self.inner.w_max }
    #[getter] fn tau_ref(&self) -> f32 { self.inner.tau_ref }
    #[getter] fn n_input(&self) -> usize { self.inner.n_input }
    #[getter] fn n_hidden(&self) -> usize { self.inner.n_hidden }
    #[getter] fn n_actions(&self) -> usize { self.inner.n_actions_real }

    // ─── Setters ─────────────────────────────────────

    #[setter] fn set_gamma(&mut self, v: f32) { self.inner.gamma = v; }
    #[setter] fn set_lr_actor(&mut self, v: f32) { self.inner.lr_actor = v; }
    #[setter] fn set_lr_critic(&mut self, v: f32) { self.inner.lr_critic = v; }
    #[setter] fn set_tau_trace(&mut self, v: f32) { self.inner.tau_trace = v; }
    #[setter] fn set_tau_e(&mut self, v: f32) { self.inner.tau_e = v; }
    #[setter] fn set_a_ltp(&mut self, v: f32) { self.inner.a_ltp = v; }
    #[setter] fn set_a_ltd(&mut self, v: f32) { self.inner.a_ltd = v; }
    #[setter] fn set_w_min(&mut self, v: f32) { self.inner.w_min = v; }
    #[setter] fn set_w_max(&mut self, v: f32) { self.inner.w_max = v; }
    #[setter] fn set_tau_ref(&mut self, v: f32) { self.inner.tau_ref = v; }

    fn __repr__(&self) -> String {
        format!("ActorCriticAgent(n_input={}, n_hidden={}, n_actions={})",
            self.inner.n_input, self.inner.n_hidden, self.inner.n_actions_real)
    }
}
