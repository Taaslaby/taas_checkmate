//! GLIF (Generalized Leaky Integrate-and-Fire) — safe Rust wrappers.
//!
//! Implements GLIF3 (after-spike current) and GLIF4 (threshold adaptation + after-spike current)
//! as defined by the Allen Institute for Brain Science (Teeter et al. 2018, Nature Communications).
//!
//! # GLIF Subtypes
//!
//! | Subtype | Mechanism | Reference |
//! |---------|-----------|-----------|
//! | GLIF1 | LIF + reset + refractory | `lif_step` |
//! | GLIF2 | GLIF1 + threshold adaptation | `lif_at_step` |
//! | **GLIF3** | GLIF1 + **after-spike current (I_ahp)** | `glif3_step` |
//! | **GLIF4** | GLIF2 + after-spike current | `glif4_step` |
//! | GLIF5 | GLIF4 + Na⁺ inactivation | planned |
//!
//! # Equations
//!
//! ## GLIF3 — After-Spike Current
//! ```text
//! dv/dt = -g_leak·(v - E_leak) + I_syn + g_ahp·(v - E_ahp)
//! dg_ahp/dt = -g_ahp / τ_ahp          ← exponential decay
//! On spike: v = v_reset, g_ahp += Δg_ahp
//! ```
//!
//! ## GLIF4 — Threshold Adaptation + After-Spike Current
//! ```text
//! dv/dt = -g_leak·(v - E_leak) + I_syn + g_ahp·(v - E_ahp)
//! dg_ahp/dt = -g_ahp / τ_ahp
//! dθ/dt = -θ / τ_θ                      ← threshold offset decays
//! Effective threshold: v_thresh_0 + θ
//! On spike: v = v_reset, g_ahp += Δg_ahp, θ += Δθ
//! ```
//!
//! # Allen Institute Default Parameters (Layer 5 mouse cortical neurons)
//!
//! | Parameter | GLIF3 Value | GLIF4 Value | Description |
//! |-----------|-------------|-------------|-------------|
//! | g_leak | 0.1 mS/cm² | 0.1 mS/cm² | Leak conductance |
//! | E_leak | -65 mV | -65 mV | Leak reversal |
//! | v_thresh | -50 mV | -50 mV | Base threshold |
//! | v_reset | -65 mV | -65 mV | Reset voltage |
//! | τ_ahp | 20 ms | 20 ms | AHP time constant |
//! | Δg_ahp | 0.008 | 0.008 | AHP increment |
//! | E_ahp | -80 mV | -80 mV | AHP reversal |
//! | τ_θ | — | 20 ms | Theta time constant |
//! | Δθ | — | 3 mV | Theta increment |

use crate::kernels;

/// GLIF3 state — LIF with after-spike current (I_ahp).
#[derive(Debug, Clone)]
pub struct Glif3State {
    pub n: usize,
    /// Membrane potential [n] (mV)
    pub v: Vec<f32>,
    /// After-spike conductance [n]
    pub g_ahp: Vec<f32>,
    /// Total input current [n]
    pub Ii: Vec<f32>,
    /// Spike output [n], 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters [n]
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
}

impl Glif3State {
    /// Create a new GLIF3 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
        }
        Self {
            n,
            v: vec![E_leak; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        kernels::glif3_step(
            &mut self.v,
            &mut self.g_ahp,
            &mut self.Ii,
            &mut self.spikes,
            &self.params,
            &mut self.refractory,
            self.tau_ref,
            None,
        );
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak at column 2
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's parameters.
    pub fn set_params(&mut self, idx: usize, g_leak: f32, E_leak: f32, v_thresh: f32, v_reset: f32,
        tau_ahp: f32, delta_g_ahp: f32, E_ahp: f32) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
        }
    }
}

/// GLIF4 state — GLIF3 + threshold adaptation (θ).
#[derive(Debug, Clone)]
pub struct Glif4State {
    pub n: usize,
    /// Membrane potential [n] (mV)
    pub v: Vec<f32>,
    /// Threshold offset [n] (effective threshold = v_thresh_0 + θ)
    pub theta: Vec<f32>,
    /// After-spike conductance [n]
    pub g_ahp: Vec<f32>,
    /// Total input current [n]
    pub Ii: Vec<f32>,
    /// Spike output [n], 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters [n]
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh_0, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
}

impl Glif4State {
    /// Create a new GLIF4 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 10];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh_0;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
            params[8 * n + j] = tau_theta;
            params[9 * n + j] = delta_theta;
        }
        Self {
            n,
            v: vec![E_leak; n],
            theta: vec![0.0; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        kernels::glif4_step(
            &mut self.v,
            &mut self.theta,
            &mut self.g_ahp,
            &mut self.Ii,
            &mut self.spikes,
            &self.params,
            &mut self.refractory,
            self.tau_ref,
            None,
        );
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak
        self.theta.fill(0.0);
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's GLIF4 parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn set_params(&mut self, idx: usize, g_leak: f32, E_leak: f32, v_thresh_0: f32, v_reset: f32,
        tau_ahp: f32, delta_g_ahp: f32, E_ahp: f32, tau_theta: f32, delta_theta: f32) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh_0;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
            self.params[8 * n + idx] = tau_theta;
            self.params[9 * n + idx] = delta_theta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glif3_state_basic() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF3 state: should fire with I=30");
        assert!(state.g_ahp[0] > 0.0, "GLIF3 state: g_ahp should accumulate");
    }

    #[test]
    fn glif4_state_basic() {
        let mut state = Glif4State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 20.0, 3.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF4 state: should fire with I=30");
        assert!(state.theta[0] > 0.0, "GLIF4 state: theta should accumulate");
        assert!(state.g_ahp[0] > 0.0, "GLIF4 state: g_ahp should accumulate");
    }

    #[test]
    fn glif3_state_reset() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        for _ in 0..100 {
            state.step(Some(&i_ext));
        }
        assert!(state.g_ahp[0] > 0.0);

        state.reset();
        assert_eq!(state.v[0], -65.0);
        assert_eq!(state.g_ahp[0], 0.0);
        assert_eq!(state.refractory[0], 0.0);
    }
}
