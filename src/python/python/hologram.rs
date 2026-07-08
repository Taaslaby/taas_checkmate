//! Hologram Bridge — full pipeline for Python.
//!
//! Wires encoding → T3 → HH → STDP → decoding as one Python class.
//!
//! Usage:
//! ```python
//! bridge = HologramBridge(n_neurons=128, dt=0.1)
//! for step in range(1000):
//!     bridge.step(pos_player=0.5, pos_npc=0.3)
//!     if bridge.spikes[0]:
//!         bridge.hebb()  # STDP update
//!     motor = bridge.decode()
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f32::consts::TAU;

use crate::core::neuron::hh::{HhLutConfig, step_hh_lut, build_config_from_channels};

/// Full hologram pipeline: phase encoding → HH → STDP → decode.
#[pyclass(name = "HologramBridge")]
pub struct PyHologramBridge {
    // HH simulator
    cfg: HhLutConfig,

    // Phase parameters
    omega: f32,      // rad/ms
    i_base: f32,     // baseline current
    i_amp: f32,      // modulation amplitude

    // STDP
    tau_trace: f32,
    decay_trace: f32,
    trace_pre: Vec<f32>,
    trace_post: Vec<f32>,
    weights: Vec<Vec<f32>>,  // [n_pre][n_post] — all-to-all
    lr_ltp: f32,
    lr_ltd: f32,
    w_min: f32,
    w_max: f32,

    // Motor decode
    decode_w: Vec<f32>,      // [n_neurons * n_motor]
    decode_b: Vec<f32>,      // [n_motor]
    motor: Vec<f32>,

    // Spike history (for decode)
    spike_buf: Vec<i32>,
    prev_v: Vec<f32>,

    // Time
    t: f32,
}

#[pymethods]
impl PyHologramBridge {
    /// Create hologram bridge.
    ///
    /// Args:
    ///     n: number of neurons (must be ≥ 2)
    ///     dt: timestep in ms
    ///     omega: T3 frequency in rad/ms (default 800)
    ///     n_motor: motor output dimensions (default 2)
    #[new]
    #[pyo3(signature = (n, dt, omega=800.0, n_motor=2))]
    pub fn new(n: usize, dt: f32, omega: f32, n_motor: usize) -> PyResult<Self> {
        if n < 2 {
            return Err(PyValueError::new_err("n must be ≥ 2"));
        }

        let channels = crate::kernels::neuron::hh::standard_channels();
        let cfg = build_config_from_channels(&channels, n, dt, -100.0, 60.0, 1024);

        let tau_trace = 10.0;
        let decay_trace = (-dt / tau_trace).exp();

        Ok(Self {
            cfg,
            omega,
            i_base: 10.0,
            i_amp: 8.0,

            tau_trace,
            decay_trace,
            trace_pre: vec![0.0; n],
            trace_post: vec![0.0; n],
            weights: vec![vec![0.5; n]; n],
            lr_ltp: 0.004,
            lr_ltd: 0.003,
            w_min: 0.0,
            w_max: 2.0,

            decode_w: vec![0.0; n * n_motor],
            decode_b: vec![0.0; n_motor],
            motor: vec![0.0; n_motor],

            spike_buf: vec![0; n],
            prev_v: vec![-65.0; n],

            t: 0.0,
        })
    }

    /// Step the simulation.
    ///
    /// Args:
    ///     phases: per-neuron phase offsets (len=n). Applied as θ_i + ω·t.
    ///
    /// Each neuron gets: I_i = i_base + i_amp · sin(ω·t + phases[i])
    ///
    /// Returns: spike count
    pub fn step(&mut self, phases: Vec<f32>) -> PyResult<usize> {
        if phases.len() != self.cfg.n {
            return Err(PyValueError::new_err(
                format!("phases length {} != n {}", phases.len(), self.cfg.n)
            ));
        }

        // 1. Phase → current (T3 carrier)
        for i in 0..self.cfg.n {
            self.cfg.Ii[i] = self.i_base + self.i_amp * (self.omega * self.t + phases[i]).sin();
        }

        // 2. Save voltages before step (for spike detection)
        // 3. HH step
        step_hh_lut(&mut self.cfg);

        // 4. Spike detection + EPSC
        let mut n_spikes = 0;
        for i in 0..self.cfg.n {
            self.spike_buf[i] = if self.prev_v[i] < 0.0 && self.cfg.v[i] >= 0.0 { 1 } else { 0 };
            if self.spike_buf[i] == 1 { n_spikes += 1; }
            self.prev_v[i] = self.cfg.v[i];
        }

        // 5. EPSC: all-to-all pre→post
        for pre in 0..self.cfg.n {
            if self.spike_buf[pre] == 1 {
                for post in 0..self.cfg.n {
                    if pre != post {
                        self.cfg.Ii[post] += self.weights[pre][post] * 5.0;
                    }
                }
            }
        }

        // 6. STDP traces
        for i in 0..self.cfg.n {
            self.trace_pre[i] = self.trace_pre[i] * self.decay_trace + self.spike_buf[i] as f32;
            self.trace_post[i] = self.trace_post[i] * self.decay_trace + self.spike_buf[i] as f32;
        }

        // 7. Motor decode (integrate spikes)
        for m in 0..self.motor.len() {
            for i in 0..self.cfg.n {
                self.motor[m] += self.spike_buf[i] as f32 * self.decode_w[i * self.motor.len() + m];
            }
        }

        self.t += self.cfg.dt;
        Ok(n_spikes)
    }

    /// Apply STDP to all synapses.
    /// Call after step() when you detect a spike.
    pub fn hebb(&mut self) {
        for pre in 0..self.cfg.n {
            for post in 0..self.cfg.n {
                if pre == post { continue; }
                if self.spike_buf[pre] == 1 {
                    self.weights[pre][post] += self.lr_ltp * self.trace_post[post];
                }
                if self.spike_buf[post] == 1 {
                    self.weights[pre][post] -= self.lr_ltd * self.trace_pre[pre];
                }
                self.weights[pre][post] = self.weights[pre][post]
                    .clamp(self.w_min, self.w_max);
            }
        }
    }

    /// STDP for a specific pre→post pair.
    pub fn hebb_pair(&mut self, pre: usize, post: usize) {
        if pre == post || pre >= self.cfg.n || post >= self.cfg.n { return; }
        if self.spike_buf[pre] == 1 {
            self.weights[pre][post] += self.lr_ltp * self.trace_post[post];
        }
        if self.spike_buf[post] == 1 {
            self.weights[pre][post] -= self.lr_ltd * self.trace_pre[pre];
        }
        self.weights[pre][post] = self.weights[pre][post]
            .clamp(self.w_min, self.w_max);
    }

    /// Decode spikes to motor output.
    /// Returns: motor array
    pub fn decode(&self) -> Vec<f32> {
        self.motor.clone()
    }

    /// Phase encode a position [0,1] to T3 phase [0, 2π].
    #[staticmethod]
    pub fn phase_encode(pos: f32) -> f32 {
        pos * TAU
    }

    /// Get membrane potentials.
    pub fn get_v(&self) -> Vec<f32> {
        self.cfg.v.clone()
    }

    /// Get input currents.
    pub fn get_Ii(&self) -> Vec<f32> {
        self.cfg.Ii.clone()
    }

    /// Get spikes from last step.
    pub fn get_spikes(&self) -> Vec<i32> {
        self.spike_buf.clone()
    }

    /// Get weight matrix.
    pub fn get_weights(&self) -> Vec<Vec<f32>> {
        self.weights.clone()
    }

    /// Get a specific weight.
    pub fn get_weight(&self, pre: usize, post: usize) -> f32 {
        if pre < self.cfg.n && post < self.cfg.n {
            self.weights[pre][post]
        } else { 0.0 }
    }

    /// Set learning rates.
    pub fn set_lr(&mut self, ltp: f32, ltd: f32) {
        self.lr_ltp = ltp;
        self.lr_ltd = ltd;
    }

    /// Set decode weights: [n_neurons * n_motor] flat array.
    pub fn set_decode_weights(&mut self, w: Vec<f32>) -> PyResult<()> {
        let expected = self.cfg.n * self.motor.len();
        if w.len() != expected {
            return Err(PyValueError::new_err(
                format!("decode_w len {} != n*motor {}*{}", w.len(), self.cfg.n, self.motor.len())
            ));
        }
        self.decode_w = w;
        Ok(())
    }

    /// Reset all state (voltages, traces, motor).
    pub fn reset(&mut self) {
        self.cfg.v.fill(-65.0);
        for i in 0..self.cfg.n {
            let s = i * self.cfg.n_gates;
            let g = &self.cfg.init_gates;
            for j in 0..self.cfg.n_gates {
                self.cfg.gates[s + j] = g[j];
            }
        }
        self.cfg.Ii.fill(0.0);
        self.trace_pre.fill(0.0);
        self.trace_post.fill(0.0);
        self.motor.fill(0.0);
        self.spike_buf.fill(0);
        self.prev_v.fill(-65.0);
        self.t = 0.0;
    }

    /// Get current simulation time in ms.
    #[getter]
    pub fn time(&self) -> f32 { self.t }

    /// Get number of neurons.
    #[getter]
    pub fn n(&self) -> usize { self.cfg.n }

    /// Get timestep.
    #[getter]
    pub fn dt(&self) -> f32 { self.cfg.dt }

    fn __repr__(&self) -> String {
        format!("HologramBridge(n={}, t={:.2}ms)", self.cfg.n, self.t)
    }
}
