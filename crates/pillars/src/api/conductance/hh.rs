use super::*;
use std::any::Any;

use crate::api::population::Population;
use crate::core::buf::AlignedVec;
use crate::core::population::I_SYN_CLAMP_RANGE;
use crate::kernels::hh_channels::{hh_poly_step_slices, ChannelMeta};

/// A data-driven conductance-based population with an arbitrary channel set.
pub struct ConductancePopulation {
    channel_meta: Vec<ChannelMeta>,
    n_channels: usize,
    n_gates: usize,
    n: usize,
    /// Membrane potential (mV), column 0.
    v: AlignedVec,
    /// Gate variables, gate-major: `[g0×n] [g1×n] …`.
    gates: AlignedVec,
    /// Leak conductance (mS/cm²).
    g_leak: AlignedVec,
    /// Leak reversal potential (mV).
    e_leak: AlignedVec,
    /// Membrane capacitance (µF) — the engine divides net current by `C_m`.
    c_m: AlignedVec,
    /// Per-neuron max conductance, channel-major: `[ch0×n] [ch1×n] …`.
    g_max: AlignedVec,
    /// Edge-detection flag per neuron (`1.0` = was below threshold last step).
    was_below: AlignedVec,
    /// Scratch buffer for the clamped input current (reused every step).
    i_scratch: AlignedVec,
    dt: f32,
    celsius: f32,
}

impl ConductancePopulation {
    /// Create a population of `n` neurons with the given channel set.
    ///
    /// `dt` is the integration step (ms); `celsius` sets the Q10 temperature
    /// correction (6.3 °C → Q10 = 1, the squid-axon reference).
    #[must_use]
    pub fn new(channel_meta: Vec<ChannelMeta>, n: usize, dt: f32, celsius: f32) -> Self {
        assert!(
            dt.is_finite() && dt > 0.0,
            "ConductancePopulation: dt must be finite and > 0, got {dt}"
        );
        assert!(
            celsius.is_finite(),
            "ConductancePopulation: celsius must be finite, got {celsius}"
        );
        let n_channels = channel_meta.len();
        let n_gates: usize = channel_meta.iter().map(|c| c.n_gates).sum();
        Self {
            channel_meta,
            n_channels,
            n_gates,
            n,
            v: AlignedVec::new(n, 0.0),
            gates: AlignedVec::new(n_gates * n, 0.0),
            g_leak: AlignedVec::new(n, 0.0),
            e_leak: AlignedVec::new(n, 0.0),
            c_m: AlignedVec::new(n, 1.0),
            g_max: AlignedVec::new(n_channels * n, 0.0),
            was_below: AlignedVec::new(n, 1.0),
            i_scratch: AlignedVec::new(n, 0.0),
            dt,
            celsius,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of ion channels (excluding the leak).
    pub fn n_channels(&self) -> usize {
        self.n_channels
    }

    /// Total number of gating variables across all channels.
    pub fn n_gates(&self) -> usize {
        self.n_gates
    }

    /// Set the membrane capacitance (µF) for all neurons.
    pub fn set_cm(&mut self, c_m: f32) {
        assert!(
            c_m.is_finite() && c_m > 0.0,
            "set_cm: capacitance must be finite and > 0, got {c_m}"
        );
        self.c_m.fill(c_m);
    }

    /// Set the leak conductance (mS/cm²) and reversal potential (mV).
    pub fn set_leak(&mut self, g_leak: f32, e_leak: f32) {
        assert!(
            g_leak.is_finite() && g_leak >= 0.0,
            "set_leak: conductance must be finite and >= 0, got {g_leak}"
        );
        assert!(
            e_leak.is_finite(),
            "set_leak: reversal must be finite, got {e_leak}"
        );
        self.g_leak.fill(g_leak);
        self.e_leak.fill(e_leak);
    }

    /// Set the max conductance (mS/cm²) of one channel for all neurons.
    ///
    /// # Panics
    /// If `channel_idx >= n_channels()`.
    pub fn set_g_max(&mut self, channel_idx: usize, g_max: f32) {
        assert!(
            channel_idx < self.n_channels,
            "set_g_max: channel {channel_idx} out of range (have {})",
            self.n_channels
        );
        assert!(
            g_max.is_finite() && g_max >= 0.0,
            "set_g_max: conductance must be finite and >= 0, got {g_max}"
        );
        let start = channel_idx * self.n;
        for v in &mut self.g_max.as_mut_slice()[start..start + self.n] {
            *v = g_max;
        }
    }

    /// Initialise the membrane to `v0` and every gate to its steady state
    /// `x_∞ = α(v0)/(α(v0)+β(v0))` at that potential.
    pub fn set_resting_state(&mut self, v0: f32) {
        self.v.fill(v0);
        let steady = steady_state_gates(v0, &self.channel_meta);
        for (gate_idx, &g) in steady.iter().enumerate() {
            self.fill_gate(gate_idx, g);
        }
    }

    fn fill_gate(&mut self, gate_idx: usize, value: f32) {
        let start = gate_idx * self.n;
        for v in &mut self.gates.as_mut_slice()[start..start + self.n] {
            *v = value;
        }
    }

    /// Advance one time step: clamp input, integrate via [`hh_poly_step_slices`],
    /// then detect spikes as rising edges through 0 mV.
    pub fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        assert_eq!(i_in.len(), self.n, "i_in length mismatch");
        assert_eq!(spikes_out.len(), self.n, "spikes_out length mismatch");
        if self.n == 0 {
            return;
        }

        // Clamp input to ±100 — matches `NeuralPopulation`'s i_syn clamp, so
        // exponential models can't overflow on extreme injected current.
        for (dst, src) in self.i_scratch.as_mut_slice().iter_mut().zip(i_in) {
            *dst = src.clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
        }

        hh_poly_step_slices(
            self.v.as_mut_slice(),
            self.gates.as_mut_slice(),
            self.i_scratch.as_slice(),
            self.g_leak.as_slice(),
            self.e_leak.as_slice(),
            self.c_m.as_slice(),
            &self.channel_meta,
            self.g_max.as_slice(),
            self.dt,
            self.celsius,
        );

        // Rising-edge spike detection (matches `HHModel` semantics).
        let v = self.v.as_slice();
        let was_below = self.was_below.as_mut_slice();
        for (i, spike_out) in spikes_out.iter_mut().enumerate() {
            let is_above = v[i] >= SPIKE_THRESHOLD_MV;
            let was = was_below[i] > 0.5;
            *spike_out = i32::from(is_above && was);
            was_below[i] = if is_above { 0.0 } else { 1.0 };
        }
    }
}

impl Population for ConductancePopulation {
    #[inline(always)]
    fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        self.step(i_in, spikes_out);
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    fn state_slice(&self, col: usize) -> &[f32] {
        if col == 0 {
            &self.v.as_slice()[..self.n]
        } else if col <= self.n_gates {
            let gate = col - 1;
            &self.gates.as_slice()[gate * self.n..(gate + 1) * self.n]
        } else {
            &[]
        }
    }

    fn param_slice(&self, col: usize) -> &[f32] {
        match col {
            0 => &self.g_leak.as_slice()[..self.n],
            1 => &self.e_leak.as_slice()[..self.n],
            2 => &self.c_m.as_slice()[..self.n],
            c if c >= 3 && c < 3 + self.n_channels => {
                let ch = c - 3;
                &self.g_max.as_slice()[ch * self.n..(ch + 1) * self.n]
            }
            _ => &[],
        }
    }

    fn param_slice_mut(&mut self, col: usize) -> &mut [f32] {
        match col {
            0 => &mut self.g_leak.as_mut_slice()[..self.n],
            1 => &mut self.e_leak.as_mut_slice()[..self.n],
            2 => &mut self.c_m.as_mut_slice()[..self.n],
            c if c >= 3 && c < 3 + self.n_channels => {
                let ch = c - 3;
                &mut self.g_max.as_mut_slice()[ch * self.n..(ch + 1) * self.n]
            }
            _ => &mut [],
        }
    }

    fn fill_param(&mut self, col: usize, value: f32) {
        match col {
            0 => self.g_leak.fill(value),
            1 => self.e_leak.fill(value),
            2 => self.c_m.fill(value),
            c if c >= 3 && c < 3 + self.n_channels => self.set_g_max(c - 3, value),
            _ => panic!(
                "fill_param: column {col} out of range (0..{})",
                3 + self.n_channels
            ),
        }
    }

    fn fill_state(&mut self, col: usize, value: f32) {
        if col == 0 {
            self.v.fill(value);
        } else if col <= self.n_gates {
            self.fill_gate(col - 1, value);
        } else {
            panic!(
                "fill_state: column {col} out of range (0..={})",
                self.n_gates
            );
        }
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
