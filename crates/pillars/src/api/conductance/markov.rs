use super::*;
use std::any::Any;

use crate::api::population::Population;
use crate::core::buf::AlignedVec;
use crate::core::population::I_SYN_CLAMP_RANGE;
use crate::kernels::hh_channels::{
    markov_open_fraction, markov_steady_state, markov_step, q10_from_celsius, MarkovGateSpec,
    MAX_MARKOV_STATES,
};

/// One ion channel in a [`MarkovPopulation`]: a list of kinetic-scheme gates
/// + reversal potential + per-neuron max conductance.
#[derive(Debug, Clone)]
pub struct MarkovChannel {
    /// Markov gates (`gateKS`), in declaration order.
    pub gates: Vec<MarkovGateSpec>,
    /// Reversal potential (mV).
    pub e_rev: f32,
    /// Max conductance (mS/cm²).
    pub g_max: f32,
}

/// A conductance-based population whose channels are kinetic schemes
/// (`ionChannelKS`/`gateKS`), integrated as master equations via the exact
/// matrix exponential. Runs **scalar** (no SIMD): the per-neuron dense
/// matrix exponential is O(N³) per gate, so this is the slow-but-general path
/// behind NeuroML kinetic-scheme channels; HH-only cells stay on
/// [`ConductancePopulation`]'s SIMD engine.
///
/// State layout is column-stride `n` matching the HH engine's convention:
/// gate-major `[gate0_state0×n] [gate0_state1×n] …`, so every occupancy lives
/// in its own `AlignedVec` slice.
pub struct MarkovPopulation {
    /// All gates flattened in (channel, gate) order; per-gate state offsets.
    gates: Vec<MarkovGateSpec>,
    /// Per-channel first global gate index (sentinelled: len = n_channels + 1).
    channel_gate_off: Vec<usize>,
    /// Total number of states across all gates.
    n_states_total: usize,
    n: usize,
    /// Membrane potential (mV), column 0.
    v: AlignedVec,
    /// Occupancies: `gates[g]` state `s` of neuron `i` at
    /// `(gate_off[g] + s) * n + i`.
    occ: AlignedVec,
    /// Per-gate first-state column index (in units of `n`).
    gate_off: Vec<usize>,
    /// Leak conductance (mS/cm²) and reversal (mV).
    g_leak: AlignedVec,
    e_leak: AlignedVec,
    /// Membrane capacitance (µF).
    c_m: AlignedVec,
    /// Per-neuron max conductance, channel-major: `[ch0×n] [ch1×n] …`.
    g_max: AlignedVec,
    /// Per-channel reversal potential (mV).
    e_rev: Vec<f32>,
    n_channels: usize,
    /// Edge-detection flag per neuron (`1.0` = was below threshold last step).
    was_below: AlignedVec,
    /// Scratch buffer for the clamped input current (reused every step).
    i_scratch: AlignedVec,
    dt: f32,
    celsius: f32,
}

impl MarkovPopulation {
    /// Create a population of `n` neurons with the given channel set.
    /// `dt` is the integration step (ms); `celsius` sets the Q10 temperature
    /// correction for transition rates.
    ///
    /// # Panics
    /// If any gate exceeds [`MAX_MARKOV_STATES`] states.
    #[must_use]
    pub fn new(channels: Vec<MarkovChannel>, n: usize, dt: f32, celsius: f32) -> Self {
        assert!(
            dt.is_finite() && dt > 0.0,
            "MarkovPopulation: dt must be finite and > 0, got {dt}"
        );
        assert!(
            celsius.is_finite(),
            "MarkovPopulation: celsius must be finite, got {celsius}"
        );
        let mut gates = Vec::new();
        let mut e_rev = Vec::with_capacity(channels.len());
        let mut gate_off = Vec::new();
        let mut channel_gate_off = Vec::with_capacity(channels.len() + 1);
        let mut n_states_total = 0usize;
        channel_gate_off.push(0);
        for ch in &channels {
            for gate in &ch.gates {
                if let Err(err) = gate.validate() {
                    panic!("MarkovPopulation: invalid gate: {err}");
                }
                gate_off.push(n_states_total);
                n_states_total += gate.n_states();
                gates.push(gate.clone());
            }
            assert!(
                ch.e_rev.is_finite(),
                "MarkovPopulation: channel e_rev must be finite, got {}",
                ch.e_rev
            );
            channel_gate_off.push(gates.len());
            e_rev.push(ch.e_rev);
        }
        let n_channels = channels.len();
        Self {
            gates,
            channel_gate_off,
            n_states_total,
            n,
            v: AlignedVec::new(n, 0.0),
            occ: AlignedVec::new(n_states_total * n, 0.0),
            gate_off,
            g_leak: AlignedVec::new(n, 0.0),
            e_leak: AlignedVec::new(n, 0.0),
            c_m: AlignedVec::new(n, 1.0),
            g_max: AlignedVec::new(n_channels * n, 0.0),
            e_rev,
            n_channels,
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

    /// Total number of states across all kinetic-scheme gates.
    pub fn n_states_total(&self) -> usize {
        self.n_states_total
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

    /// Initialise the membrane to `v0` and every gate to its steady-state
    /// occupancy distribution at that potential.
    pub fn set_resting_state(&mut self, v0: f32) {
        self.v.fill(v0);
        let occ = self.occ.as_mut_slice();
        for (g, gate) in self.gates.iter().enumerate() {
            let p = markov_steady_state(gate, v0);
            let base = self.gate_off[g];
            for (s, &pi) in p.iter().enumerate() {
                let start = (base + s) * self.n;
                for dst in &mut occ[start..start + self.n] {
                    *dst = pi;
                }
            }
        }
    }

    /// Gather the occupancy vector of gate `g` for neuron `i` into `out`.
    #[inline(always)]
    fn gather_occ(&self, g: usize, i: usize, out: &mut [f32; MAX_MARKOV_STATES]) {
        let occ = self.occ.as_slice();
        let base = self.gate_off[g];
        for (s, dst) in out.iter_mut().enumerate().take(self.gates[g].n_states()) {
            *dst = occ[(base + s) * self.n + i];
        }
    }

    /// Scatter `p` back into gate `g` of neuron `i`.
    #[inline(always)]
    fn scatter_occ(&mut self, g: usize, i: usize, p: &[f32; MAX_MARKOV_STATES]) {
        let occ = self.occ.as_mut_slice();
        let base = self.gate_off[g];
        for (s, &pi) in p.iter().enumerate().take(self.gates[g].n_states()) {
            occ[(base + s) * self.n + i] = pi;
        }
    }

    /// Advance one time step: clamp input, integrate each gate's master
    /// equation (matrix exponential), then detect spikes as rising edges
    /// through 0 mV. Scalar — the matrix exponential is per-neuron.
    ///
    /// The membrane uses **implicit Euler (derivimplicit)**, matching `HHModel`
    /// and NEURON's default: `V' = (C/dt·V + I + ΣgE) / (C/dt + Σg)`, with
    /// conductances/gates evaluated at the old voltage (semi-implicit). This
    /// removes the explicit-Euler dt stability limit for stiff channels
    /// (Ca/HCN); the master equation itself is already exact (matrix exp).
    // Индекс `i` одновременно адресует несколько параллельных буферов
    // (v, occ, g_max, i_scratch, …), поэтому итерация по диапазону здесь
    // оправдана — enumerate по одному срезу только усложнила бы доступ к
    // остальным.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        assert_eq!(i_in.len(), self.n, "i_in length mismatch");
        assert_eq!(spikes_out.len(), self.n, "spikes_out length mismatch");
        if self.n == 0 {
            return;
        }

        // Clamp input to ±100 (same contract as `ConductancePopulation`).
        for (dst, src) in self.i_scratch.as_mut_slice().iter_mut().zip(i_in) {
            *dst = src.clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
        }

        let q10 = q10_from_celsius(self.celsius);
        let dt = self.dt;
        let mut p = [0.0f32; MAX_MARKOV_STATES];

        for i in 0..self.n {
            let vi = self.v.as_slice()[i];
            let mut sum_g = 0.0f32;
            let mut sum_g_e = 0.0f32;

            for c in 0..self.n_channels {
                let g_max = self.g_max.as_slice()[c * self.n + i];
                let mut fopen = 1.0f32;
                for g in self.channel_gate_off[c]..self.channel_gate_off[c + 1] {
                    self.gather_occ(g, i, &mut p);
                    markov_step(
                        &self.gates[g],
                        &mut p[..self.gates[g].n_states()],
                        vi,
                        dt,
                        q10,
                    );
                    self.scatter_occ(g, i, &p);
                    let q = markov_open_fraction(&self.gates[g], &p[..self.gates[g].n_states()]);
                    fopen *= q.powi(self.gates[g].power as i32);
                }
                let g_ch = g_max * fopen;
                sum_g += g_ch;
                sum_g_e += g_ch * self.e_rev[c];
            }

            let g_leak = self.g_leak.as_slice()[i];
            let e_leak = self.e_leak.as_slice()[i];
            sum_g += g_leak;
            sum_g_e += g_leak * e_leak;

            let cm = self.c_m.as_slice()[i];
            let inv_dt = cm / dt;
            let i_ext = self.i_scratch.as_slice()[i];
            let v_new = (vi * inv_dt + i_ext + sum_g_e) / (inv_dt + sum_g);
            self.v.as_mut_slice()[i] = v_new;

            // Rising-edge spike detection (matches `HHModel` semantics).
            let was = self.was_below.as_slice()[i] > 0.5;
            let is_above = v_new >= SPIKE_THRESHOLD_MV;
            spikes_out[i] = i32::from(is_above && was);
            self.was_below.as_mut_slice()[i] = if is_above { 0.0 } else { 1.0 };
        }
    }
}

impl Population for MarkovPopulation {
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
        } else if col <= self.n_states_total {
            let s = col - 1;
            &self.occ.as_slice()[s * self.n..(s + 1) * self.n]
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
        } else if col <= self.n_states_total {
            let s = col - 1;
            let start = s * self.n;
            for dst in &mut self.occ.as_mut_slice()[start..start + self.n] {
                *dst = value;
            }
        } else {
            panic!(
                "fill_state: column {col} out of range (0..={})",
                self.n_states_total
            );
        }
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
