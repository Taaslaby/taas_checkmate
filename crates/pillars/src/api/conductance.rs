//! # ConductancePopulation — data-driven HH population (arbitrary channels)
//!
//! A [`Population`](crate::api::Population) for conductance-based cells whose
//! ion channels are **not** the fixed squid-axon NaV/KV layout baked into the
//! canonical [`HHModel`](crate::zoo::HHModel). The channel set is described by
//! [`ChannelMeta`] (the data-driven representation consumed by the generic
//! engine [`hh_poly_step`](crate::kernels::hh_channels::hh_poly_step)), so any
//! channel expressible as `gateHHrates` / `gateHHtauInf` (m³h, n⁴, HCN, M, …)
//! runs through the same SIMD + scalar engine.
//!
//! This is the runtime behind NeuroML `hhCell` models whose `channelDensity`
//! references non-standard `ionChannelHH` components (see `api::neuroml`).
//! Standard NaV/KV cells still prefer `HHModel` (NEURON-validated fast path).
//!
//! # Layout
//!
//! Column stride is `n` (the logical population size), matching the stride the
//! [`hh_poly_step`] engine expects. `AlignedVec` pads to a multiple of 8 for
//! the SIMD loads; the engine handles the scalar tail itself.

use std::any::Any;

use crate::api::population::Population;
use crate::core::buf::AlignedVec;
use crate::core::population::I_SYN_CLAMP_RANGE;
use crate::kernels::hh_channels::{
    hh_poly_step_slices, markov_open_fraction, markov_steady_state, markov_step, q10_from_celsius,
    ChannelMeta, MarkovGateSpec, MAX_MARKOV_STATES,
};

/// Spike-detection threshold in mV — matches `HHModel`'s `v_thresh = 0.0`
/// (rising-edge crossing through 0 mV, one spike per action potential).
const SPIKE_THRESHOLD_MV: f32 = 0.0;

/// Numerical floor on `α + β` before forming `x_∞ = α/(α+β)`. Below it both
/// rates are effectively zero and the steady state is indeterminate, so we fall
/// back to [`STEADY_STATE_NEUTRAL`]. Mirrors `hh_channels::steady_state_gates`
/// bit-for-bit so resting-state init matches the engine's own helper.
const STEADY_STATE_SUM_EPSILON: f32 = 1e-12;

/// Neutral fallback `x_∞` when `α + β` underflows [`STEADY_STATE_SUM_EPSILON`]
/// (gate with no voltage dependence at this potential).
const STEADY_STATE_NEUTRAL: f32 = 0.5;

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

    /// Advance one time step: clamp input, integrate via [`hh_poly_step`],
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

/// Steady-state gate values `x_∞ = α(v)/(α(v)+β(v))` in gate order
/// (channel-major), mirroring `hh_channels::steady_state_gates` bit-for-bit but
/// over the `ChannelMeta` representation this population stores. The two
/// containers hold identical per-gate kinetics (same order), so the fallback
/// below keeps resting-state init consistent with the engine's own helper.
fn steady_state_gates(v: f32, channel_meta: &[ChannelMeta]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channel_meta {
        for kinetics in &ch.kinetics {
            let (a, b) = kinetics.alpha_beta(v);
            let sum = a + b;
            gates.push(if sum > STEADY_STATE_SUM_EPSILON {
                a / sum
            } else {
                STEADY_STATE_NEUTRAL
            });
        }
    }
    gates
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::hh_channels::{
        build_channel_meta, pacemaking_channels, standard_channels, ClosedForm, MarkovRate,
        MarkovTransition,
    };

    /// Population size for the behavioural tests (SIMD block width, no tail).
    const N_NEURONS: usize = 8;
    /// Integration step (ms).
    const DT_MS: f32 = 0.05;
    /// Reference temperature (°C) — 6.3 is the squid-axon Q10 = 1 anchor.
    const CELSIUS: f32 = 6.3;
    /// Simulation length: 4_000 steps × 0.05 ms = 200 ms.
    const RUN_STEPS: usize = 4_000;
    /// Membrane capacitance (µF/cm²), leak conductance (mS/cm²) and reversal
    /// (mV) — the canonical squid-axon passive parameters shared with HHModel.
    const CM_UF: f32 = 1.0;
    const G_LEAK_MS: f32 = 0.3;
    const E_LEAK_MV: f32 = -54.387;
    /// Resting membrane potential (mV) for the gate steady-state init.
    const REST_MV: f32 = -65.0;
    /// Minimum spontaneous spike count expected from the pacemaking set.
    const MIN_PACEMAKING_SPIKES: i64 = 3;

    /// The standard squid-axon channel set, run through `ConductancePopulation`,
    /// must stay silent without input (quiescent at rest) — the same invariant
    /// `hh_channels` tests assert for the raw engine.
    #[test]
    fn standard_channels_quiescent_without_input() {
        let mut pop = build_population(standard_channels());

        let i_in = vec![0.0f32; N_NEURONS];
        let mut spikes = vec![0i32; N_NEURONS];
        let mut total = 0i64;
        for _ in 0..RUN_STEPS {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
        }
        assert_eq!(total, 0, "standard HH should not self-spike, got {total}");
        let v = pop.state_slice(0)[0];
        assert!(
            (-75.0..-45.0).contains(&v),
            "v should stay near rest, got {v}"
        );
    }

    /// Pacemaking channels (standard + CaV-T) must fire spontaneously at zero
    /// input — the generic engine reproduces the canonical pacemaking behavior.
    #[test]
    fn pacemaking_channels_fire_spontaneously() {
        let mut pop = build_population(pacemaking_channels());

        let i_in = vec![0.0f32; N_NEURONS];
        let mut spikes = vec![0i32; N_NEURONS];
        let mut total = 0i64;
        for _ in 0..RUN_STEPS {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
        }
        assert!(
            total >= MIN_PACEMAKING_SPIKES,
            "pacemaking HH should fire ≥{MIN_PACEMAKING_SPIKES} spikes in 200 ms, got {total}"
        );
    }

    /// Build a `ConductancePopulation` from a channel set, applying the shared
    /// canonical passive parameters and resting-state init used by both tests.
    fn build_population(
        channels: Vec<crate::kernels::hh_channels::ChannelSpec>,
    ) -> ConductancePopulation {
        let meta = build_channel_meta(&channels);
        let mut pop = ConductancePopulation::new(meta, N_NEURONS, DT_MS, CELSIUS);
        pop.set_cm(CM_UF);
        pop.set_leak(G_LEAK_MS, E_LEAK_MV);
        for (i, ch) in channels.iter().enumerate() {
            pop.set_g_max(i, ch.g_max);
        }
        pop.set_resting_state(REST_MV);
        pop
    }

    /// `MarkovPopulation` с тем же стандартным набором NaV/KV (опущенным до
    /// 2-состоянийного Маркова) должен вести себя как `ConductancePopulation`:
    /// молчать без входа и держать мембрану у покоя. Это parity-якорь между
    /// SIMD HH-движком и скалярным master-equation движком.
    #[test]
    fn markov_population_standard_channels_quiescent() {
        let channels = standard_channels();
        let mc: Vec<MarkovChannel> = channels
            .iter()
            .map(|ch| MarkovChannel {
                gates: ch
                    .gates
                    .iter()
                    .map(|g| {
                        crate::kernels::hh_channels::gate_kinetics_to_markov(
                            g.kinetics.clone(),
                            g.power,
                        )
                    })
                    .collect(),
                e_rev: ch.E_rev,
                g_max: ch.g_max,
            })
            .collect();
        let mut pop = MarkovPopulation::new(mc, N_NEURONS, DT_MS, CELSIUS);
        pop.set_cm(CM_UF);
        pop.set_leak(G_LEAK_MS, E_LEAK_MV);
        for (i, ch) in channels.iter().enumerate() {
            pop.set_g_max(i, ch.g_max);
        }
        pop.set_resting_state(REST_MV);

        let i_in = vec![0.0f32; N_NEURONS];
        let mut spikes = vec![0i32; N_NEURONS];
        let mut total = 0i64;
        for _ in 0..RUN_STEPS {
            pop.step(&i_in, &mut spikes);
            total += spikes.iter().map(|&s| i64::from(s)).sum::<i64>();
        }
        assert_eq!(
            total, 0,
            "Markov-lowered HH should not self-spike, got {total}"
        );
        let v = pop.state_slice(0)[0];
        assert!(
            (-75.0..-45.0).contains(&v),
            "v should stay near rest, got {v}"
        );
    }

    /// `MarkovPopulation::new` должен отвергнуть малформированный public
    /// `MarkovGateSpec` (выход за границы перехода) честной паникой, а не
    /// пропустить его до паники индексации внутри `markov_step`.
    #[test]
    #[should_panic(expected = "invalid gate")]
    fn markov_population_rejects_malformed_gate() {
        let bad = MarkovGateSpec {
            open: vec![0.0, 1.0],
            transitions: vec![MarkovTransition {
                from: 0,
                to: 2, // out of range
                rate: MarkovRate::Directed {
                    rate: ClosedForm::Exp {
                        rate: 1.0,
                        midpoint: 0.0,
                        scale: 10.0,
                    },
                },
            }],
            power: 1,
        };
        let ch = MarkovChannel {
            gates: vec![bad],
            e_rev: 50.0,
            g_max: 1.0,
        };
        let _ = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
    }

    /// Implicit Euler (derivimplicit) мембрана **безусловно устойчива**: при
    /// dt = 10 мс (≈3× мембранной τ = C/gL = 3.33 мс) explicit Euler осциллирует
    /// и расходится, а implicit монотонно сходится к покою без перескока.
    ///
    /// Проверка на чистой RC-мембране (без каналов): каждый шаг обязан
    /// приближать V к E_leak (никогда не пересекая его с другой стороны), что
    /// явно исключает осцилляторный/расходящийся explicit-Euler режим.
    #[test]
    fn markov_membrane_implicit_stable_at_large_dt() {
        // Leak-only cell: τ = C/gL = 1.0 / 0.3 ≈ 3.33 ms, dt = 10 ms > 2τ.
        let mut pop = MarkovPopulation::new(Vec::new(), 1, 10.0, CELSIUS);
        pop.set_cm(CM_UF);
        pop.set_leak(G_LEAK_MS, E_LEAK_MV);
        pop.set_resting_state(-20.0); // далеко от покоя (-54.387)

        let i_in = [0.0f32];
        let mut spikes = [0i32];
        let mut v = -20.0f32;
        for _ in 0..20 {
            pop.step(&i_in, &mut spikes);
            v = pop.state_slice(0)[0];
            // Никогда не пересекаем покой с другой стороны (нет осцилляции).
            assert!(v >= E_LEAK_MV, "overshoot past rest: {v}");
            assert!(v.is_finite() && v > -1000.0, "diverged: {v}");
        }
        // 20 шагов × 10 мс достаточно, чтобы монотонно сесть на покой.
        assert!(
            (v - E_LEAK_MV).abs() < 1.0,
            "should converge to rest, got {v}"
        );
    }

    #[test]
    #[should_panic(expected = "dt must be finite and > 0")]
    fn conductance_population_rejects_zero_dt() {
        let meta = build_channel_meta(&standard_channels());
        let _ = ConductancePopulation::new(meta, N_NEURONS, 0.0, CELSIUS);
    }

    #[test]
    #[should_panic(expected = "celsius must be finite")]
    fn conductance_population_rejects_nan_celsius() {
        let meta = build_channel_meta(&standard_channels());
        let _ = ConductancePopulation::new(meta, N_NEURONS, DT_MS, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "capacitance must be finite and > 0")]
    fn conductance_population_rejects_zero_cm() {
        let mut pop = build_population(standard_channels());
        pop.set_cm(0.0);
    }

    #[test]
    #[should_panic(expected = "conductance must be finite and >= 0")]
    fn conductance_population_rejects_nan_leak() {
        let mut pop = build_population(standard_channels());
        pop.set_leak(f32::NAN, E_LEAK_MV);
    }

    #[test]
    #[should_panic(expected = "reversal must be finite")]
    fn markov_population_rejects_inf_reversal() {
        let mut pop = MarkovPopulation::new(Vec::new(), N_NEURONS, DT_MS, CELSIUS);
        pop.set_leak(G_LEAK_MS, f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "dt must be finite and > 0")]
    fn markov_population_rejects_zero_dt() {
        let _ = MarkovPopulation::new(Vec::new(), N_NEURONS, 0.0, CELSIUS);
    }

    #[test]
    #[should_panic(expected = "capacitance must be finite and > 0")]
    fn markov_population_rejects_zero_cm() {
        let mut pop = MarkovPopulation::new(Vec::new(), N_NEURONS, DT_MS, CELSIUS);
        pop.set_cm(0.0);
    }

    #[test]
    #[should_panic(expected = "conductance must be finite and >= 0")]
    fn markov_population_rejects_negative_g_max() {
        let ch = MarkovChannel {
            gates: vec![],
            e_rev: 0.0,
            g_max: 1.0,
        };
        let mut pop = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
        pop.set_g_max(0, -1.0);
    }

    #[test]
    #[should_panic(expected = "e_rev must be finite")]
    fn markov_population_rejects_nan_channel_reversal() {
        let ch = MarkovChannel {
            gates: vec![],
            e_rev: f32::NAN,
            g_max: 1.0,
        };
        let _ = MarkovPopulation::new(vec![ch], N_NEURONS, DT_MS, CELSIUS);
    }
}
