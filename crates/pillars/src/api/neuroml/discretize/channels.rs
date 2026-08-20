use super::*;

use crate::kernels::hh_channels::{
    markov_open_fraction, markov_steady_state, markov_step, GateKinetics, MarkovGateSpec,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// An active (voltage-gated) channel attached to one compartment: HH gates
/// (data-driven [`GateKinetics`]) + reversal potential + absolute max
/// conductance (µS). Gate state is integrated with **exponential Euler** at the
/// old voltage, then the resulting conductance enters the Hines solve
/// (semi-implicit).
#[derive(Clone)]
pub(crate) struct ActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// Reversal potential, mV.
    pub e_rev: f32,
    /// Absolute max conductance, µS (already density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    pub(crate) gate_state: Vec<f32>,
}

impl ActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates` (1.0 = no temperature
    /// scaling, the NeuroML default without `q10Settings`).
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "ActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            gate_state,
        }
    }
}

/// Steady-state of a gate at `v`: `α/(α+β)` (neutral fallback 0.5).
///
/// Ca-dependent gates (Poirazi `cagk`/`kca`) are evaluated at the initial
/// Ca²⁺ concentration `1e-4 mM` (the resting `decayingPoolConcentrationModel`
/// value) rather than `ca = 0`, so their opening gate starts at the physical
/// resting occupancy. V-only gates ignore `ca`, so this is a no-op for them
/// (and for `gateHHInstantaneous`, whose value is recomputed algebraically
/// every step).
pub(crate) fn gate_steady_state(kinetics: GateKinetics, v: f32) -> f32 {
    let (a, b) = kinetics.alpha_beta_at(v, 1e-4);
    let sum = a + b;
    if sum > 1e-12 {
        a / sum
    } else {
        0.5
    }
}

/// One exponential-Euler step of a gate: `q' = q_inf + (q−q_inf)·exp(−dt·(α+β)·q10)`.
/// `gateHHInstantaneous` is algebraic: `q = steady_state(v, ca)` (no ODE).
pub(crate) fn gate_step(kinetics: GateKinetics, q: f32, v: f32, ca: f32, dt: f32, q10: f32) -> f32 {
    match kinetics {
        GateKinetics::Instantaneous { steady_state } => steady_state.eval_at(v, ca),
        kinetics => {
            // `alpha_beta_at` (not `alpha_beta`): Ca-dependent `Rates`/`TauInf`
            // gates (Poirazi cagk/kca) must read the compartment's actual
            // [Ca²⁺] — `alpha_beta(v)` would pin them to ca = 0.
            let (a, b) = kinetics.alpha_beta_at(v, ca);
            let sum = a + b;
            let q_inf = if sum > 1e-12 { a / sum } else { 0.5 };
            let decay = (-dt * sum.max(1e-12) * q10).exp();
            q_inf + (q - q_inf) * decay
        }
    }
}

/// A Markov (kinetic-scheme, `gateKS`/`gateFractional`) channel attached to one
/// compartment: gates integrated as a master equation (matrix exponential) +
/// reversal + absolute max conductance (µS). Parallel to [`ActiveChannel`]
/// (HH gates) — the cable solver carries both lists.
#[derive(Clone)]
pub(crate) struct MarkovActiveChannel {
    pub gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    pub e_rev: f32,
    pub g_max_us: f32,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
}

impl MarkovActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "MarkovActiveChannel: q10s length");
        let mut occ = Vec::new();
        let mut gate_off = Vec::with_capacity(gates.len());
        for gate in &gates {
            gate_off.push(occ.len());
            occ.extend(markov_steady_state(gate, v_rest));
        }
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            occ,
            gate_off,
        }
    }

    /// Advance every gate one step at `v` and return the open fraction
    /// `Π q_gate^power`. Each gate uses its own Q10 (per-gate `q10Settings`).
    pub(crate) fn step_open(&mut self, v: f32, dt: f32) -> f32 {
        let mut fopen = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            fopen *= q.powi(gate.power as i32);
        }
        fopen
    }
}

/// Exact [`sample_binomial`] draw — the **second waiting time** (geometric)
/// method: count successes via the number of failures before each success
/// (`geom(p)` waiting times), which is exact for all `n`/`p` and runs in
/// O(successes) for small `p` (the open probability at rest is tiny, so this
/// is the fast path) / O(failures) for large `p` via the swap.
///
/// `rng.gen::<f64>()` is uniform in `[0, 1)`; we map to `(0, 1]` so `ln` is
/// never `-inf` (a `0.0` draw would produce an infinite waiting time).
pub(crate) fn sample_binomial(rng: &mut StdRng, n: usize, p: f32) -> usize {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    // Sample failures when p > 1/2 (fewer geometric draws), then complement.
    if p > 0.5 {
        return n - sample_binomial(rng, n, 1.0 - p);
    }
    let ln_q = (1.0 - p as f64).ln();
    let mut successes = 0usize;
    let mut trials = 0i64;
    loop {
        // Failures before the next success: `floor(ln(u) / ln(1-p))`.
        let u = 1.0 - rng.gen::<f64>(); // (0, 1]
        let k = (u.ln() / ln_q).floor() as i64;
        trials += k + 1;
        if trials > n as i64 {
            break;
        }
        successes += 1;
        if successes == n {
            break;
        }
    }
    successes
}

/// A **stochastic** kinetic-scheme channel: a population of `number` discrete
/// ion channels (`channelPopulation`), each described by the same Markov gates.
/// The master equation still yields the per-channel occupancy (mean-field), but
/// the open fraction used for the conductance is a **binomial sample** over the
/// discrete channel count — `n_open ~ Binomial(number, p_open)`, so the
/// delivered fraction is `n_open / number`. This is the canonical source of
/// channel noise (∝ 1/√N) in small neurons/axons.
///
/// The mean of the sample equals the deterministic open fraction (so the
/// `channelDensity` and `channelPopulation` models agree on average); the
/// variance `p_open·(1−p_open)/number` is the channel noise. A seeded
/// [`StdRng`] makes the noise reproducible; `CablePopulation` re-seeds each
/// neuron from a per-neuron salt so the `n` copies of a cell diverge.
#[derive(Clone)]
pub(crate) struct StochasticMarkovActiveChannel {
    gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    q10s: Vec<f32>,
    pub(crate) e_rev: f32,
    /// Total conductance `single_channel_conductance × number` (µS) — the
    /// deterministic upper bound; the stochastic fraction scales it each step.
    pub(crate) g_max_us: f32,
    /// Number of discrete channels.
    n_channels: usize,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
    rng: StdRng,
    /// Per-channel base seed (from the compiler) — [`CablePopulation`] mixes
    /// the neuron index into it so each neuron's stream is independent.
    base_seed: u64,
}

impl StochasticMarkovActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV), and its RNG seeded with `seed`. `q10s` is
    /// parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        e_rev: f32,
        g_max_us: f32,
        n_channels: usize,
        v_rest: f32,
        seed: u64,
    ) -> Self {
        debug_assert_eq!(
            gates.len(),
            q10s.len(),
            "StochasticMarkovActiveChannel: q10s"
        );
        let mut occ = Vec::new();
        let mut gate_off = Vec::with_capacity(gates.len());
        for gate in &gates {
            gate_off.push(occ.len());
            occ.extend(markov_steady_state(gate, v_rest));
        }
        Self {
            gates,
            q10s,
            e_rev,
            g_max_us,
            n_channels,
            occ,
            gate_off,
            rng: StdRng::seed_from_u64(seed),
            base_seed: seed,
        }
    }

    /// Re-seed the RNG — used by [`CablePopulation`] to give every neuron an
    /// independent noise stream from one shared channel template.
    pub fn reseed_for_neuron(&mut self, neuron: usize) {
        self.rng = StdRng::seed_from_u64(mix_seed(self.base_seed, neuron as u64));
    }

    /// Advance every gate one step at `v` and return the **stochastic** open
    /// fraction `n_open / n_channels`, where `n_open ~ Binomial(n_channels,
    /// p_open)` and `p_open = Π q_gate^power` is the deterministic open
    /// probability (exact for a single `gateKS`, a well-defined approximation
    /// for a multi-gate channel).
    pub(crate) fn step_open(&mut self, v: f32, dt: f32) -> f32 {
        let mut p_open = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            p_open *= q.powi(gate.power as i32);
        }
        let n_open = sample_binomial(&mut self.rng, self.n_channels, p_open);
        n_open as f32 / self.n_channels as f32
    }
}
