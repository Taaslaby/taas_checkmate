use super::*;

use crate::kernels::hh_channels::{
    markov_open_fraction, markov_steady_state, markov_step, GateKinetics, MarkovGateSpec,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A **stochastic** Nernst (Ca²⁺) channel population
/// (`channelPopulationNernst`): `number` discrete channels whose combined
/// reversal is the **Nernst potential** of the compartment's [Ca²⁺] (dynamic)
/// and whose Ca²⁺ current feeds the compartment's pool. This is the
/// discrete-channel analogue of [`NernstActiveChannel`] (just as
/// [`StochasticMarkovActiveChannel`] is of [`ActiveChannel`]): the same
/// binomial open-fraction sampling, but `erev` is recomputed from [Ca²⁺] each
/// step.
#[derive(Clone)]
pub(crate) struct StochasticNernstActiveChannel {
    gates: Vec<MarkovGateSpec>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    ca_ext_mm: f32,
    /// Total conductance `single_channel_conductance × number` (µS).
    pub(crate) g_max_us: f32,
    /// Number of discrete channels.
    n_channels: usize,
    /// Flattened occupancy (gate-major): gate `g` state `s` at `gate_off[g]+s`.
    occ: Vec<f32>,
    /// Per-gate first-state offset into `occ`.
    gate_off: Vec<usize>,
    rng: StdRng,
    base_seed: u64,
}

impl StochasticNernstActiveChannel {
    /// Build a channel with every gate initialised to its steady-state
    /// occupancy at `v_rest` (mV), and its RNG seeded with `seed`. `q10s` is
    /// parallel to `gates`.
    pub fn new(
        gates: Vec<MarkovGateSpec>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        n_channels: usize,
        v_rest: f32,
        seed: u64,
    ) -> Self {
        debug_assert_eq!(
            gates.len(),
            q10s.len(),
            "StochasticNernstActiveChannel: q10s"
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
            ca_ext_mm,
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

    /// Advance every gate one step and return `(fopen, e_rev_mv)`: the
    /// **stochastic** open fraction `n_open / n_channels` and the Nernst
    /// reversal of `ca_int_mm` against `ca_ext_mm`.
    pub(crate) fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32, temp_k: f32) -> (f32, f32) {
        let mut p_open = 1.0f32;
        for (g, gate) in self.gates.iter().enumerate() {
            let off = self.gate_off[g];
            let n = gate.n_states();
            markov_step(gate, &mut self.occ[off..off + n], v, dt, self.q10s[g]);
            let q = markov_open_fraction(gate, &self.occ[off..off + n]);
            p_open *= q.powi(gate.power as i32);
        }
        let n_open = sample_binomial(&mut self.rng, self.n_channels, p_open);
        (
            n_open as f32 / self.n_channels as f32,
            nernst_reversal_mv(ca_int_mm, self.ca_ext_mm, temp_k),
        )
    }
}

/// Ca²⁺ reversal potential from the Nernst equation (LEMS
/// `channelPopulationNernst`/`channelDensityNernst`):
/// `E = (R·T/(z·F))·ln(ca_ext/ca_int)` in mV. `ca_ext <= 0` → 0 (LEMS guard).
pub(crate) fn nernst_reversal_mv(ca_int_mm: f32, ca_ext_mm: f32, temp_k: f32) -> f32 {
    if ca_ext_mm <= 0.0 {
        return 0.0;
    }
    let ca_int = ca_int_mm.max(1e-12);
    let rt_over_zf_mv = R_GAS * temp_k / (Z_CA * FARADAY) * 1000.0;
    rt_over_zf_mv * (ca_ext_mm / ca_int).ln()
}

/// Per-compartment Ca²⁺ pool: the ODE `dC/dt = iCa·flux_scale − (C−rest)/tau`,
/// integrated with exact exponential decay over the step (linear ODE, constant
/// `iCa`). Concentrations are **mM** (= mol/m³), `iCa` is **nA**, `flux_scale`
/// is precomputed from the compartment geometry (mM/ms per nA).
#[derive(Clone, Copy)]
pub(crate) struct CaPool {
    pub resting_mm: f32,
    pub tau_ms: f32,
    pub flux_scale: f32,
}

impl CaPool {
    /// One exact step of the pool ODE at constant `i_ca_nA` over `dt` (ms).
    pub(crate) fn step(&self, ca: f32, i_ca_na: f32, dt: f32) -> f32 {
        let tau = self.tau_ms.max(1e-6);
        let decay = (-dt / tau).exp();
        let steady = self.resting_mm + i_ca_na * self.flux_scale * tau;
        // LEMS clamps concentration at 0.
        (steady + (ca - steady) * decay).max(0.0)
    }
}

/// Precompute the decaying-pool flux scale `1/(2·F·V_shell)` for a compartment
/// of membrane area `area_um2` (µm²) and shell thickness `shell_thickness_um`
/// (µm). LEMS computes the shell as a spherical shell whose outer radius is
/// `sqrt(surfaceArea/4π)`. Result is **mM/ms per nA**. Test-only (the compiler
/// lowers NeuroML pools via [`thin_shell_pool_flux_scale`]).
#[cfg(test)]
pub(crate) fn decaying_pool_flux_scale(area_um2: f32, shell_thickness_um: f32) -> f32 {
    let surface_m2 = area_um2 * 1e-12;
    let r_eff = (surface_m2 / (4.0 * PI)).sqrt();
    let r_in = (r_eff - shell_thickness_um * 1e-6).max(0.0);
    let shell_vol_m3 = (4.0 * PI / 3.0) * (r_eff.powi(3) - r_in.powi(3));
    // dC/dt [mM/ms] = iCa[nA]·1e-12 / (2·F·V_shell[m³])
    1e-12 / (2.0 * FARADAY * shell_vol_m3.max(f32::MIN_POSITIVE))
}

/// Precompute the **thin-shell** pool flux scale `1/(2·F·(SA·thickness))` —
/// the exact form of custom LEMS concentration models like
/// `PoiraziCaConcentrationModel` (`iCa/(2·F·surfaceArea·shellThickness)`, the
/// published Poirazi capool), whose shell volume is `membrane area × shell
/// thickness`, **not** the exact spherical shell of the standard
/// `decayingPoolConcentrationModel`. Result is mM/ms per nA.
pub(crate) fn thin_shell_pool_flux_scale(area_um2: f32, shell_thickness_um: f32) -> f32 {
    // V[m³] = SA[m²]·thickness[m]; dC/dt [mM/ms] = iCa[nA]·1e-12/(2·F·V[m³]).
    let vol_m3 = (area_um2 * 1e-12) * (shell_thickness_um * 1e-6);
    1e-12 / (2.0 * FARADAY * vol_m3.max(f32::MIN_POSITIVE))
}

/// An HH-gated Ca²⁺ channel whose reversal potential is the **Nernst
/// potential** of the compartment's [Ca²⁺] (recomputed each step) rather than a
/// fixed `erev`. Parallel to [`ActiveChannel`] (fixed `erev`) — the cable solver
/// carries both lists, and the Nernst channel also feeds its current back into
/// the compartment's Ca²⁺ pool (`iCa`).
#[derive(Clone)]
pub(crate) struct NernstActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    pub ca_ext_mm: f32,
    /// Absolute max conductance, µS (density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    gate_state: Vec<f32>,
}

impl NernstActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "NernstActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            ca_ext_mm,
            g_max_us,
            gate_state,
        }
    }

    /// Advance every gate one step at `v` and return `(fopen, e_rev_mv)`, where
    /// the reversal is the Nernst potential of `ca_int_mm` (the compartment's
    /// current [Ca²⁺]) against `ca_ext_mm`. Each gate uses its own Q10.
    pub(crate) fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32, temp_k: f32) -> (f32, f32) {
        let mut fopen = 1.0f32;
        for (((kinetics, power), q10), q) in self
            .gates
            .iter()
            .zip(self.q10s.iter())
            .zip(self.gate_state.iter_mut())
        {
            // Ca-dependent gates (Instantaneous h2, kca tau/ss) read the
            // compartment's [Ca²⁺].
            *q = gate_step(kinetics.clone(), *q, v, ca_int_mm, dt, *q10);
            fopen *= q.powi(*power as i32);
        }
        (fopen, nernst_reversal_mv(ca_int_mm, self.ca_ext_mm, temp_k))
    }
}

/// A Ca²⁺ channel whose current is the **Goldman–Hodgkin–Katz** flux equation
/// (`channelDensityGHK2`, e.g. Poirazi 2003 `cal`/`cat`): `I = g·ghk(v, ci, co)`
/// — not ohmic, so there is no reversal potential; the current is computed
/// from the compartment's dynamic [Ca²⁺] each step and fed back into the pool.
#[derive(Clone)]
pub(crate) struct GhkActiveChannel {
    pub gates: Vec<(GateKinetics, u32)>,
    /// Per-gate Q10 rate multiplier (parallel to `gates`) — from `q10Settings`.
    pub q10s: Vec<f32>,
    /// External Ca²⁺ concentration (mM) — fixed, from the `<species>`.
    pub ca_ext_mm: f32,
    /// Absolute max conductance, µS (density × membrane area).
    pub g_max_us: f32,
    /// Gate state (one per gate, declaration order).
    gate_state: Vec<f32>,
}

impl GhkActiveChannel {
    /// Build a channel with every gate initialised to its steady state at
    /// `v_rest` (mV). `q10s` is parallel to `gates`.
    pub fn new(
        gates: Vec<(GateKinetics, u32)>,
        q10s: Vec<f32>,
        ca_ext_mm: f32,
        g_max_us: f32,
        v_rest: f32,
    ) -> Self {
        debug_assert_eq!(gates.len(), q10s.len(), "GhkActiveChannel: q10s length");
        let gate_state = gates
            .iter()
            .map(|(k, _)| gate_steady_state(k.clone(), v_rest))
            .collect();
        Self {
            gates,
            q10s,
            ca_ext_mm,
            g_max_us,
            gate_state,
        }
    }

    /// Advance every gate one step at `v` and return the open fraction. The
    /// GHK current is computed separately from `v` and the compartment's
    /// [Ca²⁺] (see [`ghk_current_na`]).
    pub(crate) fn step_open(&mut self, v: f32, dt: f32, ca_int_mm: f32) -> f32 {
        let mut fopen = 1.0f32;
        for (((kinetics, power), q10), q) in self
            .gates
            .iter()
            .zip(self.q10s.iter())
            .zip(self.gate_state.iter_mut())
        {
            // Ca-dependent gates (Instantaneous h2, kca tau/ss) read the
            // compartment's [Ca²⁺].
            *q = gate_step(kinetics.clone(), *q, v, ca_int_mm, dt, *q10);
            fopen *= q.powi(*power as i32);
        }
        fopen
    }
}

/// Goldman–Hodgkin–Katz flux current (nA per µS of conductance) — the exact
/// form in Poirazi's `cal.mod`/`cat.mod` (KTF/efun convention):
/// `f = (25/293.15)·T/2`, `nu = v/f`, `ghk = −f·(1 − (ci/co)·e^nu)·efun(nu)`,
/// `efun(z) = 1−z/2` for `|z|<1e-4` else `z/(e^z−1)`. Returns the **inward-
/// positive** current (the NHN sign convention used by the pool and the
/// membrane RHS: a negative `ghk` = inward Ca → positive flux).
pub(crate) fn ghk_current_na(v_mv: f32, ci_mm: f32, co_mm: f32, temp_k: f32) -> f32 {
    if co_mm <= 0.0 {
        return 0.0;
    }
    let f = (25.0 / 293.15) * temp_k / 2.0;
    let nu = v_mv / f;
    let efun = if nu.abs() < 1e-4 {
        1.0 - nu / 2.0
    } else {
        nu / (nu.exp() - 1.0)
    };
    // Inward-positive: −ghk = f·(1 − (ci/co)·e^nu)·efun(nu).
    f * (1.0 - (ci_mm / co_mm) * nu.exp()) * efun
}
