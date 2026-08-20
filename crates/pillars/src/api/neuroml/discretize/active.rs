use super::*;

/// Passive cable + active HH channels, integrated semi-implicitly: gates at
/// the old voltage (exponential Euler), then the Hines solve over the linear
/// system with the active conductances added to the diagonal.
#[derive(Clone)]
pub(crate) struct ActiveCableSolver {
    passive: CableSolver,
    /// Active HH channels per compartment.
    channels: Vec<Vec<ActiveChannel>>,
    /// Active Markov (kinetic-scheme) channels per compartment.
    markov_channels: Vec<Vec<MarkovActiveChannel>>,
    /// Stochastic (discrete-population) channels per compartment.
    stochastic_channels: Vec<Vec<StochasticMarkovActiveChannel>>,
    /// Nernst (Ca²⁺) HH channels per compartment — reversal computed from
    /// the compartment's [Ca²⁺] each step.
    nernst_channels: Vec<Vec<NernstActiveChannel>>,
    /// Stochastic Nernst (Ca²⁺) channel populations per compartment.
    stochastic_nernst_channels: Vec<Vec<StochasticNernstActiveChannel>>,
    /// GHK (Ca²⁺, `channelDensityGHK2`) channels per compartment — current is
    /// the Goldman–Hodgkin–Katz flux (not ohmic), computed from the
    /// compartment's [Ca²⁺] each step.
    ghk_channels: Vec<Vec<GhkActiveChannel>>,
    /// Internal [Ca²⁺] per compartment (mM).
    ca: Vec<f32>,
    /// Ca²⁺ pool (ODE params) per compartment; `None` = no tracked Ca.
    ca_pools: Vec<Option<CaPool>>,
    dt: f32,
    /// Temperature for the Nernst reversal (K).
    temp_k: f32,
    /// Per-compartment passive leak override (µS) — from per-group
    /// `channelDensity` leak densities (the model's declared leak, which may
    /// differ between segment groups). `0.0` = use the tree's uniform
    /// `g_leak_us` (from `CableParams::rm_ohm_cm2`). Parallel to `leak_rev`.
    leak_us: Vec<f32>,
    /// Per-compartment leak reversal (mV), parallel to `leak_us`.
    leak_rev: Vec<f32>,
    /// Reused per-step buffers (sized `n`, zeroed each step — no per-step
    /// allocations for active cables).
    g_extra: Vec<f32>,
    g_e_extra: Vec<f32>,
    /// Reused Ca-current buffer (nA per compartment, zeroed each step).
    i_ca: Vec<f32>,
}

impl ActiveCableSolver {
    /// Build a solver at rest (`V = e_leak`), with no active channels.
    pub fn new(tree: CompartmentTree, e_leak: f32, dt: f32, celsius: f32) -> Self {
        let n = tree.len();
        let passive = CableSolver::new(tree, e_leak, dt);
        Self {
            passive,
            channels: (0..n).map(|_| Vec::new()).collect(),
            markov_channels: (0..n).map(|_| Vec::new()).collect(),
            stochastic_channels: (0..n).map(|_| Vec::new()).collect(),
            nernst_channels: (0..n).map(|_| Vec::new()).collect(),
            stochastic_nernst_channels: (0..n).map(|_| Vec::new()).collect(),
            ghk_channels: (0..n).map(|_| Vec::new()).collect(),
            ca: vec![0.0; n],
            ca_pools: (0..n).map(|_| None).collect(),
            dt,
            temp_k: celsius + 273.15,
            leak_us: vec![0.0; n],
            leak_rev: vec![0.0; n],
            g_extra: vec![0.0; n],
            g_e_extra: vec![0.0; n],
            i_ca: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.passive.len()
    }

    /// Index of the root compartment (the soma). Test-only — production reads
    /// the soma index from the shared tree (`CablePopulation::new`).
    #[cfg(test)]
    pub fn root(&self) -> usize {
        self.passive.root()
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.passive.v(idx)
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.passive.set_v(idx, value);
    }

    /// Replace the active channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        assert!(comp < self.len(), "set_channels: {comp} out of range");
        self.channels[comp] = channels;
    }

    /// Replace the active Markov channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        assert!(
            comp < self.len(),
            "set_markov_channels: {comp} out of range"
        );
        self.markov_channels[comp] = channels;
    }

    /// Replace the stochastic (discrete-population) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_stochastic_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticMarkovActiveChannel>,
    ) {
        assert!(
            comp < self.len(),
            "set_stochastic_channels: {comp} out of range"
        );
        self.stochastic_channels[comp] = channels;
    }

    /// Replace the Nernst (Ca²⁺) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_nernst_channels(&mut self, comp: usize, channels: Vec<NernstActiveChannel>) {
        assert!(
            comp < self.len(),
            "set_nernst_channels: {comp} out of range"
        );
        self.nernst_channels[comp] = channels;
    }
    /// Replace the stochastic Nernst (Ca²⁺) channel populations of one
    /// compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_stochastic_nernst_channels(
        &mut self,
        comp: usize,
        channels: Vec<StochasticNernstActiveChannel>,
    ) {
        assert!(
            comp < self.len(),
            "set_stochastic_nernst_channels: {comp} out of range"
        );
        self.stochastic_nernst_channels[comp] = channels;
    }

    /// Replace the GHK (Ca²⁺) channels of one compartment.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_ghk_channels(&mut self, comp: usize, channels: Vec<GhkActiveChannel>) {
        assert!(comp < self.len(), "set_ghk_channels: {comp} out of range");
        self.ghk_channels[comp] = channels;
    }

    /// Override the passive leak of one compartment with an explicit
    /// conductance `g_us` (µS) at reversal `e_rev` (mV) — the per-group leak
    /// densities of the model. The tree's uniform `g_leak_us` must have been
    /// zeroed for this compartment (compile.rs) so the two don't double-count.
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_leak(&mut self, comp: usize, g_us: f32, e_rev: f32) {
        assert!(comp < self.len(), "set_leak: {comp} out of range");
        self.leak_us[comp] = g_us;
        self.leak_rev[comp] = e_rev;
    }

    /// Attach a Ca²⁺ pool to one compartment and initialise its [Ca²⁺] to
    /// `resting` (the species `initialConcentration`).
    ///
    /// # Panics
    /// If `comp >= self.len()`.
    pub fn set_ca_pool(&mut self, comp: usize, pool: CaPool, initial_mm: f32) {
        assert!(comp < self.len(), "set_ca_pool: {comp} out of range");
        self.ca_pools[comp] = Some(pool);
        self.ca[comp] = initial_mm;
    }

    /// Current internal [Ca²⁺] (mM) of a compartment.
    pub fn ca(&self, comp: usize) -> f32 {
        self.ca[comp]
    }

    /// One semi-implicit step with the solver's fixed `dt`.
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step(&mut self, i_inj: &[f32]) {
        self.step_dt(i_inj, self.dt);
    }

    /// One semi-implicit step with an explicit `dt` (for adaptive stepping).
    ///
    /// Gates are integrated at the **old** voltage (exponential Euler / exact
    /// matrix exponential), then the voltage solve is implicit — the same
    /// state convention as NEURON fixed-step (`secondorder=0`): the
    /// `markov_cable_matches_neuron_ks3` experiment showed that a
    /// fully-implicit (Picard) gate solve *worsens* parity (0.155 vs
    /// 0.014 mV), i.e. NEURON evaluates rates at the step-start voltage.
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step_dt(&mut self, i_inj: &[f32], dt: f32) {
        let n = self.len();
        self.g_extra.fill(0.0);
        self.g_e_extra.fill(0.0);
        self.i_ca.fill(0.0);
        for i in 0..n {
            // Per-group passive leak (constant conductance, semi-implicit like
            // the active channels).
            if self.leak_us[i] > 0.0 {
                self.g_extra[i] += self.leak_us[i];
                self.g_e_extra[i] += self.leak_us[i] * self.leak_rev[i];
            }
            let v = self.passive.v(i);
            for ch in &mut self.channels[i] {
                let mut fopen = 1.0f32;
                for (((kinetics, power), q10), q) in ch
                    .gates
                    .iter()
                    .zip(ch.q10s.iter())
                    .zip(ch.gate_state.iter_mut())
                {
                    // Ca-dependent gates (Instantaneous h2, kca tau/ss) read
                    // the compartment's [Ca²⁺].
                    *q = gate_step(kinetics.clone(), *q, v, self.ca[i], dt, *q10);
                    fopen *= q.powi(*power as i32);
                }
                let g_ch = ch.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * ch.e_rev;
            }
            for mc in &mut self.markov_channels[i] {
                let fopen = mc.step_open(v, dt);
                let g_ch = mc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * mc.e_rev;
            }
            for sc in &mut self.stochastic_channels[i] {
                // Stochastic: `fopen` is `n_open / n_channels`, so
                // `g_max_us · fopen = single_channel_conductance · n_open`.
                let fopen = sc.step_open(v, dt);
                let g_ch = sc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * sc.e_rev;
            }
            for nc in &mut self.nernst_channels[i] {
                // Reversal from the current [Ca²⁺] (Nernst); the Ca current also
                // feeds the compartment's Ca²⁺ pool.
                let (fopen, e_rev) = nc.step_open(v, dt, self.ca[i], self.temp_k);
                let g_ch = nc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * e_rev;
                self.i_ca[i] += g_ch * (e_rev - v);
            }
            for snc in &mut self.stochastic_nernst_channels[i] {
                // Same as `nernst_channels`, but `fopen` is the binomial sample
                // `n_open / n_channels` (channel noise).
                let (fopen, e_rev) = snc.step_open(v, dt, self.ca[i], self.temp_k);
                let g_ch = snc.g_max_us * fopen;
                self.g_extra[i] += g_ch;
                self.g_e_extra[i] += g_ch * e_rev;
                self.i_ca[i] += g_ch * (e_rev - v);
            }
            for gh in &mut self.ghk_channels[i] {
                // GHK flux current (not ohmic): explicit at the old V/[Ca],
                // like NEURON's cnexp `BREAKPOINT`. The inward-positive
                // current feeds both the membrane RHS and the Ca²⁺ pool.
                let fopen = gh.step_open(v, dt, self.ca[i]);
                let g_ch = gh.g_max_us * fopen;
                let i_in = g_ch * ghk_current_na(v, self.ca[i], gh.ca_ext_mm, self.temp_k);
                self.g_e_extra[i] += i_in;
                self.i_ca[i] += i_in;
            }
        }
        self.passive
            .step_with_extra(i_inj, &self.g_extra, &self.g_e_extra, dt);
        // Update [Ca²⁺] after the membrane solve (semi-implicit: iCa at the old
        // V/[Ca], exact exponential decay — same convention as the gates).
        for i in 0..n {
            if let Some(pool) = self.ca_pools[i] {
                self.ca[i] = pool.step(self.ca[i], self.i_ca[i], dt);
            }
        }
    }
}

/// Adaptive-step driver over [`ActiveCableSolver`] — the CVODE-analogue for the
/// cable, with **variable order** (1 ↔ 2) and step-doubling error control.
///
/// Order 1: plain implicit-Euler (backward Euler, BDF1) step-doubling — one
/// full `dt` step vs two `dt/2` steps estimate the local truncation error.
///
/// Order 2: the accepted voltage is the **Richardson extrapolation**
/// `v = 2·v_half − v_full`, which cancels the leading `h²` error term of the
/// half-step trajectory — the one-step equivalent of BDF2 (for the linear
/// cable system the two coincide). Gates/kinetics keep the half-step state
/// (semi-implicit, as in the fixed-dt solver).
///
/// Controller: `factor = 0.9·err^(−1/(p+1))` (p = 1 → −1/2, p = 2 → −1/3),
/// clamped to `[0.1, 2.0]`; `dt` stays in `[dt_min, dt_max]`. On rejection the
/// step is retried smaller at **order 1** (CVODE restarts the low order after
/// a failed step); after an accepted step with margin the order rises back to
/// 2. Stiff channels (Ca/HCN) and spikes force small `dt` automatically, while
/// slow phases take large `dt` — the same idea as NEURON's CVODE.
/// Adaptive stepping is exercised only by the module tests (the NeuroML
/// compiler runs the fixed-dt [`ActiveCableSolver`]); the whole type is
/// test-only, so production builds see an impl-less struct.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) struct AdaptiveCableSolver {
    solver: ActiveCableSolver,
    dt: f32,
    dt_min: f32,
    dt_max: f32,
    rtol: f32,
    atol: f32,
    /// Current method order: 1 (implicit Euler) or 2 (extrapolated ≈ BDF2).
    order: u8,
}

#[cfg(test)]
#[allow(dead_code)]
impl AdaptiveCableSolver {
    /// Wrap `solver` with adaptive stepping: initial `dt0`, clamped to
    /// `[dt_min, dt_max]`, with relative/absolute tolerances (mV).
    pub fn new(
        solver: ActiveCableSolver,
        dt0: f32,
        dt_min: f32,
        dt_max: f32,
        rtol: f32,
        atol: f32,
    ) -> Self {
        Self {
            solver,
            dt: dt0.clamp(dt_min, dt_max),
            dt_min,
            dt_max,
            rtol,
            atol,
            // CVODE starts at order 1; promotion happens after accepted steps.
            order: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.solver.len()
    }

    pub fn root(&self) -> usize {
        self.solver.root()
    }

    pub fn v(&self, idx: usize) -> f32 {
        self.solver.v(idx)
    }

    pub fn set_v(&mut self, idx: usize, value: f32) {
        self.solver.set_v(idx, value);
    }

    pub fn set_channels(&mut self, comp: usize, channels: Vec<ActiveChannel>) {
        self.solver.set_channels(comp, channels);
    }

    pub fn set_markov_channels(&mut self, comp: usize, channels: Vec<MarkovActiveChannel>) {
        self.solver.set_markov_channels(comp, channels);
    }

    /// Current step size (ms).
    pub fn current_dt(&self) -> f32 {
        self.dt
    }

    /// Current method order: 1 (implicit Euler) or 2 (extrapolated ≈ BDF2).
    pub fn order(&self) -> u8 {
        self.order
    }

    /// Advance one adaptive step and return the `dt` actually used (ms).
    ///
    /// # Panics
    /// If `i_inj.len() != self.len()`.
    pub fn step(&mut self, i_inj: &[f32]) -> f32 {
        loop {
            let mut full = self.solver.clone();
            full.step_dt(i_inj, self.dt);
            let mut half = self.solver.clone();
            half.step_dt(i_inj, self.dt * 0.5);
            half.step_dt(i_inj, self.dt * 0.5);

            let err = cable_error_norm(&half, &full, self.atol, self.rtol);
            // Order p: local error ~ dt^(p+1) → dt scale = err^(-1/(p+1)).
            let exponent = if self.order == 2 { -1.0 / 3.0 } else { -0.5 };
            let factor = (0.9 * err.powf(exponent)).clamp(0.1, 2.0);
            if err <= 1.0 {
                if self.order == 2 {
                    // Richardson extrapolation cancels the leading h² error
                    // term of the half-step path → second order (BDF2-eq.).
                    for i in 0..self.solver.len() {
                        half.set_v(i, 2.0 * half.v(i) - full.v(i));
                    }
                }
                self.solver = half;
                self.dt = (self.dt * factor).clamp(self.dt_min, self.dt_max);
                // Raise the order when there is margin (CVODE-style); stay at
                // 2 otherwise. The extrapolation is one-step, so promotion is
                // safe at any time.
                if self.order == 1 && err < 0.5 {
                    self.order = 2;
                }
                return self.dt;
            }
            self.dt = (self.dt * factor).clamp(self.dt_min, self.dt_max);
            // CVODE restarts at the low order after a rejected step.
            self.order = 1;
            if self.dt <= self.dt_min {
                // Can't shrink further — accept the min-dt result rather than
                // loop forever (only reachable with pathological tolerances).
                self.solver = half;
                return self.dt;
            }
        }
    }
}

/// Max-norm of the per-compartment step-doubling error, scaled by the
/// tolerance: `max_i |V_half − V_full| / (atol + rtol·max|V|)`. `≤ 1` ⇒ accept.
/// Test-only (used by [`AdaptiveCableSolver`]).
#[cfg(test)]
fn cable_error_norm(a: &ActiveCableSolver, b: &ActiveCableSolver, atol: f32, rtol: f32) -> f32 {
    let n = a.len();
    let mut err = 0.0f32;
    for i in 0..n {
        let va = a.v(i);
        let vb = b.v(i);
        let scale = atol + rtol * va.abs().max(vb.abs());
        err = err.max((va - vb).abs() / scale);
    }
    err
}
