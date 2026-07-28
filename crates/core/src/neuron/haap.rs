//! HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting) — VNC neuron model.
//!
//! Safe Rust wrapper over SIMD kernel (`kernels::haap_step`).
//!
//! ## Equations
//!
//! ```text
//! C_m·dV/dt = -g_L(V-E_L) - g_M·u·(V-E_K) - g_h·h·(V-E_h) + I_syn + I_descending
//!   τ_u·du/dt = (u_∞(V) - u)      — M-current adaptation (Kv7/KCNQ)
//!   τ_h·dh/dt = (h_∞(V) - h)      — HCN h-current (activated by hyperpolarization)
//!
//! u_∞(V) = sigmoid((V - V_half_u) / k_u)      — activates with depolarization
//! h_∞(V) = sigmoid((V_half_h - V) / k_h)      — activates with hyperpolarization
//!
//! On spike: V = V_reset, u += delta_u (adaptation bump)
//! ```
//!
//! **Note:** `τ_h` is constant in this implementation; biological HCN channels have
//! voltage-dependent kinetics `τ_h(V)` that accelerate with hyperpolarization.
//! The user passes `I_syn + I_descending` summed externally as a single `Ii` array.
//!
//! ## Biological basis
//!
//! | Current | Channel | Function | Time scale |
//! |---------|---------|----------|------------|
//! | I_M     | Kv7/KCNQ | Spike-frequency adaptation | 50-500ms |
//! | I_h     | HCN      | Rebound excitation, pacemaking | 100-1000ms |
//!
//! Ideal for Drosophila VNC motor neurons: slow adaptation prevents runaway
//! firing, while h-current provides rebound after inhibition (walking CPG).

use crate::kernels;

/// Step `n` HAAP neurons using the packed params array.
///
/// Params layout (17 × n, column-major):
///   [0]:    dt          — time step (ms), uniform
///   [n]:    C_m         — membrane capacitance (µF/cm²)
///   [2n]:   g_L         — leak conductance (mS/cm²)
///   [3n]:   E_L         — leak reversal (mV)
///   [4n]:   g_M         — M-current conductance (mS/cm²)
///   [5n]:   E_K         — potassium reversal (mV)
///   [6n]:   g_h         — HCN conductance (mS/cm²)
///   [7n]:   E_h         — HCN reversal (mV, ~-30mV)
///   [8n]:   tau_u       — M-current time constant (ms)
///   [9n]:   tau_h       — HCN time constant (ms)
///   [10n]:  V_half_u    — M-current half-activation (mV, ~-35mV)
///   [11n]:  k_u         — M-current slope factor (mV)
///   [12n]:  V_half_h    — HCN half-activation (mV, ~-75mV)
///   [13n]:  k_h         — HCN slope factor (mV, >0)
///   [14n]:  V_thresh    — spike threshold (mV)
///   [15n]:  V_reset     — after-spike reset (mV)
///   [16n]:  delta_u     — after-spike M-current increment
///
/// `n` must be multiple of 8.
pub fn step_haap(
    v: &mut [f32],
    u_state: &mut [f32],
    h_state: &mut [f32],
    Ii: &[f32],
    spikes: &mut [i32],
    params: &[f32],
) {
    let n = v.len();
    assert!(n.is_multiple_of(8), "haap_step: n must be multiple of 8");
    assert_eq!(u_state.len(), n, "haap_step: u_state length mismatch");
    assert_eq!(h_state.len(), n, "haap_step: h_state length mismatch");
    assert_eq!(Ii.len(), n, "haap_step: Ii length mismatch");
    assert_eq!(spikes.len(), n, "haap_step: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 17,
        "haap_step: params length mismatch (expected {n}*17={})",
        n * 17
    );

    kernels::haap_step(v, u_state, h_state, Ii, spikes, params);
}

/// Build packed params array from individual per-neuron arrays.
///
/// Returns `Vec<f32>` of length `n * 17`.
pub fn build_haap_params(
    n: usize,
    dt: f32,
    c_m: &[f32],
    g_L: &[f32],
    E_L: &[f32],
    g_M: &[f32],
    E_K: &[f32],
    g_h: &[f32],
    E_h: &[f32],
    tau_u: &[f32],
    tau_h: &[f32],
    V_half_u: &[f32],
    k_u: &[f32],
    V_half_h: &[f32],
    k_h: &[f32],
    V_thresh: &[f32],
    V_reset: &[f32],
    delta_u: &[f32],
) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 17];
    for i in 0..n {
        p[i] = dt;
        p[n + i] = c_m[i];
        p[2 * n + i] = g_L[i];
        p[3 * n + i] = E_L[i];
        p[4 * n + i] = g_M[i];
        p[5 * n + i] = E_K[i];
        p[6 * n + i] = g_h[i];
        p[7 * n + i] = E_h[i];
        p[8 * n + i] = tau_u[i];
        p[9 * n + i] = tau_h[i];
        p[10 * n + i] = V_half_u[i];
        p[11 * n + i] = k_u[i];
        p[12 * n + i] = V_half_h[i];
        p[13 * n + i] = k_h[i];
        p[14 * n + i] = V_thresh[i];
        p[15 * n + i] = V_reset[i];
        p[16 * n + i] = delta_u[i];
    }
    p
}

/// Convenience: build params from a uniform profile.
pub fn build_haap_params_uniform(n: usize, profile: &HaapProfile) -> Vec<f32> {
    let p = profile;
    let mut params = vec![0.0f32; n * 17];
    for i in 0..n {
        params[i] = p.dt;
        params[n + i] = p.c_m;
        params[2 * n + i] = p.g_L;
        params[3 * n + i] = p.E_L;
        params[4 * n + i] = p.g_M;
        params[5 * n + i] = p.E_K;
        params[6 * n + i] = p.g_h;
        params[7 * n + i] = p.E_h;
        params[8 * n + i] = p.tau_u;
        params[9 * n + i] = p.tau_h;
        params[10 * n + i] = p.V_half_u;
        params[11 * n + i] = p.k_u;
        params[12 * n + i] = p.V_half_h;
        params[13 * n + i] = p.k_h;
        params[14 * n + i] = p.V_thresh;
        params[15 * n + i] = p.V_reset;
        params[16 * n + i] = p.delta_u;
    }
    params
}

// ═══════════════════════════════════════════════════════
//  High-level state management
// ═══════════════════════════════════════════════════════

/// High-level HAAP neuron group state.
///
/// Manages membrane potential, M-current gating (u), HCN gating (h),
/// and per-neuron parameters for the HAAP model.
#[derive(Debug, Clone)]
pub struct HaapState {
    pub n: usize,
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub h: Vec<f32>,
    pub params: Vec<f32>,
}

/// Biophysical profile for HAAP model tuning.
#[derive(Debug, Clone, Copy)]
pub struct HaapProfile {
    pub dt: f32,
    pub c_m: f32,
    pub g_L: f32,
    pub E_L: f32,
    pub g_M: f32,
    pub E_K: f32,
    pub g_h: f32,
    pub E_h: f32,
    pub tau_u: f32,
    pub tau_h: f32,
    pub V_half_u: f32,
    pub k_u: f32,
    pub V_half_h: f32,
    pub k_h: f32,
    pub V_thresh: f32,
    pub V_reset: f32,
    pub delta_u: f32,
}

impl HaapProfile {
    /// Drosophila VNC profile — tuned for motor neuron dynamics.
    ///
    /// - Resting ~-65mV, spikes ~0mV, rebound excitation
    /// - M-current: slow adaptation (tau_u=100ms) prevents runaway
    /// - HCN: strong rebound (g_h=0.3, V_half=-75mV)
    pub const fn vnc() -> Self {
        Self {
            dt: 0.05,
            c_m: 1.0,
            g_L: 0.1,
            E_L: -65.0,
            g_M: 0.5,
            E_K: -80.0,
            g_h: 0.3,
            E_h: -30.0,
            tau_u: 100.0,
            tau_h: 200.0,
            V_half_u: -35.0,
            k_u: 5.0,
            V_half_h: -75.0,
            k_h: 6.0,
            V_thresh: -40.0,
            V_reset: -55.0,
            delta_u: 0.02,
        }
    }

    /// Slow oscillatory profile — for CPG/bursting dynamics.
    ///
    /// - Slower time constants promote oscillation
    /// - Stronger h-current for rebound bursts
    /// - Lower threshold for easier recruitment
    pub const fn cpg() -> Self {
        Self {
            dt: 0.1,
            c_m: 1.0,
            g_L: 0.08,
            E_L: -60.0,
            g_M: 1.0,
            E_K: -80.0,
            g_h: 0.8,
            E_h: -30.0,
            tau_u: 200.0,
            tau_h: 500.0,
            V_half_u: -35.0,
            k_u: 6.0,
            V_half_h: -70.0,
            k_h: 7.0,
            V_thresh: -45.0,
            V_reset: -55.0,
            delta_u: 0.05,
        }
    }

    /// Fast-spiking interneuron profile.
    ///
    /// - Fast adaptation (tau_u=20ms) for high-frequency firing
    /// - Weak HCN — less rebound, more tonic
    /// - Higher threshold for reliability
    pub const fn fast() -> Self {
        Self {
            dt: 0.05,
            c_m: 1.0,
            g_L: 0.15,
            E_L: -65.0,
            g_M: 2.0,
            E_K: -80.0,
            g_h: 0.1,
            E_h: -30.0,
            tau_u: 20.0,
            tau_h: 100.0,
            V_half_u: -30.0,
            k_u: 4.0,
            V_half_h: -80.0,
            k_h: 5.0,
            V_thresh: -35.0,
            V_reset: -50.0,
            delta_u: 0.01,
        }
    }

    /// Mammalian pyramidal cell profile.
    pub const fn pyramidal() -> Self {
        Self {
            dt: 0.05,
            c_m: 1.0,
            g_L: 0.1,
            E_L: -70.0,
            g_M: 0.8,
            E_K: -90.0,
            g_h: 0.5,
            E_h: -30.0,
            tau_u: 150.0,
            tau_h: 300.0,
            V_half_u: -40.0,
            k_u: 6.0,
            V_half_h: -80.0,
            k_h: 8.0,
            V_thresh: -50.0,
            V_reset: -60.0,
            delta_u: 0.03,
        }
    }
}

impl Default for HaapProfile {
    fn default() -> Self {
        Self::vnc()
    }
}

/// Idiomatic Rust Builder for [`HaapState`].
#[derive(Debug, Clone)]
pub struct HaapStateBuilder {
    n: usize,
    profile: HaapProfile,
    v_init: Option<f32>,
    u_init: Option<f32>,
    h_init: Option<f32>,
}

impl HaapStateBuilder {
    pub fn new(n: usize) -> Self {
        assert!(n.is_multiple_of(8), "HaapStateBuilder: n must be multiple of 8");
        Self {
            n,
            profile: HaapProfile::default(),
            v_init: None,
            u_init: None,
            h_init: None,
        }
    }

    pub fn with_profile(mut self, profile: HaapProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Override initial membrane potential (default: E_L from profile).
    pub fn with_v_init(mut self, v: f32) -> Self {
        self.v_init = Some(v);
        self
    }

    /// Override initial M-current gating (default: steady state at v_init).
    pub fn with_u_init(mut self, u: f32) -> Self {
        self.u_init = Some(u);
        self
    }

    /// Override initial HCN gating (default: steady state at v_init).
    pub fn with_h_init(mut self, h: f32) -> Self {
        self.h_init = Some(h);
        self
    }

    pub fn build(self) -> HaapState {
        let p = self.profile;
        let n = self.n;

        let v_init = self.v_init.unwrap_or(p.E_L);
        let u_ss = sigmoid(v_init, p.V_half_u, p.k_u, false); // u: activates with depolarization
        let h_ss = sigmoid(v_init, p.V_half_h, p.k_h, true);  // h: activates with hyperpolarization
        let u_init = self.u_init.unwrap_or(u_ss);
        let h_init = self.h_init.unwrap_or(h_ss);

        let params = build_haap_params_uniform(n, &p);

        HaapState {
            n,
            v: vec![v_init; n],
            u: vec![u_init; n],
            h: vec![h_init; n],
            params,
        }
    }
}

/// Compute sigmoid gating: `1/(1 + exp(-(V - V_half) / k))`.
/// If `invert` is true, uses `1/(1 + exp(-(V_half - V) / k))` for h-current.
fn sigmoid(V: f32, V_half: f32, k: f32, invert: bool) -> f32 {
    let arg = if invert {
        (V_half - V) / k.max(1e-6)
    } else {
        (V - V_half) / k.max(1e-6)
    };
    let arg_clamped = arg.clamp(-50.0, 50.0);
    1.0 / (1.0 + (-arg_clamped).exp())
}

impl HaapState {
    /// Create a new HAAP state for `n` neurons using the given profile.
    pub fn new(n: usize, profile: &HaapProfile) -> Self {
        HaapStateBuilder::new(n).with_profile(*profile).build()
    }

    /// Step simulation: update V, u, h and detect spikes.
    ///
    /// `Ii`: external/synaptic current per neuron.
    /// Returns: spike array (0/1 per neuron).
    pub fn step(&mut self, Ii: &[f32]) -> Vec<i32> {
        assert_eq!(Ii.len(), self.n, "HaapState::step: Ii length mismatch");
        let mut spikes = vec![0i32; self.n];
        kernels::haap_step(
            &mut self.v,
            &mut self.u,
            &mut self.h,
            Ii,
            &mut spikes,
            &self.params,
        );
        spikes
    }

    /// Reset all neurons to initial conditions.
    pub fn reset(&mut self, profile: &HaapProfile) {
        let v_init = profile.E_L;
        let u_ss = sigmoid(v_init, profile.V_half_u, profile.k_u, false);
        let h_ss = sigmoid(v_init, profile.V_half_h, profile.k_h, true);
        self.v.fill(v_init);
        self.u.fill(u_ss);
        self.h.fill(h_ss);
        self.params = build_haap_params_uniform(self.n, profile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haap_state_basic() {
        let mut state = HaapState::new(16, &HaapProfile::vnc());
        assert_eq!(state.n, 16);
        assert_eq!(state.v.len(), 16);
        assert_eq!(state.u.len(), 16);
        assert_eq!(state.h.len(), 16);

        let Ii = vec![15.0f32; 16];
        let spikes = state.step(&Ii);

        let total: i32 = spikes.iter().sum();
        assert!(total >= 0);
    }

    #[test]
    fn haap_spikes_with_current() {
        let mut state = HaapState::new(16, &HaapProfile::vnc());

        let mut total_spikes = 0i32;
        for _ in 0..2000 {
            let Ii = vec![15.0f32; state.n];
            let spikes = state.step(&Ii);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "HAAP should spike with I=15");
    }

    #[test]
    fn haap_silent_no_current() {
        let mut state = HaapState::new(16, &HaapProfile::vnc());

        for _ in 0..1000 {
            let Ii = vec![-10.0f32; state.n];
            let spikes = state.step(&Ii);
            assert_eq!(spikes.iter().sum::<i32>(), 0, "Should not spike with I=-10");
        }
    }

    #[test]
    fn haap_cpg_profile_oscillates() {
        let mut state = HaapState::new(8, &HaapProfile::cpg());

        let mut total_spikes = 0i32;
        for _ in 0..5000 {
            let Ii = vec![5.0f32; state.n];
            let spikes = state.step(&Ii);
            total_spikes += spikes.iter().sum::<i32>();
        }
        assert!(total_spikes > 0, "HAAP CPG should fire with I=5");
    }

    #[test]
    fn haap_reset_restores_state() {
        let mut state = HaapState::new(8, &HaapProfile::vnc());
        let Ii = vec![15.0f32; 8];
        let _ = state.step(&Ii);

        state.reset(&HaapProfile::vnc());
        assert!((state.v[0] - (-65.0)).abs() < 0.1, "Reset should restore E_L");
    }

    #[test]
    fn haap_builder_custom_params() {
        let state = HaapStateBuilder::new(8)
            .with_profile(HaapProfile::fast())
            .with_v_init(-70.0)
            .build();

        assert_eq!(state.n, 8);
        assert!((state.v[0] - (-70.0)).abs() < 0.1);
    }
}
