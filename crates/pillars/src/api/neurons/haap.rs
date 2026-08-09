//! # HAAP — Hyperpolarization-Activated Adaptive Pulse-Bursting (VNC)
//!
//! Stateful-классы поверх `run_neuron::<HAAPModel>` (17-col repacking:
//! inv_cm, inv_tau_u, inv_tau_h).
//!
//! ## Equations
//!
//! ```text
//! C_m·dV/dt = -g_L(V-E_L) - g_M·u·(V-E_K) - g_h·h·(V-E_h) + I_syn + I_descending
//!   τ_u·du/dt = (u_∞(V) - u)      — M-current adaptation (Kv7/KCNQ)
//!   τ_h·dh/dt = (h_∞(V) - h)      — HCN h-current (activated by hyperpolarization)
//!
//! On spike: V = V_reset, u += delta_u (adaptation bump)
//! ```
//!
//! Ideal for Drosophila VNC motor neurons: slow adaptation prevents runaway
//! firing, while h-current provides rebound after inhibition (walking CPG).

use crate::core::runner::run_neuron;
use crate::zoo::neurons::HAAPModel;
use std::cell::RefCell;

/// 1/x для x > 0, иначе 0 — сторож легаси-семантики τ ≤ 0 → «без эволюции».
#[inline(always)]
fn inv_or_zero(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / x
    } else {
        0.0
    }
}

/// Repack legacy 17-col params → HAAPModel 17-col (inv_cm, inv_tau_u, inv_tau_h).
fn pack_haap(params: &[f32], n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 17];
    for i in 0..n {
        p[i] = params[i];
        p[n + i] = 1.0 / params[n + i].max(1e-6); // inv_cm = 1/c_m
        for c in 2..8 {
            p[c * n + i] = params[c * n + i];
        }
        p[8 * n + i] = inv_or_zero(params[8 * n + i]); // inv_tau_u
        p[9 * n + i] = inv_or_zero(params[9 * n + i]); // inv_tau_h
        for c in 10..17 {
            p[c * n + i] = params[c * n + i];
        }
    }
    p
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

/// Build packed params array from a uniform profile.
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
    /// Кэш pillars-лейаута (17-col HAAPModel) — пересобирается при reset.
    packed: Vec<f32>,
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
        assert!(
            n.is_multiple_of(8),
            "HaapStateBuilder: n must be multiple of 8"
        );
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
        let h_ss = sigmoid(v_init, p.V_half_h, p.k_h, true); // h: activates with hyperpolarization
        let u_init = self.u_init.unwrap_or(u_ss);
        let h_init = self.h_init.unwrap_or(h_ss);

        let params = build_haap_params_uniform(n, &p);
        let packed = pack_haap(&params, n);

        HaapState {
            n,
            v: vec![v_init; n],
            u: vec![u_init; n],
            h: vec![h_init; n],
            params,
            packed,
        }
    }
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
        let n = self.n;
        unsafe {
            run_neuron::<HAAPModel>(
                self.v.as_mut_ptr(),
                self.u.as_mut_ptr(),
                self.h.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                Ii.as_ptr(),
                spikes.as_mut_ptr(),
                self.packed.as_ptr(),
                n,
            );
        }
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
        self.packed = pack_haap(&self.params, self.n);
    }
}

// ─── step_haap — свободный repacking-адаптер (python ABI) ──

// Thread-local scratch для переупаковки в свободной step_haap (без аллокаций).
thread_local! {
    static HAAP_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Step `n` HAAP neurons using the packed params array.
///
/// Params layout (17 × n, column-major):
///   `[0]`:    dt          — time step (ms), uniform
///   `[n]`:    C_m         — membrane capacitance (µF/cm²)
///   `[2n]`:   g_L         — leak conductance (mS/cm²)
///   `[3n]`:   E_L         — leak reversal (mV)
///   `[4n]`:   g_M         — M-current conductance (mS/cm²)
///   `[5n]`:   E_K         — potassium reversal (mV)
///   `[6n]`:   g_h         — HCN conductance (mS/cm²)
///   `[7n]`:   E_h         — HCN reversal (mV, ~-30mV)
///   `[8n]`:   tau_u       — M-current time constant (ms)
///   `[9n]`:   tau_h       — HCN time constant (ms)
///   `[10n]`:  V_half_u    — M-current half-activation (mV, ~-35mV)
///   `[11n]`:  k_u         — M-current slope factor (mV)
///   `[12n]`:  V_half_h    — HCN half-activation (mV, ~-75mV)
///   `[13n]`:  k_h         — HCN slope factor (mV, >0)
///   `[14n]`:  V_thresh    — spike threshold (mV)
///   `[15n]`:  V_reset     — after-spike reset (mV)
///   `[16n]`:  delta_u     — after-spike M-current increment
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

    HAAP_SCRATCH.with(|c| {
        let mut buf = c.borrow_mut();
        let need = n * 17;
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        let p = &mut buf[..need];
        for i in 0..n {
            p[i] = params[i];
            p[n + i] = 1.0 / params[n + i].max(1e-6); // inv_cm = 1/c_m
            for c in 2..8 {
                p[c * n + i] = params[c * n + i];
            }
            p[8 * n + i] = inv_or_zero(params[8 * n + i]); // inv_tau_u
            p[9 * n + i] = inv_or_zero(params[9 * n + i]); // inv_tau_h
            for c in 10..17 {
                p[c * n + i] = params[c * n + i];
            }
        }
        unsafe {
            run_neuron::<HAAPModel>(
                v.as_mut_ptr(),
                u_state.as_mut_ptr(),
                h_state.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                Ii.as_ptr(),
                spikes.as_mut_ptr(),
                p.as_ptr(),
                n,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haap_state_basic() {
        let mut state = HaapState::new(16, &HaapProfile::vnc());
        assert_eq!(state.n, 16);
        assert_eq!(state.v.len(), 16);

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
    fn haap_reset_restores_state() {
        let mut state = HaapState::new(8, &HaapProfile::vnc());
        let Ii = vec![15.0f32; 8];
        let _ = state.step(&Ii);

        state.reset(&HaapProfile::vnc());
        assert!(
            (state.v[0] - (-65.0)).abs() < 0.1,
            "Reset should restore E_L"
        );
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
