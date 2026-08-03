//! ⚡ HAAP — Hyperpolarization-Activated Adaptive Pulse-Bursting
//!
//! Uses `#[derive(NeuronModel)]` + `ComputeNeuron` for typed API.
//!
//! # Principles
//! - **Ohm's Law**: I = g·(V − E_rev), subtracted from I_inj
//! - **Forward Euler**: no gate_integrate, no hidden abstractions
//! - **Zero divisions**: all tau stored as inv_tau at init
//!
//! Equations:
//! ```text
//! I_leak = g_L·(V − E_L)
//! I_M    = g_M·u·(V − E_K)
//! I_h    = g_h·h·(V − E_h)
//! C_m·dV/dt = I_syn − I_leak − I_M − I_h
//! du/dt = (u_∞(V) − u) · inv_tau_u     (Forward Euler)
//! dh/dt = (h_∞(V) − h) · inv_tau_h     (Forward Euler)
//! Spike: V ≥ V_thresh → V = V_reset, u += delta_u
//! ```

use pillars_macros::NeuronModel;
use crate::core::lane::SimdLane;
use crate::core::compute::{StepContext, ComputeNeuron};

// ═══════════════════════════════════════════════════════════════
//  ParamIdx Enum (for manual fill_param / slice access)
// ═══════════════════════════════════════════════════════════════

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum HAAParamIdx {
    DT = 0,         // ms
    Inv_Cm = 1,     // 1/µF
    G_L = 2,        // mS/cm²
    E_L = 3,        // mV
    G_M = 4,        // mS/cm² — M-current
    E_K = 5,        // mV
    G_H = 6,        // mS/cm² — h-current
    E_H = 7,        // mV
    Inv_Tau_U = 8,  // 1/τ_u (ms⁻¹)
    Inv_Tau_H = 9,  // 1/τ_h (ms⁻¹)
    V_Half_U = 10,  // mV
    K_U = 11,       // slope
    V_Half_H = 12,  // mV
    K_H = 13,       // slope
    V_Thresh = 14,  // mV
    V_Reset = 15,   // mV
    Delta_U = 16,   // u bump on spike
}

impl From<HAAParamIdx> for usize {
    #[inline(always)] fn from(idx: HAAParamIdx) -> Self { idx as usize }
}

/// Hyperpolarization-activated adaptive bursting neuron.
///
/// # States (3)
/// - `v`: membrane potential (mV)
/// - `u`: M-current activation (0..1)
/// - `h`: h-current activation (0..1)
///
/// # Params (17)
/// - `dt`, `inv_cm`, `g_l`, `e_l`, `g_m`, `e_k`, `g_h`, `e_h`,
///   `inv_tau_u`, `inv_tau_h`, `v_half_u`, `k_u`, `v_half_h`,
///   `k_h`, `v_thresh`, `v_reset`, `delta_u`
#[derive(NeuronModel)]
pub struct HAAPModel {
    #[state] pub v: f32,
    #[state] pub u: f32,
    #[state] pub h: f32,

    #[param] pub dt: f32,
    #[param] pub inv_cm: f32,
    #[param] pub g_l: f32,
    #[param] pub e_l: f32,
    #[param] pub g_m: f32,
    #[param] pub e_k: f32,
    #[param] pub g_h: f32,
    #[param] pub e_h: f32,
    #[param] pub inv_tau_u: f32,
    #[param] pub inv_tau_h: f32,
    #[param] pub v_half_u: f32,
    #[param] pub k_u: f32,
    #[param] pub v_half_h: f32,
    #[param] pub k_h: f32,
    #[param] pub v_thresh: f32,
    #[param] pub v_reset: f32,
    #[param] pub delta_u: f32,
}

impl ComputeNeuron for HAAPModel {
    type State<S: SimdLane> = HAAPState<S>;
    type Params<S: SimdLane> = HAAPParams<S>;

    #[inline(always)]
    fn step<S: SimdLane>(
        state: &mut HAAPState<S>,
        params: &HAAPParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        let one = S::one();
        let dt = ctx.dt_simd();

        // ─── Ohmic currents: I = g·(V − E), subtracted from I_inj ──
        let i_leak = params.g_l * (state.v - params.e_l);
        let i_m    = params.g_m * state.u * (state.v - params.e_k);
        let i_h    = params.g_h * state.h * (state.v - params.e_h);

        let i_total = i_leak + i_m + i_h;
        let dv = (ctx.i_syn - i_total) * params.inv_cm;
        let v_new = state.v + dv * dt;

        // ─── Gate kinetics via pure SIMD sigmoids ──────────────────
        // u_∞ = 1/(1+exp(-k_u·(V − V_half_u)))  — opening on depolarization
        let u_arg = -params.k_u * (v_new - params.v_half_u);
        let u_inf = one / (one + u_arg.exp());
        let u_new = (state.u + dt * (u_inf - state.u) * params.inv_tau_u).clamp01();

        // h_∞ = 1/(1+exp(+k_h·(V − V_half_h)))  — opening on hyperpolarization
        let h_arg = params.k_h * (v_new - params.v_half_h);
        let h_inf = one / (one + h_arg.exp());
        let h_new = (state.h + dt * (h_inf - state.h) * params.inv_tau_h).clamp01();

        // ─── Spike detection & reset ───────────────────────────────
        let spike = v_new.cmp_ge(params.v_thresh);
        ctx.emit_spike(spike);

        state.v = spike.cond().then(params.v_reset).else_(v_new);
        let u_bump = (u_new + params.delta_u).clamp01();
        state.u = spike.cond().then(u_bump).else_(u_new);
        state.h = h_new;
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // физическая нотация: Ii — ток
    use super::*;
    use crate::core::population::run_neuron;

    fn make_vnc_params(n: usize, dt: f32) -> Vec<f32> {
        let inv_tau_u = 1.0 / 100.0;
        let inv_tau_h = 1.0 / 200.0;
        let mut p = vec![0.0f32; n * 17];
        for i in 0..n {
            p[HAAParamIdx::DT as usize * n + i] = dt;
            p[HAAParamIdx::Inv_Cm as usize * n + i] = 1.0;
            p[HAAParamIdx::G_L as usize * n + i] = 0.1;
            p[HAAParamIdx::E_L as usize * n + i] = -65.0;
            p[HAAParamIdx::G_M as usize * n + i] = 0.5;
            p[HAAParamIdx::E_K as usize * n + i] = -80.0;
            p[HAAParamIdx::G_H as usize * n + i] = 0.3;
            p[HAAParamIdx::E_H as usize * n + i] = -30.0;
            p[HAAParamIdx::Inv_Tau_U as usize * n + i] = inv_tau_u;
            p[HAAParamIdx::Inv_Tau_H as usize * n + i] = inv_tau_h;
            p[HAAParamIdx::V_Half_U as usize * n + i] = -35.0;
            p[HAAParamIdx::K_U as usize * n + i] = 5.0;
            p[HAAParamIdx::V_Half_H as usize * n + i] = -75.0;
            p[HAAParamIdx::K_H as usize * n + i] = 6.0;
            p[HAAParamIdx::V_Thresh as usize * n + i] = -40.0;
            p[HAAParamIdx::V_Reset as usize * n + i] = -55.0;
            p[HAAParamIdx::Delta_U as usize * n + i] = 0.02;
        }
        p
    }

    #[test] fn test_haap_spikes_with_current() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![15.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP should spike with I=15");
    }

    #[test] fn test_haap_silent_with_inhibition() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![-10.0; n]; let mut spikes = vec![0i32; n];
        for _ in 0..1000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HAAP should not spike with I=-10");
    }

    #[test] fn test_haap_non_aligned() {
        let n = 13; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n];
        let Ii = vec![15.0; n]; let mut spikes = vec![0i32; n];
        let mut total = 0i32;
        for _ in 0..2000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "HAAP non-aligned should spike");
    }

    #[test] fn test_haap_stress_silence() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n]; let Ii = vec![0.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(!u[i].is_nan(), "u[{i}] NaN step {step}");
                assert!(!h[i].is_nan(), "h[{i}] NaN step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "HAAP silence: no spikes");
    }

    #[test] fn test_haap_stress_explosion() {
        let n = 8; let dt = 0.05;
        let params = make_vnc_params(n, dt);
        let mut v = vec![-65.0; n]; let mut u = vec![0.0; n]; let mut h = vec![1.0; n];
        let mut dummy = vec![0.0; n]; let Ii = vec![1000.0; n]; let mut spikes = vec![0i32; n];
        for step in 0..10000 {
            unsafe { run_neuron::<HAAPModel>(
                v.as_mut_ptr(), u.as_mut_ptr(), h.as_mut_ptr(),
                dummy.as_mut_ptr(), dummy.as_mut_ptr(), dummy.as_mut_ptr(),
                Ii.as_ptr(), spikes.as_mut_ptr(), params.as_ptr(), n,
            ); }
            for i in 0..n {
                assert!(!v[i].is_nan(), "v[{i}] NaN step {step}");
                assert!(v[i].is_finite(), "v[{i}] non-finite step {step}");
                assert!(u[i].is_finite(), "u[{i}] non-finite step {step}");
                assert!(h[i].is_finite(), "h[{i}] non-finite step {step}");
            }
        }
    }
}
