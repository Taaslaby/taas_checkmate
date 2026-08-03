//! # STP — Short-Term Plasticity (Tsodyks-Markram)
//!
//! ```text
//! Facilitation (u):
//!   u' = u + u₀ · (1 - u)  on spike
//!   u' = u + (u₀ - u) · (1 - exp(-dt/τ_facil))  decay
//!
//! Depression (x):
//!   x' = x - u · x  on spike
//!   x' = x + (1 - x) · (1 - exp(-dt/τ_rec))  recovery
//!
//! Synaptic efficacy: g_eff = g_max · u · x
//! ```
//!
//! u = utilization (release probability), x = available resources (0..1).
//! τ_facil ≈ τ_rec / 5 (facilitation decays ~5× faster than depression).

use crate::core::lane::SimdLane;

/// One step of Tsodyks-Markram STP.
/// Returns (u_new, x_new).
#[inline(always)]
pub fn stp_step<S: SimdLane>(u: S, x: S, spike: S, u0: S, inv_tau_rec: S, inv_tau_facil: S, dt: S) -> (S, S) {
    let one = S::one();
    let zero = S::zero();

    let decay_rec = (-dt * inv_tau_rec).exp();
    let decay_facil = (-dt * inv_tau_facil).exp();

    // Facilitation: spike vs decay
    let u_spike = u + u0 * (one - u);
    let u_decay = u + (u0 - u) * (one - decay_facil);
    let u_new = spike.cond().then(u_spike).else_(u_decay);

    // Depression: spike vs recovery
    let x_spike = x - u_new * x;
    let x_decay = x + (one - x) * (one - decay_rec);
    let x_new = spike.cond().then(x_spike).else_(x_decay);

    (u_new.max(zero).min(one), x_new.max(zero).min(one))
}

/// Synaptic efficacy: `g_eff = g_max · u · x`
#[inline(always)]
pub fn stp_efficacy<S: SimdLane>(u: S, x: S, g_max: S) -> S {
    g_max * u * x
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    #[test]
    fn test_stp_no_spike_decay() {
        let u = f32x8::splat(0.5);
        let x = f32x8::splat(0.5);
        let spike = f32x8::splat(0.0);
        let u0 = f32x8::splat(0.2);
        let inv_tau_rec = f32x8::splat(1.0 / 100.0);
        let inv_tau_facil = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);
        let (u_new, x_new) = stp_step(u, x, spike, u0, inv_tau_rec, inv_tau_facil, dt);
        // No spike → u decays toward u0, x recovers toward 1
        assert!(u_new.to_array()[0] < 0.5, "u should decay without spike");
        assert!(x_new.to_array()[0] > 0.5, "x should recover without spike");
    }

    #[test]
    fn test_stp_spike_depression() {
        let u = f32x8::splat(0.5);
        let x = f32x8::splat(0.5);
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // true
        let u0 = f32x8::splat(0.2);
        let inv_tau_rec = f32x8::splat(1.0 / 100.0);
        let inv_tau_facil = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);
        let (u_new, x_new) = stp_step(u, x, spike, u0, inv_tau_rec, inv_tau_facil, dt);
        // On spike: u increases, x decreases (depression)
        assert!(u_new.to_array()[0] > 0.5, "u should increase on spike");
        assert!(x_new.to_array()[0] < 0.5, "x should decrease on spike");
    }
}
