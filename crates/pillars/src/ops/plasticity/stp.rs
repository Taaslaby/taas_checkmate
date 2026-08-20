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
use crate::ops::spike::spike_to_float;

/// One step of Tsodyks-Markram STP.
/// Returns (u_new, x_new).
///
/// # Spike format
/// Принимает SIMD-маску (NaN от `cmp_ge`), скалярную маску `-1.0` или
/// арифметический флаг `1.0` — как и остальные plasticity-опы.
/// ⚠️ Внутри НЕЛЬЗЯ блендить сырым `spike.cond()`: для флоата `1.0`
/// знаковый бит = 0 и SIMD-blend счёл бы спайк ложью (см. ops::spike::spike_to_float).
/// Поэтому сначала нормализуем через `spike_to_float`, потом строим маску.
#[inline(always)]
pub fn stp_step<S: SimdLane>(
    u: S,
    x: S,
    spike: S,
    u0: S,
    inv_tau_rec: S,
    inv_tau_facil: S,
    dt: S,
) -> (S, S) {
    let one = S::one();
    let zero = S::zero();

    let decay_rec = (-dt * inv_tau_rec).exp();
    let decay_facil = (-dt * inv_tau_facil).exp();

    // Нормализуем любой формат спайка в 1.0/0.0, затем строим корректную
    // SIMD-маску сравнением — сырой `spike.cond()` ломается на флоате 1.0.
    let fired = spike_to_float(spike).cmp_gt(zero).cond();

    // Facilitation: spike vs decay
    let u_spike = u + u0 * (one - u);
    let u_decay = u + (u0 - u) * (one - decay_facil);
    let u_new = fired.then(u_spike).else_(u_decay);

    // Depression: spike vs recovery
    let x_spike = x - u_new * x;
    let x_decay = x + (one - x) * (one - decay_rec);
    let x_new = fired.then(x_spike).else_(x_decay);

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

    /// Флоат-формат спайка (1.0) должен давать ТОТ ЖЕ результат, что и
    /// NaN-маска от cmp_ge. Раньше сырой spike.cond() молча трактовал 1.0
    /// как «нет спайка» (знаковый бит = 0 в vblendvps).
    #[test]
    fn test_stp_spike_float_format_matches_mask() {
        let u = f32x8::splat(0.5);
        let x = f32x8::splat(0.5);
        let u0 = f32x8::splat(0.2);
        let inv_tau_rec = f32x8::splat(1.0 / 100.0);
        let inv_tau_facil = f32x8::splat(1.0 / 20.0);
        let dt = f32x8::splat(1.0);

        let mask = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0)); // NaN-маска
        let (u_m, x_m) = stp_step(u, x, mask, u0, inv_tau_rec, inv_tau_facil, dt);

        let flt = f32x8::splat(1.0); // арифметический флаг
        let (u_f, x_f) = stp_step(u, x, flt, u0, inv_tau_rec, inv_tau_facil, dt);

        assert!(
            (u_m.to_array()[0] - u_f.to_array()[0]).abs() < 1e-6,
            "u: mask {:.4} != float {:.4}",
            u_m.to_array()[0],
            u_f.to_array()[0]
        );
        assert!(
            (x_m.to_array()[0] - x_f.to_array()[0]).abs() < 1e-6,
            "x: mask {:.4} != float {:.4}",
            x_m.to_array()[0],
            x_f.to_array()[0]
        );
        // И оба должны реально сработать как спайк:
        assert!(u_f.to_array()[0] > 0.5, "float spike must facilitate");
        assert!(x_f.to_array()[0] < 0.5, "float spike must depress");
    }

    /// Скалярная маска -1.0 (f32-лейн) тоже считается спайком.
    #[test]
    fn test_stp_scalar_neg_mask() {
        let (u_new, x_new) = stp_step(
            0.5f32,
            0.5f32,
            -1.0f32,
            0.2f32,
            1.0 / 100.0,
            1.0 / 20.0,
            1.0f32,
        );
        assert!(u_new > 0.5, "scalar -1.0 mask must facilitate: {u_new}");
        assert!(x_new < 0.5, "scalar -1.0 mask must depress: {x_new}");
    }
}
