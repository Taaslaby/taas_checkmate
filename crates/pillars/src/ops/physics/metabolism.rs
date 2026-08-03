//! # Metabolism — Energy Metabolism and Ion Homeostasis
//!
//! ## ATP Dynamics
//! ```text
//! d[ATP]/dt = r_restore · (ATP₀ - ATP) - r_consume · Na_influx
//! ```
//! ATP расходуется пропорционально числу ионов Na⁺, вошедших через спайки.
//! Восстанавливается митохондриями с постоянной времени τ_restore.
//!
//! ## Na⁺/K⁺-ATPase Pump
//! ```text
//! I_pump = -I_max · f(ATP) · f(Na⁺, K⁺)
//! f(ATP) = [ATP] / ([ATP] + K_m_ATP)
//! f(Na⁺) = 1 / (1 + (K_m_Na / [Na⁺]_i)³)
//! f(K⁺)  = [K⁺]_o / ([K⁺]_o + K_m_K)
//! ```
//! Электрогенный насос: 3 Na⁺ out, 2 K⁺ in → чистый -1 заряд → гиперполяризация.
//!
//! ## Ion Concentration Dynamics
//! ```text
//! d[Na⁺]_i / dt = -I_Na · scale_na - pump_Na_extrusion
//! ```
//!
//! # Convention
//! Все функции `#[inline(always)]`, принимают и возвращают `S: SimdLane`.
//! Никаких делений в hot path — inv_tau, inv_km предрасчитаны.

use crate::core::lane::SimdLane;
use crate::ops::spike::spike_to_float;

// ═══════════════════════════════════════════════════════════════
//  ATP Dynamics
// ═══════════════════════════════════════════════════════════════

/// ATP depletion proportional to Na⁺ influx during spikes.
///
/// ```text
/// ATP' = ATP + dt · (r_restore · (ATP₀ - ATP) - r_consume · na_influx)
/// ```
#[inline(always)]
pub fn atp_step<S: SimdLane>(
    atp: S, na_influx: S,
    k_restore: S,      // 1/τ_restore
    atp_baseline: S,   // ATP₀
    k_consume: S,      // ATP расход на единицу Na⁺
    dt: S,
) -> S {
    let restore = k_restore * (atp_baseline - atp);
    let consume = k_consume * na_influx;
    let new = atp + (restore - consume) * dt;
    new.max(S::zero()).min(atp_baseline)
}

/// Spike-triggered Na⁺ influx (approximate).
///
/// Каждый спайк пропускает ~квант Na⁺ через voltage-gated Na⁺ каналы.
/// Использует `spike_to_float` из spike.rs — принимает маску (-1.0) или float (1.0).
#[inline(always)]
pub fn na_influx_from_spike<S: SimdLane>(spike_mask: S, alpha_na: S) -> S {
    spike_to_float(spike_mask) * alpha_na
}

// ═══════════════════════════════════════════════════════════════
//  Na⁺/K⁺-ATPase Pump Current
// ═══════════════════════════════════════════════════════════════

/// Na⁺/K⁺ pump current (electrogenic, always hyperpolarizing).
///
/// Returns: pump current (negative → hyperpolarizing).
/// Вычитается из I_inj: `I_total = I_inj - i_ion - i_pump`.
///
/// Формула без лишних `recip()`:
/// ```text
/// f(ATP) = (ATP · inv_Km_ATP) / (ATP · inv_Km_ATP + 1)
/// f(Na)  = 1 / (1 + ([Na] · inv_Km_Na)⁻³)
/// f(K)   = [K] / ([K] + Km_K)
/// ```
#[inline(always)]
pub fn pump_current<S: SimdLane>(
    atp: S, na_i: S, k_o: S,
    i_pump_max: S,
    inv_km_atp: S,  // 1 / K_m_ATP
    inv_km_na: S,   // 1 / K_m_Na
    km_k: S,        // K_m_K (≈ 0.33)
) -> S {
    // f_ATP = ATP / (ATP + Km) = (ATP·inv_km) / (ATP·inv_km + 1)
    // Только одно деление вместо двух (без recip!)
    let atp_ratio = atp * inv_km_atp;
    let f_atp = atp_ratio / (atp_ratio + S::one());

    // f_Na = 1 / (1 + (Km_Na/[Na])³)
    // [Na]/Km = na_i * inv_km_na. inverse → (1 / (na_i * inv_km_na))³
    let na_ratio = na_i * inv_km_na;
    let inv_na_ratio_cubed = (na_ratio + S::splat(1e-12)).recip().cube();
        // (Km/[Na])³, clamped to avoid div-by-zero
    let f_na = S::one() / (S::one() + inv_na_ratio_cubed);

    // f_K = [K] / ([K] + Km_K) — простой Michaelis
    let f_k = k_o / (k_o + km_k);

    let pump = i_pump_max * f_atp * f_na * f_k;
    -pump  // всегда гиперполяризующий
}

// ═══════════════════════════════════════════════════════════════
//  Ion Concentration Dynamics
// ═══════════════════════════════════════════════════════════════

/// Na⁺ concentration step (intracellular).
///
/// ```text
/// [Na⁺]_i' = [Na⁺]_i + dt · (i_na_flux · scale - pump_Na_extrusion)
/// ```
#[inline(always)]
pub fn na_concentration_step<S: SimdLane>(
    na_i: S,
    i_na_flux: S,     // Na⁺ ток (NEURON convention: inward = negative)
    scale_na: S,      // конверсия ток→концентрация
    pump_extrusion: S, // экструзия Na⁺ насосом
    dt: S,
) -> S {
    // inward (negative) → -i_na_flux даёт положительный приток
    let influx = (-i_na_flux).max(S::zero()) * scale_na;
    let new = na_i + (influx - pump_extrusion) * dt;
    new.max(S::zero()).min(S::splat(1.0))
}

/// K⁺ concentration step (extracellular).
///
/// ```text
/// [K⁺]_o' = [K⁺]_o + dt · (i_k_flux · scale_k + pump_k_influx)
/// ```
#[inline(always)]
pub fn k_concentration_step<S: SimdLane>(
    k_o: S,
    i_k_flux: S,      // K⁺ ток (outward = positive)
    scale_k: S,       // конверсия ток→концентрация
    pump_k_influx: S,  // K⁺ influx через насос
    dt: S,
) -> S {
    let efflux = i_k_flux.max(S::zero()) * scale_k;
    let new = k_o + (efflux - pump_k_influx) * dt;
    new.max(S::zero()).min(S::splat(1.0))
}

// ═══════════════════════════════════════════════════════════════
//  Ornstein-Uhlenbeck Noise (colored noise)
// ═══════════════════════════════════════════════════════════════

/// Ornstein-Uhlenbeck noise step.
///
/// ```text
/// x' = x + θ · (μ - x) · dt + σ · √(dt) · ξ
/// ```
/// Где ξ — стандартный нормальный шум (Box-Muller из LCG).
///
/// LCG без `%`: `seed = (a * seed + c) - ((a * seed + c) / m).trunc() * m`
/// (использует SIMD `trunc()`, которая доступна в SimdLane).
///
/// Returns: (new_noise, new_seed) — seed обновлён на 2 шага LCG.
#[inline(always)]
pub fn ou_noise_step<S: SimdLane>(
    x: S, theta: S, mu: S, sigma: S, dt: S, seed: &mut S,
) -> S {
    let a = S::splat(1664525.0f32);
    let c = S::splat(1013904223.0f32);
    let m = S::splat(4294967296.0f32);

    // LCG без `%`: x' = a·x + c, затем frac = x' - trunc(x'/m)·m
    let next = |s: S| -> S {
        let raw = a * s + c;
        raw - (raw / m).trunc() * m
    };

    let u1 = *seed;
    *seed = next(u1);
    let u2 = *seed;
    *seed = next(*seed);  // advance one more

    // Box-Muller: scale to (0,1) avoiding log(0)
    let eps = S::splat(1e-10);
    let one_minus_eps = S::splat(1.0 - 1e-10);
    let u1_pos = (u1 / m).max(eps).min(one_minus_eps);
    let u2_pos = (u2 / m).max(eps).min(one_minus_eps);

    // ξ = √(-2·ln(u1)) · cos(2π·u2)
    let tau = S::splat(std::f32::consts::TAU);
    let radius = (-S::splat(2.0) * u1_pos.ln()).sqrt();
    let normal = radius * (tau * u2_pos).cos();

    // OU step: drift + diffusion
    let drift = theta * (mu - x) * dt;
    let diffusion = sigma * dt.sqrt() * normal;
    x + drift + diffusion
}

/// Simplified uniform noise [-1, 1].
///
/// Быстрее Box-Muller, подходит для грубой стохастики.
#[inline(always)]
pub fn uniform_noise_step<S: SimdLane>(seed: &mut S, amplitude: S) -> S {
    let a = S::splat(1664525.0f32);
    let c = S::splat(1013904223.0f32);
    let m = S::splat(4294967296.0f32);

    let raw = a * *seed + c;
    *seed = raw - (raw / m).trunc() * m;

    let u = *seed / m;  // [0, 1)
    (u * S::splat(2.0) - S::one()) * amplitude
}

/// Stochastic channel noise: Fox & Lu 1994 approximation.
///
/// ```text
/// m_noisy = m + σ · ξ
/// σ² = 2 · m · (1-m) / (N · τ) · dt
/// ```
/// Использует Box-Muller (standard normal), а не uniform,
/// для корректной дисперсии.
#[inline(always)]
pub fn channel_noise<S: SimdLane>(
    m: S,
    noise_seed: &mut S,
    n_channels: S,     // число каналов в кластере
    inv_tau: S,        // 1/τ_act
    dt: S,
) -> S {
    let one = S::one();
    let m_clamped = m.max(S::zero()).min(one);

    // σ² = 2 · m · (1-m) / (N · τ) · dt
    let variance = S::splat(2.0) * m_clamped * (one - m_clamped) / n_channels * inv_tau * dt;
    let sigma = variance.sqrt();

    // Box-Muller для standard normal
    let a = S::splat(1664525.0f32);
    let c = S::splat(1013904223.0f32);
    let m_val = S::splat(4294967296.0f32);

    let next = |s: S| -> S {
        let raw = a * s + c;
        raw - (raw / m_val).trunc() * m_val
    };

    let u1 = *noise_seed;
    *noise_seed = next(u1);
    let u2 = *noise_seed;
    *noise_seed = next(*noise_seed);

    let eps = S::splat(1e-10);
    let one_minus_eps = S::splat(1.0 - 1e-10);
    let u1_pos = (u1 / m_val).max(eps).min(one_minus_eps);
    let u2_pos = (u2 / m_val).max(eps).min(one_minus_eps);

    let tau = S::splat(std::f32::consts::TAU);
    let radius = (-S::splat(2.0) * u1_pos.ln()).sqrt();
    let normal = radius * (tau * u2_pos).cos();

    let noisy = m_clamped + sigma * normal;
    noisy.max(S::zero()).min(one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    // ─── ATP tests ──────────────────────────────────────────

    #[test]
    fn test_atp_resting() {
        let atp = f32x8::splat(1.0);
        let na = f32x8::splat(0.0);
        let k_restore = f32x8::splat(0.01);
        let baseline = f32x8::splat(1.0);
        let k_consume = f32x8::splat(0.05);
        let dt = f32x8::splat(1.0);
        let new = atp_step(atp, na, k_restore, baseline, k_consume, dt);
        assert!((new.to_array()[0] - 1.0).abs() < 1e-6,
            "ATP should stay at baseline, got {:.4}", new.to_array()[0]);
    }

    #[test]
    fn test_atp_depletion() {
        let atp = f32x8::splat(1.0);
        let na = f32x8::splat(10.0);
        let k_restore = f32x8::splat(0.01);
        let baseline = f32x8::splat(1.0);
        let k_consume = f32x8::splat(0.05);
        let dt = f32x8::splat(1.0);
        let new = atp_step(atp, na, k_restore, baseline, k_consume, dt);
        assert!((new.to_array()[0] - 0.5).abs() < 0.01,
            "ATP should deplete to ~0.5, got {:.4}", new.to_array()[0]);
    }

    #[test]
    fn test_atp_no_negative() {
        let atp = f32x8::splat(0.0);
        let na = f32x8::splat(100.0);
        let k_restore = f32x8::splat(0.0);
        let baseline = f32x8::splat(1.0);
        let k_consume = f32x8::splat(1.0);
        let dt = f32x8::splat(1.0);
        let new = atp_step(atp, na, k_restore, baseline, k_consume, dt);
        assert!(new.to_array()[0] >= 0.0,
            "ATP should never go negative, got {:.6}", new.to_array()[0]);
    }

    // ─── Na influx test ─────────────────────────────────────

    #[test]
    fn test_na_influx_on_spike() {
        let spike = f32x8::splat(1.0);
        let alpha = f32x8::splat(0.1);
        let flux = na_influx_from_spike(spike, alpha);
        assert!((flux.to_array()[0] - 0.1).abs() < 1e-6,
            "Na influx should be 0.1, got {:.4}", flux.to_array()[0]);
    }

    #[test]
    fn test_na_influx_mask() {
        let spike = f32x8::splat(-1.0);
        let alpha = f32x8::splat(0.1);
        let flux = na_influx_from_spike(spike, alpha);
        assert!((flux.to_array()[0] - 0.1).abs() < 1e-6,
            "mask -1.0 should also give 0.1, got {:.4}", flux.to_array()[0]);
    }

    #[test]
    fn test_no_na_influx_no_spike() {
        let spike = f32x8::splat(0.0);
        let alpha = f32x8::splat(0.1);
        let flux = na_influx_from_spike(spike, alpha);
        assert!((flux.to_array()[0]).abs() < 1e-6,
            "no spike → no influx, got {:.4}", flux.to_array()[0]);
    }

    // ─── Pump current tests ─────────────────────────────────

    #[test]
    fn test_pump_hyperpolarizing() {
        let atp = f32x8::splat(1.0);
        let na_i = f32x8::splat(0.1);
        let k_o = f32x8::splat(0.05);
        let i_max = f32x8::splat(1.0);
        let inv_km_atp = f32x8::splat(10.0);
        let inv_km_na = f32x8::splat(50.0);
        let km_k = f32x8::splat(0.33);
        let i = pump_current(atp, na_i, k_o, i_max, inv_km_atp, inv_km_na, km_k);
        assert!(i.to_array()[0] < 0.0,
            "pump current should be negative, got {:.4}", i.to_array()[0]);
    }

    #[test]
    fn test_pump_depends_on_atp() {
        let atp_low = f32x8::splat(0.01);
        let atp_high = f32x8::splat(1.0);
        let na_i = f32x8::splat(0.1);
        let k_o = f32x8::splat(0.05);
        let i_max = f32x8::splat(1.0);
        let inv_km_atp = f32x8::splat(10.0);
        let inv_km_na = f32x8::splat(50.0);
        let km_k = f32x8::splat(0.33);
        let i_low = pump_current(atp_low, na_i, k_o, i_max, inv_km_atp, inv_km_na, km_k);
        let i_high = pump_current(atp_high, na_i, k_o, i_max, inv_km_atp, inv_km_na, km_k);
        assert!(i_low.to_array()[0] > i_high.to_array()[0],
            "low ATP → weaker pump: {:.4} vs {:.4}",
            i_low.to_array()[0], i_high.to_array()[0]);
    }

    // ─── Ion concentration tests ────────────────────────────

    #[test]
    fn test_na_concentration_rises_on_inward_current() {
        let na_i = f32x8::splat(0.1);
        let i_na = f32x8::splat(-1.0);
        let scale = f32x8::splat(0.01);
        let pump = f32x8::splat(0.001);
        let dt = f32x8::splat(1.0);
        let new = na_concentration_step(na_i, i_na, scale, pump, dt);
        assert!(new.to_array()[0] > na_i.to_array()[0],
            "inward current → na_i rises: {:.6} > {:.6}",
            new.to_array()[0], na_i.to_array()[0]);
    }

    // ─── Noise tests ────────────────────────────────────────

    #[test]
    fn test_ou_noise_runs() {
        let x = f32x8::splat(0.0);
        let theta = f32x8::splat(0.1);
        let mu = f32x8::splat(0.0);
        let sigma = f32x8::splat(0.01);
        let dt = f32x8::splat(0.1);
        let mut seed = f32x8::splat(12345.0);
        let xn = ou_noise_step(x, theta, mu, sigma, dt, &mut seed);
        for i in 0..8 {
            assert!(xn.to_array()[i].is_finite(),
                "OU noise should be finite at idx {}", i);
        }
        let seed_diff = seed.to_array()[0] - 12345.0;
        assert!(seed_diff.abs() > 0.1,
            "seed should change after step");
    }

    #[test]
    fn test_channel_noise_bounds() {
        let m = f32x8::splat(0.5);
        let mut seed = f32x8::splat(9999.0);
        let n_ch = f32x8::splat(100.0);
        let inv_tau = f32x8::splat(1.0);
        let dt = f32x8::splat(0.1);
        let m_noisy = channel_noise(m, &mut seed, n_ch, inv_tau, dt);
        for i in 0..8 {
            let val = m_noisy.to_array()[i];
            assert!(val.is_finite(),
                "channel noise should be finite at idx {}", i);
            assert!((0.0..=1.0).contains(&val),
                "channel noise should stay in [0,1], got {:.4} at idx {}", val, i);
        }
    }
}
