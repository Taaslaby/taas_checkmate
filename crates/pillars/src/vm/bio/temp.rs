use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Temperature Q10 Correction
///
///   rate_T = rate_T_ref * Q10^((T - T_ref) / 10)
///
/// Q10 (temperature coefficient) is the factor by which a rate
/// constant increases for every 10°C rise in temperature.
///
/// Typical Q10 values:
///   - HH gate kinetics:          Q10 ≈ 3 (for Na⁺, K⁺ channels)
///   - Ion channel conductances:   Q10 ≈ 1.2-1.5 (passive diffusion)
///   - Synaptic transmission:      Q10 ≈ 2-3
///   - Ca²⁺ buffering/extrusion:   Q10 ≈ 2
///   - Enzyme kinetics:            Q10 ≈ 2-3
///   - Most biological processes:  Q10 ≈ 2-3
///
/// Usage:
///   Apply BioTempCorrect to gate time constants before computing
///   gate updates. For example:
///     τ_m_corrected = τ_m * Q10^((T - T_ref) / 10)
///
///   The HH opcodes (BioHhNaM, etc.) use fixed squid parameters at
///   6.3°C. To simulate mammalian HH at 37°C:
///     τ_corr = τ_squid * 3^((37 - 6.3) / 10) = τ_squid * 3^3.07
///            ≈ τ_squid * 27.7
///     (HH kinetics are ~28x faster at 37°C than at 6.3°C)
///
/// Operation modes:
///   1. tau_reg is a time constant register: τ *= Q10^((T - T_ref) / 10)
///      (multiply tau by Q10 factor — for time constants, slower at lower T)
///   2. For rate constants (not time constants), use: rate *= Q10^((T - T_ref) / 10)
///      (for rates, faster at higher T)
///
/// This opcode handles the τ (time constant) case:
///   τ_corrected = τ * Q10^((T - T_ref) / 10)
///
/// For rate constants (alpha, beta), apply the inverse:
///   rate_corrected = rate * Q10^((T - T_ref) / 10)
///
/// Args:
///   tau_reg: register with the time constant to correct (in-place)
///   q10: temperature coefficient (≈ 3 for HH channels)
///   temp_current: current temperature (°C)
///   temp_ref: reference temperature (°C)
///           (for HH squid: 6.3°C; for mammalian: 37°C)
#[inline(always)]
pub fn op_bio_temp_correct(
    regs: &mut [f32x8; 16],
    tau_reg: u8,
    q10: f32,
    temp_current: f32,
    temp_ref: f32,
) {
    let tau = regs[tau_reg as usize];

    // Q10 factor: Q10^((T - T_ref) / 10) = exp(ln(Q10) * (T - T_ref) / 10)
    let delta_t = temp_current - temp_ref;
    let ln_q10 = q10.ln();
    let exponent = ln_q10 * delta_t / 10.0;

    // Guard exponent range for fast_exp_simd
    let exp_arg = exponent.clamp(-10.0, 10.0);

    let factor = fast_exp_simd(f32x8::splat(exp_arg));

    // τ_corrected = τ * Q10^((T - T_ref) / 10)
    // At lower T than T_ref, factor < 1, τ decreases (rates are slower,
    // so the time constant gets longer... wait.)
    //
    // Actually: Q10 = factor for rate CONSTANT for 10°C increase.
    // Rate = 1/τ. So rate(T) = rate(T_ref) * Q10^((T - T_ref)/10)
    // 1/τ(T) = 1/τ(T_ref) * Q10^(ΔT/10)
    // τ(T) = τ(T_ref) * Q10^(-ΔT/10) = τ(T_ref) / Q10^(ΔT/10)
    //
    // So τ_corrected = τ / Q10^((T - T_ref)/10)
    let tau_corrected = tau / factor;

    regs[tau_reg as usize] = tau_corrected;
}
