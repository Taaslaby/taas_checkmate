use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Kv3 (Shaw) High-Threshold Fast-Spiking K⁺ Current
///
///   I_Kv3 = g_Kv3 * m^4 * (V - E_K)
///
/// Kv3 channels activate at HIGH voltages (> -10 mV) with VERY fast
/// kinetics (τ < 1ms). This enables fast-spiking (FS) interneurons
/// to fire at >200 Hz without spike broadening.
///
/// Crucially, Kv3 has NO INACTIVATION — it stays open as long as the
/// membrane is depolarized, enabling ultra-fast repolarization.
///
/// Distribution:
///   - Parvalbumin-positive FS interneurons (cortex, hippocampus)
///   - Auditory brainstem (MNTB) — precise timing
///   - Globus pallidus (basal ganglia)
///   - Drosophila: expressed in fast motor neurons and visual system
///
/// Parameters (Rudy & McBain, 2001):
///   V_half ≈ -15 mV,  k ≈ 10
///   τ_m    ≈ 0.5 ms (nearly voltage-independent)
///   E_K    ≈ -85 mV
#[inline(always)]
pub fn op_bio_kv3_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    m_reg: u8,
    g_max: f32,
    e_rev: f32,
    v_half: f32,
    k: f32,
    tau_m: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let m_old = regs[m_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let one = f32x8::splat(1.0);

    // m_inf = 1 / (1 + exp(-(V - v_half) / k))
    let exp_arg = -(v - f32x8::splat(v_half)) / f32x8::splat(k);
    let exp_val = fast_exp_simd(exp_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_val);

    // Exact exponential integration for m gate
    let tau_safe = f32x8::splat(tau_m).max(f32x8::splat(1e-8));
    let decay = fast_exp_simd(-f32x8::splat(dt) / tau_safe);
    let m_new = m_inf + (m_old - m_inf) * decay;

    let zero = f32x8::splat(0.0);
    let m_clamped = m_new.max(zero).min(one);
    regs[m_reg as usize] = m_clamped;

    // I_Kv3 = g * m^4 * (V - E_K)
    let m4 = m_clamped * m_clamped * m_clamped * m_clamped;
    regs[i_reg as usize] = g * m4 * (v - e);
}
