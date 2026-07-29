use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Kv1 (Shaker) Low-Threshold K⁺ Current
///
///   I_Kv1 = g_Kv1 * m^3 * h * (V - E_K)
///
/// Low-threshold, rapidly activating and inactivating K⁺ current.
/// Activates near -50 mV (well below spike threshold), controlling
/// the spike threshold and AP initiation.
///
/// Unlike Kv4 (A-type), Kv1 inactivates more slowly and from
/// a more depolarized state.
///
/// Distribution:
///   - Axon initial segment (AIS) — spike initiation
///   - Presynaptic terminals — controls NT release
///   - Drosophila: Shaker mutation causes leg shaking under
///     ether anesthesia (hence the name!)
///
/// Parameters:
///   V_half_m ≈ -40 mV, k_m ≈ 15     (low threshold!)
///   V_half_h ≈ -55 mV, k_h ≈ 5
///   τ_m ≈ 2-5 ms
///   τ_h ≈ 10-50 ms
///   E_K ≈ -85 mV
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_kv1_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    m_reg: u8,
    h_reg: u8,
    g_max: f32,
    e_rev: f32,
    v_half_m: f32,
    k_m: f32,
    tau_m: f32,
    v_half_h: f32,
    k_h: f32,
    tau_h: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let m_old = regs[m_reg as usize];
    let h_old = regs[h_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);

    // m_inf = 1 / (1 + exp(-(V - v_half_m) / k_m))
    let exp_m = fast_exp_simd((-(v - f32x8::splat(v_half_m)) / f32x8::splat(k_m)).max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_m);

    // h_inf = 1 / (1 + exp((V - v_half_h) / k_h))  — positive sign = inactivation
    let exp_h = fast_exp_simd(((v - f32x8::splat(v_half_h)) / f32x8::splat(k_h)).max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let h_inf = one / (one + exp_h);

    // Exact exponential integration
    let tau_m_safe = f32x8::splat(tau_m).max(f32x8::splat(1e-8));
    let tau_h_safe = f32x8::splat(tau_h).max(f32x8::splat(1e-8));
    let decay_m = fast_exp_simd(-f32x8::splat(dt) / tau_m_safe);
    let decay_h = fast_exp_simd(-f32x8::splat(dt) / tau_h_safe);
    let m_new = m_inf + (m_old - m_inf) * decay_m;
    let h_new = h_inf + (h_old - h_inf) * decay_h;

    let m_cl = m_new.max(zero).min(one);
    let h_cl = h_new.max(zero).min(one);
    regs[m_reg as usize] = m_cl;
    regs[h_reg as usize] = h_cl;

    // I_Kv1 = g * m^3 * h * (V - E_K)
    let m3 = m_cl * m_cl * m_cl;
    regs[i_reg as usize] = g * m3 * h_cl * (v - e);
}
