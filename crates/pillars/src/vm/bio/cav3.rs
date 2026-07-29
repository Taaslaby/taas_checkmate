use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Cav3 (T-Type) Low-Voltage Activated Ca²⁺ Current
///
///   I_CaT = g_CaT * m² * h * (V - E_Ca)
///
/// T-type Ca²⁺ channels activate at LOW voltages (> -60 mV) and
/// inactivate rapidly. They produce LOW-THRESHOLD Ca²⁺ SPIKES and
/// REBOUND BURSTING after hyperpolarization.
///
/// Key properties:
///   - "Window current" at -75 to -55 mV — tonic Ca²⁺ influx near rest
///   - Inactivated at rest (-65mV); requires hyperpolarization to
///     recover from inactivation BEFORE depolarization produces a burst
///   - Kinetics 2-3x faster than L-type
///
/// Physiological roles:
///   - Thalamic rebound bursts (sleep spindles, absence epilepsy)
///   - Pacemaking in some neurons
///   - Dendritic Ca²⁺ events (small, local)
///
/// Pharmacology:
///   - Blocked by mibefradil (partially selective)
///   - Blocked by Ni²⁺ (insensitive to dihydropyridines)
///
/// Default parameters (Huguenard & McCormick, 1992):
///   m: V_half ≈ -55 mV, k_m ≈ 7.4, τ_m ≈ 1-3 ms
///   h: V_half ≈ -80 mV, k_h ≈ 5,   τ_h ≈ 20-50 ms
///   E_Ca ≈ +120 mV
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_cav3_current(
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

    // m_inf activation: activates with depolarization
    let exp_m = fast_exp_simd((-(v - f32x8::splat(v_half_m)) / f32x8::splat(k_m)).max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_m);

    // h_inf inactivation: INACTIVATES with depolarization (positive exp sign)
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

    // I_CaT = g * m² * h * (V - E_Ca)
    regs[i_reg as usize] = g * m_cl * m_cl * h_cl * (v - e);
}
