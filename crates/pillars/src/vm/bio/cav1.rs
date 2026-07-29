use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Cav1 (L-Type) High-Voltage Activated Ca²⁺ Current
///
///   I_CaL = g_CaL * m² * (V - E_Ca)
///
/// L-type Ca²⁺ channels activate at HIGH voltages (> -30 mV) and
/// inactivate SLOWLY (Ca²⁺-dependent, not voltage-dependent).
/// Unlike T-type, they do NOT produce rebound bursts.
///
/// Physiological roles:
///   - Excitation-transcription coupling (CREB, BDNF, gene expression)
///   - Dendritic Ca²⁺ spikes
///   - Muscle contraction (cardiac and skeletal)
///   - Synaptic plasticity (LTP in some synapses)
///
/// Pharmacology:
///   - Blocked by dihydropyridines (nifedipine, nimodipine)
///   - Blocked by verapamil, diltiazem
///   - Potentiated by Bay K8644
///
/// Default parameters:
///   V_half ≈ -10 mV, k_m ≈ 5  (HVA — activates near -30mV)
///   τ_m ≈ 2-5 ms (faster than T-type activation)
///   E_Ca ≈ +120 mV
///
/// Note: Ca²⁺-dependent inactivation is NOT modeled here (requires
/// separate Ca²⁺ binding dynamics). Use BioCalciumDynamics opcode
/// to track [Ca²⁺]i and modulate I_CaL accordingly.
#[inline(always)]
pub fn op_bio_cav1_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    m_reg: u8,
    g_max: f32,
    e_rev: f32,
    v_half: f32,
    k_m: f32,
    tau_m: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let m_old = regs[m_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let one = f32x8::splat(1.0);

    // m_inf = 1 / (1 + exp(-(V - v_half) / k_m))
    let exp_arg = -(v - f32x8::splat(v_half)) / f32x8::splat(k_m);
    let exp_val = fast_exp_simd(exp_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_val);

    // Exact exponential integration
    let tau_safe = f32x8::splat(tau_m).max(f32x8::splat(1e-8));
    let decay = fast_exp_simd(-f32x8::splat(dt) / tau_safe);
    let m_new = m_inf + (m_old - m_inf) * decay;

    let zero = f32x8::splat(0.0);
    let m_clamped = m_new.max(zero).min(one);
    regs[m_reg as usize] = m_clamped;

    // I_CaL = g * m² * (V - E_Ca)
    regs[i_reg as usize] = g * m_clamped * m_clamped * (v - e);
}
