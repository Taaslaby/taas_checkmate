use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Kv4 A-type Potassium Current (I_A)
///
///   I_A = g_A * m^3 * h * (V - E_K)
///
/// Fast transient K⁺ current. Activates quickly near threshold,
/// then inactivates within 10-50ms. Controls:
///   - Spike latency (delays first spike)
///   - Interspike interval (ISI) — spike frequency adaptation
///   - Dendritic backpropagation of action potentials
///   - Threshold for rebound spiking
///
/// Standard formulation (Connor & Stevens, 1971; Hoffman et al., 1997):
///   m⁰_inf(V) = 1 / (1 + exp(-(V + V_half_m) / k_m))
///   h⁰_inf(V) = 1 / (1 + exp((V + V_half_h) / k_h))
///   τ_m(V)    ≈ 1-2 ms (weakly voltage-dependent near threshold)
///   τ_h(V)    ≈ 15-80 ms (bell-shaped, peaks near -60 mV)
///
/// Default parameters (CA1 pyramidal somatodendritic):
///   g_A    ≈ 10-50 nS (very cell-dependent)
///   V_half_m ≈ -50 mV, k_m ≈ 20
///   V_half_h ≈ -80 mV, k_h ≈ 6
///   τ_m    ≈ 1.5 ms (simplified, scalar)
///   τ_h    ≈ 25 ms (simplified, scalar)
///   E_K    ≈ -85 mV
///
/// In Drosophila: Kv4 (Shal) is expressed in Kenyon cells (MB) and
/// medulla neurons.
///
/// Args:
///   i_reg: output current register (I_A)
///   v_reg: membrane potential register
///   m_reg: activation gate (updated in place)
///   h_reg: inactivation gate (updated in place)
///   g_max: maximal conductance (nS)
///   e_rev: reversal potential (≈ -85 mV for K⁺)
///   v_half_m: half-activation (≈ -50 mV)
///   k_m: slope for activation (≈ 20)
///   tau_m: activation time constant (≈ 1.5 ms)
///   v_half_h: half-inactivation (≈ -80 mV)
///   k_h: slope for inactivation (≈ 6)
///   tau_h: inactivation time constant (≈ 25 ms)
///   dt: timestep (ms)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_a_current(
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

    // ── Gate steady-states (sigmoids) ──

    // m_inf = 1 / (1 + exp(-(V - v_half_m) / k_m))
    let exp_arg_m = -(v - f32x8::splat(v_half_m)) / f32x8::splat(k_m);
    let exp_m = fast_exp_simd(exp_arg_m.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_m);

    // h_inf = 1 / (1 + exp(+(V - v_half_h) / k_h))  — positive sign = inactivates with depolarization
    let exp_arg_h = (v - f32x8::splat(v_half_h)) / f32x8::splat(k_h);
    let exp_h = fast_exp_simd(exp_arg_h.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let h_inf = one / (one + exp_h);

    // ── Time constants ──
    let tau_m_v = f32x8::splat(tau_m);
    let tau_h_v = f32x8::splat(tau_h);
    let tau_m_safe = tau_m_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_m_v);
    let tau_h_safe = tau_h_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_h_v);
    let dt_v = f32x8::splat(dt);

    // ── Exact exponential integration ──
    let decay_m = fast_exp_simd(-dt_v / tau_m_safe);
    let decay_h = fast_exp_simd(-dt_v / tau_h_safe);
    let m_new = m_inf + (m_old - m_inf) * decay_m;
    let h_new = h_inf + (h_old - h_inf) * decay_h;

    // Clamp gates to [0, 1]
    let m_clamped = m_new.max(zero).min(one);
    let h_clamped = h_new.max(zero).min(one);

    regs[m_reg as usize] = m_clamped;
    regs[h_reg as usize] = h_clamped;

    // I_A = g * m³ * h * (V - E_K)
    let m3 = m_clamped * m_clamped * m_clamped;
    regs[i_reg as usize] = g * m3 * h_clamped * (v - e);
}
