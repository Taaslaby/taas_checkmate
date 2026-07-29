use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Kv7 / KCNQ M-Current (I_M)
///
///   I_M = g_M * m * (V - E_K)
///
/// A slow, non-inactivating K⁺ current that activates near spike threshold.
/// Called "M-current" because it's suppressed by muscarinic acetylcholine
/// receptor activation.
///
/// Biological role:
///   - Controls neuronal excitability (subthreshold "brake")
///   - Spike frequency adaptation (slow time constant 50-200ms)
///   - Target of retigabine / ezogabine (anti-epileptic, activates Kv7)
///   - Target of XE991 (blocker, increases excitability)
///
/// Standard formulation (Wang & McKinnon, 1995; Adams et al., 1982):
///   m_inf(V) = 1 / (1 + exp(-(V + V_half) / k))
///   τ_m(V)   ≈ 50-200 ms, simplified to scalar (~100 ms at 37°C)
///
/// Default parameters:
///   g_M    ≈ 0.5-5 nS (small conductance, big effect)
///   V_half ≈ -35 mV  (activates near spike threshold)
///   k      ≈ 10
///   τ_m    ≈ 100 ms  (slow!)
///   E_K    ≈ -85 mV
///
/// In Drosophila: Kv7 (KCNQ) is expressed in motor neurons and
/// peripheral nervous system — regulates locomotion and rhythmic activity.
///
/// Args:
///   i_reg: output current register (I_M)
///   v_reg: membrane potential register
///   m_reg: activation gate (updated in place)
///   g_max: maximal conductance (nS)
///   e_rev: reversal potential (≈ -85 mV for K⁺)
///   v_half: half-activation (≈ -35 mV)
///   k: slope factor (≈ 10)
///   tau_m: time constant (≈ 100 ms)
///   dt: timestep (ms)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_m_current(
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

    // Guard tau against zero
    let tau_safe = f32x8::splat(tau_m).cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), f32x8::splat(tau_m));

    // Exact exponential integration
    let decay = fast_exp_simd(-f32x8::splat(dt) / tau_safe);
    let m_new = m_inf + (m_old - m_inf) * decay;

    // Clamp m to [0, 1]
    let zero = f32x8::splat(0.0);
    let m_clamped = m_new.max(zero).min(one);

    regs[m_reg as usize] = m_clamped;

    // I_M = g * m * (V - E_K)
    regs[i_reg as usize] = g * m_clamped * (v - e);
}
