use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// HCN (Hyperpolarization-Activated Cyclic Nucleotide-Gated) Current
///
///   I_h = g_h * m * (V - E_h)
///
/// The "h-current" or "pacemaker current". UNIQUE among voltage-gated
/// channels: ACTIVATED by HYPERPOLARIZATION (negative V). At rest (-65mV),
/// I_h is partially activated; it increases with hyperpolarization.
///
/// Standard formulation (Hutcheon & Yarom, 2000; Huguenard, 1996):
///   m_inf(V) = 1 / (1 + exp((V + V_half) / k))
///   τ_m(V)   ~ bell-shaped, peak near V_half
///
/// Default parameters (CA1 pyramidal neuron, ~37°C):
///   V_half ≈ -75 mV, k = -5.5 (inverted sigmoid)
///   τ_m    ≈ 10-30 ms (simplified to piecewise or constant)
///   E_h    ≈ -30 mV (permeable to K⁺ and Na⁺)
///
/// Since I_h has slow kinetics (τ >> dt), exact exponential integration
/// is used for gate dynamics:
///   m(t+dt) = m_inf + (m_old - m_inf) * exp(-dt / τ_m)
///
/// Then the current is computed and stored in i_reg.
///
/// For Drosophila: HCN-like currents exist in LPTC (lobula plate
/// tangential cells) — key for optomotor response.
///
/// Args:
///   i_reg: output current register (I_h)
///   v_reg: membrane potential register
///   m_reg: gate variable (updated in place)
///   g_max: maximal conductance (nS)
///   e_rev: reversal potential (≈ -30 mV for h-current)
///   v_half: half-activation voltage (≈ -75 mV)
///   k: slope factor (POSITIVE: since formula is 1/(1+exp((V-v_half)/k)),
///     use k=5.5 for HCN. The inversion vs normal sigmoid (which uses
///     exp(-(V-v_half)/k)) gives HCN's characteristic hyperpolarization activation)
///   tau_m: time constant (ms, ≈ 30 ms at 37°C)
///   dt: simulation timestep (ms)
#[inline(always)]
pub fn op_bio_h_current(
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
    let vh = f32x8::splat(v_half);
    let kv = f32x8::splat(k);
    let tau_v = f32x8::splat(tau_m);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // m_inf = 1 / (1 + exp((V - v_half) / k))
    // For HCN: v_half ≈ -75, k ≈ -5.5  → activation at hyperpolarization
    let exp_arg = (v - vh) / kv;
    // Clamp to avoid exp overflow
    let exp_arg_clamped = exp_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0));
    let exp_val = fast_exp_simd(exp_arg_clamped);
    let m_inf = one / (one + exp_val);

    // Guard tau against zero
    let tau_safe = tau_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_v);

    // Exact exponential integration: m(t+dt) = m_inf + (m_old - m_inf) * exp(-dt/τ)
    let decay = fast_exp_simd(-dt_v / tau_safe);
    let m_new = m_inf + (m_old - m_inf) * decay;

    regs[m_reg as usize] = m_new;

    // I_h = g * m * (V - E_h)
    regs[i_reg as usize] = g * m_new * (v - e);
}
