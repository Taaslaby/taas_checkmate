use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Calcium Dynamics (simple 1-compartment)
///
///   d[Ca]/dt = -α * I_Ca - ([Ca] - Ca_rest) / τ_Ca
///
/// Where:
///   - α converts Ca²⁺ current to concentration change (≈ 0.1 µM/nA·ms)
///   - I_Ca is the total Ca²⁺ current (T-type, L-type, etc.)
///   - τ_Ca is Ca²⁺ buffering/extrusion time constant (≈ 50-100 ms)
///   - Ca_rest is the resting Ca²⁺ concentration (≈ 0.05-0.1 µM)
///
/// Uses forward Euler (fast enough for Ca²⁺ dynamics with dt < 1ms).
///
/// Args:
///   ca_reg: Ca²⁺ concentration in µM (updated in place)
///   i_ca_reg: register with total Ca²⁺ current (from T-type, L-type, etc.)
///   ca_rest: resting Ca²⁺ concentration (µM, default 0.05)
///   tau_ca: buffering time constant (ms, default 80)
///   alpha: conversion factor (µM/nA·ms, default 0.1)
///   dt: timestep (ms)
#[inline(always)]
pub fn op_bio_calcium_dynamics(
    regs: &mut [f32x8; 16],
    ca_reg: u8,
    i_ca_reg: u8,
    ca_rest: f32,
    tau_ca: f32,
    alpha: f32,
    dt: f32,
) {
    let ca_old = regs[ca_reg as usize];
    let i_ca = regs[i_ca_reg as usize];
    let ca_rest_v = f32x8::splat(ca_rest);
    let tau_v = f32x8::splat(tau_ca);
    let alpha_v = f32x8::splat(alpha);
    let dt_v = f32x8::splat(dt);

    // Guard tau against zero
    let tau_safe = tau_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_v);

    // dCa/dt = -α * I_Ca - (Ca - Ca_rest) / τ_Ca
    // Note: I_Ca used here is in the standard outward-positive convention.
    // Inward Ca²⁺ current (negative) INCREASES [Ca²⁺] correctly:
    //   α * I_Ca: I_Ca < 0 → -α * (negative) = +α * |I| → Ca increases ✓
    let d_ca = -alpha_v * i_ca - (ca_old - ca_rest_v) / tau_safe;
    let ca_new = ca_old + d_ca * dt_v;

    // Clamp to prevent negative concentration
    let zero = f32x8::splat(0.0);
    regs[ca_reg as usize] = ca_new.max(zero);
}

/// SK (Small-Conductance Ca²⁺-Activated K⁺) Current
///
///   I_SK = g_SK * (Ca^n / (Ca^n + Ca_th^n)) * (V - E_K)
///
/// Generates the slow afterhyperpolarization (sAHP) that lasts 20ms-1s
/// after a spike train. This is the main driver of spike frequency
/// adaptation in many neurons.
///
/// Pharmacology:
///   - Blocked by apamin (bee venom toxin, 10-100 nM)
///   - Blocked by UCL-1684 (synthetic)
///   - Potentiated by EBIO (cognitive enhancer)
///
/// Args:
///   i_reg: output current register (I_SK)
///   v_reg: membrane potential register
///   ca_reg: Ca²⁺ concentration register (µM)
///   g_max: maximal conductance (nS, typically 0.5-5 nS)
///   e_rev: reversal potential (≈ -85 mV for K⁺)
///   ca_threshold: half-activation Ca²⁺ (µM, ≈ 0.3-0.5 µM for SK2)
///   n: Hill coefficient (≈ 4-5 for SK channels)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_sk_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    ca_reg: u8,
    g_max: f32,
    e_rev: f32,
    ca_threshold: f32,
    _n: f32, // FUTURE: implement Hill via repeated multiplication when MathPow available
) {
    let v = regs[v_reg as usize];
    let ca = regs[ca_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);

    // Hill equation: P_open = Ca^n / (Ca^n + K_d^n)
    // For SIMD without pow(), we approximate n ≈ 1 (monomolecular binding)
    // For n=4: ca_n = (ca²)*(ca²); for n=5: ca_n = ca⁴*ca
    // Currently using Hill=1 — qualitatively similar with proper K_d tuning.

    // Ca / (Ca + K_d) — Hill=1 approximation
    let kd = f32x8::splat(ca_threshold);
    let open_prob = ca / (ca + kd);

    // I_SK = g_SK * open_prob * (V - E_K)
    regs[i_reg as usize] = g * open_prob * (v - e);
}

/// BK (Big-Conductance Ca²⁺- and Voltage-Activated K⁺) Current
///
///   I_BK = g_BK * m(V) * p(Ca) * (V - E_K)
///
/// Generates the fast afterhyperpolarization (fAHP, 1-2ms) that
/// contributes to spike repolarization and the initial part of the AHP.
///
/// BK channels are activated by:
///   1. Membrane depolarization (voltage-dependent)
///   2. Intracellular Ca²⁺ (Ca²⁺-dependent)
///
/// Voltage activation:
///   m_inf(V) = 1 / (1 + exp(-(V - V_half) / k_V))
///
/// Ca²⁺ activation (simplified Hill):
///   p(Ca) = Ca / (Ca + K_d_Ca)
///
/// Total:
///   I_BK = g_BK * m_inf(V) * (Ca / (Ca + K_d_Ca)) * (V - E_K)
///
/// Pharmacological note:
///   - Blocked by iberiotoxin (scorpion toxin, highly specific)
///   - Blocked by paxilline (non-peptide inhibitor)
///   - Opened by NS-1619 (synthetic BK opener)
///
/// Args:
///   i_reg: output current register (I_BK)
///   v_reg: membrane potential register
///   ca_reg: Ca²⁺ concentration register (µM)
///   g_max: maximal conductance (nS, typically 1-20 nS)
///   e_rev: reversal potential (≈ -85 mV for K⁺)
///   v_half: half-activation voltage (≈ -10 mV)
///   k_v: voltage slope (≈ 15)
///   ca_kd: Ca²⁺ half-activation (µM, ≈ 10-20 µM, requires high Ca²⁺)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_bk_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    ca_reg: u8,
    g_max: f32,
    e_rev: f32,
    v_half: f32,
    k_v: f32,
    ca_kd: f32,
) {
    let v = regs[v_reg as usize];
    let ca = regs[ca_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let one = f32x8::splat(1.0);

    // Voltage-dependent activation: m_inf = 1 / (1 + exp(-(V - V_half) / kV))
    let exp_arg = -(v - f32x8::splat(v_half)) / f32x8::splat(k_v);
    let exp_val = fast_exp_simd(exp_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let m_inf = one / (one + exp_val);

    // Ca²⁺-dependent activation (Hill=1 approx): Ca / (Ca + K_d_Ca)
    let ca_act = ca / (ca + f32x8::splat(ca_kd));

    // I_BK = g_BK * m_inf * Ca_act * (V - E_K)
    regs[i_reg as usize] = g * m_inf * ca_act * (v - e);
}

/// P/Q-Type High-Voltage Activated Ca²⁺ Current (Cav2.1)
///
///   I_Ca = g_Ca * m² * (V - E_Ca)
///
/// High-threshold Ca²⁺ channel. Activates near -30 mV.
/// Primary source of Ca²⁺ influx for synaptic transmission at
/// most central synapses. Drives vesicle release at the active zone.
///
/// Uses a single activation gate (m, no inactivation on this timescale).
///
/// Standard parameters (Cav2.1, P/Q type):
///   V_half_m ≈ -20 mV, k_m ≈ 6
///   τ_m ≈ 0.5-1 ms
///   E_Ca ≈ +120 mV (or ≈ +60 mV for reduced gradient)
///
/// Args:
///   i_reg: output current register
///   ca_reg: Ca²⁺ concentration register (updated here for calcium dynamics!)
///           The Ca²⁺ influx is: dCa/dt += -α * I_Ca (applied as part of
///           the calcium dynamics opcode separately)
///   v_reg: membrane potential register
///   m_reg: activation gate (updated in place)
///   g_max: maximal conductance (nS)
///   e_rev: reversal potential (≈ +120 mV for Ca²⁺)
///   v_half: half-activation (≈ -20 mV)
///   k_m: slope (≈ 6)
///   tau_m: time constant (≈ 1 ms)
///   dt: timestep (ms)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_ca_pq_current(
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
    let tau_safe = f32x8::splat(tau_m).cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), f32x8::splat(tau_m));
    let decay = fast_exp_simd(-f32x8::splat(dt) / tau_safe);
    let m_new = m_inf + (m_old - m_inf) * decay;

    let zero = f32x8::splat(0.0);
    let m_clamped = m_new.max(zero).min(one);
    regs[m_reg as usize] = m_clamped;

    // I_Ca = g * m² * (V - E_Ca)
    regs[i_reg as usize] = g * m_clamped * m_clamped * (v - e);
}
