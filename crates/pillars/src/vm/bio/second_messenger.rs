use wide::f32x8;

/// cAMP Dynamics (Gₛ/Gᵢ → AC → cAMP → PKA)
///
///   d[cAMP]/dt = AC_activity * (1 + Gs) - 1 - PDE * cAMP / (Km + cAMP)
///
/// cAMP is the canonical second messenger for Gs (excitatory) and Gi
/// (inhibitory) coupled receptors:
///
///   Gₛ-coupled (↑cAMP): D1 dopamine, β-adrenergic, A2A adenosine
///   Gᵢ-coupled (↓cAMP): D2 dopamine, GABA-B, CB1, A1, 5-HT1A
///
/// cAMP activates Protein Kinase A (PKA), which PHOSPHORYLATES
/// ion channels, modulating their conductance and kinetics.
///
/// Simplified forward-Euler model:
///   dcAMP/dt = Gs_activity * AC_base * cAMP_rest - PDE * (cAMP - cAMP_rest)
///
/// Where:
///   Gs_activity ∈ [0, 1] from GPCRModulation
///   AC_base = basal adenylyl cyclase activity
///   PDE = phosphodiesterase degradation rate
#[inline(always)]
pub fn op_bio_camp_dynamics(
    regs: &mut [f32x8; 16],
    camp_reg: u8,
    gs_reg: u8,
    camp_rest: f32,
    ac_base: f32,
    pde: f32,
    dt: f32,
) {
    let camp_old = regs[camp_reg as usize];
    let gs = regs[gs_reg as usize];
    let cr = f32x8::splat(camp_rest);
    let ac = f32x8::splat(ac_base);
    let pde_v = f32x8::splat(pde);
    let dt_v = f32x8::splat(dt);

    // dcAMP/dt = Gs * AC_base * camp_rest - PDE * (cAMP - cAMP_rest)
    let production = gs * ac * cr;
    let degradation = pde_v * (camp_old - cr);
    let dcamp = production - degradation;

    let camp_new = camp_old + dcamp * dt_v;

    let zero = f32x8::splat(0.0);
    regs[camp_reg as usize] = camp_new.max(zero);
}

/// IP3 Dynamics (Gₛ/Gq → PLC → IP3 → Ca²⁺ release)
///
///   d[IP3]/dt = Gq_activity * PLC_base * IP3_rest - deg * (IP3 - IP3_rest)
///
/// IP3 (inositol trisphosphate) is produced by Gq-coupled receptors:
///   Gq-coupled: mGluR1/5, M1/M3 muscarinic, 5-HT2A
///
/// IP3 binds to IP3 receptors on the endoplasmic reticulum,
/// releasing Ca²⁺ from internal stores (Ca²⁺-induced Ca²⁺ release).
#[inline(always)]
pub fn op_bio_ip3_dynamics(
    regs: &mut [f32x8; 16],
    ip3_reg: u8,
    gq_reg: u8,
    ip3_rest: f32,
    plc_base: f32,
    deg: f32,
    dt: f32,
) {
    let ip3_old = regs[ip3_reg as usize];
    let gq = regs[gq_reg as usize];
    let ir = f32x8::splat(ip3_rest);
    let plc = f32x8::splat(plc_base);
    let deg_v = f32x8::splat(deg);
    let dt_v = f32x8::splat(dt);

    // d[IP3]/dt = Gq * PLC_base * IP3_rest - deg * (IP3 - IP3_rest)
    let production = gq * plc * ir;
    let degradation = deg_v * (ip3_old - ir);
    let dip3 = production - degradation;

    let ip3_new = ip3_old + dip3 * dt_v;

    let zero = f32x8::splat(0.0);
    regs[ip3_reg as usize] = ip3_new.max(zero);
}

/// PKA Modulation Factor (from cAMP)
///
///   mod_factor = 1 + pka_gain * (cAMP / (cAMP + K_d))
///
/// PKA activation follows a Hill curve. The modulation factor is
/// multiplied with channel conductance to produce the modulated
/// conductance.
///
/// For D1 (cAMP↑ → ↑excitability):
///   mod_factor > 1 for Na⁺/Ca²⁺ channels
///
/// For GABA-B/cAMP↓ → ↓excitability):
///   mod_factor < 1 for K⁺ channels
///
/// Args:
///   mod_reg: output modulation factor (scalar multiply with g)
///   camp_reg: cAMP concentration (µM)
///   kd: cAMP half-activation for PKA (≈ 0.5-2 µM)
///   gain: maximal modulation factor (>0 for enhancement, <0 for suppression)
#[inline(always)]
pub fn op_bio_pka_modulation(
    regs: &mut [f32x8; 16],
    mod_reg: u8,
    camp_reg: u8,
    kd: f32,
    gain: f32,
) {
    let camp = regs[camp_reg as usize];
    let kd_v = f32x8::splat(kd);
    let gain_v = f32x8::splat(gain);
    let one = f32x8::splat(1.0);

    // PKA_act = cAMP / (cAMP + Kd)  — Hill=1 approximation
    let pka_act = camp / (camp + kd_v);

    // mod = 1 + gain * PKA_act
    regs[mod_reg as usize] = one + gain_v * pka_act;
}
