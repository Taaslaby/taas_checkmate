use wide::{f32x8, CmpGt, CmpLt};

/// Drug Channel Block (Hill-Langmuir binding) — Per-Neuron Dosing
///
/// Implements open-channel block:
///   g_blocked = g / (1 + [drug] / IC50)
///
/// Uses Hill coefficient = 1 (standard Langmuir isotherm).
/// Full Hill equation requires SIMD pow() — a future MathPow
/// opcode will enable arbitrary Hill coefficients.
///
/// **drug_conc is read from a register** — each neuron in the SIMD batch
/// can have its own drug concentration, enabling massively parallel
/// dose-response curves in a single VM step!
///
/// If drug ≈ 0: g_blocked ≈ g (no block)
/// If drug ≫ IC50: g_blocked ≈ 0 (full block)
/// If drug = IC50: g_blocked = g/2 (half block)
///
/// Args:
///   g_eff_reg: register to store blocked conductance
///   g_orig_reg: register with original (unblocked) conductance
///   drug_reg: register with drug concentration (per-neuron!)
///   ic50: half-maximal inhibitory concentration (scalar)
///   hill: Hill coefficient (PLANNED — not yet implemented, kept for API stability)
///
/// This can model:
/// - TTX (tetrodotoxin): Na⁺ channel block, IC50 ≈ 1 nM, Hill ≈ 1
/// - Lidocaine: local anesthetic, use-dependent Na⁺ block
/// - TEA: K⁺ channel block
/// - Nifedipine: L-type Ca²⁺ channel block
/// - APV: NMDA receptor antagonist
#[inline(always)]
pub fn op_channel_block(
    regs: &mut [f32x8; 16],
    g_eff_reg: u8,
    g_orig_reg: u8,
    drug_reg: u8,
    ic50: f32,
    _hill: f32, // FUTURE: implement Hill via MathPow opcode
) {
    let g_orig = regs[g_orig_reg as usize];
    let drug_conc = regs[drug_reg as usize];

    // Guard against div-by-zero for IC50
    let ic50_safe = if ic50.abs() < 1e-10 { 1e-10 } else { ic50 };
    let ic50_v = f32x8::splat(ic50_safe);

    // g_eff = g / (1 + drug/IC50)  — Hill=1 (standard Langmuir)
    // TODO: when MathPow opcode is available, use:
    //   ratio = drug_conc / ic50_v;
    //   ratio_pow = pow_simd(ratio, hill);  // MathPow opcode
    //   g_eff = g_orig / (1.0 + ratio_pow);
    let ratio = drug_conc / ic50_v;
    let denom = f32x8::splat(1.0) + ratio;
    let g_eff = g_orig / denom;

    regs[g_eff_reg as usize] = g_eff;
}

/// Compute drug concentration for IC50-based block with use-dependence
///
/// Some drugs (e.g., lidocaine) bind more strongly to open/inactivated channels.
/// This models the "use-dependent block":
///
///   [drug]_bound += spike * kon * [drug]_free - koff * [drug]_bound
///   g_blocked = g * (1 - [drug]_bound)
///
/// For now, this is a placeholder for more sophisticated use-dependent models.
/// The simple ChannelBlock opcode is sufficient for most PoC purposes.
#[inline(always)]
pub fn op_use_dependent_block(
    regs: &mut [f32x8; 16],
    g_eff_reg: u8,
    g_orig_reg: u8,
    bound_reg: u8,
    spike_reg: u8,
    kon: f32,
    koff: f32,
    dt: f32,
) {
    let g_orig = regs[g_orig_reg as usize];
    let bound = regs[bound_reg as usize];
    let spike = regs[spike_reg as usize];
    let kon_v = f32x8::splat(kon);
    let koff_v = f32x8::splat(koff);
    let dt_v = f32x8::splat(dt);

    // On spike, drug binds more: dbound/dt = kon * spike * (1 - bound) - koff * bound
    let free = f32x8::splat(1.0) - bound;
    let dbound = (kon_v * spike * free - koff_v * bound) * dt_v;
    let bound_new = bound + dbound;

    // Clamp bound to [0, 1]
    let bound_clamped = bound_new.cmp_gt(f32x8::splat(1.0)).blend(f32x8::splat(1.0), bound_new);
    let bound_clamped = bound_clamped.cmp_lt(f32x8::splat(0.0)).blend(f32x8::splat(0.0), bound_clamped);

    regs[bound_reg as usize] = bound_clamped;
    // g_eff = g * (1 - bound)
    regs[g_eff_reg as usize] = g_orig * (f32x8::splat(1.0) - bound_clamped);
}

/// Dopamine / Serotonin GPCR Modulation Factors
///
/// Models the slow neuromodulatory effects of dopamine (D1/D2) and serotonin (5-HT)
/// on ion channel kinetics.
///
/// D1 receptors (coupled to Gs, ↑cAMP):
///   - Increase Na⁺ current → lower firing threshold
///   - Increase L-type Ca²⁺ current
///   - Typically "excitatory" modulation
///
/// D2 receptors (coupled to Gi, ↓cAMP):
///   - Decrease Ca²⁺ current
///   - Increase K⁺ current → hyperpolarization
///   - Typically "inhibitory" modulation
///
/// 5-HT1A (Gi, ↓cAMP): opens K⁺ channels → hyperpolarization
/// 5-HT2A (Gq, ↑IP3): closes K⁺ channels → depolarization
///
/// Output: modulation factor (1.0 = no change, >1 = enhanced, <1 = suppressed)
/// to be multiplied with channel conductance.
#[inline(always)]
pub fn op_gpcr_modulation(
    regs: &mut [f32x8; 16],
    mod_out_reg: u8,
    d1_conc: f32,
    d2_conc: f32,
    d1_effect: f32,
    d2_effect: f32,
) {
    let one = f32x8::splat(1.0);
    let d1_v = f32x8::splat(d1_conc);
    let d2_v = f32x8::splat(d2_conc);
    let eff1_v = f32x8::splat(d1_effect);
    let eff2_v = f32x8::splat(d2_effect);

    // D1 effect: mod_D1 = 1 + d1_conc * d1_effect (excitatory, positive)
    // D2 effect: mod_D2 = 1 - d2_conc * d2_effect (inhibitory, subtractive)
    let mod1 = one + d1_v * eff1_v;
    let mod2 = one - d2_v * eff2_v;

    // Combined modulation: multiplicative
    regs[mod_out_reg as usize] = mod1 * mod2;
}
