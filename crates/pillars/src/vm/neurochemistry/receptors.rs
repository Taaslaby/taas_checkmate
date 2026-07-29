use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// NMDA Receptor Current with Voltage-Dependent Mg²⁺ Block
///
/// I_NMDA = g_max * [glu] * Mg_block(V, [Mg]) * (V - 0mV)
///
/// Mg_block(V) = 1 / (1 + [Mg²⁺] * exp(-a * V / kT))
///
/// Where:
/// - [glu] is the normalized glutamate concentration (0..1)
/// - [Mg²⁺] is the extracellular magnesium concentration (~1.2 mM physiological)
/// - The Mg block is voltage-dependent: at rest (−65mV), Mg²⁺ blocks the channel;
///   at depolarized potentials (> −30mV), Mg²⁺ is expelled
/// - Typical parameters: a = 0.062, k = 1.0 (giving exp(-0.062 * V) factor)
///
/// This is the molecular coincidence detector underlying Hebbian learning,
/// and the target of ketamine, memantine, and esketamine.
#[inline(always)]
pub fn op_nmda_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    glu_reg: u8,
    mg_conc: f32,
    g_max: f32,
    e_rev: f32,
) {
    let v = regs[v_reg as usize];
    let glu = regs[glu_reg as usize];
    let g_max_v = f32x8::splat(g_max);
    let mg_v = f32x8::splat(mg_conc);
    let e_v = f32x8::splat(e_rev);

    // Mg²⁺ voltage-dependent block: f(V) = 1 / (1 + [Mg] * exp(-0.062 * V / 1.0))
    // Standard Jahr & Stevens (1990) formulation
    let exp_arg = f32x8::splat(-0.062) * v;
    let exp_val = fast_exp_simd(exp_arg);
    let mg_block = f32x8::splat(1.0) / (f32x8::splat(1.0) + mg_v * exp_val);

    // I_NMDA = g_max * glu * Mg_block * (V - E_rev)
    let i = g_max_v * glu * mg_block * (v - e_v);
    regs[i_reg as usize] = i;
}

/// GABA-A Receptor Chloride Current
///
/// I_GABA = g_max * [GABA] * (1 + bzd_mod) * (1 + barb_mod) * (V - E_Cl)
///
/// Allosteric modulation:
/// - Benzodiazepines (diazepam, alprazolam): increase channel opening frequency
///   → scales conductance by (1 + bzd_effect)
/// - Barbiturates (phenobarbital): increase channel opening duration
///   → also scales conductance, but can directly gate channel at high doses
/// - Ethanol: potentiates GABA-A responses
///
/// E_Cl = -70 mV (typically, reversal potential for chloride)
#[inline(always)]
pub fn op_gaba_a_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    gaba_reg: u8,
    bzd_mod: f32,
    barb_mod: f32,
    g_max: f32,
) {
    let v = regs[v_reg as usize];
    let gaba = regs[gaba_reg as usize];
    let g_max_v = f32x8::splat(g_max);
    let bzd_v = f32x8::splat(bzd_mod);
    let barb_v = f32x8::splat(barb_mod);
    let e_cl = f32x8::splat(-70.0); // Cl⁻ reversal potential

    // Modulated conductance
    let g_mod = g_max_v * gaba * (f32x8::splat(1.0) + bzd_v) * (f32x8::splat(1.0) + barb_v);
    // I_GABA = g_mod * (V - E_Cl)
    // E_Cl is typically more negative than rest, so GABA produces inward (hyperpolarizing) current
    regs[i_reg as usize] = g_mod * (v - e_cl);
}

/// Nicotinic Acetylcholine Receptor (nAChR) Current
///
/// I_nACh = g_max * [ACh] * (1 + nico_mod) * (V - 0 mV)
///
/// Fast cation channel (Na⁺/K⁺/Ca²⁺), depolarizing (excitatory).
/// Target of:
/// - Nicotine (addiction mechanism)
/// - Varenicline / Champix (smoking cessation)
/// - Alpha-bungarotoxin (snake venom, blocker)
#[inline(always)]
pub fn op_nachr_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    ach_reg: u8,
    nico_mod: f32,
    g_max: f32,
) {
    let v = regs[v_reg as usize];
    let ach = regs[ach_reg as usize];
    let g_max_v = f32x8::splat(g_max);
    let nico_v = f32x8::splat(nico_mod);
    let e_rev = f32x8::splat(0.0); // ~0 mV for nAChR

    // Modulated conductance: g_eff = g_max * [ACh] * (1 + nicotine_modulation)
    let g_eff = g_max_v * ach * (f32x8::splat(1.0) + nico_v);
    regs[i_reg as usize] = g_eff * (v - e_rev);
}

/// GluCl (Glutamate-Gated Chloride Channel) Current
///
/// I_GluCl = g_max * [glu] * (V - E_Cl)
///
/// In Drosophila, glutamate is INHIBITORY via GluCl channels (Cl⁻ conductance).
/// This is OPPOSITE to mammals where glutamate is excitatory (AMPA/NMDA).
///
/// E_Cl = -70 mV (typical chloride reversal potential)
/// At V > -70: outward current = hyperpolarization = inhibition
/// At V < -70: inward current = depolarization (rare, only if very hyperpolarized)
///
/// Target of:
/// - Ivermectin (allosteric agonist, anthelmintic)
/// - Fipronil (blocker, insecticide)
/// - Picrotoxin (blocker)
#[inline(always)]
pub fn op_glucl_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    glu_reg: u8,
    g_max: f32,
    e_rev: f32,
) {
    let v = regs[v_reg as usize];
    let glu = regs[glu_reg as usize];
    let g_max_v = f32x8::splat(g_max);
    let e_rev_v = f32x8::splat(e_rev);
    // I_GluCl = g_max * [glu] * (V - E_Cl)
    // Cl⁻ reversal ~ -70mV → outward (hyperpolarizing) when V > E_Cl → INHIBITION
    regs[i_reg as usize] = g_max_v * glu * (v - e_rev_v);
}
