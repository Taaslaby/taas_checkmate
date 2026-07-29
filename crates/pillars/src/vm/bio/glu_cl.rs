use wide::f32x8;

/// AMPA Receptor Current (Fast Glutamate)
///
///   I_ampa = g_max * [glu] * (V - E_rev)
///
/// AMPA = workhorse of fast excitatory transmission.
/// E_rev ≈ 0 mV (same as NMDA, but NO Mg block — fast, voltage-independent).
///
/// To model synaptic kinetics, use SpikeTraceDecay/Add on glu concentration:
///   glu += spike (presynaptic spike)
///   glu *= exp(-dt / tau_ampa) (decay)
/// Then BioAMPA computes the current in one shot.
///
/// Args:
///   i_reg: output current register
///   v_reg: membrane potential register
///   glu_reg: glutamate concentration register (0..1 typically)
///   g_max: maximal conductance
///   e_rev: reversal potential (typically 0.0 mV for AMPA)
#[inline(always)]
pub fn op_bio_ampa(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    glu_reg: u8,
    g_max: f32,
    e_rev: f32,
) {
    let v = regs[v_reg as usize];
    let glu = regs[glu_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);

    // I = g_max * glu * (V - E_rev)
    regs[i_reg as usize] = g * glu * (v - e);
}

/// Chloride Channel Current (Inhibitory)
///
///   I_cl = g_max * [Cl] * (V - E_Cl)
///
/// Covers glycine receptors and GABA-C (not GABA-A, which has
/// separate benzodiazepine/barbiturate modulation via NeuroGabaA).
/// E_Cl ≈ -70 mV (near resting, shunting inhibition).
///
/// Args:
///   i_reg: output current register
///   v_reg: membrane potential register
///   cl_reg: chloride gate register (0..1 typically)
///   g_max: maximal conductance
///   e_rev: reversal potential (typically -70.0 mV for Cl⁻)
#[inline(always)]
pub fn op_bio_cl_chan(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    cl_reg: u8,
    g_max: f32,
    e_rev: f32,
) {
    let v = regs[v_reg as usize];
    let cl = regs[cl_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);

    // I = g_max * [Cl] * (V - E_Cl)
    regs[i_reg as usize] = g * cl * (v - e);
}
