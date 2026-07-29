use wide::f32x8;

/// SIMD Leak Current Step
///
/// Standard outward-positive conductance:
///   I_leak = g_leak * (V - E_leak)
///
/// Consistent with PhysOhm, PhysOhmConst, BioAMPA, BioClChan,
/// NeuroNMDA, NeuroGabaA, NeuroNicoR — all use g*(V - E_rev).
///
/// CapacitorStep uses dV/dt = -I_total / Cm, so outward
/// positive I decreases V, pushing it toward E_leak.
///
/// Note: This was previously g*(E_leak - V) (INVERTED sign).
/// Fixed to match the standard convention. See git history.
#[inline(always)]
pub fn op_bio_leak(regs: &mut [f32x8; 16], dst: u8, v_reg: u8, g_leak: f32, e_leak: f32) {
    let v = regs[v_reg as usize];
    let gl = f32x8::splat(g_leak);
    let el = f32x8::splat(e_leak);

    // I_leak = g_leak * (V - E_leak)
    // Outward positive: V > E_leak → outward (hyperpolarizing) current
    regs[dst as usize] = gl * (v - el);
}
