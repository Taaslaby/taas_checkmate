use wide::f32x8;

/// Ohm's Law with Constant Reversal Potential
/// I = g * (V - E_const)
#[inline(always)]
pub fn op_phys_ohm_const(regs: &mut [f32x8; 16], i_reg: u8, g_reg: u8, v_reg: u8, e_const: f32) {
    let g = regs[g_reg as usize];
    let v = regs[v_reg as usize];
    let e = f32x8::splat(e_const);
    regs[i_reg as usize] = g * (v - e);
}
