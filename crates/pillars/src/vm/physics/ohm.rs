use wide::f32x8;

/// Ohm's Law for Ion Channels
/// I = g * (V - E)
#[inline(always)]
pub fn op_phys_ohm(regs: &mut [f32x8; 16], i_reg: u8, g_reg: u8, v_reg: u8, e_reg: u8) {
    let g = regs[g_reg as usize];
    let v = regs[v_reg as usize];
    let e = regs[e_reg as usize];
    regs[i_reg as usize] = g * (v - e);
}
