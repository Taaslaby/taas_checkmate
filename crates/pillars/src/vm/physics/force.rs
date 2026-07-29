use wide::f32x8;

/// Hooke's Law for Springs/Muscles
/// F = -k * x
#[inline(always)]
pub fn op_phys_force_spring(regs: &mut [f32x8; 16], f_reg: u8, x_reg: u8, k_spring: f32) {
    let x = regs[x_reg as usize];
    let k = f32x8::splat(-k_spring);
    
    regs[f_reg as usize] = k * x;
}
