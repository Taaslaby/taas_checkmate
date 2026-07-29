use wide::f32x8;

/// Joule Heating
/// T_new = T_old + (I^2 * R / C_heat) * dt
#[inline(always)]
pub fn op_phys_joule_heating(regs: &mut [f32x8; 16], t_reg: u8, i_reg: u8, r_reg: u8, heat_cap: f32, dt: f32) {
    let t_old = regs[t_reg as usize];
    let i = regs[i_reg as usize];
    let r = regs[r_reg as usize];
    
    let factor = f32x8::splat(dt / heat_cap);
    
    // I^2 * R * (dt / C_heat)
    let heat = i * i * r * factor;
    
    regs[t_reg as usize] = t_old + heat;
}
