use wide::f32x8;

/// Membrane Capacitor Integration
/// dV/dt = -I_sum / Cm
/// V_new = V_old - (I_sum * dt / Cm)
#[inline(always)]
pub fn op_phys_capacitor_step(regs: &mut [f32x8; 16], v_reg: u8, i_sum_reg: u8, cm: f32, dt: f32) {
    let v_old = regs[v_reg as usize];
    let i_sum = regs[i_sum_reg as usize];
    
    let factor = f32x8::splat(dt / cm);
    
    regs[v_reg as usize] = v_old - (i_sum * factor);
}
