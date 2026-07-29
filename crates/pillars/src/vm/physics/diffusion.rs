use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// 0D Diffusion Decay
/// C_new = C0 + (C_old - C0) * exp(-dt / tau_diff)
#[inline(always)]
pub fn op_phys_diffusion_decay(regs: &mut [f32x8; 16], c_reg: u8, c0: f32, tau_diff: f32, dt: f32) {
    let c_old = regs[c_reg as usize];
    let c0_vec = f32x8::splat(c0);
    
    // Check if tau_diff is extremely small to avoid div by zero or infinity
    if tau_diff < 1e-6 {
        regs[c_reg as usize] = c0_vec;
        return;
    }
    
    let decay = fast_exp_simd(f32x8::splat(-dt / tau_diff));
    
    regs[c_reg as usize] = c0_vec + (c_old - c0_vec) * decay;
}
