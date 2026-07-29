use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// HH Na-M Gate Step (fast_exp_simd based)
/// Solves exact exponential integration for m-gate: 
/// m(t+dt) = m_inf + (m_old - m_inf) * exp(-dt / tau)
#[inline(always)]
pub fn op_bio_hh_na_m(regs: &mut [f32x8; 16], m_reg: u8, v_reg: u8, dt: f32) {
    let v = regs[v_reg as usize];
    let m_old = regs[m_reg as usize];
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // Alpha/Beta for Na-M
    let vp40 = v + f32x8::splat(40.0);
    
    // alpha = 0.1*(v+40) / (1 - exp(-(v+40)/10))
    let exp_val_a = fast_exp_simd(-vp40 / f32x8::splat(10.0));
    let denom_a = one - exp_val_a;
    let alpha_num = f32x8::splat(0.1) * vp40;
    
    // Blend to avoid div-by-zero near v = -40
    let small_mask = vp40.abs().cmp_lt(f32x8::splat(1e-6));
    let alpha = small_mask.blend(f32x8::splat(1.0), alpha_num / denom_a);
    
    // beta = 4.0 * exp(-(v+65)/18)
    let beta = f32x8::splat(4.0) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(18.0));

    // m_inf = alpha / (alpha + beta)
    let sum = alpha + beta;
    let m_inf = alpha / sum;
    
    // tau = 1 / sum
    let tau_inv = sum; // since exp(-dt / tau) == exp(-dt * tau_inv)
    
    // m_new = m_inf + (m_old - m_inf) * exp(-dt * tau_inv)
    let decay = fast_exp_simd(-dt_v * tau_inv);
    regs[m_reg as usize] = m_inf + (m_old - m_inf) * decay;
}
