use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

#[inline(always)]
pub fn op_bio_hh_k_n(regs: &mut [f32x8; 16], n_reg: u8, v_reg: u8, dt: f32) {
    let v = regs[v_reg as usize];
    let n_old = regs[n_reg as usize];
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    let vp55 = v + f32x8::splat(55.0);
    let exp_val = fast_exp_simd(-vp55 / f32x8::splat(10.0));
    let denom = one - exp_val;
    let alpha_num = f32x8::splat(0.01) * vp55;
    let small_mask = vp55.abs().cmp_lt(f32x8::splat(1e-6));
    let alpha = small_mask.blend(f32x8::splat(0.1), alpha_num / denom);
    
    let beta = f32x8::splat(0.125) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(80.0));

    let sum = alpha + beta;
    let n_inf = alpha / sum;
    let tau_inv = sum;
    let decay = fast_exp_simd(-dt_v * tau_inv);
    regs[n_reg as usize] = n_inf + (n_old - n_inf) * decay;
}
