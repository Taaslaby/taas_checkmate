use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

#[inline(always)]
pub fn op_bio_hh_na_h(regs: &mut [f32x8; 16], h_reg: u8, v_reg: u8, dt: f32) {
    let v = regs[v_reg as usize];
    let h_old = regs[h_reg as usize];
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    let alpha = f32x8::splat(0.07) * fast_exp_simd(-(v + f32x8::splat(65.0)) / f32x8::splat(20.0));
    let exp_arg = -(v + f32x8::splat(35.0)) / f32x8::splat(10.0);
    let beta = one / (one + fast_exp_simd(exp_arg));

    let sum = alpha + beta;
    let h_inf = alpha / sum;
    let tau_inv = sum;
    let decay = fast_exp_simd(-dt_v * tau_inv);
    regs[h_reg as usize] = h_inf + (h_old - h_inf) * decay;
}
