use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Decay a spike trace exponentially:
/// trace *= exp(-dt / tau)
#[inline(always)]
pub fn op_spike_trace_decay(regs: &mut [f32x8; 16], trace_reg: u8, tau: f32, dt: f32) {
    let trace = regs[trace_reg as usize];
    let dt_v = f32x8::splat(dt);
    let tau_v = f32x8::splat(tau);

    // Guard against div-by-zero
    let tau_safe = tau_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_v);
    let decay = fast_exp_simd(-dt_v / tau_safe);

    regs[trace_reg as usize] = trace * decay;
}

/// Add spike events to a trace:
/// trace += spike (where spike is 1.0 for a spike, 0.0 otherwise)
#[inline(always)]
pub fn op_spike_trace_add(regs: &mut [f32x8; 16], trace_reg: u8, spike_reg: u8) {
    let trace = regs[trace_reg as usize];
    let spike = regs[spike_reg as usize];

    regs[trace_reg as usize] = trace + spike;
}
