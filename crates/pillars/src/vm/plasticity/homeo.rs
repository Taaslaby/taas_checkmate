use wide::{f32x8, CmpLt};

/// Homeostatic Synaptic Scaling
///
/// Keeps the firing rate of a neuron stable by scaling its synaptic weights:
///
///   dw/dt = - (rate - target_rate) / tau * w
///
/// This is a form of multiplicative normalization: if the neuron fires too much,
/// all its weights are scaled down proportionally, and vice versa.
///
/// Args:
///   w_reg: weight to scale
///   rate_reg: current estimated firing rate of the neuron
///   target_rate: desired firing rate
///   tau: homeostatic time constant (slow, ~seconds to minutes)
///   dt: simulation timestep
#[inline(always)]
pub fn op_homeostatic_scale(
    regs: &mut [f32x8; 16],
    weight_reg: u8,
    rate_reg: u8,
    target_rate: f32,
    tau: f32,
    dt: f32,
) {
    let w = regs[weight_reg as usize];
    let rate = regs[rate_reg as usize];
    let target_v = f32x8::splat(target_rate);
    let factor_v = f32x8::splat(dt / tau);

    // w -= (rate - target) / tau * w * dt
    let rate_error = rate - target_v;
    let dw_homeo = -factor_v * rate_error * w;

    regs[weight_reg as usize] = w + dw_homeo;
}

/// Running Rate Estimation (low-pass filter of spike events)
///
///   rate_new = rate_old + (spike - rate_old) * (1 - exp(-dt / tau_rate))
///
/// This is an exponential moving average of spike events.
#[inline(always)]
pub fn op_rate_estimate(
    regs: &mut [f32x8; 16],
    rate_reg: u8,
    spike_reg: u8,
    tau_rate: f32,
    dt: f32,
) {
    use crate::vm::math::utils::fast_exp_simd;

    let rate_old = regs[rate_reg as usize];
    let spike = regs[spike_reg as usize];
    let tau_v = f32x8::splat(tau_rate);
    let dt_v = f32x8::splat(dt);

    // Guard against div-by-zero
    let tau_safe = tau_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_v);
    let decay = fast_exp_simd(-dt_v / tau_safe);

    // rate = (1 - decay) * spike + decay * rate_old
    regs[rate_reg as usize] = spike - spike * decay + rate_old * decay;
}
