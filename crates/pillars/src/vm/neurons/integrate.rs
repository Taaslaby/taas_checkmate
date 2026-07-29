use wide::{f32x8, CmpLt, CmpGt};
use crate::vm::math::utils::fast_exp_simd;

/// Leaky Membrane Integration (Exact Exponential)
///
/// Solves any leaky membrane equation exactly:
///   τ * dv/dt = (E_leak - v) + I_sum
///
/// Analytical solution (exact for constant I_sum over dt):
///   v(t+dt) = E_leak + I_sum*τ + (v - E_leak - I_sum*τ) * exp(-dt/τ)
///
/// **ANN mode** (one large step = linear activation):
///   Set dt ≫ τ, E_leak=0, τ=1: v ≈ I_sum (linear!)
///   Then FireReset with thresh=0 → v_out = max(0, I_sum) ≡ ReLU
///
/// **SNN mode** (many small steps):
///   Set dt = timestep, τ = 10ms: normal temporal dynamics
///   FireReset with thresh=-55mV → biological spiking
///
/// Args:
///   v_reg: membrane potential (updated in place)
///   i_reg: total input current
///   e_leak: leak reversal potential
///   tau: membrane time constant
///   dt: timestep
#[inline(always)]
pub fn op_leak_integrate(
    regs: &mut [f32x8; 16],
    v_reg: u8,
    i_reg: u8,
    e_leak: f32,
    tau: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let i = regs[i_reg as usize];
    let e_v = f32x8::splat(e_leak);
    let tau_v = f32x8::splat(tau);
    let dt_v = f32x8::splat(dt);

    let tau_safe = tau_v.cmp_lt(f32x8::splat(1e-8)).blend(f32x8::splat(1e-8), tau_v);
    let decay = fast_exp_simd(-dt_v / tau_safe);
    let v_inf = e_v + i * tau_safe;

    regs[v_reg as usize] = v_inf + (v - v_inf) * decay;
}

/// Fire & Reset Threshold (Universal Activation Function)
///
///   spike = 1.0 if v > v_thresh else 0.0
///   if spike: v = v_reset
///
/// **As ReLU** (ANN mode):
///   Set v_thresh = 0, v_reset = 0
///   After LeakIntegrate with dt ≫ τ: spike = max(0, v) ≡ ReLU
///
/// **As biological spike** (SNN mode):
///   Set v_thresh = -55mV, v_reset = -65mV
///
/// Args:
///   spike_reg: output (1.0 if spike, 0.0 otherwise)
///   v_reg: membrane potential (reset if spike)
///   v_thresh: spike threshold
///   v_reset: reset potential after spike
#[inline(always)]
pub fn op_fire_reset(
    regs: &mut [f32x8; 16],
    spike_reg: u8,
    v_reg: u8,
    v_thresh: f32,
    v_reset: f32,
) {
    let v = regs[v_reg as usize];
    let thresh_v = f32x8::splat(v_thresh);
    let reset_v = f32x8::splat(v_reset);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);

    let spike_mask = v.cmp_gt(thresh_v);
    regs[spike_reg as usize] = spike_mask.blend(one, zero);
    regs[v_reg as usize] = spike_mask.blend(reset_v, v);
}

/// Fire & Reset — Threshold read from REGISTERS (for per-neuron params).
///
/// Same as op_fire_reset but reads threshold and reset values from registers,
/// allowing per-neuron threshold/reset via LoadParam.
///
/// Args:
///   spike_reg: output (1.0 if spike, 0.0 otherwise)
///   v_reg: membrane potential (reset if spike)
///   thresh_reg: register holding spike threshold
///   reset_reg: register holding reset potential
#[inline(always)]
pub fn op_fire_reset_reg(
    regs: &mut [f32x8; 16],
    spike_reg: u8,
    v_reg: u8,
    thresh_reg: u8,
    reset_reg: u8,
) {
    let v = regs[v_reg as usize];
    let thresh_v = regs[thresh_reg as usize];
    let reset_v = regs[reset_reg as usize];
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);

    let spike_mask = v.cmp_gt(thresh_v);
    regs[spike_reg as usize] = spike_mask.blend(one, zero);
    regs[v_reg as usize] = spike_mask.blend(reset_v, v);
}
