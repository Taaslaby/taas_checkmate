use wide::f32x8;

/// ███ Learning Block: SuperSpike Surrogate Gradient ████████████████████████
///
/// Computes the surrogate gradient of the spike function (Heaviside step)
/// using the SuperSpike approximation [Zenke & Ganguli, 2018]:
///
///   dσ/dV = γ / (1 + (γ * (V - V_th))²)
///
/// This is the "fast sigmoid" surrogate, widely used in SNN training.
/// γ controls the sharpness of the surrogate (default 100.0).
///
/// Registers:
///   dJ_ds (out): Surrogate gradient d(spike)/dV
///   v_reg (in): Membrane potential
///   v_thresh (const): Spike threshold voltage
///   gamma (const): Surrogate sharpness (10-1000, default 100)
#[inline(always)]
pub fn op_learn_super_spike(
    regs: &mut [f32x8; 16],
    dJ_ds: u8,
    v_reg: u8,
    v_thresh: f32,
    gamma: f32,
) {
    let v = regs[v_reg as usize];
    let vt = f32x8::splat(v_thresh);
    let g = f32x8::splat(gamma);
    let one = f32x8::splat(1.0);

    // SuperSpike: γ / (1 + (γ*(V-V_th))²)
    let diff = v - vt;
    let arg = g * diff;
    let denom = one + arg * arg;
    regs[dJ_ds as usize] = g / denom;
}

/// ███ Learning Block: Surrogate Gradient Weight Update █████████████████████
///
/// Performs a weight update using surrogate gradients:
///
///   w += -η * dJ/ds * pre_trace
///
/// where dJ/ds is the surrogate gradient from LearnSuperSpike,
/// and pre_trace is the presynaptic spike trace (how recently pre fired).
///
/// This implements the core of SNN backpropagation:
///   ∂L/∂w = ∂L/∂s * ∂s/∂V * ∂V/∂w
/// with ∂V/∂w = pre_trace (from STDP-like pre-post pairing)
///
/// Registers:
///   w_reg (in/out): Synaptic weight
///   dJ_ds_reg (in): Upstream gradient dL/ds (from surrogate or loss)
///   pre_trace_reg (in): Presynaptic spike trace
///   lr (const): Learning rate (positive)
///   w_min (const): Minimum weight bound
///   w_max (const): Maximum weight bound
#[inline(always)]
pub fn op_learn_surrogate_update(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    dJ_ds_reg: u8,
    pre_trace_reg: u8,
    lr: f32,
    w_min: f32,
    w_max: f32,
) {
    let w = regs[w_reg as usize];
    let dJds = regs[dJ_ds_reg as usize];
    let pre_tr = regs[pre_trace_reg as usize];
    let lr_v = f32x8::splat(lr);
    let wmin_v = f32x8::splat(w_min);
    let wmax_v = f32x8::splat(w_max);

    // Δw = -η * dJ/ds * pre_trace
    let dw = -lr_v * dJds * pre_tr;
    let w_new = w + dw;

    // Clamp to [w_min, w_max]
    let clamped = w_new.max(wmin_v).min(wmax_v);
    regs[w_reg as usize] = clamped;
}
