use wide::f32x8;

/// ███ Learning Block: BCM Rule █████████████████████████████████████████████
///
/// Bienenstock-Cooper-Munro (BCM) theory of synaptic plasticity:
///
///   dw/dt = η * pre * post * (post - θ)
///   dθ/dt = (post² - θ) / τ_θ
///   θ ∈ [θ_min, θ_max]
///
/// The threshold θ slides based on postsynaptic activity (homeostatic).
/// When post < θ: LTD (depression)
/// When post > θ: LTP (potentiation)
///
/// Registers:
///   w_reg (in/out): Synaptic weight
///   pre_reg (in): Presynaptic activity / trace
///   post_reg (in): Postsynaptic firing rate
///   theta_reg (in/out): Sliding threshold θ
///   lr (const): Learning rate η
///   tau_theta (const): Time constant for θ sliding
///   theta_min (const): Minimum threshold (default ~1 Hz)
///   theta_max (const): Maximum threshold (default ~20 Hz)
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_bcm(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    pre_reg: u8,
    post_reg: u8,
    theta_reg: u8,
    lr: f32,
    tau_theta: f32,
    theta_min: f32,
    theta_max: f32,
    dt: f32,
) {
    let w = regs[w_reg as usize];
    let pre = regs[pre_reg as usize];
    let post = regs[post_reg as usize];
    let theta = regs[theta_reg as usize];
    let lr_v = f32x8::splat(lr);
    let tau_v = f32x8::splat(tau_theta.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // dw/dt = η * pre * post * (post - θ)
    let dw = lr_v * pre * post * (post - theta);
    regs[w_reg as usize] = w + dw * dt_v;

    // dθ/dt = (post² - θ) / τ_θ
    let dtheta = (post * post - theta) / tau_v;
    let new_theta = theta + dtheta * dt_v;

    // Clamp θ to [θ_min, θ_max]
    let theta_min_v = f32x8::splat(theta_min);
    let theta_max_v = f32x8::splat(theta_max);
    regs[theta_reg as usize] = new_theta.max(theta_min_v).min(theta_max_v);
}

/// ███ Learning Block: Oja's Rule ████████████████████████████████████████████
///
/// Oja's rule — Hebbian learning with weight normalization:
///
///   dw/dt = η * post * (pre - post * w)
///
/// This naturally bounds weights without explicit constraints.
/// The second term `- post² * w` acts as a forgetting factor.
///
/// Registers:
///   w_reg (in/out): Synaptic weight
///   pre_reg (in): Presynaptic activity
///   post_reg (in): Postsynaptic activity
///   lr (const): Learning rate
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_oja(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    pre_reg: u8,
    post_reg: u8,
    lr: f32,
    dt: f32,
) {
    let w = regs[w_reg as usize];
    let pre = regs[pre_reg as usize];
    let post = regs[post_reg as usize];
    let lr_v = f32x8::splat(lr);
    let dt_v = f32x8::splat(dt);

    // dw/dt = η * post * (pre - post * w)
    //       = η * (post * pre - post² * w)
    let dw = lr_v * post * (pre - post * w);
    regs[w_reg as usize] = w + dw * dt_v;
}

/// ███ Learning Block: Full STDP Window █████████████████████████████████████
///
/// Implements the complete STDP window function as a single opcode:
///
///   Δw = A₊ * exp(-Δt / τ₊)   for Δt > 0 (pre before post → LTP)
///   Δw = -A₋ * exp(Δt / τ₋)    for Δt < 0 (post before pre → LTD)
///
/// The spike time difference Δt = t_post - t_pre is estimated
/// using the spike traces: trace_pre approximates exp(-(t-t_pre)/τ₊)
///
/// Simplified as:
///   dw += A₊ * post_spike * pre_trace - A₋ * pre_spike * post_trace
///
/// Registers:
///   w_reg (in/out): Weight to update
///   pre_trace (in): Presynaptic spike trace
///   post_trace (in): Postsynaptic spike trace
///   post_spike (in): 1.0 if postsynaptic neuron spiked now
///   pre_spike (in): 1.0 if presynaptic neuron spiked now
///   a_plus (const): LTP amplitude
///   a_minus (const): LTD amplitude
#[inline(always)]
pub fn op_learn_stdp_window(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    pre_trace: u8,
    post_trace: u8,
    post_spike: u8,
    pre_spike: u8,
    a_plus: f32,
    a_minus: f32,
) {
    let w = regs[w_reg as usize];
    let pre_tr = regs[pre_trace as usize];
    let post_tr = regs[post_trace as usize];
    let post_spk = regs[post_spike as usize];
    let pre_spk = regs[pre_spike as usize];
    let ap = f32x8::splat(a_plus);
    let am = f32x8::splat(a_minus);

    // LTP: A₊ * post_spike * pre_trace
    let ltp = ap * post_spk * pre_tr;
    // LTD: -A₋ * pre_spike * post_trace
    let ltd = -am * pre_spk * post_tr;

    regs[w_reg as usize] = w + ltp + ltd;
}
