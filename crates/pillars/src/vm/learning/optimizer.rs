use wide::f32x8;

/// ███ Learning Block: Weight Clamp █████████████████████████████████████████
///
/// Clamps synaptic weight to [w_min, w_max].
/// Essential for stable training — prevents weights from exploding.
///
/// Registers:
///   w_reg (in/out): Weight to clamp
///   w_min (const): Lower bound
///   w_max (const): Upper bound
#[inline(always)]
pub fn op_learn_weight_clamp(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    w_min: f32,
    w_max: f32,
) {
    let w = regs[w_reg as usize];
    regs[w_reg as usize] = w.max(f32x8::splat(w_min)).min(f32x8::splat(w_max));
}

/// ███ Learning Block: L1 Weight Decay █████████████████████████████████████
///
/// L1 regularization for sparsity:
///   w -= α * sign(w) * dt
///
/// Pushes weights toward zero, promoting sparse connectivity.
///
/// Registers:
///   w_reg (in/out): Weight
///   alpha (const): Regularization strength
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_l1_decay(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    alpha: f32,
    dt: f32,
) {
    let w = regs[w_reg as usize];
    let a = f32x8::splat(alpha * dt);
    // signum approximation: w / (|w| + ε)
    let eps = f32x8::splat(1e-8);
    let sign = w / (w.abs() + eps);
    regs[w_reg as usize] = w - a * sign;
}

/// ███ Learning Block: L2 Weight Decay █████████████████████████████████████
///
/// L2 regularization:
///   w -= α * w * dt
///
/// Pushes weights toward zero with exponential decay.
///
/// Registers:
///   w_reg (in/out): Weight
///   alpha (const): Regularization strength
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_l2_decay(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    alpha: f32,
    dt: f32,
) {
    let w = regs[w_reg as usize];
    let a = f32x8::splat(alpha * dt);

    regs[w_reg as usize] = w - a * w;
}

/// ███ Learning Block: Momentum SGD █████████████████████████████████████████
///
/// Momentum-based stochastic gradient descent:
///
///   m = μ * m + η * g
///   w = w - m
///
/// where g is the gradient, m is the momentum buffer.
///
/// Registers:
///   w_reg (in/out): Weight
///   grad_reg (in): Gradient dL/dw
///   mom_reg (in/out): Momentum buffer
///   mu (const): Momentum coefficient (0.0-0.999)
///   lr (const): Learning rate
#[inline(always)]
pub fn op_learn_momentum(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    grad_reg: u8,
    mom_reg: u8,
    mu: f32,
    lr: f32,
) {
    let w = regs[w_reg as usize];
    let g = regs[grad_reg as usize];
    let m_old = regs[mom_reg as usize];
    let mu_v = f32x8::splat(mu);
    let lr_v = f32x8::splat(lr);

    // m = μ*m + lr*g
    let m_new = mu_v * m_old + lr_v * g;
    regs[mom_reg as usize] = m_new;

    // w = w - m
    regs[w_reg as usize] = w - m_new;
}

/// ███ Learning Block: Adam Optimizer █████████████████████████████████████
///
/// Adam (Adaptive Moment Estimation) [Kingma & Ba, 2014]:
///
///   m = β₁*m + (1-β₁)*g
///   v = β₂*v + (1-β₂)*g²
///   m_hat = m / (1 - β₁^t)
///   v_hat = v / (1 - β₂^t)
///   w = w - lr * m_hat / (√v_hat + ε)
///
/// Simplified for SIMD (t fixed for all neurons in batch):
///
/// Registers:
///   w_reg (in/out): Weight
///   grad_reg (in): Gradient dL/dw
///   m_reg (in/out): First moment estimate
///   v_reg (in/out): Second moment estimate
///   beta1 (const): First moment decay (default 0.9)
///   beta2 (const): Second moment decay (default 0.999)
///   lr (const): Learning rate
///   eps (const): Numerical stability (default 1e-8)
#[inline(always)]
pub fn op_learn_adam(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    grad_reg: u8,
    m_reg: u8,
    v_reg: u8,
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
) {
    let w = regs[w_reg as usize];
    let g = regs[grad_reg as usize];
    let m_old = regs[m_reg as usize];
    let v_old = regs[v_reg as usize];
    let b1 = f32x8::splat(beta1);
    let b2 = f32x8::splat(beta2);
    let lr_v = f32x8::splat(lr);
    let eps_v = f32x8::splat(eps);
    let one = f32x8::splat(1.0);

    // m = β₁*m + (1-β₁)*g
    let m_new = b1 * m_old + (one - b1) * g;
    // v = β₂*v + (1-β₂)*g²
    let v_new = b2 * v_old + (one - b2) * g * g;

    // w = w - lr * m / (√v + ε)
    let denom = v_new.sqrt() + eps_v;
    let update = lr_v * m_new / denom;
    regs[w_reg as usize] = w - update;

    regs[m_reg as usize] = m_new;
    regs[v_reg as usize] = v_new;
}

/// ███ Learning Block: Exponential LR Schedule █████████████████████████████
///
/// Decays the learning rate exponentially:
///
///   lr = lr₀ * decay_rate^(step / decay_steps)
///
/// Registers:
///   lr_reg (out): Computed learning rate
///   step_reg (in): Current training step (as f32)
///   lr_initial (const): Initial learning rate
///   decay_steps (const): Decay period in steps
///   decay_rate (const): Decay factor (e.g., 0.96)
#[inline(always)]
pub fn op_learn_lr_schedule(
    regs: &mut [f32x8; 16],
    lr_reg: u8,
    step_reg: u8,
    lr_initial: f32,
    decay_steps: f32,
    decay_rate: f32,
) {
    let step = regs[step_reg as usize];
    let lr0 = f32x8::splat(lr_initial);
    let steps_v = f32x8::splat(decay_steps.max(1.0));

    // lr = lr₀ * rate^(step / decay_steps) = lr₀ * exp(step * ln(rate) / decay_steps)
    let ln_rate = decay_rate.max(1e-8).ln();
    let exp_arg = f32x8::splat(ln_rate) * step / steps_v;
    let lr_val = lr0 * exp_arg.exp();

    regs[lr_reg as usize] = lr_val;
}

/// ███ Learning Block: Reward-Modulated STDP ███████████████████████████████
///
/// R-STDP [Izhikevich, 2007; Fremaux & Gerstner, 2016]:
///
///   Δw = η * R(t) * (e_pre_post(t))
///
/// where R(t) is a reward signal (dopamine-like) and e_pre_post
/// is the eligibility trace from pre-post pairing.
///
/// Simplified:
///   dw += lr * reward * (pre_tr * post_spk - post_tr * pre_spk)
///
/// Registers:
///   w_reg (in/out): Weight
///   pre_trace (in): Presynaptic spike trace
///   post_trace (in): Postsynaptic spike trace
///   reward_reg (in): Reward signal R(t) [-1, 1]
///   lr (const): Learning rate
#[inline(always)]
pub fn op_learn_reward_stdp(
    regs: &mut [f32x8; 16],
    w_reg: u8,
    pre_trace: u8,
    post_trace: u8,
    reward_reg: u8,
    lr: f32,
) {
    let w = regs[w_reg as usize];
    let pre_tr = regs[pre_trace as usize];
    let post_tr = regs[post_trace as usize];
    let reward = regs[reward_reg as usize];
    let lr_v = f32x8::splat(lr);

    // dw = lr * reward * (pre_trace - post_trace)
    // Simplified: pre_trace is high if pre fired recently → LTP
    //             post_trace is high if post fired recently → LTD
    let dw = lr_v * reward * (pre_tr - post_tr);
    regs[w_reg as usize] = w + dw;
}

/// ███ Learning Block: Eligibility Trace ███████████████████████████████████
///
/// Computes an eligibility trace for three-factor learning rules:
///
///   de/dt = -e/τ + pre_tr * post_spike (LTP eligibility)
///          - post_tr * pre_spike (LTD eligibility)
///
/// Registers:
///   e_reg (in/out): Eligibility trace
///   pre_trace (in): Presynaptic trace
///   post_trace (in): Postsynaptic trace
///   post_spike (in): Postsynaptic spike (1.0 or 0.0)
///   pre_spike (in): Presynaptic spike (1.0 or 0.0)
///   tau_e (const): Eligibility time constant
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_elig_trace(
    regs: &mut [f32x8; 16],
    e_reg: u8,
    pre_trace: u8,
    post_trace: u8,
    post_spike: u8,
    pre_spike: u8,
    tau_e: f32,
    dt: f32,
) {
    let e_old = regs[e_reg as usize];
    let pre_tr = regs[pre_trace as usize];
    let post_tr = regs[post_trace as usize];
    let post_spk = regs[post_spike as usize];
    let pre_spk = regs[pre_spike as usize];
    let tau_v = f32x8::splat(tau_e.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // de/dt = -e/τ + pre_tr * post_spk - post_tr * pre_spk
    let decay = -e_old * dt_v / tau_v;
    let ltp = pre_tr * post_spk;
    let ltd = post_tr * pre_spk;
    regs[e_reg as usize] = e_old + decay + (ltp - ltd) * dt_v;
}

/// ███ Learning Block: Firing Rate Regularization █████████████████████████
///
/// Regularizes the network firing rate toward a target:
///
///   cost = α * (rate - target_rate)²
///   applied as gradient to voltage: dV += -η * 2α * (rate - target) * ds/dV
///
/// Simplified: directly modulates firing threshold:
///   v_th += -β * (rate - target) * dt
///
/// Registers:
///   v_th_reg (in/out): Adaptive threshold
///   rate_reg (in): Estimated firing rate
///   target_rate (const): Desired firing rate
///   beta (const): Adaptation strength
///   dt (const): Time step
#[inline(always)]
pub fn op_learn_rate_reg(
    regs: &mut [f32x8; 16],
    v_th_reg: u8,
    rate_reg: u8,
    target_rate: f32,
    beta: f32,
    dt: f32,
) {
    let v_th = regs[v_th_reg as usize];
    let rate = regs[rate_reg as usize];
    let target = f32x8::splat(target_rate);
    let b = f32x8::splat(beta);
    let dt_v = f32x8::splat(dt);

    // dV_th/dt = -β * (rate - target)
    let dv_th = -b * (rate - target) * dt_v;
    regs[v_th_reg as usize] = v_th + dv_th;
}
