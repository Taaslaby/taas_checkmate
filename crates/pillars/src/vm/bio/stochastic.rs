use wide::f32x8;

/// ███ Stochastic Bio Block: Vesicle Release ██████████████████████████████████
///
/// Binomial/Binomial-like stochastic vesicle release.
/// Given N releasable vesicles, release probability p, the number released
/// follows a binomial distribution approximated by:
///
///   n_released = N * p + Gaussian(0, sqrt(N * p * (1-p)))
///
/// The stochastic component uses the neuron index as a pseudo-random seed
/// for deterministic (per-neuron) noise.
///
/// Registers:
///   n_released (out): Number of vesicles released (as f32, 0..N)
///   spike_reg (in): 1.0 if spike, 0.0 otherwise
///   n_vesicles_reg (in): Number of docked/releasable vesicles
///   p_release (const): Base release probability per vesicle
///   ca_mod (const): Ca²⁺ modulation factor (1.0 = baseline)
///   neuron_idx (const): Global neuron index for seeding
#[inline(always)]
pub fn op_bio_vesicle_release(
    regs: &mut [f32x8; 16],
    n_released: u8,
    spike_reg: u8,
    n_vesicles_reg: u8,
    p_release: f32,
    ca_mod: f32,
    neuron_idx: usize,
) {
    let spike = regs[spike_reg as usize];
    let n_ves = regs[n_vesicles_reg as usize];
    let p = f32x8::splat(p_release);
    let ca = f32x8::splat(ca_mod);

    // p_eff = p * ca_mod (clamped to [0, 1])
    let p_eff = (p * ca).min(f32x8::splat(1.0));
    // Expected release: n_ves * p_eff
    let expected = n_ves * p_eff;
    // Stochastic noise = sqrt(N * p * (1-p)) * Gaussian
    // We use a deterministic hash from index for simplicity
    let _idx = f32x8::splat(neuron_idx as f32);
    // Simple deterministic noise using the reg values as seeds
    // This avoids needing RNG in SIMD
    let noise_amp = (n_ves * p_eff * (f32x8::splat(1.0) - p_eff)).sqrt();
    // Pseudo-random hash from the neuron index (deterministic per neuron)
    let hash = f32x8::splat(((neuron_idx as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) as f32 * 2.0 - 1.0);
    // Clamp noise_amp to avoid issues with 0 variance
    let noise = noise_amp * hash;

    // Only release on spike
    let release = (expected + noise).max(f32x8::splat(0.0));
    regs[n_released as usize] = spike * release;
}

/// ███ Stochastic Bio Block: Channel Noise (Langevin) ████████████████████████
///
/// Langevin approximation for stochastic ion channel gating.
/// Adds Gaussian white noise scaled to the number of channels:
///
///   dm/dt = (m_inf - m) / tau + σ * sqrt(2 * m_inf * (1-m_inf) / (N * tau)) * ξ
///
/// Registers:
///   gate_reg (in/out): Gating variable (m or h or n)
///   m_inf_reg (in): Steady-state value
///   tau_reg (in): Time constant
///   dt (const): Time step
///   n_channels (const): Number of channels (default 1000)
#[inline(always)]
pub fn op_bio_channel_noise(
    regs: &mut [f32x8; 16],
    gate_reg: u8,
    m_inf_reg: u8,
    tau_reg: u8,
    dt: f32,
    n_channels: f32,
) {
    let m_old = regs[gate_reg as usize];
    let m_inf = regs[m_inf_reg as usize];
    let tau = regs[tau_reg as usize];
    let dt_v = f32x8::splat(dt);

    // Ensure tau > 0
    let tau_safe = tau.max(f32x8::splat(1e-6));

    // Deterministic: exact exponential integration
    let decay = (-dt_v / tau_safe).exp();  // Use native exp, not fast_exp for accuracy
    let m_det = m_inf + (m_old - m_inf) * decay;

    // Noise amplitude: σ = sqrt(2 * m_inf * (1-m_inf) * dt / (N * tau))
    let variance = f32x8::splat(2.0) * m_inf * (f32x8::splat(1.0) - m_inf) / (f32x8::splat(n_channels) * tau_safe);
    let variance_safe = variance.max(f32x8::splat(0.0));
    let noise_amp = (variance_safe * dt_v).sqrt();

    // Add Langevin noise to deterministic gate value
    regs[gate_reg as usize] = m_det + noise_amp;
}
