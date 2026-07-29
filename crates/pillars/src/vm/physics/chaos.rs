use wide::{f32x8, CmpLt};

/// ███ Cool Physics: Ornstein-Uhlenbeck Noise ████████████████████████████████
///
/// Colored noise (correlated) with mean-reversion.
///   dx = (-x / tau) * dt + sigma * sqrt(2 * dt / tau) * ξ
///
/// where ξ ~ N(0,1). Uses deterministic per-neuron hash for ξ.
///
/// Registers:
///   noise_reg (in/out): Noise value x
///   sigma (const): Standard deviation
///   tau (const): Correlation time constant
///   dt (const): Time step
///   global_idx (const): Neuron index for deterministic noise
#[inline(always)]
pub fn op_phys_ou_noise(
    regs: &mut [f32x8; 16],
    noise_reg: u8,
    sigma: f32,
    tau: f32,
    dt: f32,
    global_idx: usize,
) {
    let x = regs[noise_reg as usize];
    let dt_v = f32x8::splat(dt);
    let tau_v = f32x8::splat(tau.max(1e-6));
    let sig_v = f32x8::splat(sigma);

    // Deterministic decay: dx = (-x/tau) * dt
    let decay = -x * dt_v / tau_v;

    // Noise term: sigma * sqrt(2*dt/tau) * ξ
    // ξ from deterministic hash of index
    let noise_amp = sig_v * (f32x8::splat(2.0) * dt_v / tau_v).sqrt();

    // Per-neuron deterministic hash
    let hash_seed = global_idx.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let hash_seed2 = hash_seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let xi = f32x8::splat(((hash_seed2 as f32) * 2.0 - 1.0) / 2147483648.0_f32);

    let noise = noise_amp * xi;
    regs[noise_reg as usize] = x + decay + noise;
}

/// ███ Cool Physics: Lorenz Attractor ████████████████████████████████████████
///
/// Classic chaotic system for generating complex neuromodulatory signals.
/// Useful for simulating chaotic neural dynamics (e.g., olfactory bulb).
///
///   dx/dt = σ * (y - x)
///   dy/dt = x * (ρ - z) - y
///   dz/dt = x * y - β * z
///
/// Registers:
///   x_reg (in/out): X variable
///   y_reg (in/out): Y variable
///   z_reg (in/out): Z variable
///   sigma (const): Prandtl number (default 10.0)
///   rho (const): Rayleigh number (default 28.0 for chaos)
///   beta (const): Geometric factor (default 8/3)
///   dt (const): Time step
#[inline(always)]
pub fn op_phys_lorenz(
    regs: &mut [f32x8; 16],
    x_reg: u8,
    y_reg: u8,
    z_reg: u8,
    sigma: f32,
    rho: f32,
    beta: f32,
    dt: f32,
) {
    let x = regs[x_reg as usize];
    let y = regs[y_reg as usize];
    let z = regs[z_reg as usize];
    let s = f32x8::splat(sigma);
    let r = f32x8::splat(rho);
    let b = f32x8::splat(beta);
    let dt_v = f32x8::splat(dt);

    // dx/dt = σ*(y-x)
    let dx = s * (y - x);
    // dy/dt = x*(ρ-z) - y
    let dy = x * (r - z) - y;
    // dz/dt = x*y - β*z
    let dz = x * y - b * z;

    regs[x_reg as usize] = x + dx * dt_v;
    regs[y_reg as usize] = y + dy * dt_v;
    regs[z_reg as usize] = z + dz * dt_v;
}

/// ███ Cool Physics: Hopf Bifurcation Oscillator █████████████████████████████
///
/// Supercritical Hopf bifurcation: transition from quiescence to oscillation.
/// Generates alpha (8-12 Hz), theta (4-8 Hz), or gamma (30-80 Hz) rhythms.
///
///   dr/dt = μ * r - r³
///   dθ/dt = ω + b * r²
///
/// in Cartesian: dx/dt = (μ - r²) * x - ω*y
///               dy/dt = (μ - r²) * y + ω*x
///
/// Registers:
///   x_reg (in/out): Oscillator state x
///   y_reg (in/out): Oscillator state y
///   mu (const): Bifurcation parameter (μ < 0 = quiescent, μ > 0 = oscillating)
///   omega (const): Natural frequency in rad/s (e.g., 2*π*40 = 251.3 for gamma)
///   dt (const): Time step
#[inline(always)]
pub fn op_phys_hopf_oscillator(
    regs: &mut [f32x8; 16],
    x_reg: u8,
    y_reg: u8,
    mu: f32,
    omega: f32,
    dt: f32,
) {
    let x = regs[x_reg as usize];
    let y = regs[y_reg as usize];
    let mu_v = f32x8::splat(mu);
    let omega_v = f32x8::splat(omega);
    let dt_v = f32x8::splat(dt);

    // r² = x² + y²
    let r2 = x * x + y * y;

    // dx/dt = (μ - r²) * x - ω*y
    let dx = (mu_v - r2) * x - omega_v * y;
    // dy/dt = (μ - r²) * y + ω*x
    let dy = (mu_v - r2) * y + omega_v * x;

    regs[x_reg as usize] = x + dx * dt_v;
    regs[y_reg as usize] = y + dy * dt_v;
}

/// ███ Cool Physics: Poisson Spike Generator █████████████████████████████████
///
/// Generates Poisson-distributed spikes based on an instantaneous rate.
/// Uses deterministic per-neuron hash.
///
///   P(spike) = rate * dt  (clamped to [0, 1])
///
/// Registers:
///   spike_reg (out): 1.0 if spike, 0.0 otherwise
///   rate_reg (in): Instantaneous firing rate in Hz
///   dt (const): Time step (ms)
///   spike_counter (in/out): Cumulative spike count for LFP
///   global_idx (const): Neuron index
#[inline(always)]
pub fn op_phys_poisson_spike(
    regs: &mut [f32x8; 16],
    spike_reg: u8,
    rate_reg: u8,
    dt: f32,
    spike_counter: u8,
    global_idx: usize,
) {
    let rate = regs[rate_reg as usize];
    let dt_v = f32x8::splat(dt);

    // P(spike) = rate * dt, clamped to [0, 1]
    let p_spike = (rate * dt_v).min(f32x8::splat(1.0));

    // Deterministic threshold based on neuron index
    let hash_seed = (global_idx as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    let threshold = f32x8::splat(((hash_seed & 0xFFFFFFFF) as f32) / 4294967296.0_f32);

    // Spike if threshold < p_spike
    let spike = threshold.cmp_lt(p_spike).blend(
        f32x8::splat(1.0),
        f32x8::splat(0.0),
    );

    regs[spike_reg as usize] = spike;

    // Update spike counter
    let counter = regs[spike_counter as usize];
    regs[spike_counter as usize] = counter + spike;
}
