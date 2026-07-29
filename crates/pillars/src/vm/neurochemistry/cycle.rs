use wide::f32x8;

/// ███ Cool Chemistry: Synaptic Vesicle Cycle █████████████████████████████████
///
/// Full vesicle cycle: docked → released → recycled.
///
///   On spike: dock → released (proportional to Ca)
///   Released → dock (recycling): dD/dt = (R - D * k_dock/k_undock) / tau
///   Release pool conservation: D + R + F = total (dock + release + fused)
///
/// Simplified two-pool model:
///   docked: available for release
///   recycling: being refilled
///
///   dD/dt = -P_release * Ca_dep * D * spike + k_refill * R
///   dR/dt = +P_release * Ca_dep * D * spike - k_refill * R
///
/// Registers:
///   docked (in/out): Number of docked vesicles
///   recycled (in/out): Number of recycling vesicles
///   spike (in): 1.0 if presynaptic neuron spiked
///   ca_eff (in): Ca²⁺ efficacy (0-1, from Ca current)
///   total_vesicles (const): Total vesicle pool (~5-10 active zone)
///   p_release (const): Release probability per vesicle
///   tau_refill (const): Refilling time constant (ms)
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_vesicle_cycle(
    regs: &mut [f32x8; 16],
    docked: u8,
    recycled: u8,
    spike: u8,
    ca_eff: u8,
    total_vesicles: f32,
    p_release: f32,
    tau_refill: f32,
    dt: f32,
) {
    let d = regs[docked as usize];
    let r = regs[recycled as usize];
    let s = regs[spike as usize];
    let ca = regs[ca_eff as usize];
    let total = f32x8::splat(total_vesicles);
    let pr = f32x8::splat(p_release);
    let tau_r = f32x8::splat(tau_refill.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // Release rate: P_release * Ca * D * spike
    let release_rate = pr * ca * d * s;
    // Refill rate: R / tau_refill (first-order recovery)
    let refill_rate = r / tau_r;

    let new_d = d + (-release_rate + refill_rate) * dt_v;
    let new_r = r + (release_rate - refill_rate) * dt_v;

    // Clamp and conserve
    let new_d = new_d.max(f32x8::splat(0.0)).min(total);
    let new_r = new_r.max(f32x8::splat(0.0));
    // Conserve: D + R + F = total, but F (fused) is implicit
    let excess = (new_d + new_r) - total;
    // If excess > 0, reduce recycled pool
    let corrected_r = new_r - excess.max(f32x8::splat(0.0));
    let corrected_r = corrected_r.max(f32x8::splat(0.0));

    regs[docked as usize] = new_d.max(f32x8::splat(0.0));
    regs[recycled as usize] = corrected_r;
}

/// ███ Cool Chemistry: Neurotransmitter Reuptake █████████████████████████████
///
/// First-order clearance of neurotransmitter from the synaptic cleft.
/// Includes both diffusion and active reuptake.
///
///   d[NT]/dt = release - k_uptake * [NT] - k_diff * ([NT] - [NT]_rest)
///
/// Registers:
///   nt_conc (in/out): Neurotransmitter concentration in cleft
///   release_reg (in): Amount released this timestep
///   k_uptake (const): Active reuptake rate (1/ms, ~0.1 for GABA, ~0.05 for Glu)
///   k_diff (const): Diffusion rate (1/ms, ~1.0)
///   nt_rest (const): Resting concentration (≈ 0.0)
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_reuptake(
    regs: &mut [f32x8; 16],
    nt_conc: u8,
    release_reg: u8,
    k_uptake: f32,
    k_diff: f32,
    nt_rest: f32,
    dt: f32,
) {
    let nt = regs[nt_conc as usize];
    let rel = regs[release_reg as usize];
    let ku = f32x8::splat(k_uptake);
    let kd = f32x8::splat(k_diff);
    let rest = f32x8::splat(nt_rest);
    let dt_v = f32x8::splat(dt);

    // d[NT]/dt = release - k_uptake * [NT] - k_diff * ([NT] - [NT]_rest)
    let dnt = rel - ku * nt - kd * (nt - rest);
    let new_nt = (nt + dnt * dt_v).max(f32x8::splat(0.0));
    regs[nt_conc as usize] = new_nt;
}

/// ███ Cool Chemistry: Metabolic Energy (ATP) ████████████████████████████████
///
/// ATP consumption and recovery in neurons.
/// Each spike consumes ATP (for Na/K pump, vesicle recycling).
/// ATP is recovered via oxidative metabolism (slow).
///
///   dATP/dt = -(spike_cost * spike + pump_cost * Na/K activity)
///             + (ATP_rest - ATP) / tau_recovery
///
/// Registers:
///   atp_reg (in/out): ATP concentration (normalized, 0-1)
///   spike_reg (in): 1.0 if spike, 0.0 otherwise
///   atp_rest (const): Resting ATP level (≈ 0.95)
///   spike_cost (const): ATP consumed per spike (≈ 0.01)
///   tau_recovery (const): Recovery time constant (≈ 500 ms)
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_metabolic_atp(
    regs: &mut [f32x8; 16],
    atp_reg: u8,
    spike_reg: u8,
    atp_rest: f32,
    spike_cost: f32,
    tau_recovery: f32,
    dt: f32,
) {
    let atp = regs[atp_reg as usize];
    let spike = regs[spike_reg as usize];
    let rest = f32x8::splat(atp_rest);
    let cost = f32x8::splat(spike_cost);
    let tau = f32x8::splat(tau_recovery.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // dATP/dt = -spike_cost * spike + (ATP_rest - ATP)/tau
    let datp = -cost * spike + (rest - atp) / tau;
    let new_atp = (atp + datp * dt_v).max(f32x8::splat(0.0)).min(f32x8::splat(1.0));
    regs[atp_reg as usize] = new_atp;
}

/// ███ Cool Chemistry: Na/K-ATPase (Pump Current) █████████████████████████████
///
/// Electrogenic Na⁺/K⁺ pump: 3 Na⁺ out, 2 K⁺ in, producing net outward current.
/// Critical for maintaining ion gradients and a major energy consumer.
///
///   I_pump = I_max * (1/(1 + K_Na/[Na]_i)) * (1/(1 + K_K/[K]_o))
///
/// Registers:
///   i_pump (out): Na/K pump current (outward positive)
///   v_reg (in): Membrane potential (for voltage-dependent modulation)
///   na_in_reg (in): Intracellular Na⁺ concentration
///   k_out_reg (in): Extracellular K⁺ concentration
///   i_max (const): Maximum pump current
///   k_na (const): Half-activation Na⁺ (≈ 10 mM)
///   k_k (const): Half-activation K⁺ (≈ 1.5 mM)
#[inline(always)]
pub fn op_chem_nak_pump(
    regs: &mut [f32x8; 16],
    i_pump: u8,
    _v_reg: u8,
    na_in_reg: u8,
    k_out_reg: u8,
    i_max: f32,
    k_na: f32,
    k_k: f32,
) {
    let na = regs[na_in_reg as usize];
    let ko = regs[k_out_reg as usize];
    let imax = f32x8::splat(i_max);
    let kna = f32x8::splat(k_na);
    let kk = f32x8::splat(k_k);
    // Na activation: [Na]i/([Na]i + K_Na)
    let na_act = na / (na + kna);
    // K activation: [K]o/([K]o + K_K)
    let k_act = ko / (ko + kk);

    // I_pump = I_max * na_act * k_act
    let i = imax * na_act * k_act;
    regs[i_pump as usize] = i;
}
