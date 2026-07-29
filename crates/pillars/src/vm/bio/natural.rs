use wide::{f32x8, CmpGe};
use crate::vm::math::utils;

/// ███ Natural Bio Block: Dendritic Coupling ███████████████████████████████████
///
/// Models passive current flow between soma and dendrite:
///   I_dend = g_dend * (V_soma - V_dend)
///
/// This is the simplest form of dendritic electrotonic coupling.
/// g_dend is the coupling conductance (inverse of axial resistance).
///
/// Registers:
///   dst_current (out): Dendritic coupling current
///   v_soma (in): Somatic membrane potential
///   v_dend (in): Dendritic membrane potential
///   g_dend (in): Coupling conductance (register)
#[inline(always)]
pub fn op_bio_dendritic_coupling(
    regs: &mut [f32x8; 16],
    dst_current: u8,
    v_soma: u8,
    v_dend: u8,
    g_dend: u8,
) {
    let vs = regs[v_soma as usize];
    let vd = regs[v_dend as usize];
    let g = regs[g_dend as usize];
    // I = g * (V_soma - V_dend)
    regs[dst_current as usize] = g * (vs - vd);
}

/// ███ Natural Bio Block: Backpropagating AP █████████████████████████████████
///
/// Models backpropagation of a somatic action potential into dendrites.
/// On a somatic spike, the dendritic voltage receives a boost.
/// Between spikes, it decays toward the somatic voltage.
///
///   V_dend = V_dend + spike * bap_amp                   [if spike]
///   V_dend = V_dend + (V_soma - V_dend) * (1 - exp(-dt/tau)) [decay]
///
/// Registers:
///   v_dend (in/out): Dendritic voltage
///   v_soma (in): Somatic voltage
///   spike_reg (in): 1.0 if spike, 0.0 otherwise
///   bap_amp (const): Amplitude of backpropagation boost
///   tau (const): Time constant of dendritic decay
///   dt (const): Time step
#[inline(always)]
pub fn op_bio_backprop_ap(
    regs: &mut [f32x8; 16],
    v_dend: u8,
    v_soma: u8,
    spike_reg: u8,
    bap_amp: f32,
    tau: f32,
    dt: f32,
) {
    let vd = regs[v_dend as usize];
    let vs = regs[v_soma as usize];
    let spike = regs[spike_reg as usize];

    // On spike: boost dendritic voltage
    let boosted = vd + f32x8::splat(bap_amp) * spike;
    // Between spikes: dendrite decays toward soma
    let decay = utils::fast_exp_simd(f32x8::splat(-dt / tau.max(1e-6)));
    let decayed = vd + (vs - vd) * (f32x8::splat(1.0) - decay);
    // Pick boosted if spiking, decayed otherwise
    let mask = spike.cmp_ge(f32x8::splat(0.5));
    regs[v_dend as usize] = mask.blend(boosted, decayed);
}

/// ███ Natural Bio Block: E/I Balance Homeostasis █████████████████████████████
///
/// Maintains the ratio of excitatory to inhibitory input.
/// When E > target * I, the EI_factor decreases (inhibitory tone up).
/// When E < target * I, the EI_factor increases (inhibitory tone down).
///
///   de/dt = -(exc_rate - target * inh_rate) * dt / tau
///   clamped to [0.1, 10.0]
///
/// Registers:
///   ei_factor (in/out): E/I balance factor (modulates inh gain)
///   exc_rate (in): Firing rate of excitatory population
///   inh_rate (in): Firing rate of inhibitory population
///   target_ratio (const): Target E/I ratio (default ~4.0)
///   tau (const): Adaptation time constant
///   dt (const): Time step
#[inline(always)]
pub fn op_bio_ei_balance(
    regs: &mut [f32x8; 16],
    ei_factor: u8,
    exc_rate: u8,
    inh_rate: u8,
    target_ratio: f32,
    tau: f32,
    dt: f32,
) {
    let ef = regs[ei_factor as usize];
    let er = regs[exc_rate as usize];
    let ir = regs[inh_rate as usize];
    let target = f32x8::splat(target_ratio);
    let tau_v = f32x8::splat(tau.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // de/dt = -(E_rate - target * I_rate) / tau
    let error = er - target * ir;
    let de = -error * dt_v / tau_v;
    let new_ef = ef + de;
    // Clamp to [0.1, 10.0]
    let clamped = new_ef.max(f32x8::splat(0.1)).min(f32x8::splat(10.0));
    regs[ei_factor as usize] = clamped;
}

/// ███ Natural Bio Block: Astrocyte Tripartite Modulation ████████████████████
///
/// Astrocyte modulates synaptic transmission in response to glutamate spillover.
/// Astrocytic calcium increases when glutamate is high, then astrocytes release
/// gliotransmitters that modulate presynaptic release probability.
///
///   dCa_a/dt = (glu * k_act - Ca_a) * dt / tau_rise
///   mod = 1.0 + gain * Ca_a / (Ca_a + k_half)
///
/// Registers:
///   mod_factor (out): Synaptic modulation factor (multiply onto g_syn)
///   glu_reg (in): Synaptic glutamate concentration
///   ca_astro (in/out): Astrocytic calcium (state variable)
///   k_act (const): Activation rate by glutamate
///   k_half (const): Half-activation for Ca → modulation
///   gain (const): Modulation gain (0.5 = 50% boost)
///   tau (const): Calcium time constant
///   dt (const): Time step
#[inline(always)]
pub fn op_bio_astrocyte_mod(
    regs: &mut [f32x8; 16],
    mod_factor: u8,
    glu_reg: u8,
    ca_astro: u8,
    k_act: f32,
    k_half: f32,
    gain: f32,
    tau: f32,
    dt: f32,
) {
    let glu = regs[glu_reg as usize];
    let ca = regs[ca_astro as usize];
    let k_act_v = f32x8::splat(k_act);
    let tau_v = f32x8::splat(tau.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // dCa/dt = (glu * k_act - Ca) / tau
    let dca = (glu * k_act_v - ca) * dt_v / tau_v;
    let new_ca = (ca + dca).max(f32x8::splat(0.0));
    regs[ca_astro as usize] = new_ca;

    // mod = 1.0 + gain * Ca / (Ca + k_half)
    let kh = f32x8::splat(k_half);
    let g = f32x8::splat(gain);
    let one = f32x8::splat(1.0);
    regs[mod_factor as usize] = one + g * new_ca / (new_ca + kh);
}

/// ███ Natural Bio Block: Slow Afterhyperpolarization ████████████████████████
///
/// sAHP current mediated by Ca²⁺-activated K⁺ channels (SK).
/// Builds up during high firing and produces spike frequency adaptation.
///
///   I_sAHP = g_sahp * (Ca / (Ca + Kd)) * (V - E_K)
///
/// Registers:
///   dst_current (out): sAHP current
///   v_reg (in): Membrane potential
///   ca_reg (in): Intracellular calcium concentration
///   g_ahp (const): Max conductance
///   e_k (const): K⁺ reversal potential (usually -90 mV)
///   kd (const): Half-activation Ca²⁺ (SK ≈ 0.7 µM)
#[inline(always)]
pub fn op_bio_slow_ahp(
    regs: &mut [f32x8; 16],
    dst_current: u8,
    v_reg: u8,
    ca_reg: u8,
    g_ahp: f32,
    e_k: f32,
    kd: f32,
) {
    let v = regs[v_reg as usize];
    let ca = regs[ca_reg as usize];
    let g = f32x8::splat(g_ahp);
    let ek = f32x8::splat(e_k);
    let kd_v = f32x8::splat(kd);

    // Ca activation: Ca/(Ca + Kd)
    let act = ca / (ca + kd_v);
    let i = g * act * (v - ek);
    regs[dst_current as usize] = i;
}

/// ███ Natural Bio Block: NMDA Spike (Dendritic Coincidence Detector) ███████
///
/// Dendritic NMDA spike: when glutamate binding and depolarization coincide,
/// a regenerative NMDA current is generated (non-linear amplification).
/// Models the voltage-dependent Mg²⁺ block relief.
///
/// Computes:
///   mg_block = 1 / (1 + Mg/3.57 * exp(-0.062 * V * z))
///   I_NMDA = g_max * [glu] * mg_block * (V - E_rev)
///
/// Registers:
///   dst_current (out): NMDA spike current
///   v_reg (in): Dendritic voltage
///   glu_reg (in): Glutamate concentration
///   g_max (const): Max conductance
///   e_rev (const): Reversal potential (≈ 0 mV)
///   mg_conc (const): Extracellular Mg²⁺ concentration (≈ 1.2 mM)
#[inline(always)]
pub fn op_bio_nmda_spike(
    regs: &mut [f32x8; 16],
    dst_current: u8,
    v_reg: u8,
    glu_reg: u8,
    g_max: f32,
    e_rev: f32,
    mg_conc: f32,
) {
    let v = regs[v_reg as usize];
    let glu = regs[glu_reg as usize];
    let gm = f32x8::splat(g_max);
    let er = f32x8::splat(e_rev);

    // Mg block: 1 / (1 + Mg/3.57 * exp(-0.062 * V))
    let mg_ratio = f32x8::splat(mg_conc / 3.57);
    let exp_arg = f32x8::splat(-0.062) * v;
    let mg_exp = utils::fast_exp_simd(exp_arg);
    let mg_block = f32x8::splat(1.0) / (f32x8::splat(1.0) + mg_ratio * mg_exp);

    // I = g_max * [glu] * mg_block * (V - E_rev)
    regs[dst_current as usize] = gm * glu * mg_block * (v - er);
}
