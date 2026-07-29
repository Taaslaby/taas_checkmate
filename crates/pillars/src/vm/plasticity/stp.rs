use wide::{f32x8, CmpGt, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Tsodyks-Markram Short-Term Plasticity Model
///
/// A model of dynamic synaptic transmission with:
/// - **Facilitation**: u increases with each spike (use-dependent)
/// - **Depression**: x (fraction of available resources) decreases with each spike
/// - **Recovery**: both u and x recover exponentially between spikes
///
/// State variables (all in the state array as per-neuron values):
/// - u_reg: utilization variable (release probability, default ~0.2)
/// - x_reg: fraction of available vesicle pool (0..1)
/// - z_reg: fraction of recovered resources (0..1) — derived: z = 1 - x - y,
///   but we store it for numerical stability
///
/// Parameters:
/// - u0: baseline release probability
/// - tau_rec: recovery time constant for depression (ms)
/// - tau_facil: facilitation time constant (ms)
/// - dt: simulation timestep
///
/// Uses the full Tsodyks-Markram model:
/// - Facilitation on spike: u += U0 * (1 - u)
/// - Depression on spike: x -= u * x
/// - Exponential recovery between spikes
///
/// tau_facil is computed internally as tau_rec / 5.0,
/// which is biologically realistic (facilitation decays ~5x faster than depression).
///
/// Synaptic efficacy (readout via STPSynapticEfficacy): g_eff = g_max * u * x
#[inline(always)]
pub fn op_tsodyks_markram(
    regs: &mut [f32x8; 16],
    u_reg: u8,
    x_reg: u8,
    spike_reg: u8,
    u0: f32,
    tau_rec: f32,
    dt: f32,
) {
    let u_old = regs[u_reg as usize];
    let x_old = regs[x_reg as usize];
    let spike = regs[spike_reg as usize];

    let tau_facil = tau_rec / 5.0;
    let u0_v = f32x8::splat(u0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);

    // Guard against div-by-zero for tau_rec
    let tau_rec = if tau_rec < 1e-8 { 1e-8 } else { tau_rec };
    let tau_facil = if tau_facil < 1e-8 { 1e-8 } else { tau_facil };

    // Exponential decays
    let decay_rec = fast_exp_simd(f32x8::splat(-dt / tau_rec));
    let decay_facil = fast_exp_simd(f32x8::splat(-dt / tau_facil));

    // ── Facilitation update ──
    // Spike-triggered: u += u0 * (1 - u)
    let u_spike = u_old + u0_v * (one - u_old);
    // Decay toward baseline: u += (u0 - u) * (1 - exp(-dt/tau_facil))
    let u_decay = u_old + (u0_v - u_old) * (one - decay_facil);

    // Combine: if spike >= 0.5 use spike update, else use decay update
    let spike_mask = spike.cmp_gt(f32x8::splat(0.5));
    let u_new = spike_mask.blend(u_spike, u_decay);

    // ── Depression update ──
    // Spike-triggered: x -= u * x
    let x_spike = x_old - u_new * x_old;
    // Decay: x += (1 - x) * (1 - exp(-dt/tau_rec))
    let x_decay = x_old + (one - x_old) * (one - decay_rec);

    let x_new = spike_mask.blend(x_spike, x_decay);

    // Clamp to [0, 1] for numerical stability
    let u_clamped = u_new.cmp_gt(one).blend(one, u_new);
    let u_clamped = u_clamped.cmp_lt(zero).blend(zero, u_clamped);
    let x_clamped = x_new.cmp_gt(one).blend(one, x_new);
    let x_clamped = x_clamped.cmp_lt(zero).blend(zero, x_clamped);

    regs[u_reg as usize] = u_clamped;
    regs[x_reg as usize] = x_clamped;
}

/// Compute synaptic efficacy from Tsodyks-Markram state:
/// g_eff = g_max * u * x
#[inline(always)]
pub fn op_stp_syn_eff(
    regs: &mut [f32x8; 16],
    g_eff_reg: u8,
    u_reg: u8,
    x_reg: u8,
    g_max: f32,
) {
    let u = regs[u_reg as usize];
    let x = regs[x_reg as usize];
    let g_max_v = f32x8::splat(g_max);

    regs[g_eff_reg as usize] = g_max_v * u * x;
}
