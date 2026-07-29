use wide::{f32x8, CmpGe, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// Dual-Exponential Synapse: single step (decay)
/// Matches NEURON's Exp2Syn: A' = -A/tau1, B' = -B/tau2
/// Conductance: g = B - A
/// Current: i = g_max * (B - A) * (V - E_rev)
///
/// Args:
///   dst_i: output register for current
///   v_reg: V register
///   a_reg: A state register (rise component)
///   b_reg: B state register (decay component)
///   g_max: maximum conductance
///   e_rev: reversal potential
///   tau1: rise time constant
///   tau2: decay time constant (must be > tau1)
///   dt: timestep
#[inline(always)]
pub fn op_bio_exp2syn_step(
    regs: &mut [f32x8; 16],
    dst_i: u8,
    v_reg: u8,
    a_reg: u8,
    b_reg: u8,
    g_max: f32,
    e_rev: f32,
    tau1: f32,
    tau2: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let a = regs[a_reg as usize];
    let b = regs[b_reg as usize];
    let gm = f32x8::splat(g_max);
    let er = f32x8::splat(e_rev);
    let dt_v = f32x8::splat(dt);

    // Decay: A *= exp(-dt/tau1), B *= exp(-dt/tau2)
    let decay_a = fast_exp_simd(-dt_v / f32x8::splat(tau1));
    let decay_b = fast_exp_simd(-dt_v / f32x8::splat(tau2));
    let a_new = a * decay_a;
    let b_new = b * decay_b;

    regs[a_reg as usize] = a_new;
    regs[b_reg as usize] = b_new;

    // Conductance: g = B - A, Current: i = g_max * g * (V - E_rev)
    let g = b_new - a_new;
    // Clamp g to non-negative (numerical safety)
    let g_safe = g.cmp_lt(f32x8::splat(0.0)).blend(f32x8::splat(0.0), g);
    regs[dst_i as usize] = gm * g_safe * (v - er);
}

/// Dual-Exponential Synapse: on spike (add weight)
/// Matches NEURON's Exp2Syn NET_RECEIVE:
///   A += weight * factor, B += weight * factor
/// where factor normalizes peak conductance to 1.
///
/// Args:
///   a_reg: A state register
///   b_reg: B state register
///   spike_reg: spike flag register (1.0 if spike occurred)
///   weight: synaptic weight (conductance increment)
///   tau1: rise time constant
///   tau2: decay time constant
#[inline(always)]
pub fn op_bio_exp2syn_spike(
    regs: &mut [f32x8; 16],
    a_reg: u8,
    b_reg: u8,
    spike_reg: u8,
    weight: f32,
    tau1: f32,
    tau2: f32,
) {
    let spike = regs[spike_reg as usize];
    let a = regs[a_reg as usize];
    let b = regs[b_reg as usize];
    let wt = f32x8::splat(weight);
    let one = f32x8::splat(1.0);

    // Normalization factor (NEURON standard)
    // tp = (tau1*tau2)/(tau2-tau1) * ln(tau2/tau1)
    // factor = 1 / (-exp(-tp/tau1) + exp(-tp/tau2))
    let t1 = f32x8::splat(tau1);
    let t2 = f32x8::splat(tau2);
    
    // Guard against numerical issues when tau1 ≈ tau2
    let diff = t2 - t1;
    let diff_safe = diff.cmp_lt(f32x8::splat(1e-9)).blend(f32x8::splat(1e-9), diff);
    let tp = (t1 * t2) / diff_safe * (t2 / t1).ln();
    let factor = one / (-fast_exp_simd(-tp / t1) + fast_exp_simd(-tp / t2));

    // On spike: A += weight * factor, B += weight * factor
    let addition = wt * factor * spike;
    regs[a_reg as usize] = a + addition;
    regs[b_reg as usize] = b + addition;
}

/// Single Electrode Voltage Clamp (3-level, timed)
/// Matches NEURON's SEClamp with 3 voltage levels and durations.
///   I_clamp = (V_command - V) / R_s  (when active)
///   I_clamp = 0                      (when off, after all 3 phases)
///
/// Phase detection based on elapsed time register (t_reg).
/// The user increments t_reg by dt each step externally.
///
/// Args:
///   dst_i: output register for clamp current
///   v_reg: V register
///   t_reg: elapsed time register (incremented each step)
///   amp1, amp2, amp3: voltage levels for 3 phases (mV)
///   dur1, dur2, dur3: durations of each phase (ms)
///   r_series: series resistance (MΩ)
///   dt: timestep (for advancing time if needed)
#[inline(always)]
pub fn op_bio_se_clamp3(
    regs: &mut [f32x8; 16],
    dst_i: u8,
    v_reg: u8,
    t_reg: u8,
    amp1: f32,
    amp2: f32,
    amp3: f32,
    dur1: f32,
    dur2: f32,
    dur3: f32,
    r_series: f32,
) {
    let v = regs[v_reg as usize];
    let t = regs[t_reg as usize];
    let rs = f32x8::splat(r_series.max(1e-9));

    // Phase boundaries (cumulative)
    let d1 = f32x8::splat(dur1);
    let d2 = f32x8::splat(dur1 + dur2);
    let d3 = f32x8::splat(dur1 + dur2 + dur3);

    // Phase masks: cascaded blend for clean priority
    // Phase 1: t < d1
    // Phase 2: d1 <= t < d2
    // Phase 3: d2 <= t < d3
    // Off:     t >= d3
    let in_phase23 = t.cmp_ge(d1);  // phase 2, 3, or off
    let in_phase34 = t.cmp_ge(d2);  // phase 3 or off
    let in_off = t.cmp_ge(d3);      // off (past all phases)

    // Build V_command: cascade from phase 1 → phase 2 → phase 3 → off
    let vc = f32x8::splat(amp1);
    let vc = in_phase23.blend(f32x8::splat(amp2), vc);
    let vc = in_phase34.blend(f32x8::splat(amp3), vc);
    let vc = in_off.blend(v, vc);  // off → vc=v → I=0

    // I_clamp = (V_command - V) / R_s
    regs[dst_i as usize] = (vc - v) / rs;
}

/// Voltage Clamp: single-level version (convenience wrapper)
/// Matches NEURON's SEClamp with one level:
///   I_clamp = (V_hold - V) / R_s
#[inline(always)]
pub fn op_bio_voltage_clamp(
    regs: &mut [f32x8; 16],
    dst_i: u8,
    v_reg: u8,
    v_hold: f32,
    r_series: f32,
) {
    let v = regs[v_reg as usize];
    let vh = f32x8::splat(v_hold);
    let rs_safe = f32x8::splat(r_series.max(1e-9));
    regs[dst_i as usize] = (vh - v) / rs_safe;
}

/// Na⁺ Concentration Dynamics
///
/// Tracks intracellular [Na⁺] during spiking activity.
/// Na⁺ enters the cell through voltage-gated Na⁺ channels (per spike).
/// Recovery via Na⁺/K⁺ ATPase (coupled with ChemNaKPump opcode).
///
///   d[Na⁺]_i/dt = -α * I_Na - ([Na⁺]_i - [Na⁺]_rest) / τ
///
/// Where:
///   α converts Na⁺ current to concentration (mM per nA/µF·ms)
///   I_Na is the total Na⁺ current (inward negative → d[Na]/dt positive)
///   τ is Na⁺ extrusion time constant (≈ 10-50 ms via Na/K pump)
///
/// Nernst potential:
///   E_Na = 26.7 * ln([Na⁺]ₒ / [Na⁺]ᵢ) mV at 37°C
///
/// Args:
///   na_in_reg: intracellular [Na⁺] in mM (updated in place)
///   e_na_reg:  E_Na reversal potential in mV (output)
///   i_na_reg:  total Na⁺ current register (inward negative)
///   na_rest:   resting [Na⁺]_i (≈ 12 mM for fly neurons)
///   na_out:    extracellular [Na⁺] (≈ 145 mM, constant)
///   alpha:     conversion factor (≈ 0.01 mM per nA/µF·ms)
///   tau:       extrusion time constant (≈ 30 ms)
///   dt:        timestep
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_na_concentration(
    regs: &mut [f32x8; 16],
    na_in_reg: u8,
    e_na_reg: u8,
    i_na_reg: u8,
    na_rest: f32,
    na_out: f32,
    alpha: f32,
    tau: f32,
    dt: f32,
) {
    let na = regs[na_in_reg as usize];
    let i_na = regs[i_na_reg as usize];
    let rest = f32x8::splat(na_rest);
    let no = f32x8::splat(na_out);
    let al = f32x8::splat(alpha);
    let tau_v = f32x8::splat(tau.max(1e-6));
    let dt_v = f32x8::splat(dt);

    // d[Na]_i/dt = -α * I_Na - ([Na]_i - [Na]_rest) / τ
    // Note: I_Na is outward-positive. Inward Na⁺ current (negative) → +d[Na]
    let d_na = -al * i_na - (na - rest) / tau_v;
    let new_na = (na + d_na * dt_v).max(f32x8::splat(0.0));
    regs[na_in_reg as usize] = new_na;

    // Nernst: E_Na = 26.7 * ln([Na]_out / [Na]_i) mV at 37°C
    let ratio = (no / new_na).max(f32x8::splat(1.01)).min(f32x8::splat(100.0));
    let ln_ratio = ratio.ln();
    regs[e_na_reg as usize] = f32x8::splat(26.7) * ln_ratio;
}

/// Ca²⁺ Buffer Dynamics (Fixed Buffer)
///
/// Models Ca²⁺ binding to a fixed intracellular buffer:
///   B + Ca²⁺  <-->  CaB
///
///   d[CaB]/dt = k_on * [Ca²⁺] * (B_total - [CaB]) - k_off * [CaB]
///   d[Ca²⁺]_free/dt = -d[CaB]/dt + d[Ca²⁺]_influx/dt
///
/// More realistic than simple exponential decay because:
///   1. Buffer saturates at high [Ca²⁺]
///   2. Binding/unbinding kinetics matter for fast events
///   3. Ca²⁺ spikes (from VGCC) are buffered on ms timescale
///
/// Typical values:
///   B_total: 100-500 µM (calbindin, calretinin, parvalbumin)
///   k_on: 0.1-1 µM⁻¹ms⁻¹
///   k_off: 0.01-0.5 ms⁻¹
///   K_d = k_off/k_on: 0.5-5 µM
///
/// Args:
///   ca_free_reg: free [Ca²⁺] in µM (in/out)
///   ca_buf_reg:  Ca²⁺ bound to buffer in µM (in/out)
///   ca_influx_reg: Ca²⁺ influx from channels (µM, added this step)
///   buffer_total: total buffer concentration (µM)
///   k_on: binding rate (µM⁻¹ms⁻¹)
///   k_off: unbinding rate (ms⁻¹)
///   dt: timestep
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn op_bio_ca_buffer(
    regs: &mut [f32x8; 16],
    ca_free_reg: u8,
    ca_buf_reg: u8,
    ca_influx_reg: u8,
    buffer_total: f32,
    k_on: f32,
    k_off: f32,
    dt: f32,
) {
    let ca = regs[ca_free_reg as usize];
    let cab = regs[ca_buf_reg as usize];
    let influx = regs[ca_influx_reg as usize];
    let b_total = f32x8::splat(buffer_total);
    let kon = f32x8::splat(k_on);
    let koff = f32x8::splat(k_off);
    let dt_v = f32x8::splat(dt);
    let zero = f32x8::splat(0.0);

    // Free buffer: [B] = B_total - [CaB]
    let free_buf = (b_total - cab).max(zero);

    // Binding kinetics: d[CaB]/dt = k_on * [Ca] * [B] - k_off * [CaB]
    let binding = kon * ca * free_buf;
    let unbinding = koff * cab;
    let dcab = (binding - unbinding) * dt_v;

    // Add Ca²⁺ influx to free pool first
    let ca_with_influx = ca + influx;

    // Buffer binding removes free Ca²⁺, adds to CaB
    let new_cab = (cab + dcab).max(zero).min(b_total);
    // Conservation: d[Ca_free] = -d[CaB]
    let new_ca = (ca_with_influx - dcab).max(zero);

    regs[ca_free_reg as usize] = new_ca;
    regs[ca_buf_reg as usize] = new_cab;
}

#[inline(always)]
pub fn op_bio_sigmoid(regs: &mut [f32x8; 16], dst: u8, v_reg: u8, v_half: f32, k: f32) {
    regs[dst as usize] = crate::math::sigmoid::sigmoid(regs[v_reg as usize], v_half, k);
}

/// Sigmoid with ALL parameters from registers (for per-neuron params).
/// Same as op_bio_sigmoid but v_half and k come from registers, not float literals.
#[inline(always)]
pub fn op_bio_sigmoid_reg(regs: &mut [f32x8; 16], dst: u8, v_reg: u8, v_half_reg: u8, k_reg: u8) {
    let v = regs[v_reg as usize];
    let vh = regs[v_half_reg as usize];
    let k_v = regs[k_reg as usize];
    let one = f32x8::splat(1.0);
    
    // 1 / (1 + exp(-(V - v_half) / k))
    let exp_arg = -(v - vh) / k_v;
    let exp_val = fast_exp_simd(exp_arg);
    regs[dst as usize] = one / (one + exp_val);
}

#[inline(always)]
pub fn op_bio_tau_bell_curve(regs: &mut [f32x8; 16], dst: u8, v_reg: u8, tau_base: f32, tau_amp: f32, v_center: f32, sigma: f32) {
    let v = regs[v_reg as usize];
    let tb = f32x8::splat(tau_base);
    let ta = f32x8::splat(tau_amp);
    let vc = f32x8::splat(v_center);
    let sig = f32x8::splat(sigma);
    
    // tau_base + tau_amp * exp(-((V - v_center) / sigma)^2)
    let diff = (v - vc) / sig;
    let sq_diff = -(diff * diff);
    
    let exp_val = fast_exp_simd(sq_diff);
    regs[dst as usize] = tb + ta * exp_val;
}

#[inline(always)]
pub fn op_bio_gate_integrate(regs: &mut [f32x8; 16], gate_reg: u8, x_inf_reg: u8, tau_reg: u8, dt: f32) {
    let gate_old = regs[gate_reg as usize];
    let x_inf = regs[x_inf_reg as usize];
    let tau = regs[tau_reg as usize];
    let dt_v = f32x8::splat(dt);
    
    // exp(-dt / tau)
    // Avoid division by zero if tau is too small
    let tau_safe = tau.cmp_lt(f32x8::splat(1e-6)).blend(f32x8::splat(1e-6), tau);
    let decay = fast_exp_simd(-dt_v / tau_safe);
    
    regs[gate_reg as usize] = x_inf + (gate_old - x_inf) * decay;
}
