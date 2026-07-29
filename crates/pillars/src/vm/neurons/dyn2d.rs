use wide::{f32x8, CmpGe};

/// 2D Neuron Dynamics (Forward Euler Step)
///
/// General quadratic+linear 2D system:
///   dv/dt = c₁v² + c₂v + c₃ - u + I
///   du/dt = a(bv - u)
///
/// Specializes to Izhikevich (2003) with c₁=0.04, c₂=5, c₃=140.
/// Covers 20+ neuron types via (a, b, c, d):
///
/// | Type | a | b | c | d | Description |
/// |------|---|---|---|---|-------------|
/// | RS | 0.02 | 0.2 | -65 | 8 | Regular spiking |
/// | IB | 0.02 | 0.2 | -55 | 4 | Intrinsically bursting |
/// | CH | 0.02 | 0.2 | -50 | 2 | Chattering |
/// | FS | 0.1 | 0.2 | -65 | 2 | Fast spiking |
/// | TC | 0.02 | 0.25 | -65 | 0.05 | Thalamo-cortical |
/// | RZ | 0.1 | 0.26 | -65 | 2 | Resonator |
/// | LTS | 0.02 | 0.25 | -65 | 2 | Low-threshold spiking |
///
/// Uses forward Euler. dt ≤ 0.1ms recommended.
/// Spike detection + reset uses SEPARATE opcode Dyn2DReset.
///
/// Args:
///   v_reg: membrane potential
///   u_reg: recovery variable
///   i_reg: input current
///   a: recovery time constant
///   b: recovery sensitivity
///   dt: timestep
#[inline(always)]
pub fn op_dyn_2d_step(
    regs: &mut [f32x8; 16],
    v_reg: u8,
    u_reg: u8,
    i_reg: u8,
    a: f32,
    b: f32,
    dt: f32,
) {
    let v = regs[v_reg as usize];
    let u = regs[u_reg as usize];
    let i = regs[i_reg as usize];
    let dt_v = f32x8::splat(dt);
    let a_v = f32x8::splat(a);
    let b_v = f32x8::splat(b);

    let c1 = f32x8::splat(0.04);
    let c2 = f32x8::splat(5.0);
    let c3 = f32x8::splat(140.0);

    let dv = (c1 * v * v + c2 * v + c3 - u + i) * dt_v;
    let v_new = v + dv;

    let du = a_v * (b_v * v - u) * dt_v;
    let u_new = u + du;

    regs[v_reg as usize] = v_new;
    regs[u_reg as usize] = u_new;
}

/// 2D Dynamics Spike Reset
///
///   if v ≥ 30: v = c, u += d, spike = 1
///   else: spike = 0
///
/// Runs AFTER Dyn2DStep. Threshold 30mV is hardcoded.
///
/// Args:
///   spike_reg: output (1.0 if spike, else 0.0)
///   v_reg: membrane potential (reset to c if spike)
///   u_reg: recovery variable (increase by d if spike)
///   c: reset voltage
///   d: reset increment
#[inline(always)]
pub fn op_dyn_2d_reset(
    regs: &mut [f32x8; 16],
    spike_reg: u8,
    v_reg: u8,
    u_reg: u8,
    c: f32,
    d: f32,
) {
    let v = regs[v_reg as usize];
    let u = regs[u_reg as usize];
    let c_v = f32x8::splat(c);
    let d_v = f32x8::splat(d);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let v_thresh = f32x8::splat(30.0);

    let spike_mask = v.cmp_ge(v_thresh);
    regs[spike_reg as usize] = spike_mask.blend(one, zero);
    regs[v_reg as usize] = spike_mask.blend(c_v, v);
    let u_new = u + d_v;
    regs[u_reg as usize] = spike_mask.blend(u_new, u);
}
