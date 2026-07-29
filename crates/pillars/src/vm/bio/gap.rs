use wide::f32x8;

/// Gap Junction (Electrical Synapse) Current
///
///   I_gap = g_gap * (V_pre - V_post)
///
/// Unlike chemical synapses (which use spikes + neurotransmitters),
/// gap junctions directly couple the membrane potentials of two neurons
/// via connexin pores. Current flows bidirectionally proportional to
/// the voltage difference.
///
/// In the TNMT VM, gap junctions are typically NOT stored in the
/// Connectome delay ring buffer. Instead, they are computed fresh
/// each timestep from the current V_pre and V_post registers.
///
/// For a gap junction network, each neuron accumulates:
///   I_gap_total = Σ g_gap * (V_neighbor - V_self)
///
/// This opcode computes one such contribution. Call it once per
/// gap junction connection per timestep.
///
/// Biological notes:
///   - g_gap ≈ 0.05-0.5 nS (rectifying electrical synapses)
///   - Gap rectification (asymmetric) is not modeled here
///   - ~25% of fly lamina (LA) synapses are electrical (amacrine cells)
///   - Essential for neural synchrony and fast signal propagation
///
/// Args:
///   i_reg: output current register
///   pre_v_reg: register with pre-synaptic membrane potential
///   post_v_reg: register with post-synaptic membrane potential
///   g_gap: gap junction conductance (scalar, applied to all 8 SIMD lanes)
#[inline(always)]
pub fn op_bio_gap_junction(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    pre_v_reg: u8,
    post_v_reg: u8,
    g_gap: f32,
) {
    let pre_v = regs[pre_v_reg as usize];
    let post_v = regs[post_v_reg as usize];
    let gg = f32x8::splat(g_gap);

    // I_gap = g_gap * (V_pre - V_post)
    // Positive when pre is depolarized relative to post → inward (depolarizing) for post
    regs[i_reg as usize] = gg * (pre_v - post_v);
}
