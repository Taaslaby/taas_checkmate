use wide::f32x8;

/// STDP Potentiation (LTP — Long-Term Potentiation):
///
///   w += lr * pre_trace * post_spike
///
/// If the post-synaptic neuron fires while the pre-synaptic trace is high
/// (pre before post), the weight increases.
///
/// Args:
///   w_reg: weight to update
///   pre_trace_reg: pre-synaptic spike trace (how recently pre fired)
///   post_spike_reg: 1.0 if post-synaptic neuron spiked, 0 otherwise
///   lr: learning rate for potentiation (positive)
#[inline(always)]
pub fn op_stdp_potentiation(
    regs: &mut [f32x8; 16],
    weight_reg: u8,
    pre_trace_reg: u8,
    post_spike_reg: u8,
    lr: f32,
) {
    let w = regs[weight_reg as usize];
    let pre_tr = regs[pre_trace_reg as usize];
    let post_spk = regs[post_spike_reg as usize];
    let lr_v = f32x8::splat(lr);

    // w += lr * pre_trace * post_spike
    regs[weight_reg as usize] = w + lr_v * pre_tr * post_spk;
}

/// STDP Depression (LTD — Long-Term Depression):
///
///   w += lr * post_trace * pre_spike
///   (where lr is typically negative for depression)
///
/// If the pre-synaptic neuron fires while the post-synaptic trace is high
/// (post before pre), the weight decreases.
///
/// Args:
///   w_reg: weight to update
///   post_trace_reg: post-synaptic spike trace (how recently post fired)
///   pre_spike_reg: 1.0 if pre-synaptic neuron spiked, 0 otherwise
///   lr: learning rate for depression (typically negative)
#[inline(always)]
pub fn op_stdp_depression(
    regs: &mut [f32x8; 16],
    weight_reg: u8,
    post_trace_reg: u8,
    pre_spike_reg: u8,
    lr: f32,
) {
    let w = regs[weight_reg as usize];
    let post_tr = regs[post_trace_reg as usize];
    let pre_spk = regs[pre_spike_reg as usize];
    let lr_v = f32x8::splat(lr);

    // w += lr * post_trace * pre_spike
    regs[weight_reg as usize] = w + lr_v * post_tr * pre_spk;
}
