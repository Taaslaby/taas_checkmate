use crate::kernels::hh_channels::ChannelMeta;

/// Spike-detection threshold in mV — matches `HHModel`'s `v_thresh = 0.0`
/// (rising-edge crossing through 0 mV, one spike per action potential).
pub(crate) const SPIKE_THRESHOLD_MV: f32 = 0.0;

/// Steady-state gate values `x_∞ = α(v)/(α(v)+β(v))` in gate order
/// (channel-major) over the [`ChannelMeta`] representation this population
/// stores. Delegates the per-gate math to
/// [`GateKinetics::steady_state_at`](crate::kernels::hh_channels::GateKinetics::steady_state_at)
/// — the single source shared with `hh_channels::steady_state_gates`, so
/// resting-state init can never drift from the engine's own helper.
pub(crate) fn steady_state_gates(v: f32, channel_meta: &[ChannelMeta]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channel_meta {
        for kinetics in &ch.kinetics {
            gates.push(kinetics.steady_state_at(v));
        }
    }
    gates
}
