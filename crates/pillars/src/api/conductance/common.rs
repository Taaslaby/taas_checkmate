use crate::kernels::hh_channels::ChannelMeta;

/// Spike-detection threshold in mV — matches `HHModel`'s `v_thresh = 0.0`
/// (rising-edge crossing through 0 mV, one spike per action potential).
pub(crate) const SPIKE_THRESHOLD_MV: f32 = 0.0;

/// Numerical floor on `α + β` before forming `x_∞ = α/(α+β)`. Below it both
/// rates are effectively zero and the steady state is indeterminate, so we fall
/// back to [`STEADY_STATE_NEUTRAL`]. Mirrors `hh_channels::steady_state_gates`
/// bit-for-bit so resting-state init matches the engine's own helper.
pub(crate) const STEADY_STATE_SUM_EPSILON: f32 = 1e-12;

/// Neutral fallback `x_∞` when `α + β` underflows [`STEADY_STATE_SUM_EPSILON`]
/// (gate with no voltage dependence at this potential).
pub(crate) const STEADY_STATE_NEUTRAL: f32 = 0.5;

/// Steady-state gate values `x_∞ = α(v)/(α(v)+β(v))` in gate order
/// (channel-major), mirroring `hh_channels::steady_state_gates` bit-for-bit but
/// over the `ChannelMeta` representation this population stores. The two
/// containers hold identical per-gate kinetics (same order), so the fallback
/// below keeps resting-state init consistent with the engine's own helper.
pub(crate) fn steady_state_gates(v: f32, channel_meta: &[ChannelMeta]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channel_meta {
        for kinetics in &ch.kinetics {
            let (a, b) = kinetics.alpha_beta(v);
            let sum = a + b;
            gates.push(if sum > STEADY_STATE_SUM_EPSILON {
                a / sum
            } else {
                STEADY_STATE_NEUTRAL
            });
        }
    }
    gates
}
