//! Brain simulator module.

use super::BrainState;
use crate::data::ConnectomeData;
use crate::propagate::record_spikes;
use crate::types::RateParams;

// ─── Rate simulation ──────────────────────────────────

/// Simulate `n_steps` of Rate dynamics on a connectome.
///
/// Rate model stores firing probabilities in `state.v`.
/// Sets `state.spikes` before CSR propagation so recurrent excitation works.
pub fn sim_rate(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &RateParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let mut spike_buffer = vec![0i32; n_steps * n];
    if n_neurons == 0 || n_steps == 0 {
        return spike_buffer;
    }

    // Delegate to pillars-connectome Brain with the new pillars RateModel.
    // `dt` is in seconds (dt_ms/1000); the model emits a spike when r > 0.5.
    let mut brain = crate::Brain::<pillars::zoo::RateModel>::from_connectome(connectome);
    brain.configure(|p| {
        for i in 0..n_neurons {
            p.param_slice_mut(0)[i] = params.dt_ms / 1000.0; // dt (seconds)
            p.param_slice_mut(1)[i] = 1.0 / params.tau.max(1e-6); // inv_tau
            p.state_slice_mut(0)[i] = state.v[i]; // r
        }
    });

    let idx: Vec<u32> = (0..n_neurons as u32).collect();
    for step in 0..n_steps {
        let ext: Vec<f32> = (0..n_neurons)
            .map(|i| external_I.get(i).copied().unwrap_or(0.0))
            .collect();
        brain.inject(&idx, &ext);
        brain.step();
        record_spikes(brain.spikes(), n_neurons, &mut spike_buffer, step);
    }

    let r = brain.state(0);
    for i in 0..n_neurons {
        state.v[i] = r[i];
    }
    spike_buffer
}

