//! HAAP simulator — pillars [`NeuralPopulation<HAAPModel>`].
//!
//! HAAP (Hyperpolarization-Activated Adaptive Pulse-Bursting) is the
//! literature-atlas neuron model: an M-current (`u`, spike-frequency
//! adaptation) plus an h-current (`h`, hyperpolarization-activated rebound).
//! Unlike plain HH, adaptation bounds the firing rate — the motor-neuron
//! runaway seen with the HH conversion cannot happen.

use crate::data::ConnectomeData;
use crate::propagate::{csr_propagate_from_queue, push_spikes_to_queue, record_spikes};
use crate::types::SpikeQueue;
use pillars::core::NeuralPopulation;
use pillars::zoo::{HAAPModel, HAAPModelParamIdx};

/// Build a pillars [`NeuralPopulation<HAAPModel>`] with canonical VNC defaults
/// (the same parameter set as `make_vnc_params` in the HAAP zoo tests). The
/// per-neuron literature parameters are applied afterwards via
/// `BrainRunner::set_haap_param_column`.
pub fn haap_population(n: usize, dt: f32) -> NeuralPopulation<HAAPModel> {
    let mut pop = NeuralPopulation::<HAAPModel>::new(n);

    // Params (SoA columns) — canonical VNC defaults.
    pop.fill_param(HAAPModelParamIdx::dt.into(), dt);
    pop.fill_param(HAAPModelParamIdx::inv_cm.into(), 1.0);
    pop.fill_param(HAAPModelParamIdx::g_l.into(), 0.1);
    pop.fill_param(HAAPModelParamIdx::e_l.into(), -65.0);
    pop.fill_param(HAAPModelParamIdx::g_m.into(), 0.5);
    pop.fill_param(HAAPModelParamIdx::e_k.into(), -80.0);
    pop.fill_param(HAAPModelParamIdx::g_h.into(), 0.3);
    pop.fill_param(HAAPModelParamIdx::e_h.into(), -30.0);
    pop.fill_param(HAAPModelParamIdx::inv_tau_u.into(), 0.01);
    pop.fill_param(HAAPModelParamIdx::inv_tau_h.into(), 0.005);
    pop.fill_param(HAAPModelParamIdx::v_half_u.into(), -35.0);
    pop.fill_param(HAAPModelParamIdx::k_u.into(), 5.0);
    pop.fill_param(HAAPModelParamIdx::v_half_h.into(), -75.0);
    pop.fill_param(HAAPModelParamIdx::k_h.into(), 6.0);
    pop.fill_param(HAAPModelParamIdx::v_thresh.into(), -40.0);
    pop.fill_param(HAAPModelParamIdx::v_reset.into(), -55.0);
    pop.fill_param(HAAPModelParamIdx::delta_u.into(), 0.02);

    // State: rest at v = -65 mV with gates at their Boltzmann steady state —
    // u_∞ = 1/(1+exp(−(−65+35)/5)) ≈ 0.0025 (M closed) and
    // h_∞ = 1/(1+exp((−65+75)/6)) ≈ 0.159 (HCN mostly closed). This is the
    // quiescent fixed point, matching NEURON's `INITIAL` block; the old
    // u=0/h=1 init injected a spurious transient (fully-open HCN at rest).
    pop.fill_state(0, -65.0);
    pop.fill_state(1, 0.0025);
    pop.fill_state(2, 0.159);

    pop
}

/// Pillars-based HAAP simulation over a CSR connectome — same lifecycle as
/// [`sim_hh_pillars`](crate::sim::hh::sim_hh_pillars): add external current →
/// step population → record → CSR scatter into next step's input current.
pub fn sim_haap_pillars(
    connectome: &ConnectomeData,
    pop: &mut NeuralPopulation<HAAPModel>,
    external_I: &[f32],
    n_steps: usize,
    scale: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut queue = SpikeQueue::new(n_neurons);
    let mut spikes = vec![0i32; n];
    let mut i_in = vec![0.0f32; n]; // carry-over current buffer (like old cfg.Ii)

    for step in 0..n_steps {
        // 1. ADD external_I on top of i_in
        for i in 0..n.min(external_I.len()) {
            i_in[i] += external_I[i];
        }

        // 2. pillars HAAP step (current-based — M-current + h-current)
        pop.step(&i_in, &mut spikes);

        // 3. Push spikes to queue + record
        push_spikes_to_queue(&spikes, n_neurons, &mut queue);
        record_spikes(&spikes, n_neurons, &mut spike_buffer, step);

        // 4. Clear i_in
        i_in[..n].fill(0.0);

        // 5. CSR propagate from queue → i_in
        csr_propagate_from_queue(
            &connectome.row_ptr,
            &connectome.col_indices,
            &connectome.values,
            &connectome.nt_types,
            &mut queue,
            &mut i_in,
            scale,
        );
    }

    spike_buffer
}
