//! Brain simulator module.

use crate::data::ConnectomeData;
use crate::propagate::{
    csr_propagate_conductance_from_queue, csr_propagate_from_queue, push_spikes_to_queue,
    record_spikes,
};
use crate::types::SpikeQueue;
use pillars::core::NeuralPopulation;
use pillars::kernels::synapse::{DecayFactors, SynapticKinetics};
use pillars::zoo::{HHModel, HHParamIdx};

// ─── HH simulation via pillars (NeuralPopulation<HHModel>) ──

/// Build a pillars [`NeuralPopulation<HHModel>`] matching the legacy
/// `pillars::zoo::neurons::HHModel` parameter set **exactly**
/// (E_leak = -65 mV, gates at steady state for V=-65, Q10 = 1.0 @ 6.3°C).
///
/// This is the parity contract used by `bench_hh_parity`: the pillars engine
/// must reproduce the old `step_hh_lut` spike counts neuron-for-neuron.
pub fn hh_population_standard(n: usize, dt: f32) -> NeuralPopulation<HHModel> {
    let mut pop = NeuralPopulation::<HHModel>::new(n);

    // Params (SoA columns) — identical to old `standard_config`:
    pop.fill_param(HHParamIdx::DT.into(), dt);
    pop.fill_param(HHParamIdx::G_Na.into(), 120.0);
    pop.fill_param(HHParamIdx::E_Na.into(), 50.0);
    pop.fill_param(HHParamIdx::G_K.into(), 36.0);
    pop.fill_param(HHParamIdx::E_K.into(), -77.0);
    pop.fill_param(HHParamIdx::G_Leak.into(), 0.3);
    pop.fill_param(HHParamIdx::E_Leak.into(), -65.0);
    pop.fill_param(HHParamIdx::Q10.into(), 1.0); // 6.3°C → Q10 = 1
    pop.fill_param(HHParamIdx::V_Thresh.into(), 0.0);
    pop.fill_param(HHParamIdx::Inv_Cm.into(), 1.0);
    // CaV-T: standard HH has NO Ca²⁺ current (g_CaT = 0)
    pop.fill_param(HHParamIdx::G_CaT.into(), 0.0);
    pop.fill_param(HHParamIdx::E_CaT.into(), 120.0);

    // State: v = -65, gates at steady state @ -65 (m=0.0529, h=0.596, n=0.318,
    // c=0.19), was_below = 1.0 → first rising edge through 0 fires (1/AP).
    pop.fill_state(0, -65.0);
    pop.fill_state(1, 0.0529);
    pop.fill_state(2, 0.596);
    // n-gate steady state at -65 mV ≈ 0.318 (biological value, NOT 1/π)
    #[allow(clippy::approx_constant)]
    pop.fill_state(3, 0.318);
    // CaV-T steady-state activation at -65 mV: 1/(1+exp(-(-65+56)/6.2)) ≈ 0.19
    pop.fill_state(4, 0.19);
    pop.fill_state(5, 1.0);

    pop
}

/// Build a pillars [`NeuralPopulation<HHModel>`] with **pacemaking** channels
/// (NaV + KV + CaV-T): the T-type Ca²⁺ current (g_CaT = 0.5, E_CaT = +120 mV)
/// depolarizes the membrane past NaV threshold on its own → neurons fire
/// **spontaneously** without any input current (rhythmic VNC/CPG activity).
///
/// ⚠️ E_leak must be −54.387 mV (the true squid-axon leak reversal), NOT −65:
/// with E_leak=−65 the CaT-driven depolarization is cancelled by the leak and
/// the neuron never fires (verified empirically). This matches the legacy
/// `pacemaking_channels` reference test (`hh_step_scalar` with e_leak=−54.387).
pub fn hh_population_pacemaking(n: usize, dt: f32) -> NeuralPopulation<HHModel> {
    let mut pop = hh_population_standard(n, dt);
    pop.fill_param(HHParamIdx::E_Leak.into(), -54.387);
    pop.fill_param(HHParamIdx::G_CaT.into(), 0.5);
    pop.fill_param(HHParamIdx::E_CaT.into(), 120.0);
    pop
}

/// Simulate `n_steps` of Hodgkin-Huxley on a connectome using the pillars engine
/// ([`NeuralPopulation<HHModel>`] + SIMD).
///
/// 1. ADD external_I on top of the carry-over current buffer
/// 2. `pop.step()` — SIMD HH dynamics + edge-triggered spike detection
/// 3. Push spikes to queue → record
/// 4. Clear current buffer
/// 5. CSR propagate: `I[post] += W · nt_types[pre] · scale`
pub fn sim_hh_pillars(
    connectome: &ConnectomeData,
    pop: &mut NeuralPopulation<HHModel>,
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

        // 2. pillars HH step (Ii clamp ±100 happens inside NeuralPopulation)
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

/// Pillars-based HH with conductance-based synaptic kinetics — same lifecycle
/// as [`sim_hh_synaptic`]: decay → set I = external → compute I_syn → step →
/// record → CSR jump conductances.
pub fn sim_hh_synaptic_pillars(
    connectome: &ConnectomeData,
    pop: &mut NeuralPopulation<HHModel>,
    external_I: &[f32],
    n_steps: usize,
    kinetics: &mut SynapticKinetics,
    scale: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let mut queue = SpikeQueue::new(n_neurons);
    let mut spikes = vec![0i32; n];
    let dt = pop.param_slice(HHParamIdx::DT.into())[0];
    let factors = DecayFactors::new(dt);
    let mut i_in = vec![0.0f32; n];

    for step in 0..n_steps {
        // 1. Decay conductances
        kinetics.decay_all(&factors);

        // 2. Set I = external_I (fresh each step — fill first, no stale tail)
        i_in[..n].fill(0.0);
        for (i, &ext) in external_I.iter().enumerate().take(n) {
            i_in[i] += ext;
        }

        // 3. Compute I_syn from conductances and add to I
        kinetics.compute_currents(pop.state_slice(0), &mut i_in);

        // 4. pillars HH step
        pop.step(&i_in, &mut spikes);

        // 5. Push spikes to queue + record
        push_spikes_to_queue(&spikes, n_neurons, &mut queue);
        record_spikes(&spikes, n_neurons, &mut spike_buffer, step);

        // 6. CSR propagate: jump conductances
        csr_propagate_conductance_from_queue(
            &connectome.row_ptr,
            &connectome.col_indices,
            &connectome.values,
            &connectome.receptor_types,
            &mut queue,
            kinetics,
            scale,
        );
    }

    spike_buffer
}
