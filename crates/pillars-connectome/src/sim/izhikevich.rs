//! Brain simulator module.

use super::lif::step_izhikevich_dual_pillars;
use super::BrainState;
use crate::data::ConnectomeData;
use crate::propagate::{csr_propagate_from_queue, push_spikes_to_queue, record_spikes};
use crate::types::{IzhikevichDualParams, IzhikevichParams};
use pillars::kernels::synapse::SynapticKinetics;

// ─── Izhikevich simulation ─────────────────────────────

/// Pad a Vec to `n_padded` by replicating the last element.
pub fn pad_to<T: Copy>(v: &[T], n_padded: usize) -> Vec<T> {
    let mut padded = v.to_vec();
    padded.resize(n_padded, *v.last().unwrap_or(&v[0]));
    padded
}

/// Simulate `n_steps` of Izhikevich dynamics on a connectome via the
/// pillars-native orchestrator (`pillars-connectome::Brain`).
///
/// Note: the legacy `sim_izhikevich` dropped the CSR propagation (propagated
/// current was overwritten by `external_I`); here spikes genuinely reach
/// post-synaptic neurons — an intentional fix.
pub fn sim_izhikevich(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let mut spike_buffer = vec![0i32; n_steps * n];
    if n_neurons == 0 || n_steps == 0 {
        return spike_buffer;
    }

    let mut brain = crate::Brain::<pillars::zoo::IzhikevichModel>::from_connectome(connectome);
    brain.configure(|p| {
        for i in 0..n_neurons {
            p.param_slice_mut(0)[i] = params.dt_ms; // dt
            p.param_slice_mut(1)[i] = params.a[i]; // a
            p.param_slice_mut(2)[i] = params.b[i]; // b
            p.param_slice_mut(3)[i] = params.c[i]; // c
            p.param_slice_mut(4)[i] = params.d[i]; // d
            p.param_slice_mut(5)[i] = params.tau_ref;
            p.param_slice_mut(6)[i] = 30.0; // v_thresh (legacy hardcoded)
            p.state_slice_mut(0)[i] = state.v[i]; // v
            p.state_slice_mut(1)[i] = state.u[i]; // u
            p.state_slice_mut(2)[i] = state.refractory[i]; // refr
        }
    });

    let idx: Vec<u32> = (0..n_neurons as u32).collect();
    for step in 0..n_steps {
        let ext: Vec<f32> = (0..n_neurons)
            .map(|i| external_I.get(i).copied().unwrap_or(0.0))
            .collect();
        brain.inject(&idx, &ext);
        brain.step();
        for i in 0..n_neurons {
            spike_buffer[step * n + i] = brain.spikes()[i];
        }
    }

    // Sync back the membrane state (trace/slow_state are not part of the
    // pillars Izhikevich model — left untouched).
    let v = brain.state(0);
    let u = brain.state(1);
    let refr = brain.state(2);
    for i in 0..n_neurons {
        state.v[i] = v[i];
        state.u[i] = u[i];
        state.refractory[i] = refr[i];
    }
    spike_buffer
}

// ─── Izhikevich Dual Memory simulation ────────────────

/// Simulate `n_steps` of Izhikevich dual memory dynamics on a connectome.
///
/// Extends [`sim_izhikevich`] with a per-neuron **slow state** `m` that
/// accumulates recent spiking activity and modulates input current.
///
/// **Slow memory pathway** (Sun et al. 2026):
/// ```text
/// m[t] = τ_m · m[t-1] + (1-τ_m) · spike[t-1]
/// I += w_m · m[t-1]
/// ```
///
/// The slow state is stored in `state.slow_state`, initialized to 0.5 (neutral).
/// High `w_m` gives stronger feedback from past activity → bistability.
/// High `τ_m` gives longer memory window.
///
/// Also applies **Hebbian plasticity** on CSR weights at the end of each step:
/// `dW = η · m_pre · m_post − α · W · dt`
///
/// Recommended defaults: `tau_m = 0.999`, `w_m = 5.0`, `hebb_lr = 0.001`, `hebb_decay = 0.0001`.
pub fn sim_izhikevich_dual(
    connectome: &mut ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichDualParams,
    external_I: &[f32],
    n_steps: usize,
    hebb_lr: f32,
    hebb_decay: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let n_synapses = connectome.values.len();
    let n_padded = state.n_padded;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];

    let dt = params.dt_ms;
    let tau_m = params.tau_m;
    let w_m = params.w_m;

    // Pad params
    let padded_a = pad_to(&params.a, n_padded);
    let padded_b = pad_to(&params.b, n_padded);
    let padded_c = pad_to(&params.c, n_padded);
    let padded_d = pad_to(&params.d, n_padded);

    // I_buf = external initially. Each step the CSR-propagated recurrent
    // current accumulates on top (step 6/7 below); it is NOT reset before the
    // izhikevich step, or recurrence would never reach the dynamics.
    state.I_buf[..n].fill(0.0);
    for (i, &ext) in external_I.iter().enumerate().take(n) {
        state.I_buf[i] += ext;
    }
    for i in n..state.I_buf.len() {
        state.I_buf[i] = 0.0;
    }

    for step in 0..n_steps {
        // 2. Izhikevich dual step (SIMD) — slow state m modulates I
        step_izhikevich_dual_pillars(
            &mut state.v,
            &mut state.u,
            &mut state.trace,
            &padded_a,
            &padded_b,
            &mut state.I_buf,
            dt,
            0.99,
            &mut state.spikes,
            &padded_c,
            &padded_d,
            &mut state.refractory,
            params.tau_ref,
            &mut state.slow_state,
            tau_m,
            w_m,
        );

        // 3. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 4. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 5. Hebbian plasticity on CSR weights (in-place)
        if hebb_lr > 0.0 && n_synapses > 0 {
            pillars::kernels::learning::hebbian_csr_step(
                &mut connectome.values,
                &connectome.col_indices,
                &connectome.row_ptr,
                &state.slow_state[..n_neurons],
                n_neurons,
                hebb_lr,
                hebb_decay,
                dt,
            );
        }

        // 6. Reset I: wipe the recurrent current consumed by this step's
        //    izhikevich update, then restore external. Step 7 re-adds recurrent
        //    for the next iteration.
        state.I_buf[..n].fill(0.0);
        for (i, &ext) in external_I.iter().enumerate().take(n) {
            state.I_buf[i] += ext;
        }
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 7. CSR propagate from queue
        csr_propagate_from_queue(
            &connectome.row_ptr,
            &connectome.col_indices,
            &connectome.values,
            &connectome.nt_types,
            &mut state.queue,
            &mut state.I_buf,
            1.0,
        );
    }

    spike_buffer
}

// ─── Izhikevich simulation with synaptic kinetics ─────

/// Simulate Izhikevich dynamics with conductance-based synaptic kinetics.
///
/// Same lifecycle as `sim_lif_synaptic` but with Izhikevich neurons:
/// 1. Decay all conductances: `g *= exp(-dt/tau)` for each receptor type
/// 2. Set I_buf = external_I (reset)
/// 3. Compute I_syn from conductances and add to I_buf
/// 4. Izhikevich step (SIMD)
/// 5. Push spikes to queue → record spikes
/// 6. CSR propagate: jump conductances (not currents)
///
/// Requires `connectome.receptor_types` to be non-empty.
/// Izhikevich with conductance-based receptor kinetics via the pillars-native
/// `pillars-connectome::SynapticBrain`. Same signature as the legacy version;
/// conductance carry-over (the caller's `kinetics`) is preserved.
pub fn sim_izhikevich_synaptic(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &IzhikevichParams,
    external_I: &[f32],
    n_steps: usize,
    kinetics: &mut SynapticKinetics,
    scale: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let mut spike_buffer = vec![0i32; n_steps * n];
    if n_neurons == 0 || n_steps == 0 {
        return spike_buffer;
    }

    let mut scaled = connectome.clone();
    for w in &mut scaled.values {
        *w *= scale;
    }
    let mut brain = crate::SynapticBrain::<pillars::zoo::IzhikevichModel>::from_connectome(
        &scaled,
        params.dt_ms,
    );
    brain.set_kinetics(kinetics); // carry-over

    brain.configure(|p| {
        for i in 0..n_neurons {
            p.param_slice_mut(0)[i] = params.dt_ms; // dt
            p.param_slice_mut(1)[i] = params.a[i]; // a
            p.param_slice_mut(2)[i] = params.b[i]; // b
            p.param_slice_mut(3)[i] = params.c[i]; // c
            p.param_slice_mut(4)[i] = params.d[i]; // d
            p.param_slice_mut(5)[i] = params.tau_ref;
            p.param_slice_mut(6)[i] = 30.0; // v_thresh (legacy hardcoded)
            p.state_slice_mut(0)[i] = state.v[i]; // v
            p.state_slice_mut(1)[i] = state.u[i]; // u
            p.state_slice_mut(2)[i] = state.refractory[i]; // refr
        }
    });

    for step in 0..n_steps {
        let ext: Vec<f32> = (0..n_neurons)
            .map(|i| external_I.get(i).copied().unwrap_or(0.0))
            .collect();
        let s = brain.step(&ext);
        for i in 0..n_neurons {
            spike_buffer[step * n + i] = s[i];
        }
    }

    // Carry-over back + sync membrane state.
    *kinetics = brain.kinetics().clone();
    let v = brain.state(0);
    let u = brain.state(1);
    let refr = brain.state(2);
    for i in 0..n_neurons {
        state.v[i] = v[i];
        state.u[i] = u[i];
        state.refractory[i] = refr[i];
    }
    spike_buffer
}

