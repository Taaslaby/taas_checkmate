//! Brain simulator module.

use super::BrainState;
use crate::data::ConnectomeData;
use crate::types::LifParams;
use pillars::kernels::synapse::SynapticKinetics;

pub fn step_lif_pillars(
    v: &mut [f32],
    ii: &mut [f32],
    spikes: &mut [i32],
    params: &[f32],
    refractory: &mut [f32],
    tau_ref: f32,
) {
    let n = v.len();
    assert_eq!(ii.len(), n, "step_lif_pillars: Ii length mismatch");
    assert_eq!(spikes.len(), n, "step_lif_pillars: spikes length mismatch");
    assert_eq!(
        params.len(),
        n * 5,
        "step_lif_pillars: params length mismatch (expected {}*5={})",
        n,
        n * 5
    );
    assert_eq!(
        refractory.len(),
        n,
        "step_lif_pillars: refractory length mismatch"
    );
    assert!(
        n.is_multiple_of(8),
        "step_lif_pillars: n must be a multiple of 8 (SIMD), got {n}"
    );

    // Legacy 5-col [dt, tau_m, v_rest, v_thresh, v_reset] → separate slices.
    // dt — broadcast from slot 0 (та же конвенция, что в sim_lif configure).
    // Репак — по чанкам SIMD-ширины 8 на стеке, без heap-аллокаций; numerics
    // идентичны (step_lif поэлементный, chunks_exact(8)).
    // Замечание (ревью волны 12): api::step_lif использует `inv_tau_m =
    // if τ>0 {1/τ} else {0}`, легаси клампил `1/τ.max(1e-6)` — расхождение
    // только при τ_m ≤ 1e-6 (нефизично).
    let dt = params[0];
    let iter = v
        .chunks_exact_mut(8)
        .zip(ii.chunks_exact_mut(8))
        .zip(spikes.chunks_exact_mut(8))
        .zip(refractory.chunks_exact_mut(8))
        .enumerate();
    for (chunk, (((v8, ii8), sp8), refr8)) in iter {
        let base = chunk * 8;
        let mut tau_m = [0.0f32; 8];
        let mut v_rest = [0.0f32; 8];
        let mut v_thresh = [0.0f32; 8];
        let mut v_reset = [0.0f32; 8];
        for j in 0..8 {
            let i = base + j;
            tau_m[j] = params[n + i];
            v_rest[j] = params[2 * n + i];
            v_thresh[j] = params[3 * n + i];
            v_reset[j] = params[4 * n + i];
        }
        pillars::api::neurons::step_lif(
            v8, ii8, sp8, dt, &tau_m, &v_rest, &v_thresh, &v_reset, refr8, tau_ref,
        );
    }
}

// ─── LIF simulation ──────────────────────────────────

/// Simulate `n_steps` of LIF dynamics on a connectome.
///
/// `safe::lif_step` signature:
/// `lif_step(v: &mut [f32], ii: &mut [f32], spikes: &mut [i32],
///           params: &[f32], refractory: &mut [f32], tau_ref: f32)`
/// where params = [tau_m, v_rest, v_thresh, v_reset, _] interleaved per neuron (n * 5).
pub fn sim_lif(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &LifParams,
    external_I: &[f32],
    n_steps: usize,
    scale: f32,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let mut spike_buffer = vec![0i32; n_steps * n];
    if n_neurons == 0 || n_steps == 0 {
        return spike_buffer;
    }

    // Delegate to the pillars-native orchestrator (`pillars-connectome::Brain`).
    // Same public signature; the engine underneath is 100% pillars — the
    // "translator into legacy" (safe::lif_step + queue/CSR loop) disappears.
    // `scale` is folded into the edge weights (it scales the propagated
    // current in the legacy `csr_propagate_from_queue`).
    // Delegate to the pillars-native orchestrator (`crate::Brain`). The legacy
    // `csr_propagate_from_queue` scaled the propagated current by `scale` — we
    // fold it into the edge weights to preserve the exact same dynamics.
    let mut scaled = connectome.clone();
    for w in &mut scaled.values {
        *w *= scale;
    }
    let mut brain = crate::Brain::<pillars::zoo::LIFModel>::from_connectome(&scaled);
    brain.configure(|p| {
        let np = params.packed.len() / 5;
        for i in 0..n_neurons {
            p.param_slice_mut(0)[i] = params.packed[0]; // dt (broadcast from slot 0)
            p.param_slice_mut(1)[i] = 1.0 / params.packed[np + i].max(1e-6); // inv_tau_m
            p.param_slice_mut(2)[i] = params.packed[2 * np + i]; // v_rest
            p.param_slice_mut(3)[i] = params.packed[3 * np + i]; // v_thresh
            p.param_slice_mut(4)[i] = params.packed[4 * np + i]; // v_reset
            p.param_slice_mut(5)[i] = params.tau_ref;
            p.state_slice_mut(0)[i] = state.v[i]; // v
            p.state_slice_mut(1)[i] = state.refractory[i]; // refr
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

    // Sync back the final membrane state so callers see it.
    let v = brain.state(0);
    let refr = brain.state(1);
    for i in 0..n_neurons {
        state.v[i] = v[i];
        state.refractory[i] = refr[i];
    }
    spike_buffer
}

// ─── LIF simulation with synaptic kinetics ───────────

/// Simulate LIF dynamics with conductance-based synaptic kinetics.
///
/// Requires `connectome.receptor_types` to be non-empty. Uses per-synapse
/// receptor types (AMPA, NMDA, GABA_A, GABA_B) to compute synaptic currents
/// from conductances instead of instant current injection.
///
/// Per-step lifecycle:
/// 1. Decay all conductances: `g *= exp(-dt/τ)` for each receptor type
/// 2. Compute I_syn: `I_syn[i] = Σ g_r[i] · (V[i] - E_rev_r)`
/// 3. Set I_buf = external_I + I_syn
/// 4. LIF step
/// 5. Push spikes to queue
/// 6. CSR propagate: jump conductances instead of currents
///
/// Simulate LIF with conductance-based receptor kinetics (AMPA/NMDA/GABA) via
/// the pillars-native `pillars-connectome::SynapticBrain`. Same signature as
/// conductance carry-over (the caller's `kinetics`) is preserved across calls.
pub fn sim_lif_synaptic(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &LifParams,
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
    let mut brain =
        crate::SynapticBrain::<pillars::zoo::LIFModel>::from_connectome(&scaled, params.dt_ms);
    brain.set_kinetics(kinetics); // carry-over

    brain.configure(|p| {
        let np = params.packed.len() / 5;
        for i in 0..n_neurons {
            p.param_slice_mut(0)[i] = params.packed[0]; // dt (broadcast)
            p.param_slice_mut(1)[i] = 1.0 / params.packed[np + i].max(1e-6); // inv_tau_m
            p.param_slice_mut(2)[i] = params.packed[2 * np + i]; // v_rest
            p.param_slice_mut(3)[i] = params.packed[3 * np + i]; // v_thresh
            p.param_slice_mut(4)[i] = params.packed[4 * np + i]; // v_reset
            p.param_slice_mut(5)[i] = params.tau_ref;
            p.state_slice_mut(0)[i] = state.v[i]; // v
            p.state_slice_mut(1)[i] = state.refractory[i]; // refr
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
    let refr = brain.state(1);
    for i in 0..n_neurons {
        state.v[i] = v[i];
        state.refractory[i] = refr[i];
    }
    spike_buffer
}
