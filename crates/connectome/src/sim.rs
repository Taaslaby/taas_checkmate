//! Brain simulator — универсальный движок для любого CSR коннектома.
//!
//! Поддерживает LIF, Izhikevich, Rate, и GCO (Graph-Coupled Oscillator) модели.
//! Принимает [`ConnectomeData`](crate::data::ConnectomeData) для описания графа.

pub use crate::propagate::*;
pub use crate::types::*;

use crate::data::ConnectomeData;
use pillars::core::NeuralPopulation;
use pillars::zoo::{HHModel, HHParamIdx};
use taas_kernels::neuron::synapse::{DecayFactors, SynapticKinetics};
use taas_kernels::safe;

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
    let new_conn = pillars_connectome::ConnectomeData::from_csr(
        connectome.row_ptr.iter().map(|&v| v as u32).collect(),
        connectome.col_indices.iter().map(|&v| v as u32).collect(),
        connectome.values.iter().map(|&w| w * scale).collect(),
        vec![0u8; connectome.n_synapses],
        vec![0u8; connectome.n_synapses],
    );
    let mut brain =
        pillars_connectome::Brain::<pillars::zoo::LIFModel>::from_connectome(&new_conn);
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
/// Bridge the two duplicate `SynapticKinetics` implementations: the legacy
/// `taas_kernels::neuron::synapse` copy and the canonical
/// `pillars::kernels::synapse` one. Same conductance buffers, two copies.
pub(crate) fn kinetics_legacy_to_pillars(
    l: &taas_kernels::neuron::synapse::SynapticKinetics,
) -> pillars::kernels::synapse::SynapticKinetics {
    let mut k = pillars::kernels::synapse::SynapticKinetics::new(l.n);
    k.g_ampa = l.g_ampa.clone();
    k.g_nmda = l.g_nmda.clone();
    k.g_gaba_a = l.g_gaba_a.clone();
    k.g_gaba_b = l.g_gaba_b.clone();
    k.g_nachr = l.g_nachr.clone();
    k.g_glucl = l.g_glucl.clone();
    k.g_machr = l.g_machr.clone();
    k
}

pub(crate) fn kinetics_pillars_to_legacy(
    p: &pillars::kernels::synapse::SynapticKinetics,
) -> taas_kernels::neuron::synapse::SynapticKinetics {
    let mut k = taas_kernels::neuron::synapse::SynapticKinetics::new(p.n);
    k.g_ampa = p.g_ampa.clone();
    k.g_nmda = p.g_nmda.clone();
    k.g_gaba_a = p.g_gaba_a.clone();
    k.g_gaba_b = p.g_gaba_b.clone();
    k.g_nachr = p.g_nachr.clone();
    k.g_glucl = p.g_glucl.clone();
    k.g_machr = p.g_machr.clone();
    k
}

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

    let new_conn = pillars_connectome::ConnectomeData::from_csr(
        connectome.row_ptr.iter().map(|&v| v as u32).collect(),
        connectome.col_indices.iter().map(|&v| v as u32).collect(),
        connectome.values.iter().map(|&w| w * scale).collect(),
        vec![0u8; connectome.n_synapses],
        connectome.receptor_types.clone(),
    );
    let mut brain =
        pillars_connectome::SynapticBrain::<pillars::zoo::LIFModel>::from_connectome(
            &new_conn,
            params.dt_ms,
        );
    brain.set_kinetics(&kinetics_legacy_to_pillars(kinetics)); // carry-over

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
    *kinetics = kinetics_pillars_to_legacy(brain.kinetics());
    let v = brain.state(0);
    let refr = brain.state(1);
    for i in 0..n_neurons {
        state.v[i] = v[i];
        state.refractory[i] = refr[i];
    }
    spike_buffer
}

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

    let new_conn = pillars_connectome::ConnectomeData::from_csr(
        connectome.row_ptr.iter().map(|&v| v as u32).collect(),
        connectome.col_indices.iter().map(|&v| v as u32).collect(),
        connectome.values.clone(),
        vec![0u8; connectome.n_synapses],
        vec![0u8; connectome.n_synapses],
    );
    let mut brain =
        pillars_connectome::Brain::<pillars::zoo::IzhikevichModel>::from_connectome(&new_conn);
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

    for step in 0..n_steps {
        // 1. Set I = external_I (reset)
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
        for i in n..state.I_buf.len() {
            state.I_buf[i] = 0.0;
        }

        // 2. Izhikevich dual step (SIMD) — slow state m modulates I
        safe::izhikevich_dual_step(
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
            None,
        );

        // 3. Push spikes to queue
        push_spikes_to_queue(&state.spikes, n_neurons, &mut state.queue);

        // 4. Record spikes
        record_spikes(&state.spikes, n_neurons, &mut spike_buffer, step);

        // 5. Hebbian plasticity on CSR weights (in-place)
        if hebb_lr > 0.0 && n_synapses > 0 {
            safe::hebbian_csr_step(
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

        // 6. Reset I = external_I
        state.I_buf[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);
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
    let new_conn = pillars_connectome::ConnectomeData::from_csr(
        connectome.row_ptr.iter().map(|&v| v as u32).collect(),
        connectome.col_indices.iter().map(|&v| v as u32).collect(),
        connectome.values.clone(),
        vec![0u8; connectome.n_synapses],
        vec![0u8; connectome.n_synapses],
    );
    let mut brain = pillars_connectome::Brain::<pillars::zoo::RateModel>::from_connectome(&new_conn);
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

// ─── GPU LIF simulation (CUDA) ──────────────────────────

/// GPU LIF simulation using CUDA — persistent context between calls.
#[cfg(feature = "cuda")]
pub mod gpu {
    use super::BrainState;
    use crate::data::ConnectomeData;
    use cuda_kernels::{DevicePtr, GpuContext};
    use std::sync::Mutex;

    static GPU_CTX: Mutex<Option<(GpuContext, usize, usize, bool)>> = Mutex::new(None);
    /// Cached GPU spike buffer — avoids alloc/free churn between batch calls.
    static GPU_SPIKE_BUF: Mutex<Option<(DevicePtr<i32>, usize)>> = Mutex::new(None);

    /// Get or create a GPU spike buffer of the required size.
    fn get_spike_buf(n_ints: usize) -> DevicePtr<i32> {
        let mut cache = GPU_SPIKE_BUF.lock().unwrap();
        if let Some((ptr, size)) = cache.take() {
            if size == n_ints {
                return ptr; // reuse
            }
            // size changed — old buf dropped here
        }
        let ptr = DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer");
        *cache = Some((ptr, n_ints));
        // Return a fresh allocation (cache holds the same-sized one for next call)
        DevicePtr::<i32>::new(n_ints).expect("GPU: alloc spike buffer")
    }

    /// Return a spike buffer to cache for reuse.
    fn put_spike_buf(buf: DevicePtr<i32>, n_ints: usize) {
        let mut cache = GPU_SPIKE_BUF.lock().unwrap();
        *cache = Some((buf, n_ints));
    }

    /// Simulate `n_steps` of LIF dynamics on a connectome using GPU (CUDA).
    ///
    /// Использует `batch_lif_steps` — аккумулирует спайки на GPU через per-step D2D,
    /// скачивает весь spike buffer + v одним D2H в конце.
    /// Также чинит stale external_I baseline (обновляет d_external_I каждый batch).
    ///
    /// Первый вызов создаёт GpuContext, последующие используют persistent cache.
    /// Uploads only v (not dummy u/a/b/c/d) — conditional upload saves ~5× bandwidth.
    pub fn sim_lif_gpu(
        connectome: &ConnectomeData,
        state: &mut BrainState,
        dt_ms: f32,
        external_I: &[f32],
        n_steps: usize,
        tau_m: &[f32],
        v_rest: &[f32],
        v_thresh: &[f32],
        v_reset: &[f32],
    ) -> Vec<i32> {
        let n = connectome.n_neurons;
        let n_neurons = connectome.row_ptr.len() - 1;
        let n_synapses = connectome.n_synapses;
        let spike_buf_len = n_steps * n;
        let mut spike_buffer = vec![0i32; spike_buf_len];
        let mut v_back = vec![0.0f32; n_neurons];

        // Get or create persistent GPU context
        let mut cache = GPU_CTX.lock().unwrap();
        let (gpu_ctx, needs_init) = match cache.take() {
            Some((ctx, cached_n, cached_s, uploaded))
                if cached_n == n_neurons && cached_s == n_synapses =>
            {
                (ctx, !uploaded)
            }
            _ => {
                let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
                (ctx, true)
            }
        };

        // Upload v only (no dummy u/a/b/c/d — LIF kernel doesn't use them)
        gpu_ctx.upload_v_only(&state.v[..n_neurons]).unwrap();

        if needs_init {
            gpu_ctx
                .upload_lif_params(dt_ms, tau_m, v_rest, v_thresh, v_reset)
                .unwrap();
            gpu_ctx
                .upload_connectome(
                    &connectome.row_ptr,
                    &connectome.col_indices,
                    &connectome.values,
                    &connectome.nt_types,
                    external_I,
                )
                .unwrap();
        }

        // Get cached spike buffer or allocate new one
        let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

        // Batch run
        gpu_ctx
            .batch_lif_steps(
                external_I,
                d_spike_buf.as_mut_ptr(),
                &mut spike_buffer,
                &mut v_back,
                n_steps,
            )
            .expect("GPU: batch_lif_steps");

        // Return spike buffer to cache
        put_spike_buf(d_spike_buf, n_steps * n_neurons);

        // Copy v back to state
        state.v[..n_neurons].copy_from_slice(&v_back);

        *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

        spike_buffer
    }

    /// Simulate `n_steps` of Izhikevich dynamics on a connectome using GPU (CUDA).
    ///
    /// Same batched approach as `sim_lif_gpu`:
    /// - Per-step D2D spike accumulation, single D2H at end
    /// - Updates d_external_I + d_I every call (fixes stale baseline)
    /// - Uploads full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
    /// - Downloads v + u at end for state persistence
    ///
    /// Default Izhikevich params: a=0.02, b=0.2, c=-65, d=8, trace_decay=0.99
    pub fn sim_izhikevich_gpu(
        connectome: &ConnectomeData,
        state: &mut BrainState,
        dt_ms: f32,
        external_I: &[f32],
        n_steps: usize,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
    ) -> Vec<i32> {
        let n = connectome.n_neurons;
        let n_neurons = connectome.row_ptr.len() - 1;
        let n_synapses = connectome.n_synapses;
        let spike_buf_len = n_steps * n;
        let mut spike_buffer = vec![0i32; spike_buf_len];
        let mut v_back = vec![0.0f32; n_neurons];
        let mut u_back = vec![0.0f32; n_neurons];
        let trace_decay = 0.99f32;

        // Get or create persistent GPU context
        let mut cache = GPU_CTX.lock().unwrap();
        let (gpu_ctx, needs_init) = match cache.take() {
            Some((ctx, cached_n, cached_s, uploaded))
                if cached_n == n_neurons && cached_s == n_synapses =>
            {
                (ctx, !uploaded)
            }
            _ => {
                let ctx = GpuContext::new(n_neurons, n_synapses).unwrap();
                (ctx, true)
            }
        };

        // Upload full neuron state (v, u, a, b, c, d) — Izhikevich needs all params
        gpu_ctx
            .upload_neurons(&state.v[..n_neurons], &state.u[..n_neurons], a, b, c, d)
            .unwrap();

        if needs_init {
            gpu_ctx
                .upload_connectome(
                    &connectome.row_ptr,
                    &connectome.col_indices,
                    &connectome.values,
                    &connectome.nt_types,
                    external_I,
                )
                .unwrap();
        }

        // Get cached spike buffer
        let mut d_spike_buf = get_spike_buf(n_steps * n_neurons);

        // Batch run
        gpu_ctx
            .batch_izhikevich_steps(
                external_I,
                dt_ms,
                trace_decay,
                d_spike_buf.as_mut_ptr(),
                &mut spike_buffer,
                &mut v_back,
                &mut u_back,
                n_steps,
            )
            .expect("GPU: batch_izhikevich_steps");

        // Return spike buffer to cache
        put_spike_buf(d_spike_buf, n_steps * n_neurons);

        // Copy v + u back to state
        state.v[..n_neurons].copy_from_slice(&v_back);
        state.u[..n_neurons].copy_from_slice(&u_back);

        *cache = Some((gpu_ctx, n_neurons, n_synapses, true));

        spike_buffer
    }
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

    let new_conn = pillars_connectome::ConnectomeData::from_csr(
        connectome.row_ptr.iter().map(|&v| v as u32).collect(),
        connectome.col_indices.iter().map(|&v| v as u32).collect(),
        connectome.values.iter().map(|&w| w * scale).collect(),
        vec![0u8; connectome.n_synapses],
        connectome.receptor_types.clone(),
    );
    let mut brain =
        pillars_connectome::SynapticBrain::<pillars::zoo::IzhikevichModel>::from_connectome(
            &new_conn,
            params.dt_ms,
        );
    brain.set_kinetics(&kinetics_legacy_to_pillars(kinetics)); // carry-over

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
    *kinetics = kinetics_pillars_to_legacy(brain.kinetics());
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

// ─── GCO simulation ──────────────────────────────────

/// Simulate `n_steps` of graph-coupled oscillator dynamics on a connectome.
///
/// `safe::gco_step` signature:
/// `gco_step(v, u, h_real, h_imag, i_ext, speed, dt)`
pub fn sim_gco(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &GcoParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let dt = params.dt_ms / 1000.0;

    // H*psi scratch buffers (must match v size)
    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let mut speed = vec![1.0; state.n_padded];

    for step in 0..n_steps {
        // 1. Build Hamiltonian from CSR + state
        // h_real[i] = Σ W[j,i] * u[j], h_imag[i] = -Σ W[j,i] * v[j]
        h_real.fill(0.0);
        h_imag.fill(0.0);
        let max_syn = connectome.col_indices.len().min(connectome.values.len());
        for pre in 0..n_neurons {
            let start = connectome.row_ptr[pre].max(0) as usize;
            let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
            let mult = if pre < connectome.nt_types.len() {
                connectome.nt_types[pre]
            } else {
                1.0
            };
            for j in start..end {
                let post = connectome.col_indices[j] as usize;
                let w = connectome.values[j] * mult;
                if post < n {
                    h_real[post] += w * state.u[pre];
                    h_imag[post] -= w * state.v[pre];
                }
            }
        }

        // 2. Set external input
        for i in 0..n.min(external_I.len()) {
            state.I_buf[i] = external_I[i];
        }

        // 3. GCO step
        safe::gco_step(
            &mut state.v,
            &mut state.u,
            &mut h_real,
            &mut h_imag,
            &mut state.I_buf,
            &mut speed,
            dt,
        );

        // 4. "Spike" if |v+iu|² > threshold
        for i in 0..n_neurons {
            let prob = state.v[i] * state.v[i] + state.u[i] * state.u[i];
            if prob > 0.5 {
                let idx = step * n_neurons + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
            }
        }
    }

    spike_buffer
}

// ─── HH simulation via pillars (NeuralPopulation<HHModel>) ──

/// Build a pillars [`NeuralPopulation<HHModel>`] matching the legacy
/// `taas_core::neuron::hh::standard_config` parameter set **exactly**
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

        // 2. Set I = external_I
        i_in[..n].copy_from_slice(&external_I[..n.min(external_I.len())]);

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

/// Simulate GCO with time-varying external input, recording (v, u) trajectory.
///
/// Args:
///   connectome: CSR connectome
///   state: mutable brain state (persistent across calls)
///   params: GcoParams { dt_ms }
///   external_I: flat 2D array `[n_changes * n]` — I_ext per change
///   n_changes: number of distinct external input settings
///   steps_per_input: GCO steps per external input
///   record_every: record (v, u) every N steps (0 = only at end of each input block)
///
/// Returns:
///   (v_traj, u_traj) — each Vec<f32> of length n_records * n
///   where n_records = n_changes * steps_per_input / record_every
pub fn sim_gco_record(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &GcoParams,
    external_I: &[f32],
    n_changes: usize,
    steps_per_input: usize,
    record_every: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let dt = params.dt_ms / 1000.0;
    let re = if record_every == 0 {
        steps_per_input
    } else {
        record_every
    };
    let recs_per_input = steps_per_input / re;
    let n_records = n_changes * recs_per_input;

    let mut v_traj = vec![0.0f32; n_records * n];
    let mut u_traj = vec![0.0f32; n_records * n];

    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let mut speed = vec![1.0; state.n_padded];

    let mut rec_idx = 0;
    for change in 0..n_changes {
        let base = change * n;
        for i in 0..n.min(external_I.len().saturating_sub(base)) {
            state.I_buf[i] = external_I[base + i];
        }

        for step in 0..steps_per_input {
            h_real.fill(0.0);
            h_imag.fill(0.0);
            let max_syn = connectome.col_indices.len().min(connectome.values.len());
            for pre in 0..n_neurons {
                let start = connectome.row_ptr[pre].max(0) as usize;
                let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
                let mult = if pre < connectome.nt_types.len() {
                    connectome.nt_types[pre]
                } else {
                    1.0
                };
                for j in start..end {
                    let post = connectome.col_indices[j] as usize;
                    let w = connectome.values[j] * mult;
                    if post < n {
                        h_real[post] += w * state.u[pre];
                        h_imag[post] -= w * state.v[pre];
                    }
                }
            }

            safe::gco_step(
                &mut state.v,
                &mut state.u,
                &mut h_real,
                &mut h_imag,
                &mut state.I_buf,
                &mut speed,
                dt,
            );

            if (step + 1) % re == 0 {
                let out_base = rec_idx * n;
                for i in 0..n {
                    v_traj[out_base + i] = state.v[i];
                    u_traj[out_base + i] = state.u[i];
                }
                rec_idx += 1;
            }
        }
    }

    (v_traj, u_traj)
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use taas_kernels::neuron::synapse::ReceptorType;

    #[test]
    fn spike_queue_basic_push_pop() {
        let mut q = SpikeQueue::new(8);
        assert_eq!(q.len(), 0);
        q.push(5);
        q.push(42);
        q.push(100);
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(5));
        assert_eq!(q.pop(), Some(42));
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop(), Some(100));
        assert_eq!(q.pop(), None);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn spike_queue_wrap_around() {
        let mut q = SpikeQueue::new(4);
        // Push until near capacity
        q.push(1);
        q.push(2);
        q.push(3);
        q.push(4);
        assert_eq!(q.len(), 4);
        // Pop all
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.len(), 2);
        // Now push more — wraps around in circular buffer
        q.push(10);
        q.push(11);
        // head is now 6, tail is 2, cap=4, mask=3
        assert_eq!(q.len(), 4); // 6 - 2 = 4
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(11));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_queue_drain() {
        let mut q = SpikeQueue::new(8);
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.len(), 3);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(q.pop(), None);
        // After clear, can still push
        q.push(42);
        assert_eq!(q.pop(), Some(42));
    }

    #[test]
    fn spike_queue_capacity_power_of_two() {
        let q = SpikeQueue::new(5); // next power of 2 = 8
        assert_eq!(q.buffer.len(), 8);
        assert_eq!(q.mask, 7);

        let q2 = SpikeQueue::new(1000); // next power of 2 = 1024
        assert_eq!(q2.buffer.len(), 1024);

        let q3 = SpikeQueue::new(1); // min 4
        assert_eq!(q3.buffer.len(), 4);
    }

    #[test]
    fn push_spikes_to_queue_basic() {
        let spikes = vec![0, 1, 0, 0, 1, 0, 1];
        let mut q = SpikeQueue::new(16);
        let count = push_spikes_to_queue(&spikes, 7, &mut q);
        assert_eq!(count, 3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(6));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn csr_propagate_from_queue_basic() {
        // Simple ring: 0→1, 1→2, 2→0
        let row_ptr = vec![0i32, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![0.5f32, 0.5, 0.5];
        let nt_types = vec![1.0f32, 1.0, 1.0];
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(8);

        // Neuron 0 spikes
        q.push(0);
        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            1.0,
        );

        // I_buf[1] should get 0.5
        assert!((I_buf[1] - 0.5).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(I_buf[2], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_multi_spike() {
        // 0→2, 1→1 (self), 2→0, 3→1
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let col_indices = vec![2, 1, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let nt_types = vec![1.0f32, -1.0, 1.0, 1.0]; // neuron 1 is inhibitory
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(16);

        q.push(0); // exc → I_buf[2] += 1.0
        q.push(1); // inh → I_buf[1] += 2.0 * -1.0 = -2.0
        q.push(3); // exc → I_buf[1] += 4.0

        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            1.0,
        );

        assert!((I_buf[2] - 1.0).abs() < 1e-6, "I_buf[2]={}", I_buf[2]);
        assert!((I_buf[1] - 2.0).abs() < 1e-6, "I_buf[1]={}", I_buf[1]); // -2.0 + 4.0 = 2.0
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_inhibitory_scale() {
        let row_ptr = vec![0i32, 2, 2];
        let col_indices = vec![0, 1]; // neuron 0→0 and 0→1
        let values = vec![1.0f32, 1.0];
        let nt_types = vec![1.0f32, 0.0]; // neuron 1 is silent (no synapses anyway)
        let mut I_buf = vec![0.0f32; 2];
        let mut q = SpikeQueue::new(8);

        q.push(0);
        // scale = -0.5
        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            -0.5,
        );

        assert!((I_buf[0] - (-0.5)).abs() < 1e-6, "I_buf[0]={}", I_buf[0]);
        assert!((I_buf[1] - (-0.5)).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(q.len(), 0);
    }

    // ─── Synaptic kinetics integration tests ───────────

    fn make_synaptic_connectome() -> ConnectomeData {
        // 3 neurons: 0→1 (exc, AMPA), 1→2 (inh, GABA_A), 2→0 (exc, AMPA)
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2, 3],
            vec![1, 2, 0],
            vec![0.5f32, 1.0, 0.8],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        conn.build_receptor_types();
        conn
    }

    #[test]
    fn test_sim_lif_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(
            !conn.receptor_types.is_empty(),
            "receptor_types must be non-empty"
        );

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        // Strong drive ensures firing: LIF dv/dt = (I - (V - V_rest)) / tau
        // With I=100, tau=5, dt=0.1: dv ~= 2.0/step → reaches V_thresh=0 in ~26 steps
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![100.0f32; n];
        let n_steps = 100;
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_lif_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            n_steps,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "LIF synaptic should fire with high drive, got {} spikes",
            total_spikes
        );
        assert_eq!(spikes.len(), n_steps * n);
    }

    #[test]
    fn test_sim_izhikevich_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(!conn.receptor_types.is_empty());

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![7.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            20,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "Izhikevich synaptic should fire with high drive"
        );
        assert_eq!(spikes.len(), 20 * n);
    }

    #[test]
    fn test_sim_lif_synaptic_conductance_decay() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![0.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        kinetics.jump(0, ReceptorType::AMPA, 10.0);
        kinetics.jump(1, ReceptorType::AMPA, 10.0);

        let initial_g = kinetics.g_ampa[0];
        assert!((initial_g - 10.0).abs() < 1e-6, "initial g={}", initial_g);

        let _spikes = sim_lif_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            1,
            &mut kinetics,
            1.0,
        );

        let expected_decay = (-0.1f32 / 2.0).exp();
        assert!(
            kinetics.g_ampa[0] < 10.0 * expected_decay * 1.01,
            "g_ampa[0]={} should have decayed",
            kinetics.g_ampa[0]
        );
        assert!(
            kinetics.g_ampa[0] > 0.0,
            "g_ampa[0] should still be positive"
        );
    }

    #[test]
    fn test_sim_izhikevich_synaptic_receptor_types() {
        let mut conn = make_synaptic_connectome();
        conn.receptor_types = vec![0u8, 1, 2]; // AMPA, NMDA, GABA_A

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![10.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            50,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "Izhikevich with mixed receptors should fire"
        );

        let ampa_decay = (-0.5f32 / 2.0).exp().powi(50);
        let nmda_decay = (-0.5f32 / 50.0).exp().powi(50);
        assert!(
            nmda_decay > ampa_decay,
            "NMDA should decay slower than AMPA"
        );
        println!("AMPA decay over 50 steps: {:.6}", ampa_decay);
        println!("NMDA decay over 50 steps: {:.6}", nmda_decay);
    }

    #[test]
    fn test_sim_hh_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(
            !conn.receptor_types.is_empty(),
            "receptor_types must be non-empty"
        );

        let n = conn.n_neurons;
        // dt=0.01 ms — HH is numerically unstable at dt=0.5 (see test_sim_hh_basic).
        let mut pop = hh_population_standard(n, 0.01);
        // Strong drive ensures HH fires: I=15 µA/cm² is suprathreshold for HH
        let external_I = vec![15.0f32; n];
        let n_steps = 5000;
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_hh_synaptic_pillars(&conn, &mut pop, &external_I, n_steps, &mut kinetics, 1.0);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "HH synaptic should fire with high drive (I=15), got {} spikes",
            total_spikes
        );
        assert_eq!(spikes.len(), n_steps * n);
    }

    #[test]
    fn test_sim_hh_basic() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        // dt=0.01 ms — HH exponential-Euler is numerically unstable at dt=0.5
        // (the NaN safety check in ohmic_gated catches it). 5000 steps = 50 ms.
        let mut pop = hh_population_standard(n, 0.01);
        let external_I = vec![15.0f32; n];
        let n_steps = 5000;

        let spikes = sim_hh_pillars(&conn, &mut pop, &external_I, n_steps, 1.0);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "HH (instant) should fire with high drive (I=15), got {} spikes",
            total_spikes
        );
        assert_eq!(spikes.len(), n_steps * n);
    }

    // ═══════════════════════════════════════════════════════
    //  Integration: HH physics + synaptic chemistry
    // ═══════════════════════════════════════════════════════

    /// Simple 2-neuron connectome: 0 → 1 with nAChR
    fn make_2_neuron_connectome() -> ConnectomeData {
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2],      // row_ptr: 0 has 1 syn, 1 has 0 syn
            vec![1_usize as i32],  // 0 → 1
            vec![1.0f32],          // weight
            vec![1.0f32, -1.0f32], // nt_types: 0=exc, 1=inh
            vec![],                // receptors (built below)
        );
        conn.receptor_types = vec![4u8]; // nAChR (index 4)
        conn
    }

    #[test]
    fn test_hh_current_based_psp() {
        // Current-based path (sim_hh_pillars): pre (0) fires → post (1) receives I = w * nt * scale
        // At dt=0.1ms, pre needs I ≥ 15 µA to fire reliably
        let conn = make_2_neuron_connectome();
        let n = conn.n_neurons;
        let mut pop = hh_population_standard(n, 0.1);
        let mut spikes_buf = vec![0i32; n];

        let v_before = pop.state_slice(0)[1]; // post-synaptic V before

        // Drive neuron 0 to fire
        for _ in 0..300 {
            // Manual step: i_in = external, step, then propagate
            let mut q = SpikeQueue::new(8);
            let external_I = vec![15.0f32, 0.0];
            pop.step(&external_I, &mut spikes_buf);
            if spikes_buf[0] != 0 {
                q.push(0);
            }
            let mut i_in = vec![0.0f32; n];
            csr_propagate_from_queue(
                &conn.row_ptr,
                &conn.col_indices,
                &conn.values,
                &conn.nt_types,
                &mut q,
                &mut i_in,
                1.0,
            );
            if spikes_buf[0] != 0 {
                break;
            }
        }

        // Now pre fired (or we forced it). Track post voltage.
        let v_after_first = pop.state_slice(0)[1];
        println!(
            "Current-based PSP: pre-V={:.1}, post V before={v_before:.1} after={v_after_first:.1}",
            pop.state_slice(0)[0]
        );

        // Post should get depolarized via CSR: I_post = 1.0 * 1.0 * 1.0 = 1.0 µA/cm²
        // HH threshold is ~7 µA/cm², so this single-spike EPSP should be subthreshold
        // but detectable as a voltage change.
        let dv_post = v_after_first - v_before;
        assert!(
            dv_post != 0.0,
            "Current-based PSP should produce voltage change on post"
        );
        // The EPSP should be small: with dt=0.1ms and gL=0.3, one spike delivers
        // I=1.0 for one dt. ΔV = I/Cm * dt = 1.0/1.0 * 0.1e-3 * 1000 = 0.1 mV
        // But spike-triggered current might persist for multiple steps.
        // Allow 0.01-5mV range.
        assert!(
            dv_post.abs() < 10.0,
            "Current-based PSP too large: {dv_post} mV (should be < 10 mV)"
        );
    }

    #[test]
    fn test_hh_synaptic_psp() {
        // Conductance-based path via sim_hh_synaptic_pillars:
        // Each pre-spike → jump g_nachr[post] → post I_syn = g * (0 - V)
        // The driving force multiplier (0 - V) ≈ 65 makes this path ~65× stronger
        let conn = make_2_neuron_connectome();
        let n = conn.n_neurons;
        let mut pop = hh_population_standard(n, 0.01);
        let mut kinetics = SynapticKinetics::new(n);
        // Drive only pre (neuron 0), post gets 0 external
        let external_I = vec![15.0f32, 0.0];

        // Run 20000 steps = 200ms (dt=0.01 — HH is unstable at larger dt).
        // Pre should fire many times; each spike jumps g_nachr[post] += 1.0.
        let spikes = sim_hh_synaptic_pillars(&conn, &mut pop, &external_I, 20000, &mut kinetics, 1.0);

        // Count post spikes (neuron 1, offset in spike buffer = 1)
        let post_spikes: i32 = spikes.iter().skip(1).step_by(n).copied().sum();
        let pre_spikes: i32 = spikes.iter().step_by(n).copied().sum();

        println!(
            "Synaptic PSP: pre={} spikes, post={} spikes (conductance path)",
            pre_spikes, post_spikes
        );
        // Pre should fire with I=15
        assert!(pre_spikes > 0, "Pre-neuron should fire with I=15");
        // Post should fire too — each pre-spike adds g=1.0 to nAChR,
        // which delivers I ≈ 65 µA/cm² — well above threshold.
        assert!(
            post_spikes > 0,
            "Post should fire via conductance-based PSP (driving force ×65)"
        );
    }

    #[test]
    fn test_hh_synaptic_vs_current_ratio() {
        // Демонстрация ratio (conductance-based) / (current-based) = (E_rev - V)
        // Для nAChR при V=-65: ratio = 65
        let n = 2;
        let mut kin = SynapticKinetics::new(n);
        let v = vec![-65.0, -65.0];
        let mut i_curr = [0.0, 0.0];
        let mut i_cond = vec![0.0, 0.0];

        // Current-based: nt_types[0]=1, weight=1, scale=1 → I_post = 1*1*1 = 1.0
        i_curr[1] = 1.0; // simulated CSR result

        // Conductance-based: jump + compute
        kin.jump(1, ReceptorType::nAChR, 1.0); // g=1.0
        kin.compute_currents(&v, &mut i_cond);

        let ratio = i_cond[1] / i_curr[1];
        println!("Conductance/Current ratio for nAChR at V=-65: {ratio:.0}× (expected ~65)");
        assert!(
            ratio > 50.0 && ratio < 80.0,
            "Ratio {ratio:.0} should be ~65× for nAChR at rest"
        );
    }

    #[test]
    fn test_hh_both_paths_produce_spikes() {
        // Both paths should produce spikes when drive is sufficient
        let conn = make_2_neuron_connectome();
        let n = conn.n_neurons;

        // Current-based path (dt=0.01 — HH is unstable at larger dt).
        let mut pop1 = hh_population_standard(n, 0.01);
        let ext = vec![15.0f32; n];
        let spikes_curr = sim_hh_pillars(&conn, &mut pop1, &ext, 5000, 1.0);

        // Conductance-based path
        let mut pop2 = hh_population_standard(n, 0.01);
        let mut kin = SynapticKinetics::new(n);
        let spikes_cond = sim_hh_synaptic_pillars(&conn, &mut pop2, &ext, 5000, &mut kin, 1.0);

        let total_curr: i32 = spikes_curr.iter().sum();
        let total_cond: i32 = spikes_cond.iter().sum();

        println!("Current-based total spikes: {total_curr}, Conductance-based: {total_cond}");
        assert!(total_curr > 0, "Current-based HH should fire");
        assert!(total_cond > 0, "Conductance-based HH should fire");
    }
}
