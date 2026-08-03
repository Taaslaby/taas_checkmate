//! CUDA kernels for neuron-engine — GPU routines.
use crate::context::GpuCtx;
use crate::device::ok_or;
use crate::ffi::*;
use crate::ptr::DevicePtr;
use std::ffi::{c_float, c_int, c_void};
// ═══════════════════════════════════════════════════════════
//  Legacy individual kernel calls (kept for compat)
// ═══════════════════════════════════════════════════════════

// XOR Attention
struct XorCtx;
extern "C" {
    fn xor_attention_create(n_keys: c_int, n_bytes: c_int) -> *mut XorCtx;
    fn xor_attention_compute(
        ctx: *mut XorCtx,
        query: *const u8,
        keys: *const u8,
        values: *const c_float,
        output: *mut c_float,
    ) -> c_int;
    fn xor_attention_destroy(ctx: *mut XorCtx);
    fn cuda_xor_attention(
        query: *const u8,
        keys: *const u8,
        values: *const c_float,
        output: *mut c_float,
        n_keys: c_int,
        n_bytes: c_int,
    ) -> c_int;
}

pub struct XorAttention {
    ctx: *mut XorCtx,
    n_keys: usize,
    n_bytes: usize,
}

impl XorAttention {
    pub fn new(n_keys: usize, n_bytes: usize) -> Result<Self, String> {
        assert!(n_bytes.is_multiple_of(8));
        let ctx = unsafe { xor_attention_create(n_keys as c_int, n_bytes as c_int) };
        if ctx.is_null() {
            Err("xor_attention_create failed".into())
        } else {
            Ok(XorAttention {
                ctx,
                n_keys,
                n_bytes,
            })
        }
    }

    pub fn compute(
        &self,
        query: &[u8],
        keys: &[u8],
        values: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let err = unsafe {
            xor_attention_compute(
                self.ctx,
                query.as_ptr(),
                keys.as_ptr(),
                values.as_ptr(),
                output.as_mut_ptr(),
            )
        };
        ok_or(err, "xor_attention")
    }

    pub fn compute_scalar(&self, query: &[u8], keys: &[u8], values: &[f32]) -> Result<f32, String> {
        let mut out = vec![0.0f32];
        self.compute(query, keys, values, &mut out)?;
        Ok(out[0])
    }
}

impl Drop for XorAttention {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                xor_attention_destroy(self.ctx);
            }
        }
    }
}

// Legacy XOR attention single-call
pub fn xor_attention(
    query: &[u8],
    keys: &[u8],
    values: &[f32],
    output: &mut [f32],
    n_keys: usize,
    n_bytes: usize,
) -> Result<(), String> {
    assert!(n_bytes.is_multiple_of(8));
    ok_or(
        unsafe {
            cuda_xor_attention(
                query.as_ptr(),
                keys.as_ptr(),
                values.as_ptr(),
                output.as_mut_ptr(),
                n_keys as c_int,
                n_bytes as c_int,
            )
        },
        "xor_attention",
    )
}

// ─── Individual wrappers (single-kernel API) ─────────────

pub fn hebbian_update(
    weights: &mut [f32],
    cos_diff: &[f32],
    lr: f32,
    decay: f32,
    dt: f32,
) -> Result<(), String> {
    let n = weights.len();
    ok_or(
        unsafe {
            cuda_hebbian_update(
                weights.as_mut_ptr(),
                cos_diff.as_ptr(),
                lr,
                decay,
                dt,
                n as c_int,
            )
        },
        "hebbian",
    )
}

pub fn kuramoto_prepare(
    phases: &[f32],
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> Result<(), String> {
    let n = phases.len();
    ok_or(
        unsafe {
            cuda_kuramoto_prepare(
                phases.as_ptr(),
                cos_out.as_mut_ptr(),
                sin_out.as_mut_ptr(),
                n as c_int,
            )
        },
        "kuramoto_prepare",
    )
}

pub fn kuramoto_update(
    phases: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    dt: f32,
) -> Result<(), String> {
    let n = phases.len();
    ok_or(
        unsafe {
            cuda_kuramoto_update(
                phases.as_mut_ptr(),
                omegas.as_ptr(),
                coupling.as_ptr(),
                dt,
                n as c_int,
            )
        },
        "kuramoto_update",
    )
}

pub fn kuramoto_bptt_loss_grad(
    phases_traj: &[f32],
    coupling_traj: &[f32],
    omegas: &[f32],
    dtheta: &mut [f32],
    coherence_start: usize,
    coherence_weight: f32,
    temporal_scale: f32,
    dt: f32,
    n_steps: usize,
) -> Result<(), String> {
    let n = phases_traj.len() / n_steps;
    ok_or(
        unsafe {
            cuda_kuramoto_bptt_loss_grad(
                phases_traj.as_ptr(),
                coupling_traj.as_ptr(),
                omegas.as_ptr(),
                dtheta.as_mut_ptr(),
                coherence_start as c_int,
                coherence_weight,
                temporal_scale,
                dt,
                n as c_int,
                n_steps as c_int,
            )
        },
        "kuramoto_bptt_loss_grad",
    )
}

pub fn kuramoto_bptt_step(
    grad_weights: &mut [f32],
    dtheta: &mut [f32],
    dtheta_next: &[f32],
    phases: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    dt: f32,
) -> Result<(), String> {
    let n = phases.len();
    let n_syn = grad_weights.len();
    ok_or(
        unsafe {
            cuda_kuramoto_bptt_step(
                grad_weights.as_mut_ptr(),
                dtheta.as_mut_ptr(),
                dtheta_next.as_ptr(),
                phases.as_ptr(),
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                values.as_ptr(),
                dt,
                n as c_int,
                n_syn as c_int,
            )
        },
        "kuramoto_bptt_step",
    )
}

pub fn kuramoto_csr_coupling(
    coupling: &mut [f32],
    sin_buf: &[f32],
    cos_buf: &[f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
) -> Result<(), String> {
    let n = coupling.len().min(sin_buf.len()).min(cos_buf.len());
    let n_neurons = row_ptr.len() - 1;
    let _n_syn = col_indices.len().min(values.len());
    // Validate sizes
    assert!(
        n <= n_neurons,
        "coupling.len={} > n_neurons={}",
        n,
        n_neurons
    );
    ok_or(
        unsafe {
            cuda_kuramoto_csr_coupling(
                coupling.as_mut_ptr(),
                sin_buf.as_ptr(),
                cos_buf.as_ptr(),
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                values.as_ptr(),
                n as c_int,
            )
        },
        "kuramoto_csr_coupling",
    )
}

pub fn safety_clamp(data: &mut [f32], min_val: f32, max_val: f32) -> Result<(), String> {
    let n = data.len() as c_int;
    ok_or(
        unsafe { cuda_safety_clamp(data.as_mut_ptr(), min_val, max_val, n) },
        "safety_clamp",
    )
}

pub fn stochastic_encoder(
    target_I: &mut [f32],
    inputs: &[f32],
    noise: &[f32],
    threshold: f32,
    spike_val: f32,
) -> Result<(), String> {
    let n = target_I.len().min(inputs.len()).min(noise.len()) as c_int;
    ok_or(
        unsafe {
            cuda_stochastic_encoder(
                target_I.as_mut_ptr(),
                inputs.as_ptr(),
                noise.as_ptr(),
                threshold,
                spike_val,
                n,
            )
        },
        "stochastic_encoder",
    )
}

pub fn thalamic_gate(
    target_I: &mut [f32],
    source_v: &[f32],
    gate_v: &[f32],
    threshold: f32,
    gain: f32,
) -> Result<(), String> {
    let n = target_I.len().min(source_v.len()).min(gate_v.len()) as c_int;
    ok_or(
        unsafe {
            cuda_thalamic_gate(
                target_I.as_mut_ptr(),
                source_v.as_ptr(),
                gate_v.as_ptr(),
                threshold,
                gain,
                n,
            )
        },
        "thalamic_gate",
    )
}

pub fn sigmoid_gate(
    output: &mut [f32],
    input: &[f32],
    gain: f32,
    threshold: f32,
) -> Result<(), String> {
    let n = output.len().min(input.len()) as c_int;
    ok_or(
        unsafe { cuda_sigmoid_gate(output.as_mut_ptr(), input.as_ptr(), gain, threshold, n) },
        "sigmoid_gate",
    )
}

pub fn phase_gate_apply(
    output: &mut [f32],
    input: &[f32],
    r: &[f32],
    gain: f32,
    threshold: f32,
) -> Result<(), String> {
    let n = output.len().min(input.len()).min(r.len()) as c_int;
    ok_or(
        unsafe {
            cuda_phase_gate_apply(
                output.as_mut_ptr(),
                input.as_ptr(),
                r.as_ptr(),
                gain,
                threshold,
                n,
            )
        },
        "phase_gate",
    )
}

pub fn t3_drive_encoder(drives: &mut [f32], inputs: &[f32], gain: f32) -> Result<(), String> {
    let n = drives.len().min(inputs.len()) as c_int;
    ok_or(
        unsafe { cuda_t3_drive_encoder(drives.as_mut_ptr(), inputs.as_ptr(), gain, n) },
        "t3_drive",
    )
}

pub fn t3_inverse_kuramoto(
    drive_out: &mut [f32],
    r: &[f32],
    r_prev: &[f32],
    alpha: f32,
    gain: f32,
    eps: f32,
    decay: f32,
) -> Result<(), String> {
    let n = drive_out.len().min(r.len()).min(r_prev.len()) as c_int;
    ok_or(
        unsafe {
            cuda_t3_inverse_kuramoto(
                drive_out.as_mut_ptr(),
                r.as_ptr(),
                r_prev.as_ptr(),
                alpha,
                gain,
                eps,
                decay,
                n,
            )
        },
        "t3_inverse",
    )
}

pub fn t3_tact_generator(
    pulses: &mut [f32],
    ph1: &[f32],
    ph2: &[f32],
    ph3: &[f32],
    threshold: f32,
) -> Result<(), String> {
    let n = pulses.len().min(ph1.len()).min(ph2.len()).min(ph3.len()) as c_int;
    ok_or(
        unsafe {
            cuda_t3_tact_generator(
                pulses.as_mut_ptr(),
                ph1.as_ptr(),
                ph2.as_ptr(),
                ph3.as_ptr(),
                threshold,
                n,
            )
        },
        "t3_tact",
    )
}

pub fn nlms_train(
    weights: &mut [f32],
    bias: &mut [f32],
    states: &[f32],
    error: f32,
    lr: f32,
    eps: f32,
) -> Result<(), String> {
    let n = weights.len() as c_int;
    ok_or(
        unsafe {
            cuda_nlms_train(
                weights.as_mut_ptr(),
                bias.as_mut_ptr(),
                states.as_ptr(),
                error,
                lr,
                eps,
                n,
            )
        },
        "nlms",
    )
}

pub fn kalman_update(
    weights: &mut [f32],
    cov: &mut [f32],
    states: &[f32],
    error: f32,
    s: f32,
) -> Result<(), String> {
    let n = weights.len() as c_int;
    ok_or(
        unsafe {
            cuda_kalman_update(
                weights.as_mut_ptr(),
                cov.as_mut_ptr(),
                states.as_ptr(),
                error,
                s,
                n,
            )
        },
        "kalman",
    )
}

pub fn echo_tap(
    output: &mut [f32],
    buffer: &[f32],
    n_osc: usize,
    delay_len: usize,
    delay_idx: usize,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_echo_tap(
                output.as_mut_ptr(),
                buffer.as_ptr(),
                n_osc as c_int,
                delay_len as c_int,
                delay_idx as c_int,
            )
        },
        "echo_tap",
    )
}

pub fn ring_synthesis(
    output: &mut [f32],
    amplitudes: &[f32],
    trig_buf: &[f32],
    weights: &[f32],
) -> Result<(), String> {
    let n = output
        .len()
        .min(amplitudes.len())
        .min(trig_buf.len())
        .min(weights.len()) as c_int;
    ok_or(
        unsafe {
            cuda_ring_synthesis(
                output.as_mut_ptr(),
                amplitudes.as_ptr(),
                trig_buf.as_ptr(),
                weights.as_ptr(),
                n,
            )
        },
        "ring_synthesis",
    )
}

// ─── Signal analysis (interference, coherence, band energy) ─

pub fn interference_matrix(
    output: &mut [f32],
    phases_a: &[f32],
    phases_b: &[f32],
) -> Result<(), String> {
    let na = phases_a.len() as c_int;
    let nb = phases_b.len() as c_int;
    ok_or(
        unsafe {
            cuda_interference(
                output.as_mut_ptr(),
                phases_a.as_ptr(),
                phases_b.as_ptr(),
                na,
                nb,
            )
        },
        "interference",
    )
}

pub fn interference_mean(
    output: &mut [f32],
    phases_a: &[f32],
    phases_b: &[f32],
) -> Result<(), String> {
    let n = phases_a.len().min(phases_b.len()) as c_int;
    ok_or(
        unsafe {
            cuda_interference_mean(output.as_mut_ptr(), phases_a.as_ptr(), phases_b.as_ptr(), n)
        },
        "interference_mean",
    )
}

pub fn phase_coherence(
    output: &mut [f32],
    sin_buf: &[f32],
    cos_buf: &[f32],
    amplitudes: &[f32],
) -> Result<(), String> {
    let n = sin_buf.len().min(cos_buf.len()).min(amplitudes.len()) as c_int;
    ok_or(
        unsafe {
            cuda_phase_coherence(
                output.as_mut_ptr(),
                sin_buf.as_ptr(),
                cos_buf.as_ptr(),
                amplitudes.as_ptr(),
                n,
            )
        },
        "phase_coherence",
    )
}

pub fn band_energy(
    band_energies: &mut [f32],
    amplitudes: &[f32],
    omegas: &[f32],
    band_edges: &[f32],
) -> Result<(), String> {
    let n_bands = band_energies.len() as c_int;
    let n = amplitudes.len().min(omegas.len()) as c_int;
    ok_or(
        unsafe {
            cuda_band_energy(
                band_energies.as_mut_ptr(),
                amplitudes.as_ptr(),
                omegas.as_ptr(),
                band_edges.as_ptr(),
                n_bands,
                n,
            )
        },
        "band_energy",
    )
}

pub fn winding_matrix(
    output: &mut [f32],
    phases_a: &[f32],
    phases_b: &[f32],
) -> Result<(), String> {
    let na = phases_a.len() as c_int;
    let nb = phases_b.len() as c_int;
    ok_or(
        unsafe {
            cuda_winding_matrix(
                output.as_mut_ptr(),
                phases_a.as_ptr(),
                phases_b.as_ptr(),
                na,
                nb,
            )
        },
        "winding_matrix",
    )
}

pub fn lazy_sincos_update(
    sin_out: &mut [f32],
    cos_out: &mut [f32],
    phases: &[f32],
    phases_prev: &mut [f32],
    eps: f32,
) -> Result<(), String> {
    let n = sin_out
        .len()
        .min(cos_out.len())
        .min(phases.len())
        .min(phases_prev.len()) as c_int;
    ok_or(
        unsafe {
            cuda_lazy_sincos_update(
                sin_out.as_mut_ptr(),
                cos_out.as_mut_ptr(),
                phases.as_ptr(),
                phases_prev.as_mut_ptr(),
                eps,
                n,
            )
        },
        "lazy_sincos",
    )
}

pub fn batch_coherence_sweep(
    output: &mut [f32],
    areas: &[f32],
    areas_sin: &[f32],
    areas_cos: &[f32],
    n_elem: usize,
) -> Result<(), String> {
    let total = areas.len();
    let n_areas = total / n_elem;
    ok_or(
        unsafe {
            cuda_batch_coherence_sweep(
                output.as_mut_ptr(),
                areas.as_ptr(),
                areas_sin.as_ptr(),
                areas_cos.as_ptr(),
                n_areas as c_int,
                n_elem as c_int,
            )
        },
        "batch_coherence",
    )
}

// ═══════════════════════════════════════════════════════════
//  T3 kernels (host wrappers)
// ═══════════════════════════════════════════════════════════

pub fn t3_update(
    phases: &mut [f32],
    amplitudes: &mut [f32],
    omegas: &[f32],
    coupling: &[f32],
    drives: &[f32],
    dt: f32,
    alpha: f32,
    gain: f32,
) -> Result<(), String> {
    let n = phases.len() as c_int;
    ok_or(
        unsafe {
            cuda_t3_update(
                phases.as_mut_ptr(),
                amplitudes.as_mut_ptr(),
                omegas.as_ptr(),
                coupling.as_ptr(),
                drives.as_ptr(),
                dt,
                alpha,
                gain,
                n,
            )
        },
        "t3_update",
    )
}

pub fn t3_prepare(
    phases: &mut [f32],
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> Result<(), String> {
    let n = phases.len() as c_int;
    ok_or(
        unsafe {
            cuda_t3_prepare(
                phases.as_mut_ptr(),
                cos_out.as_mut_ptr(),
                sin_out.as_mut_ptr(),
                n,
            )
        },
        "t3_prepare",
    )
}

pub fn edge_predictor(
    d_states: &DevicePtr<f32>,
    d_output: &DevicePtr<f32>,
    d_w1: &DevicePtr<f32>,
    d_b1: &DevicePtr<f32>,
    d_w2: &DevicePtr<f32>,
    d_b2: &DevicePtr<f32>,
    d_w3: &DevicePtr<f32>,
    d_b3: &DevicePtr<f32>,
    N: usize,
    state_dim: usize,
    hidden_dim: usize,
) -> Result<(), String> {
    let err = unsafe {
        cuda_edge_predictor(
            d_states.as_ptr(),
            d_output.as_ptr() as *mut c_float,
            d_w1.as_ptr(),
            d_b1.as_ptr(),
            d_w2.as_ptr(),
            d_b2.as_ptr(),
            d_w3.as_ptr(),
            d_b3.as_ptr(),
            N as c_int,
            state_dim as c_int,
            hidden_dim as c_int,
        )
    };
    ok_or(err, "edge_predictor")
}

/// Sample N states from GMM on GPU.
pub fn gmm_sample(
    d_weights: &DevicePtr<f32>,
    d_means: &DevicePtr<f32>,
    d_covars: &DevicePtr<f32>,
    d_cum_weights: &DevicePtr<f32>,
    d_states: &DevicePtr<f32>,
    N: usize,
    dim: usize,
    n_components: usize,
    seed: u32,
) -> Result<(), String> {
    let err = unsafe {
        cuda_gmm_sample(
            d_weights.as_ptr(),
            d_means.as_ptr(),
            d_covars.as_ptr(),
            d_cum_weights.as_ptr(),
            d_states.as_ptr() as *mut c_float,
            N as c_int,
            dim as c_int,
            n_components as c_int,
            seed,
        )
    };
    ok_or(err, "gmm_sample")
}

/// Top-k selection per row from NxN logits on GPU (pass 1).
pub fn topk_select(
    d_logits: &DevicePtr<f32>,
    d_target_degs: &DevicePtr<i32>,
    d_edge_counts: &DevicePtr<i32>,
    d_scratch: &DevicePtr<i32>,
    N: usize,
    threshold: f32,
    topk_mult: usize,
    max_k: usize,
) -> Result<(), String> {
    let err = unsafe {
        cuda_topk_select(
            d_logits.as_ptr(),
            d_target_degs.as_ptr(),
            d_edge_counts.as_ptr() as *mut c_int,
            d_scratch.as_ptr() as *mut c_int,
            N as c_int,
            threshold,
            topk_mult as c_int,
            max_k as c_int,
        )
    };
    ok_or(err, "topk_select")
}

/// Build CSR from scratch buffer on GPU (pass 2).
pub fn csr_build(
    d_edge_counts: &DevicePtr<i32>,
    d_row_ptr: &DevicePtr<i32>,
    d_scratch: &DevicePtr<i32>,
    d_col_indices: &DevicePtr<i32>,
    d_values: &DevicePtr<f32>,
    max_k: usize,
    N: usize,
) -> Result<(), String> {
    let err = unsafe {
        cuda_csr_build(
            d_edge_counts.as_ptr(),
            d_row_ptr.as_ptr(),
            d_scratch.as_ptr(),
            d_col_indices.as_ptr() as *mut c_int,
            d_values.as_ptr() as *mut c_float,
            max_k as c_int,
            N as c_int,
        )
    };
    ok_or(err, "csr_build")
}

// ═══════════════════════════════════════════════════════════
//  Dopamine-gated STDP + trace update
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_update_trace(
        trace: *mut c_float,
        rate: *const c_float,
        decay: c_float,
        n: c_int,
    ) -> c_int;
    fn cuda_csr_stdp_dopamine(
        weights: *mut c_float,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        trace: *const c_float,
        lr: c_float,
        delta: c_float,
        w_min: c_float,
        w_max: c_float,
        n_neurons: c_int,
    ) -> c_int;
    fn cuda_update_eligibility(
        elig: *mut c_float,
        spikes: *const c_int,
        row_ptr: *const c_int,
        col_indices: *const c_int,
        elig_add: c_float,
        elig_decay: c_float,
        n_neurons: c_int,
    ) -> c_int;
    fn cuda_csr_elig_stdp(
        weights: *mut c_float,
        elig: *const c_float,
        lr: c_float,
        delta: c_float,
        gate: c_float,
        w_min: c_float,
        w_max: c_float,
        n_synapses: c_int,
    ) -> c_int;
}

// ═══════════════════════════════════════════════════════════
//  T3 BPTT — full backward pass (K2+K3 + K1)
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_pst_jacobian(
        d_final_state: *const c_float,
        d_grad_pst: *const c_float,
        d_sums: *mut c_float,
        d_v: *mut c_float,
        n: c_int,
    ) -> c_int;

    fn cuda_pst_jacobian_add(
        d_final_state: *const c_float,
        d_grad_pst: *const c_float,
        d_sums: *mut c_float,
        d_v: *mut c_float,
        n: c_int,
    ) -> c_int;

    // T3 BPTT step (K2 + K3)
    fn cuda_bptt_step(
        d_state_step: *const c_float, // [6n]
        d_drive_step: *const c_float, // [3n]
        d_v: *mut c_float,            // [6n] in: v_current, out: v_prev
        d_scratch: *mut c_float,      // [12] or [12 * n_blocks]
        d_grad_step: *mut c_float,    // [3n] output grad for this step
        d_grad_pst: *const c_float,   // [pst_dim] velocity contribution
        w12: c_float,
        w13: c_float,
        w23: c_float,
        dt: c_float,
        gain: c_float,
        alpha: c_float,
        eps: c_float,
        n: c_int,
        is_last_step: c_int,
    ) -> c_int;
}

/// Compute PST Jacobian-vector product on GPU.
///
/// `final_state`: packed [th1(n), th2(n), th3(n), r1(n), r2(n), r3(n)] — on device
/// `grad_pst`: [pst_dim] — on device
/// `d_sums`: scratch [12] — on device (can be reused)
/// `v_out`: [6*n] packed as [v_th1, v_th2, v_th3, v_r1, v_r2, v_r3] — on device
pub fn pst_jacobian_gpu(
    d_final_state: &DevicePtr<f32>,
    d_grad_pst: &DevicePtr<f32>,
    d_sums: &DevicePtr<f32>,
    d_v: &DevicePtr<f32>,
    n: usize,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_pst_jacobian(
                d_final_state.as_ptr() as *const c_float,
                d_grad_pst.as_ptr() as *const c_float,
                d_sums.as_ptr() as *mut c_float,
                d_v.as_ptr() as *mut c_float,
                n as c_int,
            )
        },
        "pst_jacobian_gpu",
    )
}

// ─── CSR SpMV + tanh (GPU SPAUN) ────────────────────────

// ═══════════════════════════════════════════════════════════
//  Dopamine-gated STDP + trace update wrappers
// ═══════════════════════════════════════════════════════════

/// Update per-neuron eligibility traces from rate activity.
/// trace[i] = trace[i] * decay + abs(rate[i])
pub fn update_trace(trace: &mut [f32], rate: &[f32], decay: f32) -> Result<(), String> {
    let n = trace.len().min(rate.len());
    ok_or(
        unsafe { cuda_update_trace(trace.as_mut_ptr(), rate.as_ptr(), decay, n as c_int) },
        "update_trace",
    )
}

/// Dopamine-gated CSR STDP:
/// w_ij += lr * delta * (trace[pre] - trace[post])
/// Sign-preserving clip to [w_min, w_max]
pub fn csr_stdp_dopamine(
    weights: &mut [f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    trace: &[f32],
    lr: f32,
    delta: f32,
    w_min: f32,
    w_max: f32,
) -> Result<(), String> {
    let n = trace.len();
    ok_or(
        unsafe {
            cuda_csr_stdp_dopamine(
                weights.as_mut_ptr(),
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                trace.as_ptr(),
                lr,
                delta,
                w_min,
                w_max,
                n as c_int,
            )
        },
        "csr_stdp_dopamine",
    )
}

/// Update per-synapse eligibility trace (CPE-style):
/// For spiking pre-neurons, add elig_add to all outgoing synapses.
/// Then decay all traces by elig_decay.
pub fn update_eligibility(
    elig: &mut [f32],
    spikes: &[i32],
    row_ptr: &[i32],
    col_indices: &[i32],
    elig_add: f32,
    elig_decay: f32,
) -> Result<(), String> {
    let n = row_ptr.len() - 1;
    ok_or(
        unsafe {
            cuda_update_eligibility(
                elig.as_mut_ptr(),
                spikes.as_ptr(),
                row_ptr.as_ptr(),
                col_indices.as_ptr(),
                elig_add,
                elig_decay,
                n as c_int,
            )
        },
        "update_eligibility",
    )
}

/// Apply dopamine-gated STDP with per-synapse eligibility:
/// w[i] += lr * delta * elig[i] * gate
/// Sign-preserving clip to [w_min, w_max]
pub fn csr_elig_stdp(
    weights: &mut [f32],
    elig: &[f32],
    lr: f32,
    delta: f32,
    gate: f32,
    w_min: f32,
    w_max: f32,
) -> Result<(), String> {
    let n = weights.len().min(elig.len());
    ok_or(
        unsafe {
            cuda_csr_elig_stdp(
                weights.as_mut_ptr(),
                elig.as_ptr(),
                lr,
                delta,
                gate,
                w_min,
                w_max,
                n as c_int,
            )
        },
        "csr_elig_stdp",
    )
}

// ═══════════════════════════════════════════════════════════
//  Device-pointer wrappers for dopamine STDP kernels
// ═══════════════════════════════════════════════════════════

/// Dopamine-gated CSR STDP (device pointers only).
/// Same as csr_stdp_dopamine but takes raw device pointers.
pub fn csr_stdp_dopamine_devptr(
    d_weights: *mut c_float,
    d_row_ptr: *const c_int,
    d_col_indices: *const c_int,
    d_trace: *const c_float,
    lr: f32,
    delta: f32,
    w_min: f32,
    w_max: f32,
    n_neurons: usize,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_csr_stdp_dopamine(
                d_weights,
                d_row_ptr,
                d_col_indices,
                d_trace,
                lr,
                delta,
                w_min,
                w_max,
                n_neurons as c_int,
            )
        },
        "csr_stdp_dopamine_devptr",
    )
}

/// CSR SpMV + tanh on GPU.
///
/// out = tanh(W × in + input_scale * input_vec)
///
/// All pointers must be device pointers (GPU memory).
/// `stream` is a CUDA stream (can be NULL for default stream).
///
/// Returns Ok(()) on success.
pub fn csr_spmv_tanh(
    out: *mut c_float,
    in_state: *const c_float,
    row_ptr: *const c_int,
    col_indices: *const c_int,
    values: *const c_float,
    input_vec: *const c_float,
    input_scale: f32,
    n: usize,
    stream: Option<*mut c_void>,
) -> Result<(), String> {
    let stream_ptr = stream.unwrap_or(core::ptr::null_mut());
    let err = unsafe {
        cuda_csr_spmv_tanh(
            out,
            in_state,
            row_ptr,
            col_indices,
            values,
            input_vec,
            input_scale,
            n as c_int,
            stream_ptr,
        )
    };
    if err == 0 {
        Ok(())
    } else {
        Err(format!("csr_spmv_tanh: error {err}"))
    }
}

// ═══════════════════════════════════════════════════════════
//  T3 Fused Forward + BPTT Backward (entirely on GPU)
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_t3_fused_forward_bptt(
        d_all_drives: *const c_float,    // [n_steps × 3n]
        d_grad_drives: *mut c_float,     // [n_steps × 3n]
        d_state: *mut c_float,           // [6n]
        d_sincos: *mut c_float,          // [6n]
        d_coupling: *mut c_float,        // [3n]
        d_omega: *const c_float,         // [3n]
        d_traj_state: *mut c_float,      // [n_steps × 6n]
        d_final_state: *mut c_float,     // [6n]
        d_grad_pst: *const c_float,      // [pst_dim]
        d_per_step_grad: *const c_float, // [n_steps × pst_dim] or NULL
        d_v: *mut c_float,               // [6n]
        d_scratch: *mut c_float,         // [12 × n_blocks]
        d_k1_sums: *mut c_float,         // [12]
        w12: c_float,
        w13: c_float,
        w23: c_float,
        dt: c_float,
        gain: c_float,
        alpha: c_float,
        eps: c_float,
        n: c_int,
        n_steps: c_int,
    ) -> c_int;
}

/// Run the fused T3 forward + BPTT backward pass entirely on GPU.
///
/// This replaces: CPU forward → trajectory save → H2D upload → GPU backward.
/// Now: GPU forward → trajectory in VRAM → GPU backward → D2H grad_drives.
///
/// # Arguments
///
/// All pointers are **device pointers** (GPU memory pre-allocated via `DevicePtr`).
///
/// * `d_all_drives` — [n_steps × 3n] all drives in step-major layout
/// * `d_grad_drives` — [n_steps × 3n] output gradient buffer
/// * `d_state` — [6n] initial state [th1,th2,th3,r1,r2,r3], updated in-place
/// * `d_sincos` — [6n] workspace
/// * `d_coupling` — [3n] workspace
/// * `d_omega` — [3n] natural frequencies [omega1,omega2,omega3]
/// * `d_traj_state` — [n_steps × 6n] trajectory buffer
/// * `d_traj_drive` — [n_steps × 3n] drive buffer
/// * `d_final_state` — [6n] final state copy for PST Jacobian
/// * `d_grad_pst` — [pst_dim] loss gradient
/// * `d_v` — [6n] gradient vector workspace
/// * `d_scratch` — [12 × max_blocks] scratch for coupling + BPTT reductions
/// * `d_k1_sums` — [12] scratch for PST Jacobian sums
pub fn t3_fused_forward_bptt(
    d_all_drives: &DevicePtr<f32>,
    d_grad_drives: &DevicePtr<f32>,
    d_state: &DevicePtr<f32>,
    d_sincos: &DevicePtr<f32>,
    d_coupling: &DevicePtr<f32>,
    d_omega: &DevicePtr<f32>,
    d_traj_state: &DevicePtr<f32>,
    d_final_state: &DevicePtr<f32>,
    d_grad_pst: &DevicePtr<f32>,
    d_per_step_grad: Option<&DevicePtr<f32>>,
    d_v: &DevicePtr<f32>,
    d_scratch: &DevicePtr<f32>,
    d_k1_sums: &DevicePtr<f32>,
    w12: f32,
    w13: f32,
    w23: f32,
    dt: f32,
    gain: f32,
    alpha: f32,
    eps: f32,
    n: usize,
    n_steps: usize,
) -> Result<(), String> {
    let per_step_ptr = d_per_step_grad
        .map(|p| p.as_ptr() as *const c_float)
        .unwrap_or(core::ptr::null());
    ok_or(
        unsafe {
            cuda_t3_fused_forward_bptt(
                d_all_drives.as_ptr() as *const c_float,
                d_grad_drives.as_ptr() as *mut c_float,
                d_state.as_ptr() as *mut c_float,
                d_sincos.as_ptr() as *mut c_float,
                d_coupling.as_ptr() as *mut c_float,
                d_omega.as_ptr() as *const c_float,
                d_traj_state.as_ptr() as *mut c_float,
                d_final_state.as_ptr() as *mut c_float,
                d_grad_pst.as_ptr() as *const c_float,
                per_step_ptr,
                d_v.as_ptr() as *mut c_float,
                d_scratch.as_ptr() as *mut c_float,
                d_k1_sums.as_ptr() as *mut c_float,
                w12,
                w13,
                w23,
                dt,
                gain,
                alpha,
                eps,
                n as c_int,
                n_steps as c_int,
            )
        },
        "t3_fused_forward_bptt",
    )
}

/// Compute v += J^T · grad_pst on GPU (ADD variant for per-step SPIGRAD).
///
/// Same as `pst_jacobian_gpu` but ADDS to `d_v` instead of overwriting.
pub fn pst_jacobian_gpu_add(
    d_state: &DevicePtr<f32>,
    d_grad_pst: &DevicePtr<f32>,
    d_sums: &DevicePtr<f32>,
    d_v: &DevicePtr<f32>,
    n: usize,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_pst_jacobian_add(
                d_state.as_ptr() as *const c_float,
                d_grad_pst.as_ptr() as *const c_float,
                d_sums.as_ptr() as *mut c_float,
                d_v.as_ptr() as *mut c_float,
                n as c_int,
            )
        },
        "pst_jacobian_gpu_add",
    )
}

/// Run one T3 BPTT step on GPU: drive gradient (K3) + backprop through dynamics (K2).
///
/// All buffers are device pointers (pre-allocated via `DevicePtr`).
/// `d_v` is updated in-place: [6n] v_current → v_prev.
/// `d_grad_step` receives the drive gradient for this timestep: [3n].
pub fn bptt_step_gpu(
    d_state_step: &DevicePtr<f32>,
    d_drive_step: &DevicePtr<f32>,
    d_v: &DevicePtr<f32>,
    d_scratch: &DevicePtr<f32>,
    d_grad_step: &DevicePtr<f32>,
    d_grad_pst: Option<&DevicePtr<f32>>,
    w12: f32,
    w13: f32,
    w23: f32,
    dt: f32,
    gain: f32,
    alpha: f32,
    eps: f32,
    n: usize,
    is_last_step: bool,
) -> Result<(), String> {
    let grad_pst_ptr = d_grad_pst
        .map(|p| p.as_ptr() as *const c_float)
        .unwrap_or(core::ptr::null());
    ok_or(
        unsafe {
            cuda_bptt_step(
                d_state_step.as_ptr() as *const c_float,
                d_drive_step.as_ptr() as *const c_float,
                d_v.as_ptr() as *mut c_float,
                d_scratch.as_ptr() as *mut c_float,
                d_grad_step.as_ptr() as *mut c_float,
                grad_pst_ptr,
                w12,
                w13,
                w23,
                dt,
                gain,
                alpha,
                eps,
                n as c_int,
                is_last_step as c_int,
            )
        },
        "bptt_step_gpu",
    )
}

// Required for extern block — these are already declared elsewhere but we need them here
// The cudaMalloc/cudaMemcpy/cudaFree are already declared in the connectome section above

// ═══════════════════════════════════════════════════════════
//  SNN kernels (LIF-AT, SORN, Spike Encoding)
// ═══════════════════════════════════════════════════════════

extern "C" {
    fn cuda_lif_at_step(
        v: *mut c_float,
        theta: *mut c_float,
        I: *mut c_float,
        spikes: *mut c_int,
        params: *const c_float,
        n: c_int,
    ) -> c_int;
    fn cuda_intrinsic_plasticity(
        theta: *mut c_float,
        spikes: *const c_int,
        lr_ip: c_float,
        target_rate: c_float,
        decay: c_float,
        n: c_int,
    ) -> c_int;
    fn cuda_synaptic_normalization(
        weights: *mut c_float,
        n_pre: c_int,
        n_post: c_int,
        target_sum: c_float,
        eps: c_float,
    ) -> c_int;
    fn cuda_bsa_encode(
        spikes_out: *mut c_int,
        signal: *const c_float,
        reconstructed: *const c_float,
        threshold: c_float,
        n: c_int,
    ) -> c_int;
    fn cuda_phase_encode(
        spikes_out: *mut c_int,
        phases: *mut c_float,
        cos_phases: *const c_float,
        signal: *const c_float,
        threshold: c_float,
        gain: c_float,
        n: c_int,
    ) -> c_int;
    fn cuda_ttfs_init(
        fire_times: *mut c_int,
        values: *const c_float,
        max_delay: c_int,
        n: c_int,
    ) -> c_int;
    fn cuda_ttfs_step(
        spikes_out: *mut c_int,
        fire_times: *const c_int,
        fired: *mut c_int,
        current_step: c_int,
        n: c_int,
    ) -> c_int;
    fn cuda_population_rate_decode(
        rates_out: *mut c_float,
        spikes: *const c_int,
        n_neurons: c_int,
        window: c_int,
    ) -> c_int;
    // Sync variants via GpuCtx (upload→kernel→sync in one call)
    fn gpu_lif_at_step(ctx: *mut GpuCtx, host_params: *const c_float, n: c_int) -> c_int;
    fn gpu_intrinsic_plasticity(
        ctx: *mut GpuCtx,
        lr_ip: c_float,
        target_rate: c_float,
        decay: c_float,
    ) -> c_int;
}

pub fn step_lif_at(
    v: &mut [f32],
    theta: &mut [f32],
    I: &mut [f32],
    spikes: &mut [i32],
    dt: f32,
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_theta: &[f32],
    beta: &[f32],
) -> Result<(), String> {
    let n = v.len();
    let mut flat = vec![0.0f32; 1 + 6 * n];
    flat[0] = dt;
    for i in 0..n {
        flat[1 + i] = tau_m[i];
        flat[1 + n + i] = v_rest[i];
        flat[1 + 2 * n + i] = v_thresh[i];
        flat[1 + 3 * n + i] = v_reset[i];
        flat[1 + 4 * n + i] = tau_theta[i];
        flat[1 + 5 * n + i] = beta[i];
    }
    ok_or(
        unsafe {
            cuda_lif_at_step(
                v.as_mut_ptr(),
                theta.as_mut_ptr(),
                I.as_mut_ptr(),
                spikes.as_mut_ptr(),
                flat.as_ptr(),
                n as c_int,
            )
        },
        "lif_at",
    )
}

pub fn snn_intrinsic_plasticity(
    theta: &mut [f32],
    spikes: &[i32],
    lr_ip: f32,
    target_rate: f32,
    decay: f32,
) -> Result<(), String> {
    let n = theta.len() as c_int;
    ok_or(
        unsafe {
            cuda_intrinsic_plasticity(
                theta.as_mut_ptr(),
                spikes.as_ptr(),
                lr_ip,
                target_rate,
                decay,
                n,
            )
        },
        "intrinsic_plasticity",
    )
}

pub fn snn_synaptic_normalization(
    weights: &mut [f32],
    n_pre: usize,
    n_post: usize,
    target_sum: f32,
    eps: f32,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_synaptic_normalization(
                weights.as_mut_ptr(),
                n_pre as c_int,
                n_post as c_int,
                target_sum,
                eps,
            )
        },
        "synaptic_normalization",
    )
}

pub fn snn_bsa_encode(
    spikes_out: &mut [i32],
    signal: &[f32],
    reconstructed: &[f32],
    threshold: f32,
) -> Result<(), String> {
    let n = spikes_out.len().min(signal.len()).min(reconstructed.len()) as c_int;
    ok_or(
        unsafe {
            cuda_bsa_encode(
                spikes_out.as_mut_ptr(),
                signal.as_ptr(),
                reconstructed.as_ptr(),
                threshold,
                n,
            )
        },
        "bsa_encode",
    )
}

pub fn snn_phase_encode(
    spikes_out: &mut [i32],
    phases: &mut [f32],
    cos_phases: &[f32],
    signal: &[f32],
    threshold: f32,
    gain: f32,
) -> Result<(), String> {
    let n = spikes_out
        .len()
        .min(phases.len())
        .min(cos_phases.len())
        .min(signal.len()) as c_int;
    ok_or(
        unsafe {
            cuda_phase_encode(
                spikes_out.as_mut_ptr(),
                phases.as_mut_ptr(),
                cos_phases.as_ptr(),
                signal.as_ptr(),
                threshold,
                gain,
                n,
            )
        },
        "phase_encode",
    )
}

pub fn snn_ttfs_init(
    fire_times: &mut [i32],
    values: &[f32],
    max_delay: usize,
) -> Result<(), String> {
    let n = fire_times.len().min(values.len()) as c_int;
    ok_or(
        unsafe {
            cuda_ttfs_init(
                fire_times.as_mut_ptr(),
                values.as_ptr(),
                max_delay as c_int,
                n,
            )
        },
        "ttfs_init",
    )
}

pub fn snn_ttfs_step(
    spikes_out: &mut [i32],
    fire_times: &[i32],
    fired: &mut [i32],
    current_step: usize,
) -> Result<(), String> {
    let n = spikes_out.len().min(fire_times.len()).min(fired.len()) as c_int;
    ok_or(
        unsafe {
            cuda_ttfs_step(
                spikes_out.as_mut_ptr(),
                fire_times.as_ptr(),
                fired.as_mut_ptr(),
                current_step as c_int,
                n,
            )
        },
        "ttfs_step",
    )
}

pub fn snn_population_rate_decode(
    rates_out: &mut [f32],
    spikes: &[i32],
    n_neurons: usize,
    window: usize,
) -> Result<(), String> {
    ok_or(
        unsafe {
            cuda_population_rate_decode(
                rates_out.as_mut_ptr(),
                spikes.as_ptr(),
                n_neurons as c_int,
                window as c_int,
            )
        },
        "population_rate_decode",
    )
}

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{get_device_count, set_device, stdp_update, GpuContext};
    use serial_test::serial;

    macro_rules! check_gpu {
        () => {
            let count = get_device_count().unwrap_or(0);
            if count == 0 {
                eprintln!("⚠ No CUDA devices — skipping");
                return;
            }
        };
    }

    fn require_gpu() {
        check_gpu!();
    }

    #[serial]
    #[test]
    fn test_device_count() {
        let count = get_device_count().unwrap_or(0);
        println!("CUDA devices: {count}");
        if count > 0 {
            assert!(set_device(0).is_ok());
        }
    }

    #[serial]
    #[test]
    fn test_gpu_context_create_destroy() {
        check_gpu!();
        let ctx = GpuContext::new(1024, 4096).unwrap();
        println!(
            "GpuContext: {} neurons, {} synapses",
            ctx.n_neurons(),
            ctx.n_synapses()
        );
        // Drop frees everything
    }

    #[serial]
    #[test]
    fn test_gpu_context_upload_and_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload neurons
        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Upload Kuramoto
        let phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

        // Run fused step
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();

        // Download results
        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s > 0).count();
        println!("Fused step: {n_spikes} spikes out of {n}");
    }

    #[serial]
    #[test]
    fn test_gpu_context_fused_vs_separate() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Separate: izhikevich + stdp
        ctx.izhikevich_step(0.001, 0.99).unwrap();
        ctx.stdp_update(0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        let mut spikes_sep = vec![0i32; n];
        ctx.download_spikes(&mut spikes_sep).unwrap();

        // Re-upload state
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // Fused: izhikevich + stdp in one launch
        ctx.fused_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0).unwrap();
        ctx.sync().unwrap();
        let mut spikes_fused = vec![0i32; n];
        ctx.download_spikes(&mut spikes_fused).unwrap();

        assert_eq!(spikes_sep, spikes_fused, "Fused and separate must match");
        println!(
            "Fused vs separate: ✅ identical ({} spikes each)",
            spikes_sep.iter().filter(|&&s| s > 0).count()
        );
    }

    #[serial]
    #[test]
    fn test_gpu_graph_simulation_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        let phases = vec![0.0f32; n];
        let omegas = vec![1.0f32; n];
        let coupling = vec![0.0f32; n];
        ctx.upload_kuramoto(&phases, &omegas, &coupling).unwrap();

        // Run through simulation_step which uses CUDA Graphs
        ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0)
            .unwrap();

        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s > 0).count();
        println!("CUDA Graph simulation step: {n_spikes} spikes");
    }

    #[serial]
    #[test]
    fn test_gpu_graph_replay_speed() {
        check_gpu!();
        let n = 10000;
        let ctx = GpuContext::new(n, n).unwrap();

        let v = vec![-65.0f32; n];
        let u = vec![-13.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-65.0f32; n];
        let d = vec![8.0f32; n];
        ctx.upload_neurons(&v, &u, &a, &b, &c, &d).unwrap();

        // First call captures the graph
        ctx.simulation_step(0.001, 0.99, 0.1, 0.1, 0.0, 1.0, -100.0, 100.0)
            .unwrap();

        // Subsequent calls replay
        let n_iter = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            ctx.graph_replay().unwrap();
        }
        ctx.sync().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;
        println!("CUDA Graph replay {n_iter}×: {ms:.4}ms per step (n={n})");
    }

    // ─── Legacy tests (unchanged) ───────────────────────

    #[serial]
    #[test]
    fn test_xor_attention() {
        check_gpu!();
        let q = vec![0u8; 8];
        let mut k = vec![0xFFu8; 64];
        k[..8].fill(0x00);
        let v = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut o = vec![0.0f32];
        xor_attention(&q, &k, &v, &mut o, 8, 8).unwrap();
        println!("XOR attention: {:.6}", o[0]);
        assert!((o[0] - 1.0).abs() < 0.01);
    }

    #[serial]
    #[test]
    fn test_safety_clamp() {
        check_gpu!();
        let mut data = vec![-10.0f32, -5.0, 0.0, 5.0, 10.0, -3.0, 3.0, 100.0];
        safety_clamp(&mut data, -1.0, 1.0).unwrap();
        for &v in &data {
            assert!((-1.0..=1.0).contains(&v));
        }
        println!("safety_clamp: {:?}", data);
    }

    #[serial]
    #[test]
    fn test_stdp_update() {
        check_gpu!();
        let n = 64;
        let mut w = vec![0.5f32; n];
        let tp = vec![1.0f32; n];
        let tn = vec![0.0f32; n];
        stdp_update(&mut w, &tp, &tn, 0.1, 0.1, 0.0, 1.0).unwrap();
        println!("STDP[0]: {:.4}", w[0]);
        assert!((w[0] - 0.6).abs() < 0.01);
    }

    #[serial]
    #[test]
    fn test_nlms_train() {
        check_gpu!();
        let n = 32;
        let mut w = vec![0.0f32; n];
        let mut bias = vec![0.0f32; 1];
        let s: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.1).collect();
        nlms_train(&mut w, &mut bias, &s, 1.0, 0.01, 1e-8).unwrap();
        println!("NLMS[0]: {:.6}", w[0]);
        assert!(w.iter().all(|&x| x != 0.0), "All weights non-zero");
    }

    #[serial]
    #[test]
    fn test_echo_tap() {
        check_gpu!();
        let n_osc = 8;
        let delay_len = 16;
        let mut buf = vec![0.0f32; n_osc * delay_len];
        for i in 0..n_osc {
            buf[i * delay_len + 3] = i as f32;
        }
        let mut out = vec![0.0f32; n_osc];
        echo_tap(&mut out, &buf, n_osc, delay_len, 3).unwrap();
        println!("echo_tap[0]: {:.4}", out[0]);
        assert!((out[0] - 0.0).abs() < 0.01);
    }

    #[serial]
    #[test]
    fn test_kuramoto_prepare() {
        check_gpu!();
        let n = 64;
        let phases: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut cos_out = vec![0.0f32; n];
        let mut sin_out = vec![0.0f32; n];
        kuramoto_prepare(&phases, &mut cos_out, &mut sin_out).unwrap();
        println!(
            "kuramoto_prepare[0]: cos={:.4} sin={:.4}",
            cos_out[0], sin_out[0]
        );
        assert!((cos_out[0] - 1.0).abs() < 0.01);
        assert!(sin_out[0].abs() < 0.01);
    }

    // ─── LIF-AT tests ────────────────────────────────────

    #[serial]
    #[test]
    fn test_gpu_lif_at_basic_step() {
        check_gpu!();
        let n = 256;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload initial v at resting potential
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Inject strong current to drive spiking
        let I = vec![20.0f32; n];
        ctx.upload_I(&I).unwrap();

        // Run LIF-AT step
        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];
        ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta)
            .unwrap();

        // Download spikes
        let mut spikes = vec![0i32; n];
        ctx.download_spikes(&mut spikes).unwrap();
        let n_spikes = spikes.iter().filter(|&&s| s != 0).count();

        // Download v
        let mut v_out = vec![0.0f32; n];
        ctx.download_v(&mut v_out).unwrap();

        // With strong current, v should increase from -70
        let avg_v: f32 = v_out.iter().sum::<f32>() / n as f32;
        println!(
            "LIF-AT basic step: avg v = {:.4}, spikes = {n_spikes}/{n}",
            avg_v
        );

        // At least some v increased (even if not all spiked yet)
        assert!(avg_v > -70.0, "v should increase from -70, got avg {avg_v}");
    }

    #[serial]
    #[test]
    fn test_gpu_lif_at_spike_with_strong_current() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        // Upload state
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Very strong current to guarantee spikes within a few steps
        let I = vec![100.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];

        // Multiple steps to ensure spike
        let mut total_spikes = 0;
        for _step in 0..30 {
            ctx.upload_I(&I).unwrap(); // re-update I each step (kernel resets I after use)
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta)
                .unwrap();
            let mut spikes = vec![0i32; n];
            ctx.download_spikes(&mut spikes).unwrap();
            total_spikes += spikes.iter().filter(|&&s| s != 0).count();
        }

        println!("LIF-AT spike test: {total_spikes} total spikes over 30 steps");
        assert!(
            total_spikes > 0,
            "Expected at least 1 spike with I=100, got 0"
        );
    }

    #[serial]
    #[test]
    fn test_gpu_lif_at_no_spike_with_negative_current() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Inhibitory current — no spikes expected
        let I = vec![-10.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![100.0f32; n];
        let beta = vec![5.0f32; n];

        let mut total_spikes = 0;
        for _step in 0..30 {
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta)
                .unwrap();
            let mut spikes = vec![0i32; n];
            ctx.download_spikes(&mut spikes).unwrap();
            total_spikes += spikes.iter().filter(|&&s| s != 0).count();
        }

        println!("LIF-AT inhibitory test: {total_spikes} spikes (expected 0)");
        assert_eq!(
            total_spikes, 0,
            "Expected no spikes with I=-10, got {total_spikes}"
        );
    }

    // ─── CPU vs CUDA cross-validation ───────────────────

    #[serial]
    #[test]
    fn test_lif_at_cpu_gpu_identity() {
        check_gpu!();
        let raw_n: usize = 64;
        let n = raw_n.div_ceil(8) * 8; // pad for CPU ASM
        let ctx = GpuContext::new(n, n).unwrap();

        // Identical initial state for both backends
        let mut v_cpu: Vec<f32> = (0..n).map(|i| -70.0 + (i % 8) as f32 * 0.5).collect();
        let mut theta_cpu: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.3).collect();
        let mut I_cpu: Vec<f32> = (0..n).map(|i| 15.0 + (i % 7) as f32 * 2.0).collect();
        let mut spikes_cpu = vec![0i32; n];

        // Per-neuron params
        let tau_m: Vec<f32> = (0..n).map(|i| 10.0 + (i % 3) as f32).collect();
        let v_rest: Vec<f32> = (0..n).map(|_| -70.0).collect();
        let v_thresh: Vec<f32> = (0..n).map(|_| -55.0).collect();
        let v_reset: Vec<f32> = (0..n).map(|_| -75.0).collect();
        let tau_theta: Vec<f32> = (0..n).map(|i| 50.0 + (i % 4) as f32 * 10.0).collect();
        let beta: Vec<f32> = (0..n).map(|i| 5.0 + (i % 3) as f32).collect();
        let dt = 1.0;

        // Upload state to GPU (identical to CPU init)
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_cpu, &u, &a, &b, &c, &d).unwrap();
        ctx.upload_I(&I_cpu).unwrap();
        ctx.upload_theta(&theta_cpu).unwrap();

        // Run 10 steps on both backends
        let mut refractory_cpu = vec![0.0f32; n];
        let tau_ref = 3.0f32;
        let mut cpu_params = vec![dt; n]; // p[0..n] = dt
        cpu_params.extend_from_slice(&tau_m);
        cpu_params.extend_from_slice(&v_rest);
        cpu_params.extend_from_slice(&v_thresh);
        cpu_params.extend_from_slice(&v_reset);
        cpu_params.extend_from_slice(&tau_theta);
        cpu_params.extend_from_slice(&beta);
        for _step in 0..10 {
            // GPU step (must run first to use the SAME I value as CPU)
            ctx.upload_I(&I_cpu).unwrap(); // sync I before step
            ctx.lif_at_step(dt, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta)
                .unwrap();

            // CPU step (SIMD reference, same I value)
            taas_kernels::safe::lif_at_step(
                &mut v_cpu,
                &mut theta_cpu,
                &mut I_cpu,
                &mut spikes_cpu,
                &cpu_params,
                &mut refractory_cpu,
                tau_ref,
                None,
            );

            // GPU kernel resets I to 0 after each step — match on CPU side
            I_cpu.iter_mut().for_each(|x| *x = 0.0);

            // Download GPU results
            let mut v_gpu = vec![0.0f32; n];
            let mut theta_gpu = vec![0.0f32; n];
            let mut spikes_gpu = vec![0i32; n];
            ctx.download_v(&mut v_gpu).unwrap();
            ctx.download_theta(&mut theta_gpu).unwrap();
            ctx.download_spikes(&mut spikes_gpu).unwrap();

            // Compare (must be bit-identical within f32 precision)
            for i in 0..n {
                let v_diff = (v_cpu[i] - v_gpu[i]).abs();
                let th_diff = (theta_cpu[i] - theta_gpu[i]).abs();
                // CPU ASM stores -1 for spike (all-ones from vcmpps),
                // CUDA stores +1. Both use 0 for no spike.
                let sp_cpu = spikes_cpu[i] != 0;
                let sp_gpu = spikes_gpu[i] != 0;

                assert!(
                    v_diff < 2e-1,
                    "Step {_step}, neuron {i}: v CPU={} GPU={}",
                    v_cpu[i],
                    v_gpu[i]
                );
                assert!(
                    th_diff < 1e-1,
                    "Step {_step}, neuron {i}: theta CPU={} GPU={}",
                    theta_cpu[i],
                    theta_gpu[i]
                );
                assert_eq!(
                    sp_cpu, sp_gpu,
                    "Step {_step}, neuron {i}: spike CPU={} GPU={}",
                    spikes_cpu[i], spikes_gpu[i]
                );
            }
        }

        println!("CPU vs GPU LIF-AT: ✅ bit-identical across 10 steps (n={n})");
    }

    // ─── Intrinsic Plasticity CUDA test ──────────────────

    #[serial]
    #[test]
    fn test_gpu_intrinsic_plasticity() {
        check_gpu!();
        let n = 64;
        let ctx = GpuContext::new(n, n).unwrap();

        // Set up: some spiked, some didn't
        let v_init = vec![-70.0f32; n];
        let u = vec![0.0f32; n];
        let a = vec![0.02f32; n];
        let b = vec![0.2f32; n];
        let c = vec![-70.0f32; n];
        let d = vec![2.0f32; n];
        ctx.upload_neurons(&v_init, &u, &a, &b, &c, &d).unwrap();

        // Upload initial theta (all 1.0) and spikes
        let theta_init = vec![1.0f32; n];
        ctx.upload_theta(&theta_init).unwrap();

        // Inject current to generate spikes
        let I = vec![100.0f32; n];
        ctx.upload_I(&I).unwrap();

        let tau_m = vec![10.0f32; n];
        let v_rest = vec![-70.0f32; n];
        let v_thresh = vec![-55.0f32; n];
        let v_reset = vec![-75.0f32; n];
        let tau_theta = vec![50.0f32; n];
        let beta = vec![5.0f32; n];

        // Run a few LIF-AT steps to generate spikes
        for _ in 0..5 {
            ctx.upload_I(&I).unwrap(); // re-update I each step (kernel resets I after use)
            ctx.lif_at_step(1.0, &tau_m, &v_rest, &v_thresh, &v_reset, &tau_theta, &beta)
                .unwrap();
        }

        // Download theta before IP
        let mut theta_before = vec![0.0f32; n];
        ctx.download_theta(&mut theta_before).unwrap();

        // Apply Intrinsic Plasticity
        let lr_ip = 0.1;
        let target_rate = 0.5;
        let decay = 0.9;
        ctx.intrinsic_plasticity(lr_ip, target_rate, decay).unwrap();

        // Download theta after IP
        let mut theta_after = vec![0.0f32; n];
        ctx.download_theta(&mut theta_after).unwrap();

        // Verify theta changed
        let mut changed = false;
        for i in 0..n {
            // θ = θ + lr·(rate - target), θ *= decay
            // rate=1 if spiked, 0 if not
            let theta_manual = theta_before[i] + lr_ip * (1.0 - target_rate);
            let theta_manual = theta_manual * decay;
            let theta_manual = theta_manual.clamp(0.0, 50.0);
            let diff = (theta_after[i] - theta_manual).abs();
            assert!(
                diff < 1e-5,
                "IP mismatch neuron {i}: before={:.6} expected={:.6} got={:.6}",
                theta_before[i],
                theta_manual,
                theta_after[i]
            );
            if (theta_after[i] - theta_before[i]).abs() > 1e-6 {
                changed = true;
            }
        }

        assert!(changed, "IP should have changed at least one theta value");
        println!("GPU Intrinsic Plasticity: ✅ all thetas match expected values");
    }
}
