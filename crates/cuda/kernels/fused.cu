/**
 * Fused CUDA kernels — combine multiple operations into one launch.
 *
 * Motivation: each kernel launch costs ~5-50μs. By fusing common sequential
 * operations, we eliminate launch overhead AND keep intermediate results
 * in registers/shared memory instead of global memory.
 */
#include <cuda_runtime.h>
#include <stdint.h>

#include "gpu_context.h"

// ─── Fused Izhikevich + STDP ────────────────────────────
//
// Combines: izhikevich_step + stdp_update + trace_decay
// Saves: 2 kernel launches, 2× global memory round-trips for traces
//
// Key insight: after a neuron spikes, its trace is used immediately for STDP.
// Keeping this in registers avoids writing/reading trace from global memory.

extern "C" __global__ void fused_izhikevich_stdp_kernel(
    // Neuron state
    float* v, float* u, float* trace,
    const float* a, const float* b, const float* I,
    float dt, float trace_decay,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref,
    // STDP state (same thread index = same neuron)
    float* weights, const float* traces_pre, const float* traces_post,
    float lr_ltd, float lr_ltp, float w_min, float w_max,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float vi = v[idx], ui = u[idx], Ii = I[idx];

    // ─── Refractory period: skip all dynamics (matches CPU SIMD behavior) ──
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        spikes[idx] = 0;
        return;  // v, u, and trace unchanged during refractory
    }

    // ─── Izhikevich step ───────────────────────────────
    vi += dt * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + Ii);
    ui += dt * a[idx] * (b[idx] * vi - ui);
    int spiked = (vi >= 30.0f);
    if (spiked) { vi = c[idx]; ui += d[idx]; }
    if (refractory && spiked) { refractory[idx] = tau_ref; }

    // ─── Trace decay + spike event ─────────────────────
    float tr = trace[idx] * trace_decay;
    if (spiked) tr += 1.0f;
    trace[idx] = tr;

    // ─── STDP update ───────────────────────────────────
    float w = weights[idx];
    w += lr_ltp * traces_pre[idx] - lr_ltd * traces_post[idx];
    w = fminf(w_max, fmaxf(w_min, w));
    weights[idx] = w;

    // ─── Store results ─────────────────────────────────
    v[idx] = vi;
    u[idx] = ui;
    spikes[idx] = spiked;
}

// ─── Async stream-based launch ──────────────────────────

extern "C" int gpu_fused_izhikevich_stdp_async(void* vctx,
    float dt, float trace_decay,
    float lr_ltd, float lr_ltp, float w_min, float w_max)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    fused_izhikevich_stdp_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_v, ctx->d_u, ctx->d_trace,
        ctx->d_a, ctx->d_b, ctx->d_I, dt, trace_decay,
        ctx->d_spikes, ctx->d_c, ctx->d_d,
        ctx->d_refractory, 3.0f,
        ctx->d_weights, ctx->d_traces_pre, ctx->d_traces_post,
        lr_ltd, lr_ltp, w_min, w_max, n);
    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════
//  Fused Connectome CSR Propagation (no Izhikevich)
// ═══════════════════════════════════════════════════════
//
// Single kernel launch: reset I to external_I, then CSR propagate spikes.
// Izhikevich step is called separately from Rust via gpu_izhikevich_step_async
// (which is in izhikevich.cu and knows its own kernel symbol).
//
// No H2D/D2H transfers between steps — everything on GPU.

/// Kernel: reset I_buf[i] = external_I[i] for all neurons.
/// Defined FIRST so it's visible to gpu_connectome_csr_async.
extern "C" __global__ void reset_I_kernel(
    float* I_buf, const float* external_I, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    I_buf[idx] = external_I[idx];
}

/// CSR propagation kernel: for each spiked neuron, atomicAdd its
/// synaptic weights to the I buffer of postsynaptic targets.
///
/// One thread per spiked neuron — fine for ~2-5% firing rate.
/// `weight_scale` is an additional multiplier (e.g. 0.005 for HH units).
extern "C" __global__ void csr_propagate_kernel(
    const int* spikes,
    const int* row_ptr, const int* col_indices, const float* values,
    const float* nt_types,
    float* I_buf,
    int n_neurons,
    float weight_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;
    if (!spikes[idx]) return;

    int start = row_ptr[idx];
    int end = row_ptr[idx + 1];
    float mult = nt_types[idx] * weight_scale;

    for (int j = start; j < end; j++) {
        int target = col_indices[j];
        if (target >= 0 && target < n_neurons) {
            atomicAdd(&I_buf[target], values[j] * mult);
        }
    }
}

// ─── CSR STDP kernel ───────────────────────────────────────
//
// Per-synapse STDP: for each pre-neuron, update all its outgoing
// synapses using pre-trace (LTP) and post-trace (LTD).
//
// w_ij += lr_ltp * trace_pre[i] - lr_ltd * trace_post[j]
//
// One block per pre-neuron. Each block iterates over the pre-neuron's
// outgoing synapses. Average 38 synapses per neuron for FlyWire.

extern "C" __global__ void csr_stdp_kernel(
    float* weights,
    const int* row_ptr, const int* col_indices,
    const float* trace,  // per-neuron trace buffer (d_trace), shared pre/post
    float lr_ltp, float lr_ltd, float w_min, float w_max,
    int n_neurons
) {
    int pre = blockIdx.x;
    if (pre >= n_neurons) return;

    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    float tp = trace[pre];  // pre-synaptic trace

    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        int post = col_indices[k];
        if (post < 0 || post >= n_neurons) continue;
        float tn = trace[post];  // post-synaptic trace (same buffer)
        float w = weights[k];
        w += lr_ltp * tp - lr_ltd * tn;
        weights[k] = fminf(w_max, fmaxf(w_min, w));
    }
}

extern "C" int gpu_connectome_stdp_async(void* vctx,
    float lr_ltp, float lr_ltd, float w_min, float w_max)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 128;
    csr_stdp_kernel<<<ctx->n_neurons, bs, 0, ctx->stream>>>(
        ctx->d_weights,
        ctx->d_row_ptr, ctx->d_col_indices,
        ctx->d_trace,  // use d_trace (updated by izhikevich step)
        lr_ltp, lr_ltd, w_min, w_max,
        ctx->n_neurons);
    return (int)cudaGetLastError();
}

// ─── Spike count kernel ──────────────────────────────────

extern "C" __global__ void count_spikes_kernel(
    const int* spikes, int* total, int n
) {
    __shared__ int block_sum[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = (tid < n && spikes[tid]) ? 1 : 0;
    block_sum[threadIdx.x] = local;
    __syncthreads();

    // Simple reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(total, block_sum[0]);
    }
}

/// Separate spike counting kernel launch.
/// Counts spikes from d_spikes into d_total_spikes via block reduction.
/// Called AFTER izhikevich_step to count the spikes just produced.
extern "C" int gpu_count_spikes_async(void* vctx, int* d_total_spikes) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (d_total_spikes == NULL) return 1;
    int bs = 256, n = ctx->n_neurons;
    count_spikes_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_spikes, d_total_spikes, n);
    return (int)cudaGetLastError();
}

/// CSR propagate using plastic weights (d_weights).
/// Same as gpu_connectome_csr_async but reads from d_weights
/// instead of d_csr_values, so STDP updates take effect.
extern "C" int gpu_connectome_csr_plastic_async(void* vctx)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    int gs_n = gs(n, bs);

    reset_I_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_I, ctx->d_external_I, n);

    csr_propagate_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_spikes,
        ctx->d_row_ptr, ctx->d_col_indices, ctx->d_weights,
        ctx->d_nt_types,
        ctx->d_I,
        n,
        1.0f);

    return (int)cudaGetLastError();
}

/// Combined connectome CSR step: reset I to external_I, then CSR propagate.
/// Uses d_csr_values (static weights).
extern "C" int gpu_connectome_csr_async(void* vctx)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    int gs_n = gs(n, bs);

    // 1. Reset I to external baseline
    reset_I_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_I, ctx->d_external_I, n);

    // 2. CSR propagate spikes → I_buf (atomicAdd for safety)
    csr_propagate_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_spikes,
        ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values,
        ctx->d_nt_types,
        ctx->d_I,
        n,
        1.0f);

    return (int)cudaGetLastError();
}

// ─── HH batch helper kernels ─────────────────────────────

/// Vector add: d_I[i] = d_external_I[i] + d_I_syn[i]
extern "C" __global__ void hh_prepare_step_kernel(
    float* d_I, const float* d_external_I, const float* d_I_syn, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_I[idx] = d_external_I[idx] + d_I_syn[idx];
}

/// Scalar multiply: d_I_syn[i] *= decay
extern "C" __global__ void hh_decay_I_syn_kernel(
    float* d_I_syn, float decay, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_I_syn[idx] *= decay;
}

/// CSR propagate onto I_syn (no reset) — for HH batch step.
/// Reuses csr_propagate_kernel directly with any target buffer.
static inline int csr_propagate_onto(
    GpuCtx* ctx, float* target_buf, float weight_scale
) {
    int bs = 256, n = ctx->n_neurons;
    csr_propagate_kernel<<<gs(n, bs), bs, 0, ctx->stream>>>(
        ctx->d_spikes,
        ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values,
        ctx->d_nt_types,
        target_buf,
        n,
        weight_scale);
    return (int)cudaGetLastError();
}

/// Run n_steps of HH with GPU-side CSR propagation and I_syn decay.
///
/// All per-step work stays on GPU — no H2D/D2H transfers per step.
/// Single D2H at end for spike buffer + final v.
///
/// Args:
///   external_I — host buffer [n_neurons], uploaded to d_external_I once
///   n_external — length of external_I
///   dev_spike_buf — pre-allocated device buffer [n_steps × n_neurons] ints
///   host_spike_buf — output host buffer [n_steps × n_neurons] ints
///   host_v — output host buffer [n_neurons] floats (final v)
///   host_I_syn — output host buffer [n_neurons] floats (final I_syn)
///   n_steps — number of steps
///   n_neurons — number of neurons
///   dt — HH timestep (ms)
///   tau_ref — refractory period (ms)
///   tau_syn — synaptic decay time constant (ms)
///   weight_scale — additional multiplier for CSR weights (e.g. 0.005 for HH)
///
extern "C" int gpu_context_batch_hh_steps(
    void* vctx,
    const float* external_I,
    int n_external,
    int* host_spike_buf,
    float* host_v,
    float* host_I_syn,
    int n_steps,
    int n_neurons,
    float dt,
    float tau_ref,
    float tau_syn,
    float weight_scale)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    int n = ctx->n_neurons;

    // Allocate d_hh_I_syn on first use
    if (ctx->d_hh_I_syn == NULL) {
        e = cudaMalloc(&ctx->d_hh_I_syn, (size_t)n * sizeof(float));
        if (e) return (int)e;
        e = cudaMemsetAsync(ctx->d_hh_I_syn, 0, (size_t)n * sizeof(float), ctx->stream);
        if (e) return (int)e;
    }

    // Allocate/reuse d_hh_spike_buf
    int spike_buf_len = n_steps * n;
    if (ctx->d_hh_spike_buf == NULL || ctx->n_hh_spike_buf_capacity < spike_buf_len) {
        if (ctx->d_hh_spike_buf) cudaFree(ctx->d_hh_spike_buf);
        e = cudaMalloc(&ctx->d_hh_spike_buf, (size_t)spike_buf_len * sizeof(int));
        if (e) return (int)e;
        ctx->n_hh_spike_buf_capacity = spike_buf_len;
    }

    // Upload external_I to device
    e = cudaMemcpyAsync(ctx->d_external_I, external_I,
        n_external * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;

    float decay = 1.0f - dt / tau_syn;
    if (decay < 0.0f) decay = 0.0f;

    int bs = 256;
    int gs_n = gs(n, bs);

    for (int step = 0; step < n_steps; step++) {
        // 1. Prepare I = external_I + I_syn
        hh_prepare_step_kernel<<<gs_n, bs, 0, ctx->stream>>>(
            ctx->d_I, ctx->d_external_I, ctx->d_hh_I_syn, n);

        // 2. HH step — produces spikes in d_spikes
        {
            int r = gpu_hh_step_async(vctx, dt, tau_ref);
            if (r) return r;
        }

        // 3. D2D: d_spikes → d_hh_spike_buf[step * n]
        e = cudaMemcpyAsync(ctx->d_hh_spike_buf + step * n, ctx->d_spikes,
            n * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream);
        if (e) return (int)e;

        // 4. CSR propagate spikes onto d_hh_I_syn (no reset)
        {
            int r = csr_propagate_onto(ctx, ctx->d_hh_I_syn, weight_scale);
            if (r) return r;
        }

        // 5. Decay I_syn
        if (decay > 0.0f) {
            hh_decay_I_syn_kernel<<<gs_n, bs, 0, ctx->stream>>>(
                ctx->d_hh_I_syn, decay, n);
        } else {
            cudaMemsetAsync(ctx->d_hh_I_syn, 0, (size_t)n * sizeof(float), ctx->stream);
        }
    }

    // Final D2H: spike buffer + v + I_syn
    e = cudaMemcpyAsync(host_spike_buf, ctx->d_hh_spike_buf,
        spike_buf_len * sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaMemcpyAsync(host_v, ctx->d_v,
        n_neurons * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaMemcpyAsync(host_I_syn, ctx->d_hh_I_syn,
        n_neurons * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaStreamSynchronize(ctx->stream);
    return (int)e;
}
