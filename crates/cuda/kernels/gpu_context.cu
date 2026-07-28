/**
 * GPU Context — persistent device memory + CUDA stream + graph capture.
 *
 * Keeps ALL neural state on GPU between timesteps.
 * Uses CUDA stream for async H2D/kernel/D2H pipelining.
 * Captures simulation step as CUDA Graph for zero-overhead replays.
 *
 * Async kernel launch functions live in their respective .cu files
 * (izhikevich.cu, stdp.cu, kuramoto.cu, gates.cu, fused.cu).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "gpu_context.h"

// ─── Forward declaration of destroy (used in fail label) ──

extern "C" void gpu_context_destroy(void* vctx);

// ─── ψ-wave async wrappers (defined in psi_wave.cu) ──────

extern "C" int gpu_gco_hamiltonian_async(void* vctx);
extern "C" int gpu_gco_step_async(void* vctx, float dt, float hbar);

// ─── LIF + CSR + Izhikevich + HH async wrappers ──

extern "C" int gpu_lif_step_async(void* vctx);
extern "C" int gpu_izhikevich_step_async(void* vctx, float dt, float trace_decay);
extern "C" int gpu_hh_step_async(void* vctx, float dt, float tau_ref);
extern "C" int gpu_hh_upload_state(void* vctx,
    const float* v, const float* gates, const float* Ii,
    const float* g_leak, const float* E_leak, const float* C_m,
    int n, int n_gates);
extern "C" int gpu_hh_upload_channels(void* vctx,
    const float* E_rev_arr, const float* g_max_arr,
    const unsigned int* powers, const int* n_gates_per_ch,
    int n_gates, int n_channels);
extern "C" int gpu_connectome_csr_async(void* vctx);

// ─── Get d_noise device pointer (allocates if NULL) ─────

extern "C" unsigned int* gpu_context_get_d_noise(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (!ctx->d_noise) {
        cudaError_t e = cudaMalloc(&ctx->d_noise, (size_t)ctx->n_neurons * sizeof(unsigned int));
        if (e) return NULL;
    }
    return ctx->d_noise;
}

// ─── Create GPU context (minimal — lazy alloc in uploads/steps) ───

extern "C" void* gpu_context_create(int n_neurons, int n_synapses, int n_decoders) {
    GpuCtx* ctx = (GpuCtx*)calloc(1, sizeof(GpuCtx));
    if (!ctx) return NULL;
    ctx->n_neurons = n_neurons;
    ctx->n_synapses = n_synapses;
    ctx->n_decoders = n_decoders;
    ctx->n_t3 = n_neurons;
    ctx->n_hh_spike_buf_capacity = 0;
    ctx->n_hh_gates = 0;
    ctx->n_hh_channels = 0;
    ctx->n_csr_synapses = n_synapses;

    cudaError_t e = cudaStreamCreate(&ctx->stream);
    if (e) { fprintf(stderr,"stream: %s\n",cudaGetErrorString(e)); free(ctx); return NULL; }

    ctx->psi_graph_captured = 0;
    ctx->psi_graph_dt = 0;
    ctx->psi_graph_hbar = 0;

    return ctx;
}

// ─── Destroy GPU context ─────────────────────────────────

extern "C" void gpu_context_destroy(void* vctx) {
    if (!vctx) return;
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (ctx->graph_captured) { cudaGraphExecDestroy(ctx->graph_exec); cudaGraphDestroy(ctx->graph); }
    if (ctx->psi_graph_captured) { cudaGraphExecDestroy(ctx->psi_graph_exec); cudaGraphDestroy(ctx->psi_graph); }
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    #define F(p) do { if(ctx->p) cudaFree(ctx->p); } while(0)
    F(d_v);F(d_u);F(d_trace);F(d_spikes);F(d_I);
    F(d_a);F(d_b);F(d_c);F(d_d);F(d_lif_params);
    F(d_weights);F(d_traces_pre);F(d_traces_post);
    F(d_phases);F(d_omegas);F(d_coupling);F(d_cos_buf);F(d_sin_buf);F(d_phases_prev);
    F(d_gate_out);F(d_decoder_w);F(d_decoder_bias);F(d_decoder_cov);
    F(d_amplitudes);F(d_drives);F(d_refractory);F(d_t3_prev_amps);    F(d_row_ptr); F(d_col_indices); F(d_csr_values); F(d_nt_types); F(d_external_I);
    F(d_psi_v); F(d_psi_u); F(d_psi_h_real); F(d_psi_h_imag); F(d_psi_drive);
    F(d_aber_real); F(d_aber_imag); F(d_aber_collapsed); F(d_noise);
    F(d_hh_gates); F(d_hh_g_leak); F(d_hh_E_leak); F(d_hh_C_m);
    F(d_hh_E_rev_arr); F(d_hh_g_max_arr);
    F(d_hh_powers); F(d_hh_n_gates_per_ch);
    F(d_hh_I_syn); F(d_hh_spike_buf);
    #undef F
    free(ctx);
}

// ─── Upload I (input current) to GPU ────────────────────

extern "C" int gpu_context_upload_I(void* vctx, const float* I, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_I, ctx->n_neurons*sizeof(float));
    cudaError_t e = cudaMemcpyAsync(ctx->d_I, I, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── Upload v only (LIF path — avoid dummy u/a/b/c/d uploads) ──

extern "C" int gpu_context_upload_v_only(void* vctx, const float* v, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_v, ctx->n_neurons*sizeof(float));
    cudaError_t e = cudaMemcpyAsync(ctx->d_v, v, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ═══════════════════════════════════════════════════════════
//  LUT-HH: upload state (direct computation, no LUTs)
// ═══════════════════════════════════════════════════════════

/// Upload HH per-neuron params, gate state, and channel info.
/// Allocates device arrays on first call, reuses on subsequent calls.
///
/// Args:
///   v — membrane potentials [n] (uploaded to d_v)
///   gates — gate variables [n * n_gates] row-major (uploaded to d_hh_gates)
///   Ii — input currents [n] (uploaded to d_I)
///   g_leak — leak conductances [n]
///   E_leak — leak reversals [n]
///   C_m — membrane capacitances [n]
///   n — number of neurons
///   n_gates — total gates across all channels (e.g. 3 for NaV+KV)
///   n_channels — number of channels (e.g. 2 for NaV+KV)
///   E_rev_arr — [n_channels] reversal potentials
///   g_max_arr — [n_channels] max conductances
///   powers — [n_gates] gate powers (exponents)
///   n_gates_per_ch — [n_channels] gates per channel

extern "C" int gpu_hh_upload_state(void* vctx,
    const float* v, const float* gates, const float* Ii,
    const float* g_leak, const float* E_leak, const float* C_m,
    int n, int n_gates)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;

    // Upload v and I (reuse standard d_v, d_I)
    ENSURE_ALLOC(d_v, ctx->n_neurons*sizeof(float));
    ENSURE_ALLOC(d_I, ctx->n_neurons*sizeof(float));
    e = cudaMemcpyAsync(ctx->d_v, v, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_I, Ii, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;

    // Allocate + upload gates (first call allocates)
    if (ctx->d_hh_gates == NULL || ctx->n_hh_gates != n_gates) {
        if (ctx->d_hh_gates) cudaFree(ctx->d_hh_gates);
        e = cudaMalloc(&ctx->d_hh_gates, (size_t)n * n_gates * sizeof(float));
        if (e) goto fail;
        ctx->n_hh_gates = n_gates;
    }
    // Convert AOS → SOA: host gates[n * n_gates] → tmp[n_gates * n]
    // SOA layout: gates[gate_idx * n + neuron_idx] for coalesced GPU access.
    // Use a separate scope to avoid goto-bypass-init error (C++ rule).
    {
        float* h_tmp = (float*)malloc((size_t)n * n_gates * sizeof(float));
        if (!h_tmp) { e = (cudaError_t)1; goto fail; }
        for (int ni = 0; ni < n; ni++) {
            for (int gi = 0; gi < n_gates; gi++) {
                h_tmp[gi * n + ni] = gates[ni * n_gates + gi];
            }
        }
        e = cudaMemcpyAsync(ctx->d_hh_gates, h_tmp, (size_t)n * n_gates * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream);
        free(h_tmp);
        if (e) goto fail;
    }

    // Allocate + upload per-neuron params
    #define ALLOC_AND_UPLOAD(field) do { \
        if (ctx->field == NULL) { \
            e = cudaMalloc(&ctx->field, (size_t)n * sizeof(float)); \
            if (e) goto fail; \
        } \
    } while(0)

    ALLOC_AND_UPLOAD(d_hh_g_leak);
    ALLOC_AND_UPLOAD(d_hh_E_leak);
    ALLOC_AND_UPLOAD(d_hh_C_m);

    e = cudaMemcpyAsync(ctx->d_hh_g_leak, g_leak, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_hh_E_leak, E_leak, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_hh_C_m, C_m, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;

    #undef ALLOC_AND_UPLOAD

    return (int)cudaStreamSynchronize(ctx->stream);

fail:
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

/// Upload HH channel info (E_rev, g_max, powers, n_gates_per_ch).
/// Must be called after gpu_hh_upload_state().
///
/// On GPU, HH uses direct computation with __expf() — no LUTs needed.
/// These arrays provide channel-specific parameters to the kernel.

extern "C" int gpu_hh_upload_channels(void* vctx,
    const float* E_rev_arr, const float* g_max_arr,
    const unsigned int* powers, const int* n_gates_per_ch,
    int n_gates, int n_channels)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;

    #define ALLOC_UPLOAD_CH(field, c_type, count) do { \
        if (ctx->field) cudaFree(ctx->field); \
        e = cudaMalloc(&ctx->field, (size_t)(count) * sizeof(c_type)); \
        if (e) goto fail; \
    } while(0)

    ALLOC_UPLOAD_CH(d_hh_E_rev_arr, float, n_channels);
    ALLOC_UPLOAD_CH(d_hh_g_max_arr, float, n_channels);
    ALLOC_UPLOAD_CH(d_hh_powers, unsigned int, n_gates);
    ALLOC_UPLOAD_CH(d_hh_n_gates_per_ch, int, n_channels);

    e = cudaMemcpyAsync(ctx->d_hh_E_rev_arr, E_rev_arr, n_channels*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_hh_g_max_arr, g_max_arr, n_channels*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_hh_powers, powers, n_gates*sizeof(unsigned int),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;
    e = cudaMemcpyAsync(ctx->d_hh_n_gates_per_ch, n_gates_per_ch, n_channels*sizeof(int),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) goto fail;

    ctx->n_hh_gates = n_gates;
    ctx->n_hh_channels = n_channels;

    #undef ALLOC_UPLOAD_CH

    return (int)cudaStreamSynchronize(ctx->stream);

fail:
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── Upload initial state ────────────────────────────────

extern "C" int gpu_context_upload_neurons(void* vctx,
    const float* v, const float* u, const float* a, const float* b,
    const float* c, const float* d, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int n_neurons = ctx->n_neurons;
    ENSURE_ALLOC(d_v, n_neurons*sizeof(float));
    ENSURE_ALLOC(d_u, n_neurons*sizeof(float));
    ENSURE_ALLOC(d_a, n_neurons*sizeof(float));
    ENSURE_ALLOC(d_b, n_neurons*sizeof(float));
    ENSURE_ALLOC(d_c, n_neurons*sizeof(float));
    ENSURE_ALLOC(d_d, n_neurons*sizeof(float));
    int err = 0;
    err |= (int)cudaMemcpyAsync(ctx->d_v, v, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_u, u, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_a, a, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_b, b, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_c, c, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_d, d, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

extern "C" int gpu_context_upload_kuramoto(void* vctx,
    const float* phases, const float* omegas, const float* coupling, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_phases, nn*sizeof(float));
    ENSURE_ALLOC(d_omegas, nn*sizeof(float));
    ENSURE_ALLOC(d_coupling, nn*sizeof(float));
    int err = 0;
    err |= (int)cudaMemcpyAsync(ctx->d_phases, phases, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_omegas, omegas, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_coupling, coupling, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

// ─── Upload CSR connectome arrays (row_ptr, col_indices, values) ──
// Note: values are stored in d_I (reused as weight buffer)

/// Initialize plastic weights: copy d_csr_values → d_weights.
/// Must be called after upload_connectome() to start STDP from connectome weights.
extern "C" int gpu_context_init_plastic_weights(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_weights, ctx->d_csr_values,
        ctx->n_csr_synapses * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_upload_connectome(void* vctx,
    const int* row_ptr, const int* col_indices, const float* values,
    const float* nt_types, const float* external_I,
    int n_neurons, int n_synapses)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    int err = 0;

    // Ensure row_ptr + nt_types + external_I allocated
    ENSURE_ALLOC(d_row_ptr, (ctx->n_neurons+1)*sizeof(int));
    ENSURE_ALLOC(d_nt_types, ctx->n_neurons*sizeof(float));
    ENSURE_ALLOC(d_external_I, ctx->n_neurons*sizeof(float));

    // Reallocate CSR arrays if synapse count changed or first time
    if (n_synapses > ctx->n_csr_synapses || !ctx->d_col_indices) {
        if (ctx->d_col_indices) cudaFree(ctx->d_col_indices);
        if (ctx->d_csr_values) cudaFree(ctx->d_csr_values);
        if (ctx->d_weights) cudaFree(ctx->d_weights);
        e = cudaMalloc(&ctx->d_col_indices, (size_t)n_synapses * sizeof(int));
        if (e) return (int)e;
        e = cudaMalloc(&ctx->d_csr_values, (size_t)n_synapses * sizeof(float));
        if (e) return (int)e;
        e = cudaMalloc(&ctx->d_weights, (size_t)n_synapses * sizeof(float));
        if (e) return (int)e;
        ctx->n_csr_synapses = n_synapses;
    }

    err |= (int)cudaMemcpyAsync(ctx->d_row_ptr, row_ptr, (n_neurons+1)*sizeof(int),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_col_indices, col_indices, n_synapses*sizeof(int),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_csr_values, values, n_synapses*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_nt_types, nt_types, n_neurons*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_external_I, external_I, n_neurons*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

// ─── Download / Upload results ───────────────────────────

extern "C" int gpu_context_download_spikes(void* vctx, int* host_spikes, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_spikes, ctx->d_spikes, n*sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_download_v(void* vctx, float* host_v, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_v, ctx->d_v, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_download_phases(void* vctx, float* host_phases, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_phases, ctx->n_neurons*sizeof(float));
    cudaError_t e = cudaMemcpyAsync(host_phases, ctx->d_phases, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_download_u(void* vctx, float* host_u, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_u, ctx->d_u, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_download_adex_w(void* vctx, float* host_w, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_w, ctx->d_adex_w, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_upload_adex_w(void* vctx, const float* host_w, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_adex_w, ctx->n_neurons*sizeof(float));
    cudaError_t e = cudaMemcpyAsync(ctx->d_adex_w, host_w, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_download_weights(void* vctx, float* host_w, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_w, ctx->d_weights, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int cuda_upload_csr_values(void* vctx, const float* values, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_csr_values, ctx->n_csr_synapses*sizeof(float));
    cudaError_t e = cudaMemcpyAsync(ctx->d_csr_values, values, n*sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int cuda_download_csr_values(void* vctx, float* values, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(values, ctx->d_csr_values, n*sizeof(float),
        cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── D2D copy: phases → trajectory buffer (after update step) ─────

extern "C" int gpu_context_copy_phases_to_traj_async(void* vctx, float* d_traj, int step) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    return (int)cudaMemcpyAsync(d_traj + step * ctx->n_neurons, ctx->d_phases,
        ctx->n_neurons * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream);
}

// ─── D2D copy: coupling → trajectory buffer (after CSR coupling, before update) ──

extern "C" int gpu_context_copy_coupling_to_traj_async(void* vctx, float* d_traj, int step) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_coupling, ctx->n_neurons*sizeof(float));
    return (int)cudaMemcpyAsync(d_traj + step * ctx->n_neurons, ctx->d_coupling,
        ctx->n_neurons * sizeof(float), cudaMemcpyDeviceToDevice, ctx->stream);
}

extern "C" int gpu_context_sync(void* vctx) {
    return (int)cudaStreamSynchronize(((GpuCtx*)vctx)->stream);
}

// ─── Pre-allocate all arrays (for graph capture path) ─────
// CUDA graph capture forbids cudaMalloc, so we must ensure
// all arrays exist before calling gpu_graph_begin_capture.
extern "C" int gpu_context_ensure_all(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int n = ctx->n_neurons;
    int ns = ctx->n_synapses;
    int nd = ctx->n_decoders;
    ENSURE_ALLOC(d_v, n*sizeof(float));
    ENSURE_ALLOC(d_u, n*sizeof(float));
    ENSURE_ALLOC(d_trace, n*sizeof(float));
    ENSURE_ALLOC(d_spikes, n*sizeof(int));
    ENSURE_ALLOC(d_I, n*sizeof(float));
    ENSURE_ALLOC(d_a, n*sizeof(float));
    ENSURE_ALLOC(d_b, n*sizeof(float));
    ENSURE_ALLOC(d_c, n*sizeof(float));
    ENSURE_ALLOC(d_d, n*sizeof(float));
    ENSURE_ALLOC(d_lif_params, (1+6*n)*sizeof(float));
    ENSURE_ALLOC(d_weights, ns*sizeof(float));
    ENSURE_ALLOC(d_traces_pre, n*sizeof(float));
    ENSURE_ALLOC(d_traces_post, n*sizeof(float));
    ENSURE_ALLOC(d_phases, n*sizeof(float));
    ENSURE_ALLOC(d_omegas, n*sizeof(float));
    ENSURE_ALLOC(d_coupling, n*sizeof(float));
    ENSURE_ALLOC(d_cos_buf, n*sizeof(float));
    ENSURE_ALLOC(d_sin_buf, n*sizeof(float));
    ENSURE_ALLOC(d_phases_prev, n*sizeof(float));
    ENSURE_ALLOC(d_gate_out, n*sizeof(float));
    ENSURE_ALLOC(d_decoder_w, nd*sizeof(float));
    ENSURE_ALLOC(d_decoder_bias, sizeof(float));
    ENSURE_ALLOC(d_decoder_cov, nd*sizeof(float));
    ENSURE_ALLOC(d_amplitudes, n*sizeof(float));
    ENSURE_ALLOC(d_drives, n*sizeof(float));
    ENSURE_ALLOC(d_refractory, n*sizeof(float));
    ENSURE_ALLOC(d_t3_prev_amps, n*sizeof(float));
    ENSURE_ALLOC(d_adex_w, n*sizeof(float));
    ENSURE_ALLOC(d_noise, n*sizeof(unsigned int));
    ENSURE_ALLOC(d_row_ptr, (n+1)*sizeof(int));
    ENSURE_ALLOC(d_col_indices, ns*sizeof(int));
    ENSURE_ALLOC(d_csr_values, ns*sizeof(float));
    ENSURE_ALLOC(d_nt_types, n*sizeof(float));
    ENSURE_ALLOC(d_external_I, n*sizeof(float));
    ENSURE_ALLOC(d_psi_v, n*sizeof(float));
    ENSURE_ALLOC(d_psi_u, n*sizeof(float));
    ENSURE_ALLOC(d_psi_h_real, n*sizeof(float));
    ENSURE_ALLOC(d_psi_h_imag, n*sizeof(float));
    ENSURE_ALLOC(d_psi_drive, n*sizeof(float));
    ENSURE_ALLOC(d_aber_real, n*sizeof(float));
    ENSURE_ALLOC(d_aber_imag, n*sizeof(float));
    ENSURE_ALLOC(d_aber_collapsed, n*sizeof(int));
    return (int)cudaStreamSynchronize(ctx->stream);
}

// ─── Batch LIF steps with GPU spike accumulation ──────────
//
// Runs `n_steps` of LIF + CSR propagate on GPU.
// Accumulates spikes in `dev_spike_buf` via per-step D2D copies from d_spikes.
// Downloads entire spike buffer + final v in ONE D2H call at the end.
// Also updates d_external_I (per-step baseline) — fixes stale external_I bug.
//
// Args:
//   external_I — current external input (updates both d_external_I and d_I)
//   n_external — length of external_I array
//   dev_spike_buf — pre-allocated device buffer [n_steps × n_neurons] ints
//   host_spike_buf — output host buffer [n_steps × n_neurons] ints
//   host_v — output host buffer [n_neurons] floats (final v)
//   n_steps — number of LIF steps to run
//   n_neurons — number of neurons (must match context)

extern "C" int gpu_context_batch_lif_steps(
    void* vctx,
    const float* external_I,
    int n_external,
    int* dev_spike_buf,
    int* host_spike_buf,
    float* host_v,
    int n_steps,
    int n_neurons)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    int n = ctx->n_neurons;

    ENSURE_ALLOC(d_external_I, n*sizeof(float));
    ENSURE_ALLOC(d_I, n*sizeof(float));

    // 1. Update external baseline (d_external_I) + working I (d_I)
    //    This fixes the stale external_I bug — baseline is now current.
    e = cudaMemcpyAsync(ctx->d_external_I, external_I, n_external * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(ctx->d_I, external_I, n_external * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;

    // 2. Run all LIF steps — accumulate spikes on GPU, NO per-step D2H
    for (int step = 0; step < n_steps; step++) {
        // LIF step — produces spikes in d_spikes
        int r = gpu_lif_step_async(vctx);
        if (r) return r;

        // D2D copy: d_spikes → position in dev_spike_buf (no host transfer!)
        e = cudaMemcpyAsync(dev_spike_buf + step * n, ctx->d_spikes,
            n * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream);
        if (e) return (int)e;

        // CSR propagate — resets I to d_external_I (just updated), then adds postsynaptic currents
        r = gpu_connectome_csr_async(vctx);
        if (r) return r;
    }

    // 3. Single D2H download: entire spike buffer + final v
    e = cudaMemcpyAsync(host_spike_buf, dev_spike_buf,
        n_steps * n * sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaMemcpyAsync(host_v, ctx->d_v, n_neurons * sizeof(float),
        cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── Batch Izhikevich steps with GPU spike accumulation ──────
//
// Same pattern as batch_lif_steps but uses izhikevich_step_async
// and also downloads u (recovery variable) at the end.
//
// Args:
//   external_I — current external input (updates both d_external_I and d_I)
//   n_external — length of external_I array
//   dt — timestep in ms
//   trace_decay — trace decay factor for STDP
//   dev_spike_buf — pre-allocated device buffer [n_steps × n_neurons] ints
//   host_spike_buf — output host buffer [n_steps × n_neurons] ints
//   host_v — output host buffer [n_neurons] floats (final v)
//   host_u — output host buffer [n_neurons] floats (final u)
//   n_steps — number of Izhikevich steps to run
//   n_neurons — number of neurons (must match context)

extern "C" int gpu_context_batch_izhikevich_steps(
    void* vctx,
    const float* external_I,
    int n_external,
    float dt,
    float trace_decay,
    int* dev_spike_buf,
    int* host_spike_buf,
    float* host_v,
    float* host_u,
    int n_steps,
    int n_neurons)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    int n = ctx->n_neurons;

    ENSURE_ALLOC(d_external_I, n*sizeof(float));
    ENSURE_ALLOC(d_I, n*sizeof(float));

    // 1. Update external baseline + working I
    e = cudaMemcpyAsync(ctx->d_external_I, external_I, n_external * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(ctx->d_I, external_I, n_external * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;

    // 2. Run all Izhikevich steps — accumulate spikes on GPU
    for (int step = 0; step < n_steps; step++) {
        int r = gpu_izhikevich_step_async(vctx, dt, trace_decay);
        if (r) return r;

        // D2D copy: d_spikes → dev_spike_buf
        e = cudaMemcpyAsync(dev_spike_buf + step * n, ctx->d_spikes,
            n * sizeof(int), cudaMemcpyDeviceToDevice, ctx->stream);
        if (e) return (int)e;

        // CSR propagate
        r = gpu_connectome_csr_async(vctx);
        if (r) return r;
    }

    // 3. Single D2H: spike buffer + final v + final u
    e = cudaMemcpyAsync(host_spike_buf, dev_spike_buf,
        n_steps * n * sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaMemcpyAsync(host_v, ctx->d_v, n_neurons * sizeof(float),
        cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaMemcpyAsync(host_u, ctx->d_u, n_neurons * sizeof(float),
        cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;

    e = cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── CUDA Graph infrastructure ──────────────────────────

extern "C" int gpu_graph_begin_capture(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (ctx->graph_captured) {
        cudaGraphExecDestroy(ctx->graph_exec);
        cudaGraphDestroy(ctx->graph);
        ctx->graph_captured = 0;
    }
    return (int)cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal);
}

extern "C" int gpu_graph_end_capture(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaStreamEndCapture(ctx->stream, &ctx->graph);
    if (e) return (int)e;
    e = cudaGraphInstantiate(&ctx->graph_exec, ctx->graph, NULL, NULL, 0);
    if (e) { cudaGraphDestroy(ctx->graph); return (int)e; }
    ctx->graph_captured = 1;
    return 0;
}

extern "C" int gpu_graph_replay(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (!ctx->graph_captured) return 1;
    return (int)cudaGraphLaunch(ctx->graph_exec, ctx->stream);
}

extern "C" int gpu_graph_is_captured(void* vctx) {
    return ((GpuCtx*)vctx)->graph_captured ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════
//  ψ-wave CUDA Graph — runs combined_step without host launch overhead
// ═══════════════════════════════════════════════════════════
//
// Usage: upload ψ-wave state + drive, then capture once with given dt/hbar.
//        Replay for each subsequent step.  If dt/hbar change, the graph is
//        automatically invalidated and re-captured on the next replay.

extern "C" int gpu_psi_graph_replay(void* vctx, float dt, float hbar) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;

    // Invalidate if params changed
    if (ctx->psi_graph_captured &&
        (ctx->psi_graph_dt != dt || ctx->psi_graph_hbar != hbar))
    {
        cudaGraphExecDestroy(ctx->psi_graph_exec);
        cudaGraphDestroy(ctx->psi_graph);
        ctx->psi_graph_captured = 0;
    }

    if (!ctx->psi_graph_captured) {
        // Capture: hamiltonian (reset + CSR matmul) + step_normalize
        e = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeThreadLocal);
        if (e) return (int)e;

        // Use existing async wrappers — kernel launches are captured on stream
        gpu_gco_hamiltonian_async(vctx);
        gpu_gco_step_async(vctx, dt, hbar);

        e = cudaStreamEndCapture(ctx->stream, &ctx->psi_graph);
        if (e) return (int)e;

        e = cudaGraphInstantiate(&ctx->psi_graph_exec, ctx->psi_graph, NULL, NULL, 0);
        if (e) { cudaGraphDestroy(ctx->psi_graph); return (int)e; }

        ctx->psi_graph_captured = 1;
        ctx->psi_graph_dt = dt;
        ctx->psi_graph_hbar = hbar;

        // First step: launch graph (capture recorded but didn't execute)
        e = cudaGraphLaunch(ctx->psi_graph_exec, ctx->stream);
        return (int)e;
    }

    // Subsequent steps: replay pre-captured graph
    return (int)cudaGraphLaunch(ctx->psi_graph_exec, ctx->stream);
}

extern "C" int gpu_psi_graph_invalidate(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    if (ctx->psi_graph_captured) {
        cudaGraphExecDestroy(ctx->psi_graph_exec);
        cudaGraphDestroy(ctx->psi_graph);
        ctx->psi_graph_captured = 0;
    }
    return 0;
}
