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

extern "C" int gpu_psi_hamiltonian_async(void* vctx);
extern "C" int gpu_psi_step_async(void* vctx, float dt, float hbar);

// ─── Create GPU context ──────────────────────────────────

extern "C" void* gpu_context_create(int n_neurons, int n_synapses, int n_decoders) {
    GpuCtx* ctx = (GpuCtx*)calloc(1, sizeof(GpuCtx));
    if (!ctx) return NULL;
    ctx->n_neurons = n_neurons;
    ctx->n_synapses = n_synapses;
    ctx->n_decoders = n_decoders;
    ctx->n_t3 = n_neurons; // T3 uses same count by default

    cudaError_t e = cudaStreamCreate(&ctx->stream);
    if (e) { fprintf(stderr,"stream: %s\n",cudaGetErrorString(e)); free(ctx); return NULL; }

    #define ALLOC(n,s) do { \
        e = cudaMalloc(&ctx->n, (size_t)(s)); \
        if (e) { fprintf(stderr,#n": %s\n",cudaGetErrorString(e)); goto fail; } \
    } while(0)

    ALLOC(d_v, n_neurons*sizeof(float)); ALLOC(d_u, n_neurons*sizeof(float));
    ALLOC(d_trace, n_neurons*sizeof(float)); ALLOC(d_spikes, n_neurons*sizeof(int));
    ALLOC(d_I, n_neurons*sizeof(float)); ALLOC(d_a, n_neurons*sizeof(float));
    ALLOC(d_b, n_neurons*sizeof(float)); ALLOC(d_c, n_neurons*sizeof(float));
    ALLOC(d_d, n_neurons*sizeof(float));
    ALLOC(d_lif_params, (1+6*n_neurons)*sizeof(float));
    ALLOC(d_weights, n_synapses*sizeof(float));
    ALLOC(d_traces_pre, n_neurons*sizeof(float)); ALLOC(d_traces_post, n_neurons*sizeof(float));
    ALLOC(d_phases, n_neurons*sizeof(float)); ALLOC(d_omegas, n_neurons*sizeof(float));
    ALLOC(d_coupling, n_neurons*sizeof(float)); ALLOC(d_cos_buf, n_neurons*sizeof(float));
    ALLOC(d_sin_buf, n_neurons*sizeof(float)); ALLOC(d_phases_prev, n_neurons*sizeof(float));
    ALLOC(d_gate_out, n_neurons*sizeof(float));
    ALLOC(d_decoder_w, n_decoders*sizeof(float)); ALLOC(d_decoder_bias, sizeof(float));
    ALLOC(d_decoder_cov, n_decoders*sizeof(float));
    ALLOC(d_echo_buf, n_neurons*DELAY_LEN*sizeof(float));
    // T3 arrays
    ALLOC(d_amplitudes, n_neurons*sizeof(float));
    ALLOC(d_drives, n_neurons*sizeof(float));
    ALLOC(d_refractory, n_neurons*sizeof(float));
    ALLOC(d_t3_prev_amps, n_neurons*sizeof(float));
    ALLOC(d_adex_w, n_neurons*sizeof(float));
    // CSR connectome arrays (allocated but uploaded separately)
    ALLOC(d_row_ptr, (n_neurons+1)*sizeof(int));
    ALLOC(d_col_indices, n_synapses*sizeof(int));
    ALLOC(d_csr_values, n_synapses*sizeof(float));
    ALLOC(d_nt_types, n_neurons*sizeof(float));
    ALLOC(d_external_I, n_neurons*sizeof(float));
    ALLOC(d_psi_v, n_neurons*sizeof(float));
    ALLOC(d_psi_u, n_neurons*sizeof(float));
    ALLOC(d_psi_h_real, n_neurons*sizeof(float));
    ALLOC(d_psi_h_imag, n_neurons*sizeof(float));
    ALLOC(d_psi_drive, n_neurons*sizeof(float));
    ALLOC(d_aber_real, n_neurons*sizeof(float));
    ALLOC(d_aber_imag, n_neurons*sizeof(float));
    ALLOC(d_aber_collapsed, n_neurons*sizeof(int));
    ctx->n_csr_synapses = n_synapses;

    // Zero-initialize key state
    #define ZERO(n,s) do { \
        e = cudaMemsetAsync(ctx->n, 0, (size_t)(s), ctx->stream); \
        if (e) goto fail; \
    } while(0)
    ZERO(d_v,n_neurons*sizeof(float)); ZERO(d_u,n_neurons*sizeof(float));
    ZERO(d_trace,n_neurons*sizeof(float)); ZERO(d_spikes,n_neurons*sizeof(int));
    ZERO(d_I,n_neurons*sizeof(float));
    ZERO(d_weights,n_synapses*sizeof(float)); ZERO(d_phases,n_neurons*sizeof(float));
    ZERO(d_cos_buf,n_neurons*sizeof(float)); ZERO(d_sin_buf,n_neurons*sizeof(float));
    ZERO(d_amplitudes,n_neurons*sizeof(float)); ZERO(d_drives,n_neurons*sizeof(float));
    ZERO(d_t3_prev_amps,n_neurons*sizeof(float));    ZERO(d_refractory, n_neurons*sizeof(float));
    ZERO(d_adex_w, n_neurons*sizeof(float));
    ZERO(d_row_ptr, (n_neurons+1)*sizeof(int));
    ZERO(d_col_indices, n_synapses*sizeof(int));
    ZERO(d_csr_values, n_synapses*sizeof(float));
    ZERO(d_nt_types, n_neurons*sizeof(float));
    ZERO(d_external_I, n_neurons*sizeof(float));
    ZERO(d_psi_v, n_neurons*sizeof(float));
    ZERO(d_psi_u, n_neurons*sizeof(float));
    ZERO(d_psi_h_real, n_neurons*sizeof(float));
    ZERO(d_psi_h_imag, n_neurons*sizeof(float));
    ZERO(d_psi_drive, n_neurons*sizeof(float));
    ZERO(d_aber_real, n_neurons*sizeof(float));
    ZERO(d_aber_imag, n_neurons*sizeof(float));
    ZERO(d_aber_collapsed, n_neurons*sizeof(int));

    ctx->psi_graph_captured = 0;
    ctx->psi_graph_dt = 0;
    ctx->psi_graph_hbar = 0;

    e = cudaStreamSynchronize(ctx->stream);
    if (e) goto fail;
    return ctx;

fail:
    gpu_context_destroy(ctx);
    return NULL;
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
    F(d_gate_out);F(d_decoder_w);F(d_decoder_bias);F(d_decoder_cov);    F(d_echo_buf);
    F(d_amplitudes);F(d_drives);F(d_refractory);F(d_t3_prev_amps);    F(d_row_ptr); F(d_col_indices); F(d_csr_values); F(d_nt_types); F(d_external_I);
    F(d_psi_v); F(d_psi_u); F(d_psi_h_real); F(d_psi_h_imag); F(d_psi_drive);
    F(d_aber_real); F(d_aber_imag); F(d_aber_collapsed);
    #undef F
    free(ctx);
}

// ─── Upload I (input current) to GPU ────────────────────

extern "C" int gpu_context_upload_I(void* vctx, const float* I, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_I, I, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── Upload initial state ────────────────────────────────

extern "C" int gpu_context_upload_neurons(void* vctx,
    const float* v, const float* u, const float* a, const float* b,
    const float* c, const float* d, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
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
    int err = 0;
    err |= (int)cudaMemcpyAsync(ctx->d_phases, phases, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_omegas, omegas, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_coupling, coupling, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

// ─── Upload CSR connectome arrays (row_ptr, col_indices, values) ──
// Note: values are stored in d_I (reused as weight buffer)

extern "C" int gpu_context_upload_connectome(void* vctx,
    const int* row_ptr, const int* col_indices, const float* values,
    const float* nt_types, const float* external_I,
    int n_neurons, int n_synapses)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
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
    ctx->n_csr_synapses = n_synapses;
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

extern "C" int gpu_context_download_adex_w(void* vctx, float* host_w, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(host_w, ctx->d_adex_w, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_upload_adex_w(void* vctx, const float* host_w, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_adex_w, host_w, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

extern "C" int gpu_context_sync(void* vctx) {
    return (int)cudaStreamSynchronize(((GpuCtx*)vctx)->stream);
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
        gpu_psi_hamiltonian_async(vctx);
        gpu_psi_step_async(vctx, dt, hbar);

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
