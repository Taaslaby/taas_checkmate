/**
 * STDP plasticity CUDA kernels + host-callable wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" __global__ void stdp_update_kernel(
    float* w, const float* tp, const float* tn,
    float lr_ltd, float lr_ltp, float w_min, float w_max, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    w[idx] = fminf(w_max, fmaxf(w_min, w[idx] + lr_ltp * tp[idx] - lr_ltd * tn[idx]));
}

extern "C" __global__ void homeostatic_kernel(
    float* w, const float* tp, const float* tn,
    float lr_ltd, float lr_ltp, float tgt, float w_max, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    w[idx] += lr_ltp * tp[idx] - lr_ltd * tn[idx] + 0.001f * (tgt/n - w[idx]);
    w[idx] = fminf(w_max, fmaxf(0.0f, w[idx]));
}

extern "C" __global__ void hebbian_update_kernel(
    float* w, const float* cd, float lr, float decay, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    w[idx] += lr * cd[idx] - decay * w[idx] * dt;
}

#include "gpu_context.h"

// ─── Host wrappers ───────────────────────────────────────

extern "C" int cuda_stdp_update(float* w, const float* tp, const float* tn,
    float lr_ltd, float lr_ltp, float w_min, float w_max, int n) {
    // Host → device → host round-trip (w/tp/tn are HOST buffers).
    int bs = 256, gs = (n+bs-1)/bs;
    float* d_w = NULL; float* d_tp = NULL; float* d_tn = NULL;
    cudaError_t e = cudaMalloc(&d_w, (size_t)n * sizeof(float));
    if (e) return (int)e;
    e = cudaMalloc(&d_tp, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_w); return (int)e; }
    e = cudaMalloc(&d_tn, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_w); cudaFree(d_tp); return (int)e; }
    if ((e = cudaMemcpy(d_tp, tp, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_tn, tn, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_w, w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice))) {
        cudaFree(d_w); cudaFree(d_tp); cudaFree(d_tn); return (int)e;
    }
    stdp_update_kernel<<<gs, bs>>>(d_w, d_tp, d_tn, lr_ltd, lr_ltp, w_min, w_max, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(w, d_w, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_tp); cudaFree(d_tn);
    return (int)e;
}

extern "C" int cuda_hebbian_update(float* w, const float* cd,
    float lr, float decay, float dt, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    hebbian_update_kernel<<<gs, bs>>>(w, cd, lr, decay, dt, n);
    cudaDeviceSynchronize(); return 0;
}

// ─── Async stream-based launch ────────────────────────────

extern "C" int gpu_stdp_update_async(void* vctx,
    float lr_ltd, float lr_ltp, float w_min, float w_max)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int ns = ctx->n_synapses;
    ENSURE_ALLOC(d_weights, ns*sizeof(float));
    ENSURE_ALLOC(d_traces_pre, ctx->n_neurons*sizeof(float));
    ENSURE_ALLOC(d_traces_post, ctx->n_neurons*sizeof(float));
    int bs = 256, n = ns;
    stdp_update_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_weights, ctx->d_traces_pre, ctx->d_traces_post,
        lr_ltd, lr_ltp, w_min, w_max, n);
    return (int)cudaGetLastError();
}
