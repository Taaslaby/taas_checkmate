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
    int bs = 256, gs = (n+bs-1)/bs;
    stdp_update_kernel<<<gs, bs>>>(w, tp, tn, lr_ltd, lr_ltp, w_min, w_max, n);
    cudaDeviceSynchronize(); return 0;
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
    int bs = 256, n = ctx->n_synapses;
    stdp_update_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_weights, ctx->d_traces_pre, ctx->d_traces_post,
        lr_ltd, lr_ltp, w_min, w_max, n);
    return (int)cudaGetLastError();
}
