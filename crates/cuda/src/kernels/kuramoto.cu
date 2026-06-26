/**
 * Kuramoto oscillator CUDA kernels + host-callable wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

extern "C" __global__ void kuramoto_prepare_kernel(
    const float* phases, float* cos_out, float* sin_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float ph = phases[idx];
    cos_out[idx] = cosf(ph);
    sin_out[idx] = sinf(ph);
}

extern "C" __global__ void kuramoto_update_kernel(
    float* phases, const float* omegas, const float* coupling, float dt, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    phases[idx] += dt * (omegas[idx] + coupling[idx]);
}

extern "C" __global__ void lazy_sincos_update_kernel(
    float* sin_out, float* cos_out, const float* phases,
    float* phases_prev, float eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (fabsf(phases[idx] - phases_prev[idx]) > eps) {
        sin_out[idx] = sinf(phases[idx]);
        cos_out[idx] = cosf(phases[idx]);
        phases_prev[idx] = phases[idx];
    }
}

extern "C" __global__ void batch_coherence_sweep_kernel(
    float* output, const float* areas, const float* areas_sin, const float* areas_cos,
    int n_areas, int n_elem) {
    int a = blockIdx.x, b = threadIdx.x;
    if (a >= n_areas || b >= n_areas) return;
    float sum = 0;
    for (int k = 0; k < n_elem; k++)
        sum += areas_sin[a*n_elem + k] * areas_sin[b*n_elem + k] +
               areas_cos[a*n_elem + k] * areas_cos[b*n_elem + k];
    output[a*n_areas + b] = sum / n_elem;
}

#include "gpu_context.h"

// ─── Host wrappers ───────────────────────────────────────

extern "C" int cuda_kuramoto_prepare(
    const float* phases, float* cos_out, float* sin_out, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    kuramoto_prepare_kernel<<<gs, bs>>>(phases, cos_out, sin_out, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_kuramoto_update(
    float* phases, const float* omegas, const float* coupling, float dt, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    kuramoto_update_kernel<<<gs, bs>>>(phases, omegas, coupling, dt, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_lazy_sincos_update(
    float* sin_out, float* cos_out, const float* phases,
    float* phases_prev, float eps, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    lazy_sincos_update_kernel<<<gs, bs>>>(sin_out, cos_out, phases, phases_prev, eps, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_batch_coherence_sweep(
    float* output, const float* areas, const float* areas_sin, const float* areas_cos,
    int n_areas, int n_elem) {
    batch_coherence_sweep_kernel<<<n_areas, n_areas>>>(output, areas, areas_sin, areas_cos, n_areas, n_elem);
    cudaDeviceSynchronize(); return 0;
}

// ─── Async stream-based launches ──────────────────────────

extern "C" int gpu_kuramoto_prepare_async(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    kuramoto_prepare_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(ctx->d_phases, ctx->d_cos_buf, ctx->d_sin_buf, n);
    return (int)cudaGetLastError();
}

extern "C" int gpu_kuramoto_update_async(void* vctx, float dt) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    kuramoto_update_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(ctx->d_phases, ctx->d_omegas, ctx->d_coupling, dt, n);
    return (int)cudaGetLastError();
}
