/**
 * T3 (TriOscillator) fused CUDA kernels.
 *
 * Combines Stuart-Landau amplitude update + Kuramoto phase update
 * in a single kernel launch — avoiding redundant global memory traffic.
 *
 * ## Formulas
 *
 * ```
 * r += dt · ((α - r²)·r + drive · gain)
 * θ += dt · (ω + coupling + drive · gain / (r + ε))
 * ```
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "gpu_context.h"

// ─── Device kernels ──────────────────────────────────────

/**
 * Fused T3 update: Stuart-Landau amplitudes + Kuramoto phases.
 *
 * Each thread handles one oscillator.
 *
 * @param phases      [in/out] oscillator phases θ
 * @param amplitudes  [in/out] Stuart-Landau amplitudes r
 * @param omegas      [in]     natural frequencies ω
 * @param coupling    [in]     Kuramoto coupling term
 * @param drives      [in]     external drive inputs
 * @param dt          [in]     timestep
 * @param alpha       [in]     Stuart-Landau nonlinearity parameter
 * @param gain        [in]     drive gain
 * @param n           [in]     number of oscillators
 */
extern "C" __global__ void t3_update_kernel(
    float* phases, float* amplitudes,
    const float* omegas, const float* coupling, const float* drives,
    float dt, float alpha, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float r = amplitudes[idx];
    float drive = drives[idx];
    float cpl = coupling[idx];
    float omega = omegas[idx];
    float eps = 0.1f; // amplitude floor

    // Stuart-Landau: r += dt · ((α - r²)·r + drive · gain)
    float r_dot = (alpha - r * r) * r + drive * gain;
    amplitudes[idx] = r + dt * r_dot;

    // Kuramoto: θ += dt · (ω + coupling + drive · gain / (r + ε))
    float drive_term = drive * gain / (r + eps);
    phases[idx] += dt * (omega + cpl + drive_term);
}

/**
 * T3 prepare kernel: compute sin/cos from phases + wrap to [-π, π].
 *
 * Mirror of kuramoto_prepare_kernel but with phase wrapping.
 */
extern "C" __global__ void t3_prepare_kernel(
    float* phases, float* cos_out, float* sin_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ph = phases[idx];
    // Wrap phase to [-π, π]
    float wrapped = fmodf(ph + M_PI, 2.0f * M_PI);
    if (wrapped < 0) wrapped += 2.0f * M_PI;
    wrapped -= M_PI;

    phases[idx] = wrapped;
    cos_out[idx] = cosf(wrapped);
    sin_out[idx] = sinf(wrapped);
}

/**
 * T3 Stuart-Landau only kernel (for amplitude-only updates).
 */
extern "C" __global__ void t3_stuart_landau_kernel(
    float* amplitudes, const float* drives,
    float dt, float alpha, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float r = amplitudes[idx];
    float drive = drives[idx];
    float r_dot = (alpha - r * r) * r + drive * gain;
    amplitudes[idx] = r + dt * r_dot;
}

/**
 * Upload T3 oscillator state to device.
 */
extern "C" int gpu_t3_upload(
    void* vctx,
    const float* phases, const float* amplitudes,
    const float* omegas,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(ctx->d_phases, phases, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_amplitudes, amplitudes, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_omegas, omegas, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

/**
 * Upload T3 drives to device.
 */
extern "C" int gpu_t3_upload_drives(
    void* vctx, const float* drives, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_drives, drives,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

/**
 * Upload T3 coupling to device.
 */
extern "C" int gpu_t3_upload_coupling(
    void* vctx, const float* coupling, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_coupling, coupling,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

/**
 * Download T3 oscillator state from device.
 */
extern "C" int gpu_t3_download(
    void* vctx,
    float* phases, float* amplitudes,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(phases, ctx->d_phases,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    err |= (int)cudaMemcpyAsync(amplitudes, ctx->d_amplitudes,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

/**
 * Download T3 cos/sin buffers from device (for CPU coupling computation).
 */
extern "C" int gpu_t3_download_cos_sin(
    void* vctx,
    float* cos_out, float* sin_out,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(cos_out, ctx->d_cos_buf,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    err |= (int)cudaMemcpyAsync(sin_out, ctx->d_sin_buf,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return err;
}

/**
 * Upload T3 omegas to device (done once at init).
 */
extern "C" int gpu_t3_upload_omegas(
    void* vctx, const float* omegas, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(ctx->d_omegas, omegas,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    cudaStreamSynchronize(ctx->stream);
    return (int)e;
}

// ─── Host wrappers ───────────────────────────────────────

extern "C" int cuda_t3_update(
    float* phases, float* amplitudes,
    const float* omegas, const float* coupling, const float* drives,
    float dt, float alpha, float gain, int n) {
    int bs = 256, gs = (n + bs - 1) / bs;
    t3_update_kernel<<<gs, bs>>>(phases, amplitudes, omegas, coupling, drives,
        dt, alpha, gain, n);
    cudaDeviceSynchronize();
    return 0;
}

extern "C" int cuda_t3_prepare(
    float* phases, float* cos_out, float* sin_out, int n) {
    int bs = 256, gs = (n + bs - 1) / bs;
    t3_prepare_kernel<<<gs, bs>>>(phases, cos_out, sin_out, n);
    cudaDeviceSynchronize();
    return 0;
}

// ─── Async transfers (no sync inside — caller syncs once) ─

extern "C" int gpu_t3_upload_async(
    void* vctx,
    const float* phases, const float* amplitudes,
    const float* omegas,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(ctx->d_phases, phases, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_amplitudes, amplitudes, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    err |= (int)cudaMemcpyAsync(ctx->d_omegas, omegas, n * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    return err;
}

extern "C" int gpu_t3_upload_drives_async(void* vctx, const float* drives, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    return (int)cudaMemcpyAsync(ctx->d_drives, drives,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
}

extern "C" int gpu_t3_upload_coupling_async(void* vctx, const float* coupling, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    return (int)cudaMemcpyAsync(ctx->d_coupling, coupling,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
}

extern "C" int gpu_t3_upload_omegas_async(void* vctx, const float* omegas, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    return (int)cudaMemcpyAsync(ctx->d_omegas, omegas,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
}

extern "C" int gpu_t3_download_async(
    void* vctx,
    float* phases, float* amplitudes,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(phases, ctx->d_phases,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    err |= (int)cudaMemcpyAsync(amplitudes, ctx->d_amplitudes,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return err;
}

extern "C" int gpu_t3_download_cos_sin_async(
    void* vctx,
    float* cos_out, float* sin_out,
    int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int err = 0;
    err |= (int)cudaMemcpyAsync(cos_out, ctx->d_cos_buf,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    err |= (int)cudaMemcpyAsync(sin_out, ctx->d_sin_buf,
        n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return err;
}

// ─── Async stream-based launches ─────────────────────────

extern "C" int gpu_t3_prepare_async(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_t3;
    t3_prepare_kernel<<<gs(n, bs), bs, 0, ctx->stream>>>(
        ctx->d_phases, ctx->d_cos_buf, ctx->d_sin_buf, n);
    return (int)cudaGetLastError();
}

extern "C" int gpu_t3_update_async(void* vctx,
    float dt, float alpha, float gain) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_t3;
    t3_update_kernel<<<gs(n, bs), bs, 0, ctx->stream>>>(
        ctx->d_phases, ctx->d_amplitudes,
        ctx->d_omegas, ctx->d_coupling, ctx->d_drives,
        dt, alpha, gain, n);
    return (int)cudaGetLastError();
}

extern "C" int gpu_t3_stuart_landau_async(void* vctx,
    float dt, float alpha, float gain) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_t3;
    t3_stuart_landau_kernel<<<gs(n, bs), bs, 0, ctx->stream>>>(
        ctx->d_amplitudes, ctx->d_drives, dt, alpha, gain, n);
    return (int)cudaGetLastError();
}
