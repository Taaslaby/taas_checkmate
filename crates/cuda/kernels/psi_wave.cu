/**
 * ψ-wave CUDA kernels — Kuramoto-like phase dynamics on connectome.
 *
 * Kernels:
 *   psi_hamiltonian_kernel    — CSR matmul: h_real=W·u, h_imag=-W·v
 *   psi_step_normalize_kernel — Euler step + per-neuron normalization
 *
 * Host wrappers (sync + async):
 *   cuda_psi_step             — one-shot: alloc scratch, run, free
 *   gpu_psi_hamiltonian_async — async CSR matmul via GpuCtx
 *   gpu_psi_step_async        — async Euler+normalize via GpuCtx
 *   gpu_psi_upload_async      — upload ψ-wave state to GPU
 *   gpu_psi_download_async    — download ψ-wave state from GPU
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "gpu_context.h"

// ═══════════════════════════════════════════════════════════
//  Device kernels
// ═══════════════════════════════════════════════════════════

/// Zero out the hamiltonian scratch buffers.
/// One thread per neuron.
extern "C" __global__ void reset_psi_hamiltonian_kernel(
    float* h_real, float* h_imag, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h_real[i] = 0.0f;
    h_imag[i] = 0.0f;
}

/// CSR hamiltonian matmul: for each source neuron j, accumulate
///   h_real[i] += W[j,i] * u[j]
///   h_imag[i] -= W[j,i] * v[j]
/// for all outgoing synapses j → i.
///
/// One thread per source neuron — iterate over its outgoing edges.
/// Uses atomicAdd for thread-safe accumulation at targets.
extern "C" __global__ void psi_hamiltonian_kernel(
    const float* v, const float* u,
    const int* row_ptr, const int* col_indices, const float* values,
    float* h_real, float* h_imag,
    int n_neurons
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_neurons) return;

    int start = row_ptr[j];
    int end = row_ptr[j + 1];
    float vj = v[j];
    float uj = u[j];

    for (int k = start; k < end; k++) {
        int i = col_indices[k];
        float w = values[k];
        atomicAdd(&h_real[i], w * uj);
        atomicAdd(&h_imag[i], -w * vj);
    }
}

/// Euler step + per-neuron normalization.
///   v_i += dt * h_real_i / hbar
///   u_i += dt * (h_imag_i + i_ext_i) / hbar   (i_ext may be NULL)
///   nrm  = sqrt(v_i² + u_i² + 1e-9)
///   v_i /= nrm, u_i /= nrm
///
/// One thread per neuron.
extern "C" __global__ void psi_step_normalize_kernel(
    float* v, float* u,
    const float* h_real, const float* h_imag,
    const float* i_ext,
    float dt, float hbar,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float vv = v[i];
    float uu = u[i];

    // Euler step
    vv += dt * h_real[i] / hbar;
    float i_ext_val = i_ext ? fminf(100.0f, fmaxf(-100.0f, i_ext[i])) : 0.0f;
    uu += dt * (h_imag[i] + i_ext_val) / hbar;

    // Normalize — preserve |ψ_i| = 1
    float nrm = sqrtf(vv * vv + uu * uu + 1e-9f);
    v[i] = vv / nrm;
    u[i] = uu / nrm;
}

// ═══════════════════════════════════════════════════════════
//  Synchronous host wrappers (allocate scratch internally)
// ═══════════════════════════════════════════════════════════

/// One-shot ψ-wave step: allocates scratch, runs CSR matmul + Euler/normalize, frees.
///
/// Args:
///   v, u          — ψ-wave state (device pointers, read-write)
///   row_ptr       — CSR row pointer (device)
///   col_indices   — CSR column indices (device)
///   values        — CSR values (device)
///   i_ext         — external drive per neuron (device, may be NULL)
///   dt, hbar      — timestep and Planck constant
///   n_neurons     — number of neurons
extern "C" int cuda_psi_step(
    float* v, float* u,
    const int* row_ptr, const int* col_indices, const float* values,
    const float* i_ext,
    float dt, float hbar,
    int n_neurons
) {
    float *h_real, *h_imag;
    cudaError_t e;

    e = cudaMalloc(&h_real, n_neurons * sizeof(float));
    if (e) { fprintf(stderr, "cuda_psi_step: cudaMalloc h_real failed: %s\n", cudaGetErrorString(e)); return (int)e; }
    e = cudaMalloc(&h_imag, n_neurons * sizeof(float));
    if (e) { fprintf(stderr, "cuda_psi_step: cudaMalloc h_imag failed: %s\n", cudaGetErrorString(e)); cudaFree(h_real); return (int)e; }

    int bs = 256;
    int gs_n = (n_neurons + bs - 1) / bs;

    // 1. Zero scratch
    reset_psi_hamiltonian_kernel<<<gs_n, bs>>>(h_real, h_imag, n_neurons);

    // 2. CSR matmul
    psi_hamiltonian_kernel<<<gs_n, bs>>>(v, u, row_ptr, col_indices, values, h_real, h_imag, n_neurons);

    // 3. Euler + normalize
    psi_step_normalize_kernel<<<gs_n, bs>>>(v, u, h_real, h_imag, i_ext, dt, hbar, n_neurons);

    e = cudaDeviceSynchronize();
    cudaFree(h_real);
    cudaFree(h_imag);
    return (int)e;
}

// ═══════════════════════════════════════════════════════════
//  Async stream-based wrappers (for GpuCtx persistent memory)
// ═══════════════════════════════════════════════════════════

/// Asynchronous: reset hamiltonian scratch → CSR matmul.
/// Does NOT sync — caller must sync() or call gpu_psi_step_async.
extern "C" int gpu_psi_hamiltonian_async(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_psi_v, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_u, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_h_real, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_h_imag, nn*sizeof(float));
    int bs = 256, n = nn;
    int gs_n = gs(n, bs);

    // 1. Zero scratch buffers
    reset_psi_hamiltonian_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_psi_h_real, ctx->d_psi_h_imag, n);

    // 2. CSR matmul: h_real = W·u, h_imag = -W·v
    psi_hamiltonian_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_psi_v, ctx->d_psi_u,
        ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values,
        ctx->d_psi_h_real, ctx->d_psi_h_imag,
        n);

    return (int)cudaGetLastError();
}

/// Asynchronous: Euler step + normalize (must be called after gpu_psi_hamiltonian_async).
/// Applies external drive from d_psi_drive (uploaded via gpu_psi_upload_drive_async).
extern "C" int gpu_psi_step_async(void* vctx, float dt, float hbar) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_psi_v, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_u, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_h_real, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_h_imag, nn*sizeof(float));
    ENSURE_ALLOC(d_psi_drive, nn*sizeof(float));
    int bs = 256, n = nn;
    int gs_n = gs(n, bs);

    // d_psi_drive holds pre-uploaded external drive (or zeros if not set)
    float* i_ext = ctx->d_psi_drive;

    psi_step_normalize_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_psi_v, ctx->d_psi_u,
        ctx->d_psi_h_real, ctx->d_psi_h_imag,
        i_ext, dt, hbar, n);

    return (int)cudaGetLastError();
}

/// Combined async: hamiltonian + step (convenience wrapper).
/// Equivalent to calling gpu_psi_hamiltonian_async then gpu_psi_step_async.
extern "C" int gpu_psi_combined_step_async(void* vctx, float dt, float hbar) {
    int err = gpu_psi_hamiltonian_async(vctx);
    if (err) return err;
    return gpu_psi_step_async(vctx, dt, hbar);
}

/// Upload ψ-wave state (v, u) to GPU.
extern "C" int gpu_psi_upload_async(void* vctx,
    const float* host_v, const float* host_u, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    e = cudaMemcpyAsync(ctx->d_psi_v, host_v, n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(ctx->d_psi_u, host_u, n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    return (int)e;
}

/// Download ψ-wave state (v, u) from GPU.
extern "C" int gpu_psi_download_async(void* vctx,
    float* host_v, float* host_u, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    e = cudaMemcpyAsync(host_v, ctx->d_psi_v, n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(host_u, ctx->d_psi_u, n * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    return (int)e;
}

/// Upload ψ-wave external drive (i_ext) to d_psi_drive buffer.
extern "C" int gpu_psi_upload_drive_async(void* vctx,
    const float* host_i_ext, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e = cudaMemcpyAsync(
        ctx->d_psi_drive, host_i_ext,
        n * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    return (int)e;
}

/// Upload Aberton state (real, imag) to GPU.
extern "C" int gpu_aber_upload_async(void* vctx,
    const float* host_real, const float* host_imag, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    e = cudaMemcpyAsync(ctx->d_aber_real, host_real, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(ctx->d_aber_imag, host_imag, n*sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    return (int)e;
}

/// Download Aberton state and collapsed flags from GPU.
extern "C" int gpu_aber_download_async(void* vctx,
    float* host_real, float* host_imag, int* host_collapsed, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    cudaError_t e;
    e = cudaMemcpyAsync(host_real, ctx->d_aber_real, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(host_imag, ctx->d_aber_imag, n*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (e) return (int)e;
    e = cudaMemcpyAsync(host_collapsed, ctx->d_aber_collapsed, n*sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    return (int)e;
}

// ═══════════════════════════════════════════════════════════
//  Aberton CUDA kernels
// ═══════════════════════════════════════════════════════════

/// Fused Aberton step + collapse kernel.
///
///   ar += dt * (-gamma * ar - omega * ai + gain * driver_v)
///   ai += dt * (-gamma * ai + omega * ar + gain * driver_u)
///   if ar² + ai² > threshold²: collapse, set collapsed[i]=1
///
/// driver_v/driver_u point to the field driving the abertons.
/// Typically ψ-wave state (d_psi_v, d_psi_u) or synaptic field (d_psi_h_real, d_psi_h_imag).
///
/// All Aberton parameters are uniform across neurons (passed as kernel args).
extern "C" __global__ void aberton_step_kernel(
    float* ar, float* ai,              // Aberton state (read-write)
    const float* driver_v, const float* driver_u,  // driving field
    int* collapsed,                     // output: 1 if collapsed this step
    float gamma, float omega, float gain,
    float threshold, float collapse_scale,
    float dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ari = ar[i];
    float aii = ai[i];

    // Euler step
    ari += dt * (-gamma * ari - omega * aii + gain * driver_v[i]);
    aii += dt * (-gamma * aii + omega * ari + gain * driver_u[i]);

    // Collapse check
    float amp2 = ari * ari + aii * aii;
    if (amp2 > threshold * threshold) {
        collapsed[i] = 1;
        ari = collapse_scale * driver_v[i];
        aii = collapse_scale * driver_u[i];
    } else {
        collapsed[i] = 0;
    }

    ar[i] = ari;
    ai[i] = aii;
}

/// Asynchronous: Aberton step + collapse.
/// driver_v/driver_u: driving field device pointers.
/// If NULL, defaults to ψ-wave state (d_psi_v, d_psi_u).
extern "C" int gpu_aber_step_async(void* vctx,
    float gamma, float omega, float gain,
    float threshold, float collapse_scale,
    float dt, int use_synaptic_field)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    int gs_n = gs(n, bs);

    // Choose driver: synaptic field (h_real/h_imag) or ψ-wave state
    const float* driver_v = use_synaptic_field ? ctx->d_psi_h_real : ctx->d_psi_v;
    const float* driver_u = use_synaptic_field ? ctx->d_psi_h_imag : ctx->d_psi_u;

    aberton_step_kernel<<<gs_n, bs, 0, ctx->stream>>>(
        ctx->d_aber_real, ctx->d_aber_imag,
        driver_v, driver_u,
        ctx->d_aber_collapsed,
        gamma, omega, gain, threshold, collapse_scale,
        dt, n);

    return (int)cudaGetLastError();
}

/// Combined async: ψ-wave hamiltonian → ψ-wave step → Aberton step.
/// This is the main workhorse for GPU-accelerated Ψ+Orch-OR experiments.
///
/// Flow:
///   1. Reset hamiltonian scratch → CSR matmul (h = W·ψ)
///   2. Euler + normalize ψ-wave (using h as drive, plus external i_ext)
///   3. Aberton step + collapse (using ψ or h as driver)
extern "C" int gpu_psi_aberton_combined_async(void* vctx,
    float dt, float hbar,
    float aber_gamma, float aber_omega, float aber_gain,
    float aber_threshold, float aber_collapse_scale,
    int use_synaptic_field)
{
    int err;

    // 1. ψ-wave hamiltonian
    err = gpu_psi_hamiltonian_async(vctx);
    if (err) return err;

    // 2. ψ-wave step
    err = gpu_psi_step_async(vctx, dt, hbar);
    if (err) return err;

    // 3. Aberton step
    err = gpu_aber_step_async(vctx,
        aber_gamma, aber_omega, aber_gain,
        aber_threshold, aber_collapse_scale,
        dt, use_synaptic_field);
    if (err) return err;

    return 0;
}
