/**
 * Spiking neuron CUDA kernels (Izhikevich, LIF, AdEx, SAW).
 * Host wrappers expect device-allocated pointers (Rust manages memory).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ─── Device kernels ──────────────────────────────────────

extern "C" __global__ void izhikevich_step_kernel(
    float* v, float* u, float* trace,
    const float* a, const float* b, const float* I,
    float dt, float trace_decay,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float vi = v[idx], ui = u[idx], Ii = I[idx];

    // Refractory period: skip all dynamics (matches CPU SIMD behavior)
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        spikes[idx] = 0;
        return;  // v, u, and trace unchanged during refractory
    }

    vi += dt * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + Ii);
    ui += dt * a[idx] * (b[idx] * vi - ui);
    int spiked = (vi >= 30.0f) ? 1 : 0;
    if (spiked) { vi = c[idx]; ui += d[idx]; }
    if (refractory && spiked) { refractory[idx] = tau_ref; }
    if (trace) { trace[idx] *= trace_decay; if (spiked) trace[idx] += 1.0f; }
    v[idx] = vi; u[idx] = ui; spikes[idx] = spiked;
}

extern "C" __global__ void lif_step_kernel(
    float* v, float* I, int* spikes, const float* params, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float dt = params[0];
    float tau_m = params[1+idx], v_rest = params[1+n+idx];
    float v_thresh = params[1+2*n+idx], v_reset = params[1+3*n+idx];
    float vi = v[idx];
    vi += dt / tau_m * (-(vi - v_rest) + I[idx]);
    int spiked = (vi >= v_thresh) ? 1 : 0;
    v[idx] = spiked ? v_reset : vi; spikes[idx] = spiked;
}

extern "C" __global__ void adex_step_kernel(
    float* v, float* u, float* I, int* spikes, const float* params, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float dt = params[0];
    float tau_m = params[1+idx], tau_w = params[1+n+idx];
    float v_rest = params[1+2*n+idx], v_thresh = params[1+3*n+idx];
    float v_reset = params[1+4*n+idx], delta_t = params[1+5*n+idx];
    float ada_a = params[1+6*n+idx], ada_b = params[1+7*n+idx];
    float vi = v[idx], ui = u[idx];
    float e = delta_t * expf((vi - v_thresh) / fmaxf(delta_t, 1e-6f));
    vi += dt / tau_m * (-(vi - v_rest) + delta_t * e - ui + I[idx]);
    ui += dt / tau_w * (ada_a * (vi - v_rest) - ui);
    int spiked = (vi >= 30.0f) ? 1 : 0;
    if (spiked) { vi = v_reset; ui += ada_b; }
    v[idx] = vi; u[idx] = ui; spikes[idx] = spiked;
}

#include "gpu_context.h"

// ─── Host wrappers (device pointers) ─────────────────────

extern "C" int cuda_izhikevich_step(
    float* v, float* u, float* trace,
    const float* a, const float* b, const float* I,
    float dt, float trace_decay,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref, int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    izhikevich_step_kernel<<<gs, bs>>>(v,u,trace,a,b,I,dt,trace_decay,spikes,c,d,refractory,tau_ref,n);
    cudaDeviceSynchronize(); return (int)cudaGetLastError();
}

extern "C" int cuda_lif_step(
    float* v, float* I, int* spikes, const float* params, int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    lif_step_kernel<<<gs, bs>>>(v, I, spikes, params, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_adex_step(
    float* v, float* u, float* I, int* spikes, const float* params, int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    adex_step_kernel<<<gs, bs>>>(v, u, I, spikes, params, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_saw_step(
    float* v, float* u, float* trace,
    const float* a, const float* b, const float* I,
    float dt, float trace_decay, float noise_amp,
    int* spikes,
    float* c, float* d,   // SAW reset params: c=reset voltage, d=u_add
    float* refractory, float tau_ref, int n
) {
    // SAW uses izhikevich dynamics with its own c,d reset params (not a,b!)
    return cuda_izhikevich_step(v, u, trace, a, b, I, dt, trace_decay, spikes, c, d, refractory, tau_ref, n);
}

// ─── Async stream-based launch (for GPU context) ──────────

extern "C" int gpu_izhikevich_step_async(void* vctx, float dt, float trace_decay) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256, n = ctx->n_neurons;
    izhikevich_step_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_v, ctx->d_u, ctx->d_trace,
        ctx->d_a, ctx->d_b, ctx->d_I, dt, trace_decay,
        ctx->d_spikes, ctx->d_c, ctx->d_d,
        ctx->d_refractory, 3.0f, n);  // tau_ref=3.0 matches CPU DEFAULT_TAU_REF
    return (int)cudaGetLastError();
}
