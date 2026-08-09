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
    const float* a, const float* b, float* I,
    float dt, float trace_decay, float v_thresh,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float vi = v[idx], ui = u[idx], Ii = fminf(100.0f, fmaxf(-100.0f, I[idx]));

    // Refractory period: skip all dynamics but clear I to prevent accumulation
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        spikes[idx] = 0;
        I[idx] = 0.0f;  // BUGFIX: drain incoming spikes during refractory
        return;  // v, u, and trace unchanged during refractory
    }

    vi += dt * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + Ii);
    ui += dt * a[idx] * (b[idx] * vi - ui);
    int spiked = (vi >= v_thresh) ? 1 : 0;
    if (spiked) { vi = c[idx]; ui += d[idx]; }
    if (refractory && spiked) { refractory[idx] = tau_ref; }
    if (trace) { trace[idx] *= trace_decay; if (spiked) trace[idx] += 1.0f; }
    v[idx] = vi; u[idx] = ui; spikes[idx] = spiked;
    I[idx] = 0.0f;  // BUGFIX: reset current buffer for next step's propagation
}

extern "C" __global__ void lif_step_kernel(
    float* v, float* I, int* spikes, const float* params,
    float* refractory, float tau_ref, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float dt = params[0];
    float tau_m = params[1+idx], v_rest = params[1+n+idx];
    float v_thresh = params[1+2*n+idx], v_reset = params[1+3*n+idx];

    // Refractory period: skip all dynamics but clear I to prevent accumulation
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        v[idx] = v_reset;  // Hold at reset potential during refractory
        spikes[idx] = 0;
        I[idx] = 0.0f;  // BUGFIX: drain incoming spikes during refractory
        return;
    }

    // BUGFIX: Exponential Euler (was Forward Euler)
    // Exponential Euler: v = v_rest + (v - v_rest)*exp(-dt/tau) + I*(1 - exp(-dt/tau))
    // Matches CPU SIMD kernel (lif_step). Exact for constant I, no truncation error.
    float vi = v[idx];
    float Ii = fminf(100.0f, fmaxf(-100.0f, I[idx]));
    float decay = __expf(-dt / fmaxf(tau_m, 1e-6f));
    vi = v_rest + (vi - v_rest) * decay + Ii * (1.0f - decay);
    int spiked = (vi >= v_thresh) ? 1 : 0;
    if (spiked) {
        v[idx] = v_reset;
        if (refractory) refractory[idx] = tau_ref;
    } else {
        v[idx] = vi;
    }
    spikes[idx] = spiked;
    I[idx] = 0.0f;  // BUGFIX: reset current buffer for next step's propagation
}

extern "C" __global__ void adex_step_kernel(
    float* v, float* u, float* I, int* spikes, const float* params,
    float* refractory, float tau_ref,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // ─── Refractory period: skip all dynamics, clear I ──
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - params[0]);
        spikes[idx] = 0;
        I[idx] = 0.0f;  // BUGFIX: drain current during refractory
        return;
    }

    float dt = params[0];
    float tau_m = params[1+idx], tau_w = params[1+n+idx];
    float v_rest = params[1+2*n+idx], v_thresh = params[1+3*n+idx];
    float v_reset = params[1+4*n+idx], delta_t = params[1+5*n+idx];
    float ada_a = params[1+6*n+idx], ada_b = params[1+7*n+idx];
    float vi = v[idx], ui = u[idx];
    float Ii = fminf(100.0f, fmaxf(-100.0f, I[idx]));

    // Exp overflow guard: clamp arg to prevent Inf
    float arg = fminf((vi - v_thresh) / fmaxf(delta_t, 1e-6f), 20.0f);
    float e = __expf(arg);  // BUGFIX: use fast __expf, no double delta_t multiply

    vi += dt / tau_m * (-(vi - v_rest) + delta_t * e - ui + Ii);
    ui += dt / tau_w * (ada_a * (vi - v_rest) - ui);
    int spiked = (vi >= v_thresh) ? 1 : 0;
    if (spiked) { vi = v_reset; ui += ada_b; }
    if (refractory && spiked) { refractory[idx] = tau_ref; }
    v[idx] = vi; u[idx] = ui; spikes[idx] = spiked;
    I[idx] = 0.0f;  // BUGFIX: reset current for next step's propagation
}

#include "gpu_context.h"

// ─── Host wrappers (device pointers) ─────────────────────

extern "C" int cuda_izhikevich_step(
    float* v, float* u, float* trace,
    const float* a, const float* b, float* I,
    float dt, float trace_decay,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref,
    float v_thresh,
    int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    izhikevich_step_kernel<<<gs, bs>>>(v,u,trace,a,b,I,dt,trace_decay,v_thresh,spikes,c,d,refractory,tau_ref,n);
    cudaDeviceSynchronize(); return (int)cudaGetLastError();
}

extern "C" int cuda_lif_step(
    float* v, float* I, int* spikes, const float* params,
    float* refractory, float tau_ref, int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    lif_step_kernel<<<gs, bs>>>(v, I, spikes, params, refractory, tau_ref, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_adex_step(
    float* v, float* u, float* I, int* spikes, const float* params,
    float* refractory, float tau_ref,
    int n
) {
    int bs = 256, gs = (n+bs-1)/bs;
    adex_step_kernel<<<gs, bs>>>(v, u, I, spikes, params, refractory, tau_ref, n);
    cudaDeviceSynchronize(); return 0;
}

// ─── SAW step kernel with per-thread LCG noise ─────────
//
// Same Izhikevich dynamics as izhikevich_step_kernel, but adds
// noise_amp * rand_f32() to v before the dynamics step.
// Uses per-thread LCG seeded from thread index for deterministic
// noise across calls (no device-side RNG state needed).

__device__ inline float saw_lcg(unsigned int* state) {
    *state = *state * 1103515245 + 12345;
    return (*state >> 16) * (1.0f / 65536.0f);
}

extern "C" __global__ void saw_step_kernel(
    float* v, float* u, float* trace,
    const float* a, const float* b, float* I,
    float dt, float trace_decay, float noise_amp, float v_thresh,
    int* spikes, const float* c, const float* d,
    float* refractory, float tau_ref,
    unsigned int* noise_state,  // [n] per-neuron LCG state (in/out)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Initialize LCG state on first call per neuron
    unsigned int ns = noise_state[idx];
    if (ns == 0) ns = (unsigned int)(idx * 7919 + 104729);

    float vi = v[idx], ui = u[idx], Ii = fminf(100.0f, fmaxf(-100.0f, I[idx]));

    // Refractory period: drain I to prevent accumulation
    float ref = (refractory) ? refractory[idx] : 0.0f;
    if (ref > 0.0f) {
        refractory[idx] = fmaxf(0.0f, ref - dt);
        spikes[idx] = 0;
        I[idx] = 0.0f;  // BUGFIX: drain current during refractory
        noise_state[idx] = ns;
        return;
    }

    // Add noise before dynamics
    vi += noise_amp * (saw_lcg(&ns) - 0.5f);

    // Izhikevich dynamics (same as izhikevich_step_kernel)
    vi += dt * (0.04f * vi * vi + 5.0f * vi + 140.0f - ui + Ii);
    ui += dt * a[idx] * (b[idx] * vi - ui);
    int spiked = (vi >= v_thresh) ? 1 : 0;
    if (spiked) { vi = c[idx]; ui += d[idx]; }
    if (refractory && spiked) { refractory[idx] = tau_ref; }
    if (trace) { trace[idx] *= trace_decay; if (spiked) trace[idx] += 1.0f; }

    v[idx] = vi; u[idx] = ui; spikes[idx] = spiked;
    I[idx] = 0.0f;  // BUGFIX: reset current for next step's propagation
    noise_state[idx] = ns;
}

extern "C" int cuda_saw_step(
    float* v, float* u, float* trace,
    const float* a, const float* b, float* I,
    float dt, float trace_decay, float noise_amp,
    int* spikes,
    float* c, float* d,
    float* refractory, float tau_ref,
    unsigned int* noise_state,  // [n] per-neuron LCG state (persistent or temp)
    float v_thresh,
    int n
) {
    // Require persistent noise_state — temp allocation destroys GA reproducibility
    // because ns==0 check re-seeds LCG every step.
    if (noise_state == NULL) return -1;
    int bs = 256, gs = (n+bs-1)/bs;
    saw_step_kernel<<<gs, bs>>>(v, u, trace, a, b, I, dt, trace_decay, noise_amp, v_thresh,
        spikes, c, d, refractory, tau_ref, noise_state, n);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

// ─── Async stream-based launch (for GPU context) ──────────

extern "C" int gpu_izhikevich_step_async(void* vctx, float dt, float trace_decay) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    // Lazy-alloc all arrays the kernel needs
    ENSURE_ALLOC(d_v, nn*sizeof(float));
    ENSURE_ALLOC(d_u, nn*sizeof(float));
    ENSURE_ALLOC(d_trace, nn*sizeof(float));
    ENSURE_ALLOC(d_a, nn*sizeof(float));
    ENSURE_ALLOC(d_b, nn*sizeof(float));
    ENSURE_ALLOC(d_I, nn*sizeof(float));
    ENSURE_ALLOC(d_spikes, nn*sizeof(int));
    ENSURE_ALLOC(d_c, nn*sizeof(float));
    ENSURE_ALLOC(d_d, nn*sizeof(float));
    ENSURE_ALLOC(d_refractory, nn*sizeof(float));
    int bs = 256, n = nn;
    izhikevich_step_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_v, ctx->d_u, ctx->d_trace,
        ctx->d_a, ctx->d_b, ctx->d_I, dt, trace_decay, 30.0f,
        ctx->d_spikes, ctx->d_c, ctx->d_d,
        ctx->d_refractory, 3.0f, n);
    return (int)cudaGetLastError();
}

// ─── Async LIF step via GPU context ───────────────────────
//
// Requires LIF params already uploaded to d_lif_params
// (use gpu_context_upload_lif_params or upload via d_lif_params directly).

extern "C" int gpu_lif_step_async(void* vctx) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    // Lazy-alloc all arrays the kernel needs
    ENSURE_ALLOC(d_v, nn*sizeof(float));
    ENSURE_ALLOC(d_I, nn*sizeof(float));
    ENSURE_ALLOC(d_spikes, nn*sizeof(int));
    ENSURE_ALLOC(d_lif_params, (1+6*nn)*sizeof(float));
    ENSURE_ALLOC(d_refractory, nn*sizeof(float));
    int bs = 256, n = nn;
    lif_step_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_v, ctx->d_I, ctx->d_spikes, ctx->d_lif_params,
        ctx->d_refractory, 3.0f, n);
    return (int)cudaGetLastError();
}

// ─── Upload LIF params to d_lif_params (sync) ────────────
//
// Packs host arrays into flat layout: [dt, tau_m[0..n], v_rest[0..n],
// v_thresh[0..n], v_reset[0..n]] and copies to device.
// Total: 1 + 4*n floats.

extern "C" int gpu_context_upload_lif_params(void* vctx,
    float dt,
    const float* tau_m, const float* v_rest,
    const float* v_thresh, const float* v_reset,
    int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    ENSURE_ALLOC(d_lif_params, (1 + 6*ctx->n_neurons)*sizeof(float));
    int n_params = 1 + 4 * n;

    // Allocate host-side flat buffer
    float* h_params = (float*)malloc((size_t)n_params * sizeof(float));
    if (!h_params) return 1;

    h_params[0] = dt;
    for (int i = 0; i < n; i++) {
        h_params[1 + i] = tau_m[i];
        h_params[1 + n + i] = v_rest[i];
        h_params[1 + 2*n + i] = v_thresh[i];
        h_params[1 + 3*n + i] = v_reset[i];
    }

    // Use sync copy to avoid pageable-memory staging issues on non-default stream.
    // h_params is malloc'd (pageable), so cudaMemcpyAsync would block anyway.
    cudaError_t e = cudaMemcpy(ctx->d_lif_params, h_params,
        (size_t)n_params * sizeof(float), cudaMemcpyHostToDevice);
    free(h_params);
    return (int)e;
}
