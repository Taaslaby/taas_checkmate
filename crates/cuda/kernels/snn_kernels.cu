/**
 * SNN CUDA kernels — LIF-AT, SORN (Intrinsic Plasticity + SynNorm),
 * and Spike Encoding (BSA, Phase, TTFS, Population Rate Decoder).
 *
 * All kernels follow the project's existing patterns:
 * - extern "C" __global__ void for device kernels
 * - Host wrapper with bs=256, gs=(n+bs-1)/bs
 * - Async variants use GpuCtx from gpu_context.h
 * - --use_fast_math compatible
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════
//  LIF-AT (LIF with Adaptive Threshold)
// ═══════════════════════════════════════════════════════════
//
// dv/dt = (v_rest - v + I) / tau_m
// dθ/dt = -θ / tau_θ
// v >= v_thresh + θ → spike, v = v_reset, θ += β
//
// Params layout (flat array of scalars + per-neuron arrays):
//   [0]    = dt  (scalar)
//   [1..N] = tau_m[0..N-1]
//   [N+1..2N] = v_rest[0..N-1]
//   [2N+1..3N] = v_thresh[0..N-1]
//   [3N+1..4N] = v_reset[0..N-1]
//   [4N+1..5N] = tau_theta[0..N-1]
//   [5N+1..6N] = beta[0..N-1]
// Total: 1 + 6*N floats

extern "C" __global__ void lif_at_step_kernel(
    float* v, float* theta, float* I, int* spikes,
    const float* params, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dt = params[0];
    float tau_m  = params[1 + idx];
    float v_rest = params[1 + n + idx];
    float v_thresh = params[1 + 2*n + idx];
    float v_reset  = params[1 + 3*n + idx];
    float tau_theta = params[1 + 4*n + idx];
    float beta_param = params[1 + 5*n + idx];

    float vi = v[idx];
    float th = theta[idx];
    float Ii = I[idx];

    // Exponential Euler for membrane: v = v_rest + (v - v_rest) * exp(-dt/τ) + I * (1 - exp(-dt/τ))
    float decay = __expf(-dt / fmaxf(tau_m, 1e-6f));
    vi = v_rest + (vi - v_rest) * decay + Ii * (1.0f - decay);

    // Exponential Euler for adaptive threshold: θ *= exp(-dt/τ_θ)
    float theta_decay = __expf(-dt / fmaxf(tau_theta, 1e-6f));
    th *= theta_decay;

    // Effective threshold = v_thresh + θ
    int spiked = (vi >= v_thresh + th) ? 1 : 0;
    if (spiked) {
        vi = v_reset;
        th += beta_param;
    }

    v[idx] = vi;
    theta[idx] = th;
    spikes[idx] = spiked;
    I[idx] = 0.0f;  // BUGFIX: reset current for next step's propagation
}

// ═══════════════════════════════════════════════════════════
//  SORN — Intrinsic Plasticity
// ═══════════════════════════════════════════════════════════
//
// θ += lr_ip · (rate - target_rate)
// θ *= decay
// θ = clamp(θ, 0, 50)

extern "C" __global__ void intrinsic_plasticity_kernel(
    float* theta, const int* spikes,
    float lr_ip, float target_rate, float decay, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float rate = (spikes[idx] != 0) ? 1.0f : 0.0f;
    float th = theta[idx];
    th += lr_ip * (rate - target_rate);
    th *= decay;
    th = fminf(50.0f, fmaxf(0.0f, th));
    theta[idx] = th;
}

// ═══════════════════════════════════════════════════════════
//  SORN — Synaptic Normalization (per column)
// ═══════════════════════════════════════════════════════════
//
// w_ij * target_sum / max(Σ_k w_kj, eps)

extern "C" __global__ void synaptic_normalization_kernel(
    float* weights, int n_pre, int n_post,
    float target_sum, float eps
) {
    int j = blockIdx.x;  // one block per post-synaptic neuron
    if (j >= n_post) return;

    // Reduce: sum over i (pre-synaptic neurons)
    float sum_w = 0.0f;
    for (int i = threadIdx.x; i < n_pre; i += blockDim.x) {
        sum_w += weights[i * n_post + j];
    }

    // Block-level reduction (shared memory)
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum_w;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total = sdata[0];
        if (total > eps) {
            float scale = target_sum / total;
            for (int i = 0; i < n_pre; i++) {
                weights[i * n_post + j] *= scale;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  BSA (Bens Spiker Algorithm) — simplified per-element
// ═══════════════════════════════════════════════════════════
//
// BSA is inherently sequential (reconstruction depends on previous spikes).
// This kernel handles the per-element error check and spike decision.
// The reconstruction buffer update is done sequentially on CPU.
//
// err[t] = signal[t] - reconstructed[t]
// spike if |err[t]| > threshold

extern "C" __global__ void bsa_encode_kernel(
    int* spikes_out, const float* signal,
    const float* reconstructed, float threshold, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float err = signal[idx] - reconstructed[idx];
    spikes_out[idx] = (fabsf(err) > threshold) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════
//  Phase Encoding
// ═══════════════════════════════════════════════════════════
//
// θ += signal[i] * gain
// cos(θ) > threshold → spike

extern "C" __global__ void phase_encode_kernel(
    int* spikes_out, float* phases, const float* cos_phases,
    const float* signal, float threshold, float gain, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    phases[idx] += signal[idx] * gain;
    spikes_out[idx] = (cos_phases[idx] > threshold) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════
//  TTFS (Time-To-First-Spike) — check step
// ═══════════════════════════════════════════════════════════
//
// On step 0 (current_step == 0): initialize fire_times from values
// On each step: spike if current_step >= fire_times[i] && !fired[i]

extern "C" __global__ void ttfs_init_kernel(
    int* fire_times, const float* values, int max_delay, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = fminf(1.0f, fmaxf(0.0f, values[idx]));
    fire_times[idx] = (int)((1.0f - val) * max_delay);
}

extern "C" __global__ void ttfs_step_kernel(
    int* spikes_out, const int* fire_times,
    int* fired, int current_step, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (!fired[idx] && current_step >= fire_times[idx]) {
        spikes_out[idx] = 1;
        fired[idx] = 1;
    } else {
        spikes_out[idx] = 0;
    }
}

// ═══════════════════════════════════════════════════════════
//  Population Rate Decoding
// ═══════════════════════════════════════════════════════════
//
// rate[i] = count(spikes[t*n + i] != 0 for t in 0..window) / window

extern "C" __global__ void population_rate_decode_kernel(
    float* rates_out, const int* spikes,
    int n_neurons, int window
) {
    int i = blockIdx.x;
    if (i >= n_neurons) return;

    int count = 0;
    for (int t = threadIdx.x; t < window; t += blockDim.x) {
        if (spikes[t * n_neurons + i] != 0) count++;
    }

    extern __shared__ int scount[];
    scount[threadIdx.x] = count;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) scount[threadIdx.x] += scount[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rates_out[i] = (float)scount[0] / (float)window;
    }
}

// ═══════════════════════════════════════════════════════════
//  Host wrappers (single-kernel API)
// ═══════════════════════════════════════════════════════════

#define LAUNCH_SNN(name, ...) do { \
    int bs = 256, gs = (n + bs - 1) / bs; \
    name##_kernel<<<gs, bs>>>(__VA_ARGS__, n); \
    cudaDeviceSynchronize(); \
} while(0)

#define LAUNCH_SNN_1D(name, n, ...) do { \
    int bs = 256, gs = ((n) + bs - 1) / bs; \
    name##_kernel<<<gs, bs>>>(__VA_ARGS__); \
    cudaDeviceSynchronize(); \
} while(0)

extern "C" int cuda_lif_at_step(
    float* v, float* theta, float* I, int* spikes,
    const float* params, int n
) {
    LAUNCH_SNN(lif_at_step, v, theta, I, spikes, params);
    return 0;
}

extern "C" int cuda_intrinsic_plasticity(
    float* theta, const int* spikes,
    float lr_ip, float target_rate, float decay, int n
) {
    LAUNCH_SNN(intrinsic_plasticity, theta, spikes, lr_ip, target_rate, decay);
    return 0;
}

extern "C" int cuda_synaptic_normalization(
    float* weights, int n_pre, int n_post,
    float target_sum, float eps
) {
    int smem = 256 * sizeof(float);
    synaptic_normalization_kernel<<<n_post, 256, smem>>>(
        weights, n_pre, n_post, target_sum, eps);
    cudaDeviceSynchronize();
    return 0;
}

extern "C" int cuda_bsa_encode(
    int* spikes_out, const float* signal,
    const float* reconstructed, float threshold, int n
) {
    LAUNCH_SNN(bsa_encode, spikes_out, signal, reconstructed, threshold);
    return 0;
}

extern "C" int cuda_phase_encode(
    int* spikes_out, float* phases, const float* cos_phases,
    const float* signal, float threshold, float gain, int n
) {
    LAUNCH_SNN(phase_encode, spikes_out, phases, cos_phases, signal, threshold, gain);
    return 0;
}

extern "C" int cuda_ttfs_init(
    int* fire_times, const float* values, int max_delay, int n
) {
    LAUNCH_SNN(ttfs_init, fire_times, values, max_delay);
    return 0;
}

extern "C" int cuda_ttfs_step(
    int* spikes_out, const int* fire_times,
    int* fired, int current_step, int n
) {
    LAUNCH_SNN(ttfs_step, spikes_out, fire_times, fired, current_step);
    return 0;
}

extern "C" int cuda_population_rate_decode(
    float* rates_out, const int* spikes,
    int n_neurons, int window
) {
    int smem = 256 * sizeof(int);
    population_rate_decode_kernel<<<n_neurons, 256, smem>>>(
        rates_out, spikes, n_neurons, window);
    cudaDeviceSynchronize();
    return 0;
}

// ═══════════════════════════════════════════════════════════
//  Sync wrappers — gpu_context upload + kernel + sync pattern
// ═══════════════════════════════════════════════════════════

#include "gpu_context.h"

// LIF-AT via GPU context: upload params to d_lif_params, launch kernel, sync
// NOTE: No async variant because params must be on device first.
// Use cuda_lif_at_step() directly for host-managed params.
extern "C" int gpu_lif_at_step(void* vctx, const float* host_params, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_v, nn*sizeof(float));
    ENSURE_ALLOC(d_adex_w, nn*sizeof(float));
    ENSURE_ALLOC(d_I, nn*sizeof(float));
    ENSURE_ALLOC(d_spikes, nn*sizeof(int));
    ENSURE_ALLOC(d_lif_params, (1+6*nn)*sizeof(float));

    int n_params = 1 + 6 * n;
    cudaError_t e = cudaMemcpyAsync(ctx->d_lif_params, host_params,
        (size_t)n_params * sizeof(float), cudaMemcpyHostToDevice, ctx->stream);
    if (e) return (int)e;

    int bs = 256;
    lif_at_step_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_v, ctx->d_adex_w, ctx->d_I, ctx->d_spikes,
        ctx->d_lif_params, n);
    e = cudaGetLastError();
    if (e) return (int)e;

    return (int)cudaStreamSynchronize(ctx->stream);
}

extern "C" int gpu_intrinsic_plasticity(void* vctx,
    float lr_ip, float target_rate, float decay) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_adex_w, nn*sizeof(float));
    ENSURE_ALLOC(d_spikes, nn*sizeof(int));
    int bs = 256, n = nn;
    intrinsic_plasticity_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(
        ctx->d_adex_w, ctx->d_spikes, lr_ip, target_rate, decay, n);
    cudaError_t e = cudaGetLastError();
    if (e) return (int)e;
    return (int)cudaStreamSynchronize(ctx->stream);
}
