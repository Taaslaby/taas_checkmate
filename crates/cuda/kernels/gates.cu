/**
 * Gate and I/O CUDA kernels + host-callable wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gpu_context.h"

// ─── Device kernels ──────────────────────────────────────

extern "C" __global__ void safety_clamp_kernel(float* data, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = fminf(max_val, fmaxf(min_val, data[idx]));
}

extern "C" __global__ void stochastic_encoder_kernel(
    float* target_I, const float* inputs, const float* noise,
    float threshold, float spike_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    target_I[idx] = (inputs[idx] + noise[idx] > threshold) ? spike_val : 0.0f;
}

extern "C" __global__ void thalamic_gate_kernel(
    float* target_I, const float* source_v, const float* gate_v,
    float threshold, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (gate_v[idx] > threshold) target_I[idx] = source_v[idx] * gain;
}

extern "C" __global__ void sigmoid_gate_kernel(
    float* output, const float* input, float gain, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = 1.0f / (1.0f + expf(-gain * (input[idx] - threshold)));
}

extern "C" __global__ void phase_gate_apply_kernel(
    float* output, const float* input, const float* r,
    float gain, float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float gate = 1.0f / (1.0f + expf(-gain * (r[idx] - threshold)));
    output[idx] = input[idx] * gate;
}

extern "C" __global__ void t3_drive_encoder_kernel(
    float* drives, const float* inputs, float gain, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    drives[idx] = gain * inputs[idx];
}

extern "C" __global__ void t3_inverse_kuramoto_kernel(
    float* drive_out, const float* r, const float* r_prev,
    float alpha, float gain, float eps, float decay, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Original InverseKuramoto formula from t3-reservoir:
    // Drive(t) = (ṙ - (α - r²)·r) / (g · (r + ε))
    // where ṙ = r - r_prev
    float r_now = r[idx];
    float r_dot = r_now - r_prev[idx];
    float denom = gain * (r_now + eps);
    float drive = 0.0f;
    if (denom > 1e-10f) {
        drive = (r_dot - (alpha - r_now * r_now) * r_now) / denom;
    }
    // Apply decay from previous drive (low-pass filter)
    drive = decay * drive_out[idx] + (1.0f - decay) * drive;
    drive_out[idx] = fminf(10.0f, fmaxf(-10.0f, drive));
}

extern "C" __global__ void t3_tact_generator_kernel(
    float* pulses, const float* ph1, const float* ph2, const float* ph3,
    float threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s = sinf(ph1[idx]) + sinf(ph2[idx]) + sinf(ph3[idx]);
    pulses[idx] = (s > threshold) ? s : 0.0f;
}

#include "gpu_context.h"

// ─── Host wrappers ───────────────────────────────────────

#define LAUNCH(name, ...) do { \
    int bs = 256, gs = (n + bs - 1) / bs; \
    name##_kernel<<<gs, bs>>>(__VA_ARGS__, n); \
    cudaDeviceSynchronize(); \
} while(0)

extern "C" int cuda_safety_clamp(float* data, float min_val, float max_val, int n) {
    LAUNCH(safety_clamp, data, min_val, max_val);
    return 0;
}

// ─── Async stream-based launches ──────────────────────────

extern "C" int gpu_safety_clamp_async(void* vctx, float min_val, float max_val) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int nn = ctx->n_neurons;
    ENSURE_ALLOC(d_v, nn*sizeof(float));
    int bs = 256, n = nn;
    safety_clamp_kernel<<<gs(n,bs), bs, 0, ctx->stream>>>(ctx->d_v, min_val, max_val, n);
    return (int)cudaGetLastError();
}
extern "C" int cuda_stochastic_encoder(float* ti, const float* in, const float* no, float th, float sv, int n) {
    LAUNCH(stochastic_encoder, ti, in, no, th, sv);
    return 0;
}
extern "C" int cuda_thalamic_gate(float* ti, const float* sv, const float* gv, float th, float g, int n) {
    LAUNCH(thalamic_gate, ti, sv, gv, th, g);
    return 0;
}
extern "C" int cuda_sigmoid_gate(float* out, const float* in, float g, float th, int n) {
    LAUNCH(sigmoid_gate, out, in, g, th);
    return 0;
}
extern "C" int cuda_phase_gate_apply(float* out, const float* in, const float* r, float g, float th, int n) {
    LAUNCH(phase_gate_apply, out, in, r, g, th);
    return 0;
}
extern "C" int cuda_t3_drive_encoder(float* dr, const float* in, float g, int n) {
    LAUNCH(t3_drive_encoder, dr, in, g);
    return 0;
}
extern "C" int cuda_t3_inverse_kuramoto(float* dout, const float* r, const float* rp, float a, float g, float e, float dec, int n) {
    LAUNCH(t3_inverse_kuramoto, dout, r, rp, a, g, e, dec);
    return 0;
}
extern "C" int cuda_t3_tact_generator(float* p, const float* p1, const float* p2, const float* p3, float th, int n) {
    LAUNCH(t3_tact_generator, p, p1, p2, p3, th);
    return 0;
}
