/**
 * Decoder training and signal processing CUDA kernels + host wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ─── Device kernels ──────────────────────────────────────

extern "C" __global__ void nlms_train_kernel(
    float* w, float* bias, const float* s, float err, float lr, float eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float si = s[idx];
    w[idx] += lr * err * si / (si*si + eps);
    if (idx == 0) *bias += lr * err;
}

extern "C" __global__ void kalman_update_kernel(
    float* w, float* cov, const float* s, float err, float sig, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = cov[idx] / (sig + cov[idx] + 1e-8f);
    w[idx] += g * err;
    cov[idx] *= (1.0f - g);
}

extern "C" __global__ void echo_tap_kernel(
    float* out, const float* buf, int n_osc, int delay_len, int delay_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_osc) return;
    out[idx] = buf[idx * delay_len + delay_idx];
}

extern "C" __global__ void ring_synthesis_kernel(
    float* out, const float* a, const float* tb, const float* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = w[idx] * a[idx] * tb[idx];
}

// ─── Host wrappers ───────────────────────────────────────

extern "C" int cuda_nlms_train(float* w, float* bias, const float* s,
    float err, float lr, float eps, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    nlms_train_kernel<<<gs, bs>>>(w, bias, s, err, lr, eps, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_kalman_update(float* w, float* cov, const float* s,
    float err, float sig, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    kalman_update_kernel<<<gs, bs>>>(w, cov, s, err, sig, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_echo_tap(float* out, const float* buf,
    int n_osc, int delay_len, int delay_idx) {
    echo_tap_kernel<<<n_osc, 1>>>(out, buf, n_osc, delay_len, delay_idx);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_ring_synthesis(float* out, const float* a,
    const float* tb, const float* w, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    ring_synthesis_kernel<<<gs, bs>>>(out, a, tb, w, n);
    cudaDeviceSynchronize(); return 0;
}
