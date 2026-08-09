/**
 * Decoder training and signal processing CUDA kernels + host wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ─── Device kernels ──────────────────────────────────────

// NOTE: normalization divergence — this kernel normalizes per-lane
// (w += lr·err·si/(si²+eps)), while the canonical pillars NLMS
// (kernels/decode.rs::nlms_train) normalizes by the GLOBAL norm² of the
// whole state vector (mu = lr·err/(Σs²+eps), bias += mu). A correct port
// needs a device reduction for the norm; t3::CudaBackend uses the scalar
// canonical path for n < 1024, so this divergence only affects n ≥ 1024.
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
    // Canonical pillars formula (kernels/decode.rs::kalman_update):
    //   K = cov·s / (sig + 1e-8),  w += K·err,  cov -= K·s·cov (floored 1e-12)
    // where `s` is the states vector and `sig` the innovation variance
    // R + Σ(cov·s²). The previous body (`g = cov/(sig+cov+ε)`, `cov *= 1-g`)
    // ignored the states vector entirely — a different, inconsistent
    // algorithm that could even GROW covariance for negative states.
    float k = cov[idx] * s[idx] / (sig + 1e-8f);
    w[idx] += k * err;
    cov[idx] = fmaxf(cov[idx] - k * s[idx] * cov[idx], 1e-12f);
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
    // Host → device → host round-trip (w/bias/s are HOST buffers).
    int bs = 256, gs = (n+bs-1)/bs;
    float* d_w = NULL; float* d_bias = NULL; float* d_s = NULL;
    cudaError_t e = cudaMalloc(&d_w, (size_t)n * sizeof(float));
    if (e) return (int)e;
    e = cudaMalloc(&d_s, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_w); return (int)e; }
    e = cudaMalloc(&d_bias, sizeof(float));
    if (e) { cudaFree(d_w); cudaFree(d_s); return (int)e; }
    if ((e = cudaMemcpy(d_s, s, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_w, w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_bias, bias, sizeof(float), cudaMemcpyHostToDevice))) {
        cudaFree(d_w); cudaFree(d_s); cudaFree(d_bias); return (int)e;
    }
    nlms_train_kernel<<<gs, bs>>>(d_w, d_bias, d_s, err, lr, eps, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(w, d_w, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    if (!e) e = cudaMemcpy(bias, d_bias, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_s); cudaFree(d_bias);
    return (int)e;
}

extern "C" int cuda_kalman_update(float* w, float* cov, const float* s,
    float err, float sig, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    kalman_update_kernel<<<gs, bs>>>(w, cov, s, err, sig, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_echo_tap(float* out, const float* buf,
    int n_osc, int delay_len, int delay_idx) {
    // Host → device → host round-trip (out/buf are HOST buffers).
    size_t buf_bytes = (size_t)n_osc * (size_t)delay_len * sizeof(float);
    float* d_out = NULL; float* d_buf = NULL;
    cudaError_t e = cudaMalloc(&d_out, (size_t)n_osc * sizeof(float));
    if (e) return (int)e;
    e = cudaMalloc(&d_buf, buf_bytes);
    if (e) { cudaFree(d_out); return (int)e; }
    e = cudaMemcpy(d_buf, buf, buf_bytes, cudaMemcpyHostToDevice);
    if (e) { cudaFree(d_out); cudaFree(d_buf); return (int)e; }
    echo_tap_kernel<<<n_osc, 1>>>(d_out, d_buf, n_osc, delay_len, delay_idx);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(out, d_out, (size_t)n_osc * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out); cudaFree(d_buf);
    return (int)e;
}

extern "C" int cuda_ring_synthesis(float* out, const float* a,
    const float* tb, const float* w, int n) {
    // Host → device → host round-trip (out/a/tb/w are HOST buffers).
    int bs = 256, gs = (n+bs-1)/bs;
    float* d_out = NULL; float* d_a = NULL; float* d_tb = NULL; float* d_w = NULL;
    cudaError_t e = cudaMalloc(&d_out, (size_t)n * sizeof(float));
    if (e) return (int)e;
    e = cudaMalloc(&d_a, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_out); return (int)e; }
    e = cudaMalloc(&d_tb, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_out); cudaFree(d_a); return (int)e; }
    e = cudaMalloc(&d_w, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_out); cudaFree(d_a); cudaFree(d_tb); return (int)e; }
    if ((e = cudaMemcpy(d_a, a, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_tb, tb, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_w, w, (size_t)n * sizeof(float), cudaMemcpyHostToDevice))) {
        cudaFree(d_out); cudaFree(d_a); cudaFree(d_tb); cudaFree(d_w); return (int)e;
    }
    ring_synthesis_kernel<<<gs, bs>>>(d_out, d_a, d_tb, d_w, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(out, d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out); cudaFree(d_a); cudaFree(d_tb); cudaFree(d_w);
    return (int)e;
}
