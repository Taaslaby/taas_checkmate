/**
 * Interference, coherence, signal analysis CUDA kernels + host wrappers.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ─── Device kernels ──────────────────────────────────────

extern "C" __global__ void interference_kernel(
    float* out, const float* pa, const float* pb, int na, int nb) {
    int i = blockIdx.y, j = threadIdx.x;
    if (i >= na || j >= nb) return;
    out[i*nb + j] = 0.5f * (1.0f + cosf(pa[i] - pb[j]));
}

extern "C" __global__ void interference_mean_kernel(
    float* output, const float* pa, const float* pb, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? 0.5f * (1.0f + cosf(pa[i] - pb[i])) : 0.0f;
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[0] = sdata[0] / (float)n;
}

extern "C" __global__ void phase_coherence_kernel(
    float* out, const float* sin_buf, const float* cos_buf,
    const float* amps, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? amps[i] * (sin_buf[i] + cos_buf[i]) : 0.0f;
    sdata[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = sdata[0];
}

extern "C" __global__ void band_energy_kernel(
    float* be, const float* amps, const float* om,
    const float* edges, int nb, int n) {
    int b = blockIdx.x;
    if (b >= nb) return;
    float lo = edges[b], hi = edges[b+1], e = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (om[i] >= lo && om[i] < hi) e += amps[i] * amps[i];
    extern __shared__ float sd[];
    sd[threadIdx.x] = e; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) be[b] = sd[0];
}

extern "C" __global__ void winding_matrix_kernel(
    float* out, const float* pa, const float* pb, int na, int nb) {
    int i = blockIdx.y, j = threadIdx.x;
    if (i >= na || j >= nb) return;
    out[i*nb + j] = sinf(pa[i] - pb[j]);
}

// ─── Host wrappers ───────────────────────────────────────

extern "C" int cuda_interference(float* out, const float* pa, const float* pb, int na, int nb) {
    // Host → device → host round-trip (out/pa/pb are HOST buffers).
    size_t o_bytes = (size_t)na * (size_t)nb * sizeof(float);
    float* d_out = NULL; float* d_pa = NULL; float* d_pb = NULL;
    cudaError_t e = cudaMalloc(&d_out, o_bytes);
    if (e) return (int)e;
    e = cudaMalloc(&d_pa, (size_t)na * sizeof(float));
    if (e) { cudaFree(d_out); return (int)e; }
    e = cudaMalloc(&d_pb, (size_t)nb * sizeof(float));
    if (e) { cudaFree(d_out); cudaFree(d_pa); return (int)e; }
    if ((e = cudaMemcpy(d_pa, pa, (size_t)na * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_pb, pb, (size_t)nb * sizeof(float), cudaMemcpyHostToDevice))) {
        cudaFree(d_out); cudaFree(d_pa); cudaFree(d_pb); return (int)e;
    }
    dim3 grid(1, na);
    interference_kernel<<<grid, nb>>>(d_out, d_pa, d_pb, na, nb);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(out, d_out, o_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_out); cudaFree(d_pa); cudaFree(d_pb);
    return (int)e;
}

extern "C" int cuda_interference_mean(float* out, const float* pa, const float* pb, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    interference_mean_kernel<<<gs, bs, bs*sizeof(float)>>>(out, pa, pb, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_phase_coherence(float* out, const float* sb, const float* cb, const float* a, int n) {
    int bs = 256, gs = (n+bs-1)/bs;
    phase_coherence_kernel<<<gs, bs, bs*sizeof(float)>>>(out, sb, cb, a, n);
    cudaDeviceSynchronize(); return 0;
}

extern "C" int cuda_band_energy(float* be, const float* a, const float* om, const float* ed, int nb, int n) {
    // Host → device → host round-trip (be/a/om/ed are HOST buffers).
    float* d_be = NULL; float* d_a = NULL; float* d_om = NULL; float* d_ed = NULL;
    cudaError_t e = cudaMalloc(&d_be, (size_t)nb * sizeof(float));
    if (e) return (int)e;
    e = cudaMalloc(&d_a, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_be); return (int)e; }
    e = cudaMalloc(&d_om, (size_t)n * sizeof(float));
    if (e) { cudaFree(d_be); cudaFree(d_a); return (int)e; }
    e = cudaMalloc(&d_ed, (size_t)nb * sizeof(float));
    if (e) { cudaFree(d_be); cudaFree(d_a); cudaFree(d_om); return (int)e; }
    if ((e = cudaMemcpy(d_a, a, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_om, om, (size_t)n * sizeof(float), cudaMemcpyHostToDevice)) ||
        (e = cudaMemcpy(d_ed, ed, (size_t)nb * sizeof(float), cudaMemcpyHostToDevice))) {
        cudaFree(d_be); cudaFree(d_a); cudaFree(d_om); cudaFree(d_ed); return (int)e;
    }
    band_energy_kernel<<<nb, 256, 256*sizeof(float)>>>(d_be, d_a, d_om, d_ed, nb, n);
    e = cudaDeviceSynchronize();
    if (!e) e = cudaMemcpy(be, d_be, (size_t)nb * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_be); cudaFree(d_a); cudaFree(d_om); cudaFree(d_ed);
    return (int)e;
}

extern "C" int cuda_winding_matrix(float* out, const float* pa, const float* pb, int na, int nb) {
    dim3 grid(1, na);
    winding_matrix_kernel<<<grid, nb>>>(out, pa, pb, na, nb);
    cudaDeviceSynchronize(); return 0;
}
