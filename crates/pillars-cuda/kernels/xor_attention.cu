/**
 * XOR Attention CUDA — persistent memory API.
 *
 * Formula per key:
 *   dist = popcnt(query XOR key[i])
 *   sim  = max(0, 1 - dist / (n_bits))
 *   out  = Σ sim_i × value_i
 *
 * API: init → compute (×N) → destroy
 * Device memory allocated once, reused across calls.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ─── Helper ──────────────────────────────────────────────

static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// ─── Device kernel (warp-level reduction) ────────────────
//
// Вместо shared memory reduction с log2(N) барьерами __syncthreads,
// используем warp shuffle (__shfl_xor_sync) — регистровые операции,
// без shared memory, без барьеров внутри варпа.
//
// Алгоритм:
//   1. Каждый тред вычисляет свой вклад (sim × value)
//   2. Warp-level reduction: __shfl_xor_sync × 5 (32→16→8→4→2→1)
//   3. Каждый warp пишет свой результат в warp_sums[warp_id]
//   4. Единственный __syncthreads
//   5. Первый warp редуцирует warp_sums финальным shuffle
//
// Итого: 1 __syncthreads вместо log2(block_size), 0 динамической shared memory.

__device__ __forceinline__ int popcnt64(uint64_t x) { return __popcll(x); }

extern "C" __global__ void xor_attention_kernel(
    const uint8_t* __restrict__ query,
    const uint8_t* __restrict__ keys,
    const float*   __restrict__ values,
    float*         __restrict__ output,
    int n_keys, int n_bytes
) {
    // ─── 1. Каждый тред считает свой вклад ────────────
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < n_keys) {
        int n_words = n_bytes / 8;
        const uint64_t* qw = (const uint64_t*)query;
        const uint64_t* kw = (const uint64_t*)keys + (uint64_t)idx * n_words;
        int dist = 0;
        #pragma unroll 4
        for (int w = 0; w < n_words; w++) dist += popcnt64(qw[w] ^ kw[w]);
        int nb = n_bytes * 8;
        float sim = (dist > nb) ? 0.0f : 1.0f - (float)dist / (float)nb;
        val = fmaxf(0.0f, sim) * values[idx];
    }

    // ─── 2. Warp-level reduction (регистры, 0 shared memory) ─
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

    // ─── 3. Cross-warp: каждый warp пишет в shared memory ──
    __shared__ float warp_sums[32]; // макс 1024/32 = 32 warp'ов
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads(); // единственный барьер

    // ─── 4. Первый warp редуцирует warp_sums ─────────────
    if (warp_id == 0) {
        float fval = (threadIdx.x < (blockDim.x + 31) / 32)
            ? warp_sums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            fval += __shfl_xor_sync(0xFFFFFFFF, fval, offset);
        if (threadIdx.x == 0) atomicAdd(output, fval);
    }
}

// ─── Opaque context struct ───────────────────────────────

typedef struct {
    uint8_t* d_query;
    uint8_t* d_keys;
    float*   d_values;
    float*   d_output;
    int n_keys;
    int n_bytes;
} XorCtx;

extern "C" void* xor_attention_create(int n_keys, int n_bytes) {
    XorCtx* ctx = (XorCtx*)calloc(1, sizeof(XorCtx));
    if (!ctx) return NULL;
    ctx->n_keys = n_keys;
    ctx->n_bytes = n_bytes;

    cudaError_t e;
    e = cudaMalloc(&ctx->d_query, (size_t)n_bytes);
    if (e) { fprintf(stderr,"query alloc: %s\n",cudaGetErrorString(e)); free(ctx); return NULL; }
    e = cudaMalloc(&ctx->d_keys, (size_t)n_keys * (size_t)n_bytes);
    if (e) { fprintf(stderr,"keys alloc: %s\n",cudaGetErrorString(e)); cudaFree(ctx->d_query); free(ctx); return NULL; }
    e = cudaMalloc(&ctx->d_values, (size_t)n_keys * sizeof(float));
    if (e) { fprintf(stderr,"values alloc: %s\n",cudaGetErrorString(e)); cudaFree(ctx->d_query); cudaFree(ctx->d_keys); free(ctx); return NULL; }
    e = cudaMalloc(&ctx->d_output, sizeof(float));
    if (e) { fprintf(stderr,"output alloc: %s\n",cudaGetErrorString(e)); cudaFree(ctx->d_query); cudaFree(ctx->d_keys); cudaFree(ctx->d_values); free(ctx); return NULL; }

    return ctx;
}

#define H2D(dst, src, size) do {                                    \
    cudaError_t _e = cudaMemcpy((dst), (src), (size), cudaMemcpyHostToDevice); \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "H2D " #dst ": %s\n", cudaGetErrorString(_e));         \
        return (int)_e;                                               \
    }                                                                 \
} while(0)

extern "C" int xor_attention_compute(
    void* vctx,
    const uint8_t* query,
    const uint8_t* keys,
    const float* values,
    float* output
) {
    XorCtx* ctx = (XorCtx*)vctx;
    int n_keys = ctx->n_keys;
    int n_bytes = ctx->n_bytes;

    // Copy inputs to device (with error checks)
    H2D(ctx->d_query,  query,  (size_t)n_bytes);
    H2D(ctx->d_keys,   keys,   (size_t)n_keys * (size_t)n_bytes);
    H2D(ctx->d_values, values, (size_t)n_keys * sizeof(float));

    // Zero output
    cudaError_t e = cudaMemset(ctx->d_output, 0, sizeof(float));
    if (e != cudaSuccess) { fprintf(stderr, "memset: %s\n", cudaGetErrorString(e)); return (int)e; }

    // Launch kernel (0 dynamic shared memory — используем warp shuffle)
    int block_size = (n_keys <= 1024) ? next_pow2(n_keys) : 256;
    int grid_size = (n_keys + block_size - 1) / block_size;

    xor_attention_kernel<<<grid_size, block_size, 0>>>(
        ctx->d_query, ctx->d_keys, ctx->d_values, ctx->d_output, n_keys, n_bytes
    );

    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(e)); return (int)e; }

    // Copy result back
    e = cudaMemcpy(output, ctx->d_output, sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { fprintf(stderr, "D2H: %s\n", cudaGetErrorString(e)); return (int)e; }

    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "sync: %s\n", cudaGetErrorString(e)); return (int)e; }
    return 0;
}

extern "C" void xor_attention_destroy(void* vctx) {
    if (!vctx) return;
    XorCtx* ctx = (XorCtx*)vctx;
    if (ctx->d_query)  cudaFree(ctx->d_query);
    if (ctx->d_keys)   cudaFree(ctx->d_keys);
    if (ctx->d_values) cudaFree(ctx->d_values);
    if (ctx->d_output) cudaFree(ctx->d_output);
    free(ctx);
}

// ─── Legacy single-call wrapper (kept for compatibility) ──

extern "C" int cuda_xor_attention(
    const uint8_t* query, const uint8_t* keys, const float* values,
    float* output, int n_keys, int n_bytes
) {
    void* ctx = xor_attention_create(n_keys, n_bytes);
    if (!ctx) return 1;
    int err = xor_attention_compute(ctx, query, keys, values, output);
    xor_attention_destroy(ctx);
    return err;
}

// ─── Device queries ────────────────────────────────────

extern "C" int cuda_get_device_count(int* count) {
    return (int)cudaGetDeviceCount(count);
}
extern "C" int cuda_set_device(int device) {
    return (int)cudaSetDevice(device);
}
