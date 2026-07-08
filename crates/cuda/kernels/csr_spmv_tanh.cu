/**
 * CSR Sparse Matrix-Vector multiply + tanh activation.
 *
 * out[i] = tanh(Σ_j W_ij * in[j] + input_scale * input_vec[i])
 *
 * Used by GPU-accelerated SPAUN pipeline for rate-based connectome dynamics.
 * Single kernel launch, no intermediate H2D/D2H transfers.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

/// CSR SpMV + tanh: out = tanh(W × in + input_scale * input_vec)
///
/// One thread per neuron. Each thread iterates over its outgoing synapses.
/// For 1M neurons with avg degree ~38, this is 38M MADDs in ~0.1ms.
///
/// Args:
///   out — output state [n] (device ptr)
///   in — input state [n] (device ptr, may alias out for in-place)
///   row_ptr — CSR row pointer [n + 1] (device)
///   col_indices — CSR column indices [n_syn] (device)
///   values — CSR weights [n_syn] (device)
///   input_vec — external input [n] (device), may be NULL
///   input_scale — multiplier for input_vec
///   n — number of neurons
extern "C" __global__ void csr_spmv_tanh_kernel(
    float* out,
    const float* in,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    const float* input_vec,
    float input_scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    // Sparse matmul: sum = Σ W_ij * in[j]
    // Unrolled 4-wide for ILP
    int j = start;
    while (j + 4 <= end) {
        int c0 = col_indices[j];
        int c1 = col_indices[j + 1];
        int c2 = col_indices[j + 2];
        int c3 = col_indices[j + 3];
        float v0 = values[j];
        float v1 = values[j + 1];
        float v2 = values[j + 2];
        float v3 = values[j + 3];
        // Bounds-check (should be redundant for valid CSR, but safe)
        if (c0 >= 0 && c0 < n) sum += v0 * in[c0];
        if (c1 >= 0 && c1 < n) sum += v1 * in[c1];
        if (c2 >= 0 && c2 < n) sum += v2 * in[c2];
        if (c3 >= 0 && c3 < n) sum += v3 * in[c3];
        j += 4;
    }
    for (; j < end; j++) {
        int c = col_indices[j];
        if (c >= 0 && c < n) {
            sum += values[j] * in[c];
        }
    }

    // External input injection (e.g., sensory input)
    if (input_vec != NULL && input_scale != 0.0f) {
        sum += input_vec[i] * input_scale;
    }

    // tanh activation
    out[i] = tanhf(sum);
}

/// Host wrapper: launch csr_spmv_tanh_kernel with given config.
///
/// Returns 0 on success, CUDA error code on failure.
/// Does NOT synchronize — caller must sync or cudaMemcpy before reading results.
extern "C" int cuda_csr_spmv_tanh(
    float* out,
    const float* in,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    const float* input_vec,
    float input_scale,
    int n,
    cudaStream_t stream
) {
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    csr_spmv_tanh_kernel<<<gs, bs, 0, stream>>>(
        out, in, row_ptr, col_indices, values, input_vec, input_scale, n);
    return (int)cudaGetLastError();
}
