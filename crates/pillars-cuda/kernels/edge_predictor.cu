/**
 * EdgePredictor CUDA kernel — MLP: (s_i, s_j) → logit
 *
 * Architecture: 3-layer MLP (8→128→128→1) with ReLU.
 * Launched as 1D grid covering all N×N pairs.
 *
 * Each thread processes one pair (i,j):
 *   1. Load s_i[0..3], s_j[0..3] from global memory
 *   2. Forward through 3 Linear layers with ReLU
 *   3. Write logit to output[i * N + j]
 *
 * Weights are read from global memory (cached in L1/L2).
 */
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void edge_predictor_kernel(
    const float* __restrict__ states,   // [N × state_dim]
    float* __restrict__ output,          // [N × N] logits
    const float* __restrict__ w1,        // [8 × hidden]
    const float* __restrict__ b1,        // [hidden]
    const float* __restrict__ w2,        // [hidden × hidden]
    const float* __restrict__ b2,        // [hidden]
    const float* __restrict__ w3,        // [hidden × 1]
    const float* __restrict__ b3,        // [1]
    int N,
    int state_dim,
    int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    // Self-edges: output -inf so they're never selected in top-k
    if (i == j) {
        output[idx] = -1e20f;
        return;
    }
    int sd = state_dim;
    int hd = hidden_dim;

    // ── Load s_i, s_j ──
    float si[4], sj[4];
    #pragma unroll
    for (int d = 0; d < sd; d++) {
        si[d] = states[i * sd + d];
        sj[d] = states[j * sd + d];
    }

    // ── Layer 1: 8 → hidden, ReLU ──
    float h1[128];  // hidden_dim = 128
    for (int k = 0; k < hd; k++) {
        float sum = b1[k];
        // s_i part: first 4 input dims
        for (int d = 0; d < sd; d++) {
            sum += si[d] * w1[d * hd + k];
        }
        // s_j part: next 4 input dims
        for (int d = 0; d < sd; d++) {
            sum += sj[d] * w1[(sd + d) * hd + k];
        }
        h1[k] = fmaxf(0.0f, sum);  // ReLU
    }

    // ── Layer 2: hidden → hidden, ReLU ──
    float h2[128];
    for (int k = 0; k < hd; k++) {
        float sum = b2[k];
        for (int d = 0; d < hd; d++) {
            sum += h1[d] * w2[d * hd + k];
        }
        h2[k] = fmaxf(0.0f, sum);
    }

    // ── Layer 3: hidden → 1 (linear, no activation) ──
    float logit = b3[0];
    for (int d = 0; d < hd; d++) {
        logit += h2[d] * w3[d];
    }

    output[idx] = logit;
}

// ─── Host wrapper (synchronous) ──────────────────────────

extern "C" int cuda_edge_predictor(
    const float* d_states,
    float* d_output,
    const float* d_w1, const float* d_b1,
    const float* d_w2, const float* d_b2,
    const float* d_w3, const float* d_b3,
    int N, int state_dim, int hidden_dim)
{
    int total = N * N;
    int bs = 256;
    int gs = (total + bs - 1) / bs;
    gs = min(gs, 65535);  // CUDA grid dim limit

    edge_predictor_kernel<<<gs, bs>>>(
        d_states, d_output,
        d_w1, d_b1, d_w2, d_b2, d_w3, d_b3,
        N, state_dim, hidden_dim);

    cudaDeviceSynchronize();
    return 0;
}
