/**
 * Top-k Select — GPU per-row top-k selection from EdgePredictor logits.
 *
 * Pipeline for each neuron i:
 *   1. Load logits[i * N .. (i+1) * N]
 *   2. Create (j, logit) pairs, skip self-edge (j == i)
 *   3. Selection sort for top (k * topk_mult) candidates
 *   4. Filter by threshold (sigmoid(logit) > threshold)
 *   5. Take first k candidates
 *   6. Write count and selected targets to output (scratch buffer)
 *
 * Each row is processed by ONE thread (simple approach for N <= 4096).
 * N blocks x 1 thread per block.
 *
 * SCRATCH BUFFER LAYOUT: fixed stride max_k to avoid data races.
 *   neuron i: scratch[i * max_k .. i * max_k + edge_count[i]]
 */

#include <cuda_runtime.h>
#include <math.h>

/**
 * Sigmoid helper.
 */
__device__ inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

#define MAX_CANDIDATES 768

struct Candidate {
    int j;
    float val;
};

/**
 * Insert-sorted array: maintains descending order by val.
 */
__device__ inline void insert_candidate(Candidate* buf, int* count, int max_count, int j, float val) {
    if (*count < max_count) {
        int pos = (*count)++;
        while (pos > 0 && buf[pos - 1].val < val) {
            buf[pos] = buf[pos - 1];
            pos--;
        }
        buf[pos].j = j;
        buf[pos].val = val;
        return;
    }
    if (val > buf[max_count - 1].val) {
        int pos = max_count - 1;
        while (pos > 0 && buf[pos - 1].val < val) {
            buf[pos] = buf[pos - 1];
            pos--;
        }
        buf[pos].j = j;
        buf[pos].val = val;
    }
}

/**
 * Top-k selection per row.
 *
 * Writes to d_scratch with FIXED stride max_k (no data races between rows).
 * d_scratch layout: [N x max_k], row i at [i * max_k .. i * max_k + edge_count[i]]
 */
extern "C" __global__ void topk_select_kernel(
    const float* __restrict__ d_logits,
    const int* __restrict__ d_target_degs,
    int* __restrict__ d_edge_counts,
    int* __restrict__ d_scratch,
    int N,
    float threshold,
    int topk_mult,
    int max_k)
{
    int i = blockIdx.x;
    if (i >= N) return;

    int k = d_target_degs[i];
    if (k <= 0) {
        d_edge_counts[i] = 0;
        return;
    }
    k = min(k, N - 1);

    int max_candidates = min(k * topk_mult, MAX_CANDIDATES);
    int base = i * N;

    Candidate candidates[MAX_CANDIDATES];
    int count = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float logit = d_logits[base + j];
        float prob = sigmoid_f(logit);
        if (prob <= threshold) continue;
        insert_candidate(candidates, &count, max_candidates, j, logit);
    }

    int out_count = min(count, k);
    d_edge_counts[i] = out_count;

    // Write to scratch with FIXED stride max_k (no overlap between rows)
    int scratch_base = i * max_k;
    for (int c = 0; c < out_count; c++) {
        d_scratch[scratch_base + c] = candidates[c].j;
    }
}

/**
 * Second pass: build CSR from scratch buffer.
 * Called after CPU computes row_ptr from d_edge_counts (exclusive prefix sum).
 *
 * @param d_edge_counts   [N] edge count per neuron
 * @param d_row_ptr       [N+1] CPU-computed prefix sum
 * @param d_scratch       [N x max_k] scratch buffer from topk pass
 * @param d_col_indices   [total_edges] output CSR col_indices
 * @param d_values        [total_edges] output CSR values (filled with 1.0 placeholder)
 * @param max_k           fixed stride in d_scratch
 * @param N               number of neurons
 */
extern "C" __global__ void csr_build_kernel(
    const int* __restrict__ d_edge_counts,
    const int* __restrict__ d_row_ptr,
    const int* __restrict__ d_scratch,
    int* __restrict__ d_col_indices,
    float* __restrict__ d_values,
    int max_k,
    int N)
{
    int i = blockIdx.x;
    if (i >= N) return;

    int cnt = d_edge_counts[i];
    int start = d_row_ptr[i];
    int scratch_base = i * max_k;

    for (int c = 0; c < cnt; c++) {
        d_col_indices[start + c] = d_scratch[scratch_base + c];
        d_values[start + c] = 1.0f;
    }
}

// --- Host wrapper: top-k select (pass 1) ------------------

extern "C" int cuda_topk_select(
    const float* d_logits,
    const int* d_target_degs,
    int* d_edge_counts,
    int* d_scratch,
    int N,
    float threshold,
    int topk_mult,
    int max_k)
{
    topk_select_kernel<<<N, 1>>>(
        d_logits, d_target_degs, d_edge_counts, d_scratch,
        N, threshold, topk_mult, max_k);

    cudaDeviceSynchronize();
    return 0;
}

// --- Host wrapper: CSR build (pass 2) --------------------

extern "C" int cuda_csr_build(
    const int* d_edge_counts,
    const int* d_row_ptr,
    const int* d_scratch,
    int* d_col_indices,
    float* d_values,
    int max_k,
    int N)
{
    csr_build_kernel<<<N, 1>>>(
        d_edge_counts, d_row_ptr, d_scratch, d_col_indices, d_values,
        max_k, N);

    cudaDeviceSynchronize();
    return 0;
}
