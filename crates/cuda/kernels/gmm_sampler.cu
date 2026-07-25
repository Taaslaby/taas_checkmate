/**
 * GMM Sampler — GPU batch sampling from Gaussian Mixture Model.
 *
 * Kernel 1: `gmm_sample_kernel`
 *   N threads (one per sample).
 *   Each thread: pick component via cumulative weight binary search,
 *   then Box-Muller sample from N(mean[c], variance[c]).
 *
 * Uses per-thread XORShift32 RNG seeded by (global_seed + thread_id).
 * Writes states as flat [N × dim] f32 array.
 */

#include <cuda_runtime.h>
#include <math.h>

/**
 * XORShift32 — fast per-thread RNG.
 */
__device__ inline unsigned int xorshift32(unsigned int state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/**
 * Uniform f32 in (0, 1] from XORShift32 state.
 */
__device__ inline float randf(unsigned int *state) {
    *state = xorshift32(*state);
    // 23 bits of mantissa, shift → (0, 1]
    return ((*state & 0x007FFFFFu) | 0x00800000u) / 16777216.0f;
}

/**
 * Box-Muller: N(0, 1) from two uniforms.
 * Returns z1 (first Gaussian), stores z2 in *partner for next call.
 */
__device__ inline float box_muller(float u1, float u2, float *partner) {
    float r = sqrtf(fmaxf(1e-10f, -2.0f * logf(u1)));
    float theta = 2.0f * M_PI * u2;
    *partner = r * sinf(theta);
    return r * cosf(theta);
}

/**
 * Sample N states from GMM.
 *
 * @param d_weights    [n_components] component weights (must sum to ~1)
 * @param d_means      [n_components × dim] component means
 * @param d_covars     [n_components × dim] diagonal covariances
 * @param d_states     [N × dim] output states, flat row-major
 * @param d_cum_weights [n_components] prefix-sum of weights (pre-computed on host)
 * @param N            number of states to sample
 * @param dim          state dimension
 * @param n_components number of GMM components
 * @param seed         global RNG seed
 */
extern "C" __global__ void gmm_sample_kernel(
    const float* __restrict__ d_weights,
    const float* __restrict__ d_means,
    const float* __restrict__ d_covars,
    const float* __restrict__ d_cum_weights,
    float* __restrict__ d_states,
    int N,
    int dim,
    int n_components,
    unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Per-thread RNG state: seed by (global_seed, thread_idx)
    unsigned int rng_state = seed ^ (idx * 0x9E3779B9u);

    // ── Pick component by cumulative weight (binary search) ──
    float u = randf(&rng_state);
    int lo = 0, hi = n_components - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (d_cum_weights[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    int c = lo;

    // ── Sample state ──
    float partner = 0.0f;
    int base = c * dim;
    for (int d = 0; d < dim; d++) {
        float u1 = randf(&rng_state);
        float u2 = randf(&rng_state);
        float z = box_muller(u1, u2, &partner);
        float sigma = sqrtf(fmaxf(1e-10f, d_covars[base + d]));
        d_states[idx * dim + d] = d_means[base + d] + z * sigma;
    }
}

// ─── Host wrapper ──────────────────────────────────────────

extern "C" int cuda_gmm_sample(
    const float* d_weights,
    const float* d_means,
    const float* d_covars,
    const float* d_cum_weights,
    float* d_states,
    int N,
    int dim,
    int n_components,
    unsigned int seed)
{
    int bs = 256;
    int gs = (N + bs - 1) / bs;
    gs = min(gs, 65535);

    gmm_sample_kernel<<<gs, bs>>>(
        d_weights, d_means, d_covars, d_cum_weights,
        d_states, N, dim, n_components, seed);

    cudaDeviceSynchronize();
    return 0;
}
