/**
 * Dopamine-gated STDP on CSR connectomes + trace update from rate activity.
 *
 * Two kernels:
 *
 * 1. `update_trace_kernel` — updates per-neuron traces from rate activity:
 *      trace[i] = decay * trace[i] + abs(state[i])
 *    (rate-based approximation of spike-triggered trace)
 *
 * 2. `csr_stdp_dopamine_kernel` — dopamine-gated STDP:
 *      w_ij += lr · δ · (trace[pre] - trace[post])
 *      w = sign-preserving clip(w, w_min, w_max)
 *
 * Both kernels are independent of Izhikevich dynamics — they work on
 * any continuous rate representation (tanh activations, Kuramoto, etc.).
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ─── Trace update from rate activity ─────────────────────
//
// Updates per-neuron eligibility traces using absolute rate activity:
//   trace[i] = trace[i] * decay + abs(rate[i])
//
// This is a rate-based approximation of spike-triggered traces.
// For spiking neurons, use the izhikevich kernel's trace update instead.

extern "C" __global__ void update_trace_kernel(
    float* trace,
    const float* rate,
    float decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    trace[idx] = trace[idx] * decay + fabsf(rate[idx]);
}

// ─── Dopamine-gated CSR STDP ─────────────────────────────
//
// For each pre-neuron, iterate over its outgoing synapses:
//   w += lr · δ · (trace[pre] - trace[post])
//
// δ = dopamine signal (from reward / contrastive learning)
// Positive δ → strengthen pre-post correlated synapses
// Negative δ → weaken pre-post correlated synapses
//
// One block per pre-neuron. Uses shared memory for trace[pre] broadcast.

extern "C" __global__ void csr_stdp_dopamine_kernel(
    float* weights,          // [n_synapses] — plastic weights, updated in-place
    const int* row_ptr,      // [n_neurons + 1] — CSR row pointers
    const int* col_indices,  // [n_synapses] — CSR column indices
    const float* trace,      // [n_neurons] — per-neuron eligibility trace
    float lr,
    float delta,             // dopamine signal δ
    float w_min,
    float w_max,
    int n_neurons
) {
    int pre = blockIdx.x;
    if (pre >= n_neurons) return;

    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    float tp = trace[pre];  // pre-synaptic trace

    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        int post = col_indices[k];
        if (post < 0 || post >= n_neurons) continue;

        float tn = trace[post];  // post-synaptic trace
        float w = weights[k];
        w += lr * delta * (tp - tn);

        // Sign-preserving clip
        if (w >= 0.0f) {
            w = fminf(w_max, fmaxf(w_min, w));
        } else {
            w = fminf(w_max, fmaxf(-w_max, w));
            // Don't let sign flip — keep negative
            if (w > 0.0f) w = -w_min;
        }

        weights[k] = w;
    }
}

// ─── Simple scalar eligibility trace (per-synapse, CPE-style) ──
//
// For each spiking pre-neuron, add ELIG_ADD to all outgoing synapses,
// then decay the entire trace buffer by ELIG_DECAY.
//
// This is CPE-style: per-synapse eligibility (not per-neuron).

extern "C" __global__ void update_eligibility_kernel(
    float* elig,             // [n_synapses] — per-synapse eligibility trace
    const int* spikes,       // [n_neurons] — binary spike flags (0/1)
    const int* row_ptr,      // [n_neurons + 1] — CSR row pointers
    const int* col_indices,  // [n_synapses] — column indices
    float elig_add,
    float elig_decay,
    int n_neurons
) {
    int pre = blockIdx.x;
    if (pre >= n_neurons) return;

    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    int spiked = (spikes != NULL && pre < n_neurons) ? spikes[pre] : 0;

    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        float e = elig[k] * elig_decay;
        if (spiked) {
            e += elig_add;
        }
        elig[k] = e;
    }
}

/// Apply dopamine-gated STDP with per-synapse eligibility:
///   w[i] += lr · δ · elig[i] · gate
///   w = sign-preserving clip(w, w_min, w_max)
///
/// This is the CPE-style update: weight change = f(δ, eligibility, gate).
extern "C" __global__ void csr_elig_stdp_kernel(
    float* weights,          // [n_synapses] — plastic weights
    const float* elig,       // [n_synapses] — per-synapse eligibility trace
    float lr,
    float delta,             // dopamine δ
    float gate,
    float w_min,
    float w_max,
    int n_synapses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;

    float w = weights[idx];
    w += lr * delta * elig[idx] * gate;

    // Sign-preserving clip
    if (w >= 0.0f) {
        w = fminf(w_max, fmaxf(w_min, w));
    } else {
        w = fminf(w_max, fmaxf(-w_max, w));
        if (w > 0.0f) w = -w_min;
    }

    weights[idx] = w;
}

// ═════════════════════════════════════════════════════════
//  Host wrappers
// ═════════════════════════════════════════════════════════

extern "C" int cuda_update_trace(
    float* trace, const float* rate, float decay, int n
) {
    int bs = 256, gs = (n + bs - 1) / bs;
    update_trace_kernel<<<gs, bs>>>(trace, rate, decay, n);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

extern "C" int cuda_csr_stdp_dopamine(
    float* weights,
    const int* row_ptr, const int* col_indices,
    const float* trace,
    float lr, float delta, float w_min, float w_max,
    int n_neurons
) {
    int bs = 128;
    csr_stdp_dopamine_kernel<<<n_neurons, bs>>>(
        weights, row_ptr, col_indices, trace,
        lr, delta, w_min, w_max, n_neurons);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

extern "C" int cuda_update_eligibility(
    float* elig,
    const int* spikes,
    const int* row_ptr, const int* col_indices,
    float elig_add, float elig_decay,
    int n_neurons
) {
    int bs = 128;
    update_eligibility_kernel<<<n_neurons, bs>>>(
        elig, spikes, row_ptr, col_indices,
        elig_add, elig_decay, n_neurons);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

extern "C" int cuda_csr_elig_stdp(
    float* weights,
    const float* elig,
    float lr, float delta, float gate,
    float w_min, float w_max,
    int n_synapses
) {
    int bs = 256, gs = (n_synapses + bs - 1) / bs;
    csr_elig_stdp_kernel<<<gs, bs>>>(
        weights, elig, lr, delta, gate, w_min, w_max, n_synapses);
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}
