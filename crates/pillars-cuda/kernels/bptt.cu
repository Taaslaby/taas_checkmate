/**
 * BPTT backward pass through Kuramoto dynamics — full GPU acceleration.
 *
 * ## Flow
 *
 * 1. **Loss gradient kernel** — compute dL/dθ from coherence + temporal loss
 * 2. **BPTT step kernel** — backward through one Kuramoto step:
 *    - Weight gradient: dL/dW_ij += dL/dθ_i(t+1) · dt · sin(θ_j - θ_i)
 *    - Diagonal backprop: dθ_i(t) += dL/dθ_i(t+1) · (1 - dt·ΣW_ik·cos(θ_k-θ_i))
 *    - Cross-term backprop: dθ_j(t) += dL/dθ_i(t+1) · dt·W_ij · cos(θ_j-θ_i)  [atomicAdd]
 *
 * ## Memory layout (trajectory on GPU)
 *
 * - phases_traj [n_steps × n_neurons] — row-major, step × neuron
 * - coupling_traj [n_steps × n_neurons] — row-major, coupling at each step
 * - dtheta_traj [n_steps × n_neurons] — computed from loss, then updated in-place during BPTT
 * - grad_weights [n_synapses] — accumulated weight gradients (one D2H at end)
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "gpu_context.h"

// ─── Forward declarations for functions called by gpu_bptt_epoch_async ───
// These are defined in other .cu files OR later in this file.
extern "C" int gpu_kuramoto_prepare_async(void* vctx);
extern "C" int gpu_kuramoto_csr_coupling_async(void* vctx);
extern "C" int gpu_kuramoto_update_async(void* vctx, float dt);
extern "C" int gpu_context_copy_phases_to_traj_async(void* vctx, float* d_traj, int step);
extern "C" int gpu_context_copy_coupling_to_traj_async(void* vctx, float* d_traj, int step);
extern "C" int gpu_bptt_loss_grad_async(void* vctx,
    float* d_phases_traj, float* d_coupling_traj,
    float* d_dtheta, int coherence_start,
    float coherence_weight, float temporal_scale, float dt,
    int n_steps);
extern "C" int gpu_bptt_step_async(void* vctx,
    float* d_grad_weights, float* d_dtheta,
    const float* d_dtheta_next, const float* d_phases,
    float dt, int n_synapses);

// ═══════════════════════════════════════════════════════════
//  Kernel 1: Coherence + Temporal loss gradients
// ═══════════════════════════════════════════════════════════
//
// Computes initial dtheta[t][i] from:
//
// Coherence: R = |Σ exp(i·θⱼ)| / N
//   dR/dθᵢ = (S·cosᵢ - C·sinᵢ) / (R·N²)
//   dL/dθᵢ += -coherence_weight · dR/dθᵢ
//
// Temporal: L = |dθ/dt - ω - coupling|²
//   dv = (θ_i(t+1) - θ_i(t))/dt - ω_i - coupling_i(t)
//   dL/dθ_i(t)   += temporal_scale · 2·dv · (-1/dt)
//   dL/dθ_i(t+1) += temporal_scale · 2·dv · (1/dt)
//
// Output: dtheta[t][i] — initial gradient w.r.t. each phase
//
// Block size: 256 threads. Each block handles one timestep.
// Grid: [n_steps, n_blocks_per_step]

/// Compute sin_sum[t] and cos_sum[t] (order parameter sums) for coherence gradient.
/// Also compute initial dtheta from temporal loss.
///
/// phases_traj [n_steps × n]: step-major trajectory
/// coupling_traj [n_steps × n]: coupling at each step
/// omegas [n]: natural frequencies
/// dtheta [n_steps × n]: OUTPUT — initialized with temporal + coherence gradient
/// coherence_start: first timestep to include in coherence
/// coherence_weight: weight of coherence loss (negative)
/// temporal_scale: temporal_weight / (n_steps * n) normalization
/// dt: timestep
/// n_neurons, n_steps
__global__ void bptt_loss_grad_kernel(
    const float* phases_traj,      // [n_steps × n]
    const float* coupling_traj,    // [n_steps × n]
    const float* omegas,           // [n]
    float* dtheta,                 // [n_steps × n] output
    int coherence_start,
    float coherence_weight,
    float temporal_scale,
    float dt,
    int n_neurons,
    int n_steps)
{
    // ─── Phase 1: Compute sin_sum[t], cos_sum[t] ───
    // Use shared memory to sum within each block
    extern __shared__ float shared[];
    float* sin_sums = shared;
    float* cos_sums = &sin_sums[blockDim.x];

    int tid = threadIdx.x;
    int step = blockIdx.x;
    int off = step * n_neurons;

    // Each thread accumulates sin/cos for its assigned neurons
    float local_s = 0.0f;
    float local_c = 0.0f;
    for (int i = tid; i < n_neurons; i += blockDim.x) {
        float th = phases_traj[off + i];
        local_s += __sinf(th);
        local_c += __cosf(th);
    }
    sin_sums[tid] = local_s;
    cos_sums[tid] = local_c;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sin_sums[tid] += sin_sums[tid + s];
            cos_sums[tid] += cos_sums[tid + s];
        }
        __syncthreads();
    }

    float sum_s = sin_sums[0];
    float sum_c = cos_sums[0];
    float r = __fsqrt_rn(sum_s * sum_s + sum_c * sum_c) / n_neurons;
    float r_inv_n = (r > 1e-10f) ? 1.0f / (r * n_neurons * n_neurons) : 0.0f;

    // ─── Phase 2: Compute dtheta from temporal + coherence ───
    // Also compute temporal gradient for next step
    float next_temporal_contrib = 0.0f;        if (step < n_steps - 1) {
        // Compute temporal gradient at step t
        int next_off = (step + 1) * n_neurons;
        for (int i = tid; i < n_neurons; i += blockDim.x) {
            float dtheta_dt = (phases_traj[next_off + i] - phases_traj[off + i]) / dt;
            float dv = dtheta_dt - omegas[i] - coupling_traj[off + i];

            // dL/dθ_i(t): accumulate from temporal loss
            float grad_t = temporal_scale * 2.0f * dv * (-1.0f / dt);

            // Coherence gradient (for steps >= coherence_start)
            float coherence_grad = 0.0f;
            if (step >= coherence_start && r > 1e-10f) {
                float th = phases_traj[off + i];
                float s_term = sum_s * __cosf(th);
                float c_term = sum_c * __sinf(th);
                coherence_grad = -coherence_weight * (s_term - c_term) * r_inv_n;
            }

            dtheta[off + i] = grad_t + coherence_grad;
        }
    } else {
        // Last step: only coherence gradient, no temporal
        if (step >= coherence_start && r > 1e-10f) {
            for (int i = tid; i < n_neurons; i += blockDim.x) {
                float th = phases_traj[off + i];
                float s_term = sum_s * __cosf(th);
                float c_term = sum_c * __sinf(th);
                dtheta[off + i] = -coherence_weight * (s_term - c_term) * r_inv_n;
            }
        } else {
            for (int i = tid; i < n_neurons; i += blockDim.x) {
                dtheta[off + i] = 0.0f;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Kernel 1b: Add temporal gradient to next step
// ═══════════════════════════════════════════════════════════
//
// The temporal gradient adds dL/dθ_i(t+1) from the term at step t.
// We process this as a separate kernel after loss_grad to avoid
// complex cross-step synchronization.
__global__ void bptt_add_temporal_grad_kernel(
    float* dtheta,                 // [n_steps × n]
    const float* phases_traj,      // [n_steps × n]
    const float* coupling_traj,    // [n_steps × n]
    const float* omegas,           // [n]
    float temporal_scale,
    float dt,
    int n_neurons,
    int n_steps)
{
    int tid = threadIdx.x;
    int step = blockIdx.x;
    if (step >= n_steps - 1) return;  // last step has no temporal gradient to add

    int off = step * n_neurons;
    int next_off = (step + 1) * n_neurons;

    for (int i = tid; i < n_neurons; i += blockDim.x) {
        float dtheta_dt = (phases_traj[next_off + i] - phases_traj[off + i]) / dt;
        float dv = dtheta_dt - omegas[i] - coupling_traj[off + i];
        float grad_t_next = temporal_scale * 2.0f * dv * (1.0f / dt);
        // atomicAdd in case multiple steps contribute to same dtheta entry
        // (they don't — each step writes to its own next_off, but it's safe)
        atomicAdd(&dtheta[next_off + i], grad_t_next);
    }
}

// ═══════════════════════════════════════════════════════════
//  Kernel 2: One BPTT step — weight gradients + backprop
// ═══════════════════════════════════════════════════════════
//
// For each neuron i (one block per neuron):
//
// g_next = dtheta_next[i]
//
// For each outgoing synapse j:
//   sin_diff = sin(θ_j - θ_i) = sin_j·cos_i - cos_j·sin_i
//   cos_diff = cos(θ_j - θ_i) = cos_j·cos_i + sin_j·sin_i
//
//   grad_weights[syn_idx] += g_next * dt * sin_diff        (1, unique index)
//   if j != i && |w| > 0:
//     atomicAdd(&dtheta[j], g_next * dt * w * cos_diff)    (2, cross-term)
//   d_coupling_i += w * cos_diff                           (3, accumulate)
//
// dtheta[i] += g_next * (1.0 - dt * d_coupling_i)          (4, diagonal)
//
__global__ void bptt_step_kernel(
    float* grad_weights,         // [n_synapses] accumulated weight gradients
    float* dtheta,               // [n] output: dL/dθ at this step (MUST be zeroed before)
    const float* dtheta_next,    // [n] input: dL/dθ at next step
    const float* phases,         // [n] phases at this step
    const int* row_ptr,          // [n+1] CSR row pointers
    const int* col_indices,      // [n_synapses] CSR column indices
    const float* values,         // [n_synapses] CSR weights
    float dt,
    int n_neurons,
    int n_synapses)
{
    int i = blockIdx.x;  // one block per neuron
    if (i >= n_neurons) return;

    float g_next = dtheta_next[i];
    if (fabsf(g_next) < 1e-12f) {
        // No gradient to propagate — just zero this step's dtheta
        if (threadIdx.x == 0) dtheta[i] = 0.0f;
        return;
    }

    float th_i = phases[i];
    float cos_i = __cosf(th_i);
    float sin_i = __sinf(th_i);

    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int n_syn = end - start;

    // Each thread handles a subset of synapses
    float d_coupling_i = 0.0f;
    float diag_grad = 0.0f;

    for (int idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
        // Bounds check
        if (idx >= n_synapses) continue;
        int j = col_indices[idx];
        if (j < 0 || j >= n_neurons) continue;
        float w = values[idx];
        if (fabsf(w) < 1e-20f) continue;

        float th_j = phases[j];
        float sin_j = __sinf(th_j);
        float cos_j = __cosf(th_j);

        // sin(θ_j - θ_i) = sin_j·cos_i - cos_j·sin_i
        float sin_diff = sin_j * cos_i - cos_j * sin_i;
        // cos(θ_j - θ_i) = cos_j·cos_i + sin_j·sin_i
        float cos_diff = cos_j * cos_i + sin_j * sin_i;

        // Weight gradient (per-synapse, unique index — no atomic needed)
        grad_weights[idx] += g_next * dt * sin_diff;

        // Accumulate diagonal: Σ W_ij · cos(θ_j - θ_i)
        d_coupling_i += w * cos_diff;

        // Cross-term: dθ_j(t) += g_next · dt · W_ij · cos(θ_j - θ_i)
        // needs atomicAdd because multiple neurons can project to j
        if (j != i && fabsf(w) > 1e-20f) {
            float cross = g_next * dt * w * cos_diff;
            atomicAdd(&dtheta[j], cross);
        }
    }

    // ── Diagonal backprop ──
    // dθ_i(t) += g_next · (1 - dt · Σ W_ik · cos(θ_k - θ_i))
    // MUST use atomicAdd: cross-terms from OTHER neurons projecting to i
    // may have already accumulated into dtheta[i] via atomicAdd above.
    if (threadIdx.x == 0) {
        atomicAdd(&dtheta[i], g_next * (1.0f - dt * d_coupling_i));
    }
}


// ═══════════════════════════════════════════════════════════
//  Kernel 3: Zero out dtheta array for a single step
// ═══════════════════════════════════════════════════════════
__global__ void bptt_zero_dtheta_kernel(
    float* dtheta,    // [n] dtheta at one step
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dtheta[i] = 0.0f;
}


// ═══════════════════════════════════════════════════════════
//  Kernel 4: Zero grad_weights buffer
// ═══════════════════════════════════════════════════════════
__global__ void zero_grad_weights_kernel(float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 0.0f;
}

// ═══════════════════════════════════════════════════════════
//  Kernel 5: Weight update — SGD + sparsity + homeostasis + clamp
// ═══════════════════════════════════════════════════════════
//
// w -= lr * (grad_weights + sparsity_scale * sign(w) + homeostasis_scale * sign(w))
// clamp(w, -weight_clamp, weight_clamp)
//
// Updates ctx->d_csr_values in-place. grad_weights is NOT cleared by this kernel.
__global__ void weight_update_kernel(
    float* values,             // [n_synapses] in-place: weights to update
    const float* grad_weights, // [n_synapses] accumulated BPTT gradients
    float lr,
    float sparsity_scale,      // sparsity_weight / n_synapses
    float homeostasis_scale,   // 2 * homeostasis_weight * (mean|w| - target) / n_synapses
    float weight_clamp,         // max abs weight (0 = no clamp)
    int n_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_synapses) return;

    float w = values[idx];
    float g = grad_weights[idx];

    // NaN guard (use finite check compatible with --use_fast_math)
    if (isinf(g) || isnan(g)) g = 0.0f;

    // Sparsity + homeostasis gradients (sign of weight)
    float sign_w = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
    g += sparsity_scale * sign_w + homeostasis_scale * sign_w;

    // SGD
    w -= lr * g;

    // Weight clamp
    if (weight_clamp > 0.0f) {
        w = fminf(weight_clamp, fmaxf(-weight_clamp, w));
    }

    values[idx] = w;
}

// ═══════════════════════════════════════════════════════════
//  Kernel 6: Gradient norm — L2 reduction (two-pass)
// ═══════════════════════════════════════════════════════════
//
// Pass 1 (many blocks): each block computes sum(grad_i²) for its slice,
// writes partial sum to d_partial[blockIdx.x].
//
// Pass 2 (single block): reduces d_partial → sqrt() → writes to d_norm[0].
//
// Kernel 7: Gradient clipping — scale if norm > max_norm
//
// grad[i] *= max_norm / norm  (if norm > max_norm && norm > 1e-10)

__global__ void grad_norm_sq_partial_kernel(
    const float* grad,       // [n]
    float* partial_sums,     // [gridDim.x], output per-block sum of squares
    int n)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bs = blockDim.x;
    int start = bid * bs;
    int end = min(start + bs, n);

    float local_sum = 0.0f;
    for (int i = start + tid; i < end; i += bs) {
        float g = grad[i];
        local_sum += g * g;
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Tree reduction
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[bid] = shared[0];
}

__global__ void grad_norm_final_kernel(
    const float* partial_sums,  // [n_partials]
    float* grad_norm,            // [1] output: sqrt(sum(g²))
    int n_partials)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < n_partials; i += bs) {
        local_sum += partial_sums[i];
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) grad_norm[0] = sqrtf(shared[0]);
}

__global__ void grad_clip_kernel(
    float* grad,              // [n] modified in-place
    const float* grad_norm,   // [1] pre-computed norm on GPU
    float max_norm,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float norm = grad_norm[0];
    if (norm > max_norm && norm > 1e-10f) {
        float scale = max_norm / norm;
        grad[i] *= scale;
    }
}

// ─── Async wrappers for weight ops ──────────────────────────

/// Zero out the grad_weights buffer on device.
extern "C" int gpu_zero_grad_weights_async(void* vctx, float* d_grad, int n) {
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    zero_grad_weights_kernel<<<gs, bs, 0, ctx->stream>>>(d_grad, n);
    return (int)cudaGetLastError();
}

/// Apply weight update: SGD + sparsity + homeostasis + clamp.
/// Updates ctx->d_csr_values in-place using the accumulated gradients.
extern "C" int gpu_weight_update_async(void* vctx,
    const float* d_grad_weights,
    float lr, float sparsity_scale, float homeostasis_scale,
    float weight_clamp)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int ns = ctx->n_csr_synapses;
    int bs = 256;
    int gs = (ns + bs - 1) / bs;
    weight_update_kernel<<<gs, bs, 0, ctx->stream>>>(
        ctx->d_csr_values, d_grad_weights,
        lr, sparsity_scale, homeostasis_scale, weight_clamp, ns);
    return (int)cudaGetLastError();
}

/// Compute gradient L2 norm via two-pass reduction (async).
///
/// Pass 1: grad_norm_sq_partial_kernel — each block writes sum(g²) to d_partial.
/// Pass 2: grad_norm_final_kernel — single block reduces partials → sqrt → d_norm.
///
/// All buffers must be device pointers.
/// d_partial must be at least ((n + 255) / 256) floats.
extern "C" int gpu_grad_norm_async(void* vctx,
    const float* d_grad, float* d_norm, float* d_partial, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    // Pass 1: partial sums
    grad_norm_sq_partial_kernel<<<gs, bs, bs * sizeof(float), ctx->stream>>>(
        d_grad, d_partial, n);
    // Pass 2: reduce partials → sqrt → d_norm[0]
    int bs2 = min(gs, 256);
    grad_norm_final_kernel<<<1, bs2, bs2 * sizeof(float), ctx->stream>>>(
        d_partial, d_norm, gs);
    return (int)cudaGetLastError();
}

/// Clip gradients by norm (async).
///
/// If d_norm[0] > max_norm, scales gradients: g *= max_norm / norm.
/// Reads d_norm from GPU memory — no D2H transfer needed.
extern "C" int gpu_grad_clip_async(void* vctx,
    float* d_grad, const float* d_norm, float max_norm, int n)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    grad_clip_kernel<<<gs, bs, 0, ctx->stream>>>(
        d_grad, d_norm, max_norm, n);
    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════════
//  Fused BPTT epoch — forward + loss + BPTT + clip as ONE async call
//  (allows CUDA Graph capture of the entire epoch)
// ═══════════════════════════════════════════════════════════
//
// Does:
//   1. Forward: copy_init → for each step: prepare → coupling → copy_coupling → update → copy_phases
//   2. Last coupling: prepare → coupling → copy_coupling
//   3. Loss grad: bptt_loss_grad (2 kernels)
//   4. Zero grad_weights
//   5. BPTT backward: for each step in reverse: bptt_step
//   6. Grad norm + clip (if grad_clip_norm > 0)
//
// All data stays on GPU — ctx->d_phases, ctx->d_coupling, ctx->d_omegas,
// ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values are used internally.
// Trajectory buffers are passed as explicit device pointers.
//
// Weight update NOT included — it needs per-epoch scalar params (homeostasis)
// and should remain outside the graph.

extern "C" int gpu_bptt_epoch_async(void* vctx,
    float* d_phases_traj,       // [n_steps × n]
    float* d_coupling_traj,     // [n_steps × n]
    float* d_dtheta_traj,       // [n_steps × n]
    float* d_grad_weights,      // [n_synapses]
    float* d_grad_norm,         // [1] or NULL if no clipping
    float* d_grad_norm_partial, // [(n_synapses+255)/256] or NULL
    int n_steps,
    float dt,
    int coherence_start,
    float coherence_weight,
    float temporal_weight,
    float grad_clip_norm)       // 0 = no clipping
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int n = ctx->n_neurons;
    int ns = ctx->n_csr_synapses;

    // ─── 1. Forward — initial phases → trajectory ───
    // Use existing D2D copy wrappers (handle offset arithmetic internally)
    gpu_context_copy_phases_to_traj_async(vctx, d_phases_traj, 0);

    for (int step = 0; step < n_steps - 1; step++) {
        gpu_kuramoto_prepare_async(vctx);
        gpu_kuramoto_csr_coupling_async(vctx);
        gpu_context_copy_coupling_to_traj_async(vctx, d_coupling_traj, step);
        gpu_kuramoto_update_async(vctx, dt);
        gpu_context_copy_phases_to_traj_async(vctx, d_phases_traj, step + 1);
    }

    // ─── 2. Last coupling ───
    gpu_kuramoto_prepare_async(vctx);
    gpu_kuramoto_csr_coupling_async(vctx);
    gpu_context_copy_coupling_to_traj_async(vctx, d_coupling_traj, n_steps - 1);

    // ─── 3. Loss gradient ───
    float temporal_scale = temporal_weight / ((float)((n_steps - 1) > 0 ? n_steps - 1 : 1) * n);
    gpu_bptt_loss_grad_async(vctx,
        d_phases_traj, d_coupling_traj, d_dtheta_traj,
        coherence_start, coherence_weight, temporal_scale, dt, n_steps);

    // ─── 4. Zero grad_weights ───
    gpu_zero_grad_weights_async(vctx, d_grad_weights, ns);

    // ─── 5. BPTT backward ───
    for (int bw_step = n_steps - 2; bw_step >= 0; bw_step--) {
        float* dtheta_step = d_dtheta_traj + bw_step * n;
        float* dtheta_next = d_dtheta_traj + (bw_step + 1) * n;
        float* phases_step = d_phases_traj + bw_step * n;
        gpu_bptt_step_async(vctx, d_grad_weights,
            dtheta_step, dtheta_next, phases_step, dt, ns);
    }

    // ─── 6. Gradient norm + clip (if enabled) ───
    if (grad_clip_norm > 0.0f && d_grad_norm != NULL && d_grad_norm_partial != NULL) {
        gpu_grad_norm_async(vctx, d_grad_weights, d_grad_norm, d_grad_norm_partial, ns);
        gpu_grad_clip_async(vctx, d_grad_weights, d_grad_norm, grad_clip_norm, ns);
    }

    return (int)cudaGetLastError();
}

// ═══════════════════════════════════════════════════════════
//  Host wrappers
// ═══════════════════════════════════════════════════════════

extern "C" int cuda_kuramoto_bptt_loss_grad(
    const float* phases_traj,
    const float* coupling_traj,
    const float* omegas,
    float* dtheta,
    int coherence_start,
    float coherence_weight,
    float temporal_scale,
    float dt,
    int n_neurons,
    int n_steps)
{
    int bs = 256;
    size_t shared_mem = 2 * bs * sizeof(float);

    // Kernel 1: coherence + temporal gradient
    bptt_loss_grad_kernel<<<n_steps, bs, shared_mem>>>(
        phases_traj, coupling_traj, omegas, dtheta,
        coherence_start, coherence_weight, temporal_scale, dt,
        n_neurons, n_steps);
    cudaDeviceSynchronize();

    // Kernel 1b: add temporal gradient to next step
    bptt_add_temporal_grad_kernel<<<n_steps, bs>>>(
        dtheta, phases_traj, coupling_traj, omegas,
        temporal_scale, dt, n_neurons, n_steps);
    cudaDeviceSynchronize();

    return 0;
}

extern "C" int cuda_kuramoto_bptt_step(
    float* grad_weights,
    float* dtheta,
    const float* dtheta_next,
    const float* phases,
    const int* row_ptr,
    const int* col_indices,
    const float* values,
    float dt,
    int n_neurons,
    int n_synapses)
{
    int bs = 256;

    // Zero out dtheta before accumulating cross-terms
    int gs = (n_neurons + bs - 1) / bs;
    bptt_zero_dtheta_kernel<<<gs, bs>>>(dtheta, n_neurons);
    cudaDeviceSynchronize();

    // BPTT step: one block per neuron
    bptt_step_kernel<<<n_neurons, bs>>>(
        grad_weights, dtheta, dtheta_next, phases,
        row_ptr, col_indices, values,
        dt, n_neurons, n_synapses);
    cudaDeviceSynchronize();

    return 0;
}


// ─── Async stream-based wrappers ──────────────────────────

/// Async version: compute loss gradient on GPU (no sync inside).
/// Stores sin_sum, cos_sum in dtheta_staging[0], dtheta_staging[1] per step.
extern "C" int gpu_bptt_loss_grad_async(void* vctx,
    float* d_phases_traj, float* d_coupling_traj,
    float* d_dtheta, int coherence_start,
    float coherence_weight, float temporal_scale, float dt,
    int n_steps)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int n = ctx->n_neurons;
    int bs = 256;
    size_t shared_mem = 2 * bs * sizeof(float);

    bptt_loss_grad_kernel<<<n_steps, bs, shared_mem, ctx->stream>>>(
        d_phases_traj, d_coupling_traj, ctx->d_omegas, d_dtheta,
        coherence_start, coherence_weight, temporal_scale, dt,
        n, n_steps);

    bptt_add_temporal_grad_kernel<<<n_steps, bs, 0, ctx->stream>>>(
        d_dtheta, d_phases_traj, d_coupling_traj, ctx->d_omegas,
        temporal_scale, dt, n, n_steps);

    return (int)cudaGetLastError();
}

/// Async BPTT step (no sync inside).
/// Uses ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values for connectivity.
/// Takes trajectory data (d_dtheta, d_phases) as explicit pointers.
///
/// NOTE: does NOT zero d_dtheta. Caller MUST ensure d_dtheta is properly initialized
/// (e.g., from bptt_loss_grad for the loss gradient, or from a previous BPTT step).
/// The bptt_step_kernel uses atomicAdd so loss gradient contributions are preserved.
extern "C" int gpu_bptt_step_async(void* vctx,
    float* d_grad_weights, float* d_dtheta,
    const float* d_dtheta_next, const float* d_phases,
    float dt, int n_synapses)
{
    GpuCtx* ctx = (GpuCtx*)vctx;
    int n = ctx->n_neurons;
    int bs = 256;

    // BPTT step — use ctx's CSR arrays internally
    // d_dtheta is NOT zeroed here — it already contains the loss gradient + backprop from later steps
    // The bptt_step_kernel atomicAdds to dtheta[i], preserving existing contributions.
    bptt_step_kernel<<<n, bs, 0, ctx->stream>>>(
        d_grad_weights, d_dtheta, d_dtheta_next, d_phases,
        ctx->d_row_ptr, ctx->d_col_indices, ctx->d_csr_values,
        dt, n, n_synapses);

    return (int)cudaGetLastError();
}
