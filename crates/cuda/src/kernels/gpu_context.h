#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include <cuda_runtime.h>

typedef struct {
    float* d_v; float* d_u; float* d_trace; int* d_spikes; float* d_I;
    float* d_a; float* d_b; float* d_c; float* d_d;
    float* d_lif_params;
    float* d_weights; float* d_traces_pre; float* d_traces_post;
    float* d_phases; float* d_omegas; float* d_coupling;
    float* d_cos_buf; float* d_sin_buf; float* d_phases_prev;
    float* d_gate_out;
    float* d_decoder_w; float* d_decoder_bias;
    float* d_decoder_cov; float* d_echo_buf;
    // CSR connectome arrays (for GPU-accelerated spike propagation)
    int* d_row_ptr;
    int* d_col_indices;
    float* d_csr_values;
    float* d_nt_types;
    float* d_external_I;
    int n_csr_synapses;
    float* d_refractory;
    // T3 (TriOscillator) arrays
    float* d_amplitudes;
    float* d_drives;
    float* d_t3_prev_amps;
    float* d_adex_w;
    // ψ-wave arrays
    float* d_psi_v;
    float* d_psi_u;
    float* d_psi_h_real;
    float* d_psi_h_imag;
    float* d_psi_drive;       // external drive (i_ext) for ψ-wave
    // Aberton arrays
    float* d_aber_real;
    float* d_aber_imag;
    int*   d_aber_collapsed;  // per-step collapse flags
    int n_neurons; int n_synapses; int n_decoders; int n_t3;
    cudaStream_t stream;
    cudaGraph_t graph;          // SNN capture graph
    cudaGraphExec_t graph_exec; // SNN replay instance
    int graph_captured;         // SNN graph flag
    cudaGraph_t psi_graph;          // ψ-wave CUDA Graph
    cudaGraphExec_t psi_graph_exec;
    int psi_graph_captured;
    float psi_graph_dt, psi_graph_hbar; // cached params for invalidation
} GpuCtx;

// Echo state delay buffer length (per neuron)
#define DELAY_LEN 256

static inline int gs(int n, int bs) { return (n + bs - 1) / bs; }

#endif // GPU_CONTEXT_H
