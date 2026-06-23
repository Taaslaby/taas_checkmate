//! CSR sparse operations: propagation, Hamiltonian, STDP.
//! Ported from `kuramoto_engine/c/engine_neuron.c`
//! Hamiltonian and STDP parallelized with rayon.

use rayon::prelude::*;

// ─── Safe CSR access helpers ─────────────────────────────

#[inline(always)]
pub fn bound_row(row_ptr: &[i32], idx: usize, n: usize) -> usize {
    if idx >= row_ptr.len() { return 0; }
    let r = row_ptr[idx];
    if r < 0 { 0 } else { r.min(n as i32) as usize }
}

#[inline(always)]
pub fn bound_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() { return 0; }
    let c = col_indices[idx];
    debug_assert!(c >= 0 && (c as usize) < n, "bound_col: invalid column {} at idx {}, n={}, col_indices.len={}", c, idx, n, col_indices.len());
    if c < 0 || c as usize >= n { panic!("bound_col: out of range column {} at idx {}, n={}", c, idx, n); }
    c as usize
}

// ─── Sparse Spike Propagation ────────────────────────────

/// Propagate spikes through a CSR synaptic matrix.
/// Sequential — fan-in on input_buffer (data race with rayon).
pub fn propagate_spikes(
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    spiked_indices: &[i32], nt_types: &[f32],
    input_buffer: &mut [f32], weight_scale: f32,
) {
    let n_neurons = row_ptr.len() - 1;
    let n_input = input_buffer.len();
    let max_syn = col_indices.len().min(values.len());

    for &spike_idx in spiked_indices.iter() {
        let i = spike_idx as usize;
        if i >= n_neurons { continue; }
        let mult = nt_types.get(i).copied().unwrap_or(1.0) * weight_scale;

        // Direct indexing like bench_connectome_cpu — row_ptr values are synapse indices
        let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);

        let mut j = start;
        while j + 4 <= end {
            let t0 = col_indices.get(j).copied().unwrap_or(0) as usize;
            let t1 = col_indices.get(j + 1).copied().unwrap_or(0) as usize;
            let t2 = col_indices.get(j + 2).copied().unwrap_or(0) as usize;
            let t3 = col_indices.get(j + 3).copied().unwrap_or(0) as usize;
            if t0 < n_input { input_buffer[t0] += values[j] * mult; }
            if t1 < n_input { input_buffer[t1] += values[j + 1] * mult; }
            if t2 < n_input { input_buffer[t2] += values[j + 2] * mult; }
            if t3 < n_input { input_buffer[t3] += values[j + 3] * mult; }
            j += 4;
        }
        for j in j..end {
            let col = col_indices.get(j).copied().unwrap_or(0) as usize;
            if col < n_input { input_buffer[col] += values[j] * mult; }
        }
    }
}

// ─── Hamiltonian ─────────────────────────────────────────

/// Compute ψ-wave Hamiltonian in parallel: H_real = W * u, H_imag = -W * v.
pub fn compute_hamiltonian(
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    v: &[f32], u: &[f32],
    h_real: &mut [f32], h_imag: &mut [f32],
) {
    let n = h_real.len().min(h_imag.len());
    let max_syn = col_indices.len().min(values.len());
    h_real.par_iter_mut().zip(h_imag.par_iter_mut()).enumerate().for_each(|(i, (hr, hi))| {
        // Direct row_ptr indexing — row_ptr values are SYNAPSE indices, not neuron indices
        let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);
        let mut hr_acc = 0.0f32;
        let mut hi_acc = 0.0f32;
        let mut j = start;
        while j + 4 <= end {
            let c0 = bound_col(col_indices, j, n);
            let c1 = bound_col(col_indices, j + 1, n);
            let c2 = bound_col(col_indices, j + 2, n);
            let c3 = bound_col(col_indices, j + 3, n);
            let w0 = values[j]; let w1 = values[j + 1]; let w2 = values[j + 2]; let w3 = values[j + 3];
            hr_acc += w0 * u[c0] + w1 * u[c1] + w2 * u[c2] + w3 * u[c3];
            hi_acc -= w0 * v[c0] + w1 * v[c1] + w2 * v[c2] + w3 * v[c3];
            j += 4;
        }
        for j in j..end {
            let col = bound_col(col_indices, j, n);
            let w = values[j];
            hr_acc += w * u[col];
            hi_acc -= w * v[col];
        }
        *hr = hr_acc;
        *hi = hi_acc;
    });
}

// ─── Full ψ-wave step ────────────────────────────────────

/// Full ψ-wave step. Sequential (each row depends on previous state).
pub fn step_psi_wave_full(
    v: &mut [f32], u: &mut [f32],
    row_ptr: &[i32], col_indices: &[i32], values: &[f32],
    dt: f32, hbar: f32,
) {
    let n = v.len().min(u.len());
    let max_syn = col_indices.len().min(values.len());
    let dt_hbar = dt / hbar;
    for i in 0..n {
        // Direct row_ptr indexing — row_ptr values are SYNAPSE indices
        let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);
        let mut hr = 0.0f32;
        let mut hi = 0.0f32;
        let mut j = start;
        while j + 4 <= end {
            let c0 = bound_col(col_indices, j, n);
            let c1 = bound_col(col_indices, j + 1, n);
            let c2 = bound_col(col_indices, j + 2, n);
            let c3 = bound_col(col_indices, j + 3, n);
            let w0 = values[j]; let w1 = values[j + 1]; let w2 = values[j + 2]; let w3 = values[j + 3];
            hr += w0 * u[c0] + w1 * u[c1] + w2 * u[c2] + w3 * u[c3];
            hi -= w0 * v[c0] + w1 * v[c1] + w2 * v[c2] + w3 * v[c3];
            j += 4;
        }
        for j in j..end {
            let col = bound_col(col_indices, j, n);
            let w = values[j];
            hr += w * u[col];
            hi -= w * v[col];
        }
        let vi = v[i] + hr * dt_hbar;
        let ui = u[i] + hi * dt_hbar;
        let nrm = vi * vi + ui * ui + 1e-9;
        let inv = 1.0 / nrm.sqrt();
        v[i] = vi * inv;
        u[i] = ui * inv;
    }
}

// ─── STDP LTD (sparse) ──────────────────────────────────

/// Apply STDP LTD — sequential (Fn closure can't mutate weights from par_iter).
pub fn apply_stdp_ltd(
    row_ptr: &[i32], col_indices: &[i32],
    weights: &mut [f32],
    spiked_indices: &[i32], traces: &[f32],
    learning_rate: f32,
) {
    let num_neurons = row_ptr.len() - 1;
    let max_syn = col_indices.len().min(weights.len());
    for &spike_idx in spiked_indices.iter() {
        let i = spike_idx as usize;
        if i >= num_neurons { continue; }
        // Direct row_ptr indexing — row_ptr values are SYNAPSE indices
        let start = row_ptr.get(i).copied().unwrap_or(0).max(0) as usize;
        let end = (row_ptr.get(i + 1).copied().unwrap_or(0) as usize).min(max_syn);
        for j in start..end {
            let target_idx = bound_col(col_indices, j, num_neurons);
            weights[j] -= learning_rate * traces[target_idx];
            if weights[j] < 1e-5 { weights[j] = 1e-5; }
        }
    }
}
