#![allow(non_snake_case)] // I_buffer — физическая нотация

use super::topology::ValidatedCsrTopology;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ─── STDP LTD (sparse) ──────────────────────────────────

/// Apply STDP LTD — sequential (Fn closure can't mutate weights from par_iter).
pub fn apply_stdp_ltd(
    row_ptr: &[i32],
    col_indices: &[i32],
    weights: &mut [f32],
    spiked_indices: &[i32],
    traces: &[f32],
    learning_rate: f32,
) {
    let num_neurons = row_ptr
        .len()
        .checked_sub(1)
        .expect("apply_stdp_ltd: row_ptr must not be empty");
    let topology = ValidatedCsrTopology::new(row_ptr, col_indices, num_neurons);
    apply_stdp_ltd_with_topology(topology, weights, spiked_indices, traces, learning_rate);
}

/// Apply STDP LTD using a topology validated once at construction.
pub fn apply_stdp_ltd_with_topology(
    topology: ValidatedCsrTopology<'_>,
    weights: &mut [f32],
    spiked_indices: &[i32],
    traces: &[f32],
    learning_rate: f32,
) {
    let n = topology.n_neurons;
    assert_eq!(
        weights.len(),
        topology.col_indices.len(),
        "apply_stdp_ltd: weights length mismatch"
    );
    assert_eq!(traces.len(), n, "apply_stdp_ltd: traces length mismatch");

    for &spike_idx in spiked_indices {
        let i = spike_idx as usize;
        if i >= n {
            continue;
        }
        let (start, end) = topology.bounds(i);
        for j in start..end {
            let target_idx = topology.col_indices[j] as usize;
            weights[j] -= learning_rate * traces[target_idx];
            if weights[j] < 1e-5 {
                weights[j] = 1e-5;
            }
        }
    }
}

/// Compute Hamiltonian using a topology validated once by
/// [`ValidatedCsrTopology::new`].
pub fn compute_hamiltonian_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    v: &[f32],
    u: &[f32],
    h_real: &mut [f32],
    h_imag: &mut [f32],
) {
    let n = topology.n_neurons;
    assert_eq!(
        values.len(),
        topology.col_indices.len(),
        "CSR: values length mismatch"
    );
    assert_eq!(v.len(), n, "compute_hamiltonian: v length mismatch");
    assert_eq!(u.len(), n, "compute_hamiltonian: u length mismatch");
    assert_eq!(
        h_real.len(),
        n,
        "compute_hamiltonian: h_real length mismatch"
    );
    assert_eq!(
        h_imag.len(),
        n,
        "compute_hamiltonian: h_imag length mismatch"
    );

    #[cfg(feature = "rayon")]
    h_real
        .par_iter_mut()
        .zip(h_imag.par_iter_mut())
        .enumerate()
        .for_each(|(i, (hr, hi))| {
            let (start, end) = topology.bounds(i);
            hamiltonian_row_topology(topology, values, v, u, start, end, hr, hi);
        });
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let (start, end) = topology.bounds(i);
        hamiltonian_row_topology(
            topology,
            values,
            v,
            u,
            start,
            end,
            &mut h_real[i],
            &mut h_imag[i],
        );
    }
}

#[inline]
fn hamiltonian_row_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    v: &[f32],
    u: &[f32],
    start: usize,
    end: usize,
    hr: &mut f32,
    hi: &mut f32,
) {
    let mut hr_acc = 0.0f32;
    let mut hi_acc = 0.0f32;
    let mut j = start;
    while j + 4 <= end {
        let c0 = topology.col_indices[j] as usize;
        let c1 = topology.col_indices[j + 1] as usize;
        let c2 = topology.col_indices[j + 2] as usize;
        let c3 = topology.col_indices[j + 3] as usize;
        let w0 = values[j];
        let w1 = values[j + 1];
        let w2 = values[j + 2];
        let w3 = values[j + 3];
        hr_acc += w0 * u[c0] + w1 * u[c1] + w2 * u[c2] + w3 * u[c3];
        hi_acc -= w0 * v[c0] + w1 * v[c1] + w2 * v[c2] + w3 * v[c3];
        j += 4;
    }
    for j in j..end {
        let col = topology.col_indices[j] as usize;
        let w = values[j];
        hr_acc += w * u[col];
        hi_acc -= w * v[col];
    }
    *hr = hr_acc;
    *hi = hi_acc;
}

/// Rate step using a topology validated once at graph construction time.
pub fn rate_step_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    r: &mut [f32],
    i_buffer: &mut [f32],
    nt_types: &[f32],
    external_i: &[f32],
    decay: f32,
    inv_decay: f32,
) {
    let n = topology.n_neurons;
    assert_eq!(
        values.len(),
        topology.col_indices.len(),
        "CSR: values length mismatch"
    );
    assert_eq!(r.len(), n, "rate_step: rates length mismatch");
    assert_eq!(i_buffer.len(), n, "rate_step: I_buffer length mismatch");
    assert_eq!(nt_types.len(), n, "rate_step: nt_types length mismatch");
    assert_eq!(external_i.len(), n, "rate_step: external_I length mismatch");
    i_buffer.fill(0.0);

    for i in 0..n {
        let ri = r[i];
        if ri < 1e-6 {
            continue;
        }
        let scale = ri * nt_types[i];
        let (start, end) = topology.bounds(i);
        let mut j = start;
        while j + 4 <= end {
            let c0 = topology.col_indices[j] as usize;
            let c1 = topology.col_indices[j + 1] as usize;
            let c2 = topology.col_indices[j + 2] as usize;
            let c3 = topology.col_indices[j + 3] as usize;
            let w0 = values[j];
            let w1 = values[j + 1];
            let w2 = values[j + 2];
            let w3 = values[j + 3];
            if c0 < n {
                i_buffer[c0] += w0 * scale;
            }
            if c1 < n {
                i_buffer[c1] += w1 * scale;
            }
            if c2 < n {
                i_buffer[c2] += w2 * scale;
            }
            if c3 < n {
                i_buffer[c3] += w3 * scale;
            }
            j += 4;
        }
        for j in j..end {
            let col = topology.col_indices[j] as usize;
            if col < n {
                i_buffer[col] += values[j] * scale;
            }
        }
    }
    for i in 0..n {
        let target = (i_buffer[i] + external_i[i]).clamp(0.0, 1.0);
        r[i] = r[i] * decay + target * inv_decay;
    }
}
