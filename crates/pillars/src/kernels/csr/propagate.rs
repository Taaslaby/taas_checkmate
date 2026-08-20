use super::topology::ValidatedCsrTopology;

// ─── Sparse Spike Propagation ────────────────────────────

/// Propagate spikes through a CSR synaptic matrix.
/// Sequential — fan-in on input_buffer (data race with rayon).
pub fn propagate_spikes(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    spiked_indices: &[i32],
    nt_types: &[f32],
    input_buffer: &mut [f32],
    weight_scale: f32,
) {
    let n_neurons = row_ptr
        .len()
        .checked_sub(1)
        .expect("propagate_spikes: row_ptr must not be empty");
    let topology = ValidatedCsrTopology::new(row_ptr, col_indices, n_neurons);
    propagate_spikes_with_topology(
        topology,
        values,
        spiked_indices,
        nt_types,
        input_buffer,
        weight_scale,
    );
}

/// Propagate spikes through a topology validated once at construction.
///
/// This is the allocation-free hot-path counterpart of [`propagate_spikes`].
/// The topology validation is not repeated; only O(1) buffer checks remain.
///
/// For compatibility with the legacy boundary, a short `nt_types` slice uses
/// `1.0` for missing entries and a short `input_buffer` truncates destinations
/// instead of panicking. Production callers should provide buffers of length
/// `topology.n_neurons()`.
pub fn propagate_spikes_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    spiked_indices: &[i32],
    nt_types: &[f32],
    input_buffer: &mut [f32],
    weight_scale: f32,
) {
    let n = topology.n_neurons;
    assert_eq!(
        values.len(),
        topology.col_indices.len(),
        "propagate_spikes: values length mismatch"
    );

    for &spike_idx in spiked_indices {
        let i = spike_idx as usize;
        if i >= n {
            continue;
        }
        let mult = nt_types.get(i).copied().unwrap_or(1.0) * weight_scale;
        let (start, end) = topology.bounds(i);
        for j in start..end {
            let target = topology.col_indices[j] as usize;
            if target < input_buffer.len() {
                input_buffer[target] += values[j] * mult;
            }
        }
    }
}
