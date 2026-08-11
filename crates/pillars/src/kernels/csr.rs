//! # ⬡ CSR sparse operations — propagation, Hamiltonian, STDP
//!
//! Порт `core::sparse`: propagation, Hamiltonian, rate-step, STDP.
//! Hamiltonian parallelized with rayon (cfg-gated).
//!
//! Ported from `toyon/c/engine_neuron.c`.

#![allow(non_snake_case)]

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::sync::Arc;

// ─── Safe CSR access helpers ─────────────────────────────

/// Safe CSR row pointer access: clamp index and validate bounds.
///
/// Внутри модуля не используется (функции индексируют `row_ptr.get()`
/// напрямую) — публичный API для `core::sparse` и python-ABI.
#[inline(always)]
pub fn bound_row(row_ptr: &[i32], idx: usize, n: usize) -> usize {
    if idx >= row_ptr.len() {
        return 0;
    }
    let r = row_ptr[idx];
    if r < 0 {
        0
    } else {
        r.min(n as i32) as usize
    }
}

/// Safe CSR column index access: validate column is within [0, n). Panics on invalid.
#[inline(always)]
pub fn bound_col(col_indices: &[i32], idx: usize, n: usize) -> usize {
    if idx >= col_indices.len() {
        return 0;
    }
    let c = col_indices[idx];
    debug_assert!(
        c >= 0 && (c as usize) < n,
        "bound_col: invalid column {} at idx {}, n={}, col_indices.len={}",
        c,
        idx,
        n,
        col_indices.len()
    );
    if c < 0 || c as usize >= n {
        panic!(
            "bound_col: out of range column {} at idx {}, n={}",
            c, idx, n
        );
    }
    c as usize
}

// ─── Validated CSR topology ──────────────────────────────

/// Borrowed CSR topology validated once for reuse across simulation steps.
///
/// The wrapper owns no buffers and therefore remains cheap to copy and safe to
/// share across rayon workers. Synaptic values/weights stay outside the wrapper
/// so mutable learning kernels can update them without violating the borrow
/// contract.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedCsrTopology<'a> {
    row_ptr: &'a [i32],
    col_indices: &'a [i32],
    n_neurons: usize,
}

impl<'a> ValidatedCsrTopology<'a> {
    /// Validate and borrow a CSR topology. This is the O(edges) boundary check.
    pub fn new(row_ptr: &'a [i32], col_indices: &'a [i32], n_neurons: usize) -> Self {
        assert_eq!(
            row_ptr.len(),
            n_neurons
                .checked_add(1)
                .expect("CSR: neuron count overflows row_ptr length"),
            "CSR: row_ptr must have n + 1 entries"
        );
        assert_eq!(
            row_ptr.first().copied(),
            Some(0),
            "CSR: row_ptr must start at 0"
        );
        let mut previous = 0usize;
        for (i, &raw) in row_ptr.iter().enumerate() {
            assert!(raw >= 0, "CSR: row_ptr[{i}] is negative ({raw})");
            let offset = raw as usize;
            assert!(
                offset >= previous,
                "CSR: row_ptr must be monotonic at index {i} ({offset} < {previous})"
            );
            assert!(
                offset <= col_indices.len(),
                "CSR: row_ptr[{i}]={offset} exceeds {} synapses",
                col_indices.len()
            );
            previous = offset;
        }
        assert_eq!(
            previous,
            col_indices.len(),
            "CSR: final row_ptr offset must equal synapse count"
        );
        for (i, &raw) in col_indices.iter().enumerate() {
            assert!(
                raw >= 0 && (raw as usize) < n_neurons,
                "CSR: col_indices[{i}]={raw} out of range for {n_neurons} neurons"
            );
        }
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    #[inline(always)]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    #[inline(always)]
    pub fn row_ptr(&self) -> &'a [i32] {
        self.row_ptr
    }

    #[inline(always)]
    pub fn col_indices(&self) -> &'a [i32] {
        self.col_indices
    }

    /// Construct a view from an already validated owner without rescanning it.
    #[inline(always)]
    fn from_validated(row_ptr: &'a [i32], col_indices: &'a [i32], n_neurons: usize) -> Self {
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    #[inline(always)]
    fn bounds(&self, row: usize) -> (usize, usize) {
        assert!(row < self.n_neurons, "CSR: row {row} out of range");
        (self.row_ptr[row] as usize, self.row_ptr[row + 1] as usize)
    }
}

/// Owning, immutable CSR topology for long-lived simulation objects.
///
/// Construction performs the O(edges) validation once. `as_topology()` creates
/// a borrowed view in O(1), so a runner can reuse the validated structure on
/// every timestep without a self-referential struct or unsafe code. Synaptic
/// values remain outside this type because learning kernels mutate them.
#[derive(Clone, Debug)]
pub struct ValidatedCsrGraph {
    row_ptr: Arc<[i32]>,
    col_indices: Arc<[i32]>,
    n_neurons: usize,
}

impl ValidatedCsrGraph {
    /// Copy CSR topology into an owned validated snapshot.
    ///
    /// The copy is intentional: existing `ConnectomeData` keeps public mutable
    /// `Vec`s for ABI compatibility, while this snapshot remains immutable and
    /// safe to share across simulation steps and rayon workers.
    pub fn from_slices(row_ptr: &[i32], col_indices: &[i32], n_neurons: usize) -> Self {
        let row_ptr: Arc<[i32]> = Arc::from(row_ptr);
        let col_indices: Arc<[i32]> = Arc::from(col_indices);
        let _ = ValidatedCsrTopology::new(&row_ptr, &col_indices, n_neurons);
        Self {
            row_ptr,
            col_indices,
            n_neurons,
        }
    }

    /// Number of neurons represented by this topology.
    #[inline(always)]
    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Borrow the validated topology in O(1), without rescanning CSR arrays.
    #[inline(always)]
    pub fn as_topology(&self) -> ValidatedCsrTopology<'_> {
        ValidatedCsrTopology::from_validated(&self.row_ptr, &self.col_indices, self.n_neurons)
    }

    /// Return the owned row pointer snapshot.
    #[inline(always)]
    pub fn row_ptr(&self) -> &[i32] {
        &self.row_ptr
    }

    /// Return the owned column-index snapshot.
    #[inline(always)]
    pub fn col_indices(&self) -> &[i32] {
        &self.col_indices
    }
}

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

// ─── Hamiltonian ─────────────────────────────────────────

/// Compute graph-coupled oscillator Hamiltonian: H_real = W * u, H_imag = -W * v.
pub fn compute_hamiltonian(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    v: &[f32],
    u: &[f32],
    h_real: &mut [f32],
    h_imag: &mut [f32],
) {
    let n = h_real.len();
    let topology = ValidatedCsrTopology::new(row_ptr, col_indices, n);
    compute_hamiltonian_with_topology(topology, values, v, u, h_real, h_imag);
}

/// Compute a scatter-oriented Hamiltonian using a validated topology.
///
/// This matches the connectome convention used by the simulation runners:
/// each CSR row is a pre-neuron and contributes to its post-neuron column.
/// `h_real[post] += W[pre, post] * nt[pre] * u[pre]` and
/// `h_imag[post] -= W[pre, post] * nt[pre] * v[pre]`.
pub fn compute_hamiltonian_scatter_with_topology(
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    nt_types: &[f32],
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
    assert!(v.len() >= n, "CSR: v shorter than topology");
    assert!(u.len() >= n, "CSR: u shorter than topology");
    assert!(h_real.len() >= n, "CSR: h_real shorter than topology");
    assert!(h_imag.len() >= n, "CSR: h_imag shorter than topology");

    for pre in 0..n {
        let mult = nt_types.get(pre).copied().unwrap_or(1.0);
        let (start, end) = topology.bounds(pre);
        for j in start..end {
            let post = topology.col_indices[j] as usize;
            let w = values[j] * mult;
            h_real[post] += w * u[pre];
            h_imag[post] -= w * v[pre];
        }
    }
}

/// Full graph-coupled oscillator step using a validated topology.
///
/// The update is intentionally sequential: each row observes the state written
/// by the preceding row, matching the legacy kernel's in-place semantics.
pub fn step_gco_full_with_topology(
    v: &mut [f32],
    u: &mut [f32],
    topology: ValidatedCsrTopology<'_>,
    values: &[f32],
    dt: f32,
    scale: f32,
) {
    let n = topology.n_neurons;
    assert_eq!(v.len(), n, "step_gco_full: v length mismatch");
    assert_eq!(u.len(), n, "step_gco_full: u length mismatch");
    assert_eq!(
        values.len(),
        topology.col_indices.len(),
        "step_gco_full: values length mismatch"
    );

    let dt_scale = dt * scale;
    for i in 0..n {
        let (start, end) = topology.bounds(i);
        let mut hr = 0.0f32;
        let mut hi = 0.0f32;
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
            hr += w0 * u[c0] + w1 * u[c1] + w2 * u[c2] + w3 * u[c3];
            hi -= w0 * v[c0] + w1 * v[c1] + w2 * v[c2] + w3 * v[c3];
            j += 4;
        }
        for j in j..end {
            let col = topology.col_indices[j] as usize;
            let w = values[j];
            hr += w * u[col];
            hi -= w * v[col];
        }
        let vi = v[i] + hr * dt_scale;
        let ui = u[i] + hi * dt_scale;
        let nrm = vi * vi + ui * ui + 1e-9;
        let inv = 1.0 / nrm.sqrt();
        v[i] = vi * inv;
        u[i] = ui * inv;
    }
}

// ─── Full oscillator step ───────────────────────────────

/// Full graph-coupled oscillator step. Sequential (each row depends on previous state).
/// `scale` = time-energy scaling factor (analogous to 1/ħ in Hamiltonian evolution).
pub fn step_gco_full(
    v: &mut [f32],
    u: &mut [f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    dt: f32,
    scale: f32,
) {
    let n = v.len();
    let topology = ValidatedCsrTopology::new(row_ptr, col_indices, n);
    step_gco_full_with_topology(v, u, topology, values, dt, scale);
}

// ─── Rate-based step ────────────────────────────────────

/// One step of rate-based dynamics (scatter SpMV + clamp + exponential Euler).
///
/// Computes:
///   `I[post] += W[pre,post] * r[pre] * nt[pre]` (scatter SpMV = `W^T @ r`)
///   r = r * decay + clamp(I + I_ext, 0, 1) * (1 - decay)
///
/// Sequential — scatter has fan-in on I (data race with rayon).
/// Loop-unrolled 4-wide for instruction-level parallelism.
///
/// Uses pre-allocated I_buffer (zeroed each call) to avoid per-step allocations.
///
/// Args:
///   r: rates `[n]`, mutated in-place
///   I_buffer: scratch buffer `[n]`, filled and used internally (user-managed)
///   row_ptr, col_indices, values: CSR synaptic matrix (`W[i,j] = weight from i to j`)
///   nt_types: neurotransmitter sign (+1 exc, -1 inh) per pre-neuron `[n]`
///   external_I: external input current `[n]`
///   decay: exp(-dt/tau)
///   inv_decay: 1 - decay
pub fn rate_step(
    r: &mut [f32],
    I_buffer: &mut [f32],
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    external_I: &[f32],
    decay: f32,
    inv_decay: f32,
) {
    let n = r.len();
    let topology = ValidatedCsrTopology::new(row_ptr, col_indices, n);
    rate_step_with_topology(
        topology, values, r, I_buffer, nt_types, external_I, decay, inv_decay,
    );
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_graph_snapshot_matches_borrowed_topology() {
        let row_ptr = [0i32, 2, 3];
        let col_indices = [1i32, 0, 1];
        let graph = ValidatedCsrGraph::from_slices(&row_ptr, &col_indices, 2);
        assert_eq!(graph.n_neurons(), 2);
        assert_eq!(graph.row_ptr(), &row_ptr);
        assert_eq!(graph.col_indices(), &col_indices);

        let view = graph.as_topology();
        assert_eq!(view.n_neurons(), 2);
        assert_eq!(view.row_ptr(), &row_ptr);
        assert_eq!(view.col_indices(), &col_indices);
    }

    #[test]
    #[should_panic(expected = "row_ptr must not be empty")]
    fn propagate_spikes_rejects_empty_csr() {
        propagate_spikes(&[], &[], &[], &[], &[], &mut [], 1.0);
    }

    #[test]
    #[should_panic(expected = "row_ptr must be monotonic")]
    fn rate_step_rejects_non_monotonic_csr() {
        let mut rates = vec![0.0f32; 2];
        let mut scratch = vec![0.0f32; 2];
        rate_step(
            &mut rates,
            &mut scratch,
            &[0, 2, 1],
            &[0, 1],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            0.9,
            0.1,
        );
    }

    #[test]
    #[should_panic(expected = "traces length mismatch")]
    fn apply_stdp_ltd_rejects_short_traces() {
        apply_stdp_ltd(&[0, 1], &[0], &mut [1.0], &[0], &[], 0.1);
    }

    #[test]
    #[should_panic(expected = "final row_ptr offset")]
    fn topology_rejects_truncated_final_row() {
        ValidatedCsrTopology::new(&[0, 1, 1], &[0, 1], 2);
    }

    #[test]
    fn scatter_hamiltonian_matches_legacy_orientation() {
        let row_ptr = [0i32, 5, 6, 7];
        let col_indices = [1i32, 2, 0, 2, 1, 0, 2];
        let values = [0.25f32, -0.5, 0.75, 0.125, 0.03125, -0.2, 0.0625];
        let nt = [1.0f32, -1.0, 0.5];
        let topology = ValidatedCsrTopology::new(&row_ptr, &col_indices, 3);
        let v = [0.2f32, -0.4, 0.7, 99.0];
        let u = [0.8f32, 0.1, -0.3, 99.0];
        let mut actual_r = [0.0f32; 4];
        let mut actual_i = [0.0f32; 4];
        compute_hamiltonian_scatter_with_topology(
            topology,
            &values,
            &nt,
            &v,
            &u,
            &mut actual_r,
            &mut actual_i,
        );

        let mut expected_r = [0.0f32; 4];
        let mut expected_i = [0.0f32; 4];
        for pre in 0..3 {
            for j in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_indices[j] as usize;
                let w = values[j] * nt[pre];
                expected_r[post] += w * u[pre];
                expected_i[post] -= w * v[pre];
            }
        }
        assert_eq!(actual_r, expected_r);
        assert_eq!(actual_i, expected_i);
    }

    #[test]
    fn raw_and_validated_paths_are_bit_identical() {
        let row_ptr = [0i32, 5, 6, 7];
        let col_indices = [1i32, 2, 0, 2, 1, 0, 2];
        let values = [0.25f32, -0.5, 0.75, 0.125, 0.03125, -0.2, 0.0625];
        let topology = ValidatedCsrTopology::new(&row_ptr, &col_indices, 3);
        let spiked = [0i32, 2];
        let nt = [1.0f32, -1.0, 0.5];

        let mut raw_input = [0.0f32; 3];
        let mut validated_input = [0.0f32; 3];
        propagate_spikes(
            &row_ptr,
            &col_indices,
            &values,
            &spiked,
            &nt,
            &mut raw_input,
            1.25,
        );
        propagate_spikes_with_topology(topology, &values, &spiked, &nt, &mut validated_input, 1.25);
        assert_eq!(raw_input, validated_input);

        let v = [0.2f32, -0.4, 0.7];
        let u = [0.8f32, 0.1, -0.3];
        let mut raw_hr = [0.0f32; 3];
        let mut raw_hi = [0.0f32; 3];
        let mut validated_hr = [0.0f32; 3];
        let mut validated_hi = [0.0f32; 3];
        compute_hamiltonian(
            &row_ptr,
            &col_indices,
            &values,
            &v,
            &u,
            &mut raw_hr,
            &mut raw_hi,
        );
        compute_hamiltonian_with_topology(
            topology,
            &values,
            &v,
            &u,
            &mut validated_hr,
            &mut validated_hi,
        );
        assert_eq!(raw_hr, validated_hr);
        assert_eq!(raw_hi, validated_hi);

        let mut raw_v = v;
        let mut raw_u = u;
        let mut validated_v = v;
        let mut validated_u = u;
        step_gco_full(
            &mut raw_v,
            &mut raw_u,
            &row_ptr,
            &col_indices,
            &values,
            0.01,
            0.8,
        );
        step_gco_full_with_topology(
            &mut validated_v,
            &mut validated_u,
            topology,
            &values,
            0.01,
            0.8,
        );
        assert_eq!(raw_v, validated_v);
        assert_eq!(raw_u, validated_u);

        let mut raw_rates = [0.2f32, 0.4, 0.1];
        let mut raw_scratch = [0.0f32; 3];
        let mut validated_rates = raw_rates;
        let mut validated_scratch = [0.0f32; 3];
        rate_step(
            &mut raw_rates,
            &mut raw_scratch,
            &row_ptr,
            &col_indices,
            &values,
            &nt,
            &[0.1, 0.0, 0.2],
            0.9,
            0.1,
        );
        rate_step_with_topology(
            topology,
            &values,
            &mut validated_rates,
            &mut validated_scratch,
            &nt,
            &[0.1, 0.0, 0.2],
            0.9,
            0.1,
        );
        assert_eq!(raw_rates, validated_rates);
        assert_eq!(raw_scratch, validated_scratch);

        let mut raw_weights = values;
        let mut validated_weights = values;
        apply_stdp_ltd(
            &row_ptr,
            &col_indices,
            &mut raw_weights,
            &spiked,
            &[0.1, 0.2, 0.3],
            0.05,
        );
        apply_stdp_ltd_with_topology(
            topology,
            &mut validated_weights,
            &spiked,
            &[0.1, 0.2, 0.3],
            0.05,
        );
        assert_eq!(raw_weights, validated_weights);
    }

    #[test]
    fn propagate_spikes_basic() {
        // 3 neurons, synapses: 0→1 (w=2), 1→2 (w=3)
        let row_ptr = vec![0i32, 1, 2, 2];
        let col_indices = vec![1i32, 2];
        let values = vec![2.0f32, 3.0];
        let spiked = vec![0i32, 1];
        let nt = vec![1.0f32, 1.0, 1.0];
        let mut buf = vec![0.0f32; 3];

        propagate_spikes(&row_ptr, &col_indices, &values, &spiked, &nt, &mut buf, 1.0);
        assert_eq!(buf[1], 2.0);
        assert_eq!(buf[2], 3.0);
    }

    #[test]
    fn rate_step_converges() {
        let n = 4;
        let row_ptr = vec![0i32, 0, 0, 0, 0];
        let col_indices: Vec<i32> = vec![];
        let values: Vec<f32> = vec![];
        let nt = vec![1.0f32; n];
        let mut r = vec![0.0f32; n];
        let mut ibuf = vec![0.0f32; n];
        let ext = vec![0.5f32; n];

        for _ in 0..50 {
            rate_step(
                &mut r,
                &mut ibuf,
                &row_ptr,
                &col_indices,
                &values,
                &nt,
                &ext,
                0.9,
                0.1,
            );
        }
        for i in 0..n {
            assert!((r[i] - 0.5).abs() < 1e-2, "r[{i}]={}", r[i]);
        }
    }

    #[test]
    fn gco_full_preserves_norm() {
        let n = 8;
        let row_ptr = vec![0i32, 1, 2, 3, 4, 5, 6, 7, 8];
        let col_indices: Vec<i32> = (0..n as i32).collect();
        let values = vec![0.1f32; n];
        let mut v = vec![1.0f32; n];
        let mut u = vec![0.0f32; n];

        step_gco_full(&mut v, &mut u, &row_ptr, &col_indices, &values, 0.01, 1.0);
        let norm = (v[0] * v[0] + u[0] * u[0]).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm={norm}");
    }
}
