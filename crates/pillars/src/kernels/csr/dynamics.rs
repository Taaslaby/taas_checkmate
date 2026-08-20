#![allow(non_snake_case)] // I_buffer/H_real/H_imag — физическая нотация

use super::plasticity::{compute_hamiltonian_with_topology, rate_step_with_topology};
use super::topology::ValidatedCsrTopology;

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
