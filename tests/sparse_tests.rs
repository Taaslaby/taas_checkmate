//! Unit tests for sparse module — CSR propagation, Hamiltonian, Graph-Coupled Oscillator, STDP LTD.

// Allow non_snake_case for mathematical variable names
#![allow(non_snake_case)]

use taas_checkmate::sparse;

fn make_padded(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// Build a simple ring CSR: neuron i → (i+1) % n with weight 1.0
fn ring_csr(n: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n_pad = make_padded(n);
    let mut row_ptr = vec![0i32; n_pad + 1];
    let mut col_indices = vec![0i32; n]; // one edge per neuron
    let values = vec![1.0f32; n];
    for i in 0..n {
        row_ptr[i] = i as i32;
        col_indices[i] = ((i + 1) % n) as i32;
    }
    row_ptr[n] = n as i32;
    // Fill padding: duplicate last row
    for item in row_ptr[n..=n_pad].iter_mut() {
        *item = n as i32;
    }
    (row_ptr, col_indices, values)
}

// ─── Propagate Spikes ────────────────────────────────────

#[test]
fn test_propagate_spikes_basic() {
    let n = 8;
    let (row_ptr, col_indices, values) = ring_csr(n);
    let n_pad = make_padded(n);

    let spiked = vec![0i32]; // neuron 0 spikes
    let nt_types = vec![1.0f32; n_pad];
    let mut input_buf = vec![0.0f32; n_pad];
    let weight_scale = 1.0;

    sparse::propagate_spikes(
        &row_ptr,
        &col_indices,
        &values,
        &spiked,
        &nt_types,
        &mut input_buf,
        weight_scale,
    );

    // Neuron 0 → neuron 1 with weight 1.0
    assert!(
        (input_buf[1] - 1.0).abs() < 1e-5,
        "Neuron 1 should get 1.0, got {}",
        input_buf[1]
    );
    // All others should be 0
    for (i, &val) in input_buf[2..n].iter().enumerate() {
        assert!(
            val.abs() < 1e-5,
            "Neuron {} should get 0, got {}",
            i + 2,
            val
        );
    }
}

#[test]
fn test_propagate_spikes_multiple() {
    let n = 8;
    let (row_ptr, col_indices, values) = ring_csr(n);
    let n_pad = make_padded(n);

    let spiked = vec![0i32, 2, 5]; // multiple spikes
    let nt_types = vec![1.0f32; n_pad];
    let mut input_buf = vec![0.0f32; n_pad];

    sparse::propagate_spikes(
        &row_ptr,
        &col_indices,
        &values,
        &spiked,
        &nt_types,
        &mut input_buf,
        1.0,
    );

    // Neuron 0 → 1, Neuron 2 → 3, Neuron 5 → 6
    assert!(
        (input_buf[1] - 1.0).abs() < 1e-5,
        "N1 should get 1.0, got {}",
        input_buf[1]
    );
    assert!(
        (input_buf[3] - 1.0).abs() < 1e-5,
        "N3 should get 1.0, got {}",
        input_buf[3]
    );
    assert!(
        (input_buf[6] - 1.0).abs() < 1e-5,
        "N6 should get 1.0, got {}",
        input_buf[6]
    );
}

#[test]
fn test_propagate_spikes_weight_scale() {
    let n = 8;
    let (row_ptr, col_indices, values) = ring_csr(n);
    let n_pad = make_padded(n);

    let spiked = vec![0i32];
    let nt_types = vec![2.0f32; n_pad]; // double strength
    let mut input_buf = vec![0.0f32; n_pad];

    sparse::propagate_spikes(
        &row_ptr,
        &col_indices,
        &values,
        &spiked,
        &nt_types,
        &mut input_buf,
        1.0,
    );

    // Neuron 0 → 1 with nt_type=2.0, weight=1.0 → 2.0
    assert!(
        (input_buf[1] - 2.0).abs() < 1e-5,
        "N1 should get 2.0, got {}",
        input_buf[1]
    );
}

// ─── Compute Hamiltonian ────────────────────────────────

#[test]
fn test_compute_hamiltonian_basic() {
    let n = 8;
    let (row_ptr, col_indices, values) = ring_csr(n);
    let n_pad = make_padded(n);

    let v = vec![1.0f32; n_pad];
    let u = vec![2.0f32; n_pad];
    let mut h_real = vec![0.0f32; n_pad];
    let mut h_imag = vec![0.0f32; n_pad];

    sparse::compute_hamiltonian(
        &row_ptr,
        &col_indices,
        &values,
        &v,
        &u,
        &mut h_real,
        &mut h_imag,
    );

    // Ring: each neuron i receives from (i-1+n)%n
    // For neuron 1: receives from neuron 0 with u[0]=2.0, weight=1.0 → h_real[1]=2.0
    //             h_imag = -W*v[0] = -2.0
    for i in 1..n {
        assert!(
            (h_real[i] - 2.0).abs() < 1e-5,
            "h_real[{}] should be 2.0, got {}",
            i,
            h_real[i]
        );
        assert!(
            (h_imag[i] - (-1.0)).abs() < 1e-5,
            "h_imag[{}] should be -1.0, got {}",
            i,
            h_imag[i]
        );
    }
}

// ─── Graph-Coupled Oscillator Full Step ────────────────────────────────────

#[test]
fn test_gco_full_step_preserves_norm() {
    let n = 8;
    let (row_ptr, col_indices, values) = ring_csr(n);
    let n_pad = make_padded(n);

    let mut v = vec![1.0f32; n_pad];
    let mut u = vec![0.0f32; n_pad];
    // Normalize initial
    for i in 0..n {
        let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
        v[i] /= nrm;
        u[i] /= nrm;
    }
    let dt = 0.05;
    let hbar = 1.0;

    sparse::step_gco_full(&mut v, &mut u, &row_ptr, &col_indices, &values, dt, hbar);

    // Norm should stay close to 1.0
    for i in 0..n {
        let nrm = (v[i] * v[i] + u[i] * u[i]).sqrt();
        assert!(
            (nrm - 1.0).abs() < 0.01,
            "Graph-Coupled Oscillator norm should stay ~1, got {} at {}",
            nrm,
            i
        );
    }
}

// ─── Apply STDP LTD ──────────────────────────────────────

#[test]
fn test_apply_stdp_ltd_basic() {
    let n = 8;
    let (row_ptr, col_indices, mut weights) = ring_csr(n);

    let spiked = vec![1i32];
    let traces = vec![0.0f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let lr = 0.01;

    sparse::apply_stdp_ltd(&row_ptr, &col_indices, &mut weights, &spiked, &traces, lr);

    // Neuron 1 spikes → LTD on its outgoing synapse (→ neuron 2)
    // weight[1] -= 0.01 * traces[2] = -0.01 * 0.0 = 0.0 (no change, trace of target is 0)
    // But in a ring: neuron 1 → neuron 2 with weight[2]
    // apply_stdp_ltd: for each spike at i, for each col j, weight[j] -= lr * traces[target]
    // So weight[1] should decrease by lr * traces[col_indices[1]] = 0.01 * traces[2] = 0.01 * 0.0 = 0
    // Hmm, let me reconsider. traces[target_idx] where target_idx = col_indices[j].
    // For neuron 1 -> 2, col_indices[1] = 2. traces[2] = 0.0.
    // So weight[1] -= 0.01 * 0.0 = 0.0. No change.
    // Let's test with spiked at 0 and traces[1] = 0.5:

    // Reset and test a more interesting case
    let (_row_ptr2, _col_indices2, mut weights2) = ring_csr(n);
    let spiked2 = vec![0i32];
    let traces2 = vec![0.0f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    sparse::apply_stdp_ltd(
        &_row_ptr2,
        &_col_indices2,
        &mut weights2,
        &spiked2,
        &traces2,
        lr,
    );

    // Neuron 0 spikes → target = col_indices[0] = 1. traces[1] = 0.5
    // weight[0] -= 0.01 * 0.5 = 0.005
    assert!(
        (weights2[0] - 0.995).abs() < 1e-5,
        "STDP LTD weight[0] should be 0.995, got {}",
        weights2[0]
    );
}

#[test]
fn test_apply_stdp_ltd_clamp_low() {
    let n = 8;
    let (row_ptr, col_indices, mut weights) = ring_csr(n);

    let spiked = vec![0i32];
    let traces = vec![0.0f32, 1e10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // huge trace → clamp
    let lr = 1.0;

    sparse::apply_stdp_ltd(&row_ptr, &col_indices, &mut weights, &spiked, &traces, lr);

    // weight[0] -= 1.0 * 1e10 for col_indices[0]=1 → should be clamped to 1e-5
    assert!(
        (weights[0] - 1e-5).abs() < 1e-5,
        "LTD should clamp to 1e-5, got {}",
        weights[0]
    );
}
