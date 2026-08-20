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
