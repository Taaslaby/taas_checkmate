use super::*;

#[test]
fn dopamine_traces_step_basic() {
    let n = 8;
    let mut trace_pre = vec![0.0f32; n];
    let mut trace_post = vec![0.0f32; n];
    let mut eligibility = vec![0.0f32; n];
    let pre_spikes = vec![1i32; n];
    let post_spikes = vec![1i32; n];
    dopamine_traces_step(
        &mut trace_pre,
        &mut trace_post,
        &mut eligibility,
        &pre_spikes,
        &post_spikes,
        1.0,
        20.0,
        20.0,
        200.0,
        0.01,
        0.01,
    );
    assert!(trace_pre[0] > 0.0, "trace_pre should increase");
    assert!(trace_post[0] > 0.0, "trace_post should increase");
}

#[test]
fn dopamine_weight_update_basic() {
    let n = 8;
    let mut weights = vec![0.5f32; n];
    let eligibility = vec![1.0f32; n];
    dopamine_weight_update(&mut weights, &eligibility, 1.0, 0.1, 0.0, 1.0);
    for i in 0..n {
        assert!(
            (weights[i] - 0.6).abs() < 0.01,
            "dopamine_weight[{i}]: {}",
            weights[i]
        );
    }
}

#[test]
fn dopamine_simd_scalar_tail_consistent() {
    // 9 elements: lane 0 is SIMD, lane 8 is scalar tail.
    let n = 9;
    let mut tp = vec![0.2f32; n];
    let mut tpo = vec![0.1f32; n];
    let mut el = vec![0.05f32; n];
    let pre = vec![1i32; n];
    let post = vec![0i32; n];
    dopamine_traces_step(
        &mut tp, &mut tpo, &mut el, &pre, &post, 0.5, 20.0, 20.0, 200.0, 0.01, 0.01,
    );
    for i in 0..n {
        assert!(
            (tp[i] - tp[0]).abs() < 1e-6,
            "trace_pre SIMD/tail mismatch: [0]={} [{}]={}",
            tp[0],
            i,
            tp[i]
        );
    }
}
