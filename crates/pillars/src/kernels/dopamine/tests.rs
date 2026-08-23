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

#[test]
fn dopamine_weight_update_preserves_zero_eligibility_weights() {
    // Регрессия: синапсы с e == 0 (замороженные/врождённые пути) не должны
    // ре-клэмпиться к границам — клэмп часть ОБНОВЛЕНИЯ, а не глобальный проход.
    let n = 16;
    let mut weights = vec![0.0f32; n];
    weights[0] = 22.0; // выше w_max — врождённый grown-хвост
    weights[1] = -3.0; // ниже w_min
    weights[2] = 0.5; // обычный синапс под обновление
    let mut eligibility = vec![0.0f32; n];
    eligibility[2] = 1.0; // только этот синапс реально обновляется
    dopamine_weight_update(&mut weights, &eligibility, 1.0, 0.1, 0.0, 6.0);
    assert_eq!(weights[0], 22.0, "e=0: вес выше w_max должен сохраниться");
    assert_eq!(weights[1], -3.0, "e=0: вес ниже w_min должен сохраниться");
    assert!(
        (weights[2] - 0.6).abs() < 0.01,
        "e!=0: обновление + клэмп: {}",
        weights[2]
    );
}
