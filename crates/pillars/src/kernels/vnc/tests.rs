use super::*;

#[test]
fn vnc_mapping_basic() {
    let mn_rates = vec![10.0f32, 20.0, 5.0];
    let balance_gain = vec![1.0f64; 6];
    let contact_boost = vec![0.0f64; 6];
    let n_actuators = 4;
    let map_mn_idx = vec![0i32, 1, 2];
    let map_act_idx = vec![0i32, 1, 2];
    let map_scale = vec![1.0f32, 1.0, 0.5];
    let map_leg_idx = vec![0i32, 1, 2];
    let mirror_left = vec![0i32];
    let mirror_right = vec![1i32];
    let mut ctrl_out = vec![0.0f64; n_actuators];

    vnc_step_mapping(
        &mn_rates,
        &balance_gain,
        &contact_boost,
        0.5,
        n_actuators,
        &map_mn_idx,
        &map_act_idx,
        &map_scale,
        &map_leg_idx,
        &mirror_left,
        &mirror_right,
        &mut ctrl_out,
    );

    // rate 10·1=10 → clamp 1 → 1·0.5 = 0.5
    assert!(
        (ctrl_out[0] - 0.5).abs() < 1e-9,
        "ctrl[0] should be 0.5, got {}",
        ctrl_out[0]
    );
    // rate 20·1=20 → clamp 1 → 1·0.5 = 0.5
    assert!(
        (ctrl_out[1] - 0.5).abs() < 1e-9,
        "ctrl[1] should be 0.5, got {}",
        ctrl_out[1]
    );
    // rate 5·1=5 → clamp 1 → 1·scale(0.5)=0.5 → 0.5·gain(0.5) = 0.25
    assert!(
        (ctrl_out[2] - 0.25).abs() < 1e-9,
        "ctrl[2] should be 0.25, got {}",
        ctrl_out[2]
    );
    assert_eq!(ctrl_out[3], 0.0, "unmapped actuator stays 0");
}

#[test]
fn vnc_mapping_symmetrizes() {
    let mn_rates = vec![10.0f32, 30.0];
    let balance_gain = vec![1.0f64; 6];
    let contact_boost = vec![0.0f64; 6];
    let map_mn_idx = vec![0i32, 1];
    let map_act_idx = vec![0i32, 1];
    let map_scale = vec![1.0f32, 1.0];
    let map_leg_idx = vec![0i32, 1];
    let mirror_left = vec![0i32];
    let mirror_right = vec![1i32];
    let mut ctrl_out = vec![0.0f64; 2];

    vnc_step_mapping(
        &mn_rates,
        &balance_gain,
        &contact_boost,
        0.1,
        2,
        &map_mn_idx,
        &map_act_idx,
        &map_scale,
        &map_leg_idx,
        &mirror_left,
        &mirror_right,
        &mut ctrl_out,
    );

    assert!(
        (ctrl_out[0] - ctrl_out[1]).abs() < 1e-12,
        "mirrored actuators must be equal: {} vs {}",
        ctrl_out[0],
        ctrl_out[1]
    );
}
