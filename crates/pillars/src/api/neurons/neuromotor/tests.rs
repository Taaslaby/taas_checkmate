use super::*;

#[test]
fn neuromotor_wrapper_basic() {
    let n = 16;
    let dt = 1.0;
    let mut ca = vec![0.0f32; n];
    let mut tension = vec![0.0f32; n];
    let spikes = vec![1i32; n];
    let resets = vec![0i32; n];

    let d_ca = vec![0.5f32; n];
    let tau_pump = vec![20.0f32; n];
    let eta_reset = vec![0.8f32; n];
    let t_max = vec![0.001f32; n];
    let kd = vec![0.3f32; n];
    let l0 = vec![1.0f32; n];
    let v_max = vec![5.0f32; n];

    let mut joint_df = vec![0.0f32; 8];
    let mut joint_stf = vec![0.0f32; 8];
    let mut joint_dmp = vec![0.0f32; 8];
    let k0 = vec![1e-5f32; 8];
    let beta = vec![1e-4f32; 8];
    let b0 = vec![1e-6f32; 8];
    let alpha = vec![1e-5f32; 8];

    step_neuromotor(
        &mut ca,
        &mut tension,
        &spikes,
        &resets,
        dt,
        &d_ca,
        &tau_pump,
        &eta_reset,
        &t_max,
        &kd,
        &l0,
        &v_max,
        None,
        None,
        Some(&mut joint_df),
        Some(&mut joint_stf),
        Some(&mut joint_dmp),
        Some(&k0),
        Some(&beta),
        Some(&b0),
        Some(&alpha),
    );

    assert!(ca[0] > 0.0, "Ca2+ should accumulate");
    assert!(tension[0] > 0.0, "Tension should be generated");
    assert!(joint_stf[0] > 1e-5, "Joint stiffness should increase");
    assert!(joint_dmp[0] > 1e-6, "Joint damping should increase");
}

#[test]
fn test_neuromotor_state_struct_high_level() {
    let mut state = NeuromotorState::new(16);
    let spikes = vec![1i32; 16];
    let resets = vec![0i32; 16];

    state.step(&spikes, &resets, 1.0, None, None);

    assert!(state.ca[0] > 0.0);
    assert!(state.tension[0] > 0.0);
    assert!(state.joint_stiffness[0] > 1e-5);
    assert!(state.joint_damping[0] > 1e-6);
}

#[test]
fn test_human_scale_muscles() {
    let human_muscles = 640;
    let mut state = NeuromotorStateBuilder::new(human_muscles)
        .with_profile(BiomechanicsProfile::mammal())
        .build();

    let spikes = vec![1i32; human_muscles];
    let resets = vec![0i32; human_muscles];

    for _ in 0..1000 {
        state.step(&spikes, &resets, 1.0, None, None);
    }

    assert!(state.tension[0] > 1.0);
    assert!(state.joint_stiffness[0] > 1.0);
}

#[test]
fn test_drosophila_profile_slow_fiber_pool() {
    let n = 8;
    let mut state = NeuromotorStateBuilder::new(n)
        .with_profile(BiomechanicsProfile::drosophila_slow())
        .build();

    let spikes = vec![1i32; n];
    let resets = vec![0i32; n];

    for _ in 0..100 {
        state.step(&spikes, &resets, 0.5, None, None);
    }

    assert!(state.tension[0] < 1e-4, "Slow fiber tension should be tiny");
    assert!(state.tension[0] > 0.0, "Should have some tension");
}

#[test]
fn test_drosophila_fast_fiber_pool() {
    let n = 8;
    let mut state = NeuromotorStateBuilder::new(n)
        .with_profile(BiomechanicsProfile::drosophila_fast())
        .build();

    let spikes = vec![1i32; n];
    let resets = vec![0i32; n];

    for _ in 0..100 {
        state.step(&spikes, &resets, 0.5, None, None);
    }

    assert!(state.tension[0] > 0.0, "Fast fiber should have tension");
}

#[test]
fn test_drosophila_tdt_profile() {
    let n = 8;
    let mut state = NeuromotorStateBuilder::new(n)
        .with_profile(BiomechanicsProfile::drosophila_tdt())
        .build();

    let spikes = vec![1i32; n];
    let resets = vec![0i32; n];

    for _ in 0..200 {
        state.step(&spikes, &resets, 0.5, None, None);
    }

    assert!(state.tension[0] > state.tension[1] * 0.5);
}

#[test]
fn test_drosophila_leg_config_hexapod() {
    let config = DrosophilaLegConfig::hexapod();
    let channels = config.total_channels;

    assert!(channels > 0 && channels.is_multiple_of(8));
    assert!(config.include_tdt);
    let total_tdt: usize = config
        .pools
        .iter()
        .flat_map(|p| p.iter())
        .filter(|pool| (pool.profile.t_max - 2e-4).abs() < 1e-5)
        .map(|p| p.n_fibers)
        .sum();
    assert_eq!(total_tdt, 3 * 2, "TDT: 3 fibers × 2 middle legs = 6 total");

    // Build and test state
    let state = config.build_state();
    assert_eq!(state.n_muscles, channels);
    assert!(state.t_max.contains(&2e-4), "TDT tension must exist");
}

/// Антагонистичная пара: флексор (чётный) тянет сустав в одну сторону,
/// экстензор (нечётный) — в другую. Здесь проверяется знак крутящего
/// момента и то, что жёсткость/демпфирование растут с активацией
/// (zoo-модель; шаг — api-мост, поэтому тест живёт здесь, а не в zoo).
#[test]
fn physics_antagonist_outputs_obey_sign_and_impedance_baselines() {
    let n = 8;
    let mut state = vec![0.0; n];
    let mut tension = vec![0.0; n];
    let spikes = vec![1_i32, 0, 0, 1, 1, 0, 0, 0];
    let resets = vec![0; n];
    let d_ca = vec![0.5; n];
    let tau_pump = vec![20.0; n];
    let eta_reset = vec![0.8; n];
    let t_max = vec![1.0; n];
    let kd = vec![0.3; n];
    let l0 = vec![1.0; n];
    let v_max = vec![5.0; n];
    let mut joint_df = vec![0.0; 4];
    let mut joint_stiffness = vec![0.0; 4];
    let mut joint_damping = vec![0.0; 4];
    let k0 = vec![2.0; 4];
    let beta = vec![3.0; 4];
    let b0 = vec![0.4; 4];
    let alpha = vec![0.5; 4];

    step_neuromotor(
        &mut state,
        &mut tension,
        &spikes,
        &resets,
        1.0,
        &d_ca,
        &tau_pump,
        &eta_reset,
        &t_max,
        &kd,
        &l0,
        &v_max,
        None,
        None,
        Some(&mut joint_df),
        Some(&mut joint_stiffness),
        Some(&mut joint_damping),
        Some(&k0),
        Some(&beta),
        Some(&b0),
        Some(&alpha),
    );

    assert!(
        joint_df[0] < 0.0,
        "flexor at index 0 should beat extensor at index 1"
    );
    assert!(
        joint_df[1] > 0.0,
        "extensor at index 3 should beat flexor at index 2"
    );
    assert!(joint_stiffness[0] > k0[0]);
    assert!(joint_damping[0] > b0[0]);
    assert!(joint_stiffness[2] > k0[2]);
    assert!(joint_damping[2] > b0[2]);
}
