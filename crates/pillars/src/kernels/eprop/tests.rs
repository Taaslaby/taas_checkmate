use super::*;

#[test]
fn eprop_lif_spikes_and_eligibility() {
    let n = 8;
    let mut v = vec![-65.0f32; n];
    let mut e = vec![0.0f32; n];
    let mut psi = vec![0.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut refr = vec![0.0f32; n];
    let ii = vec![30.0f32; n];
    let pre = vec![1i32; n];
    let mut total = 0usize;
    for _ in 0..200 {
        total += eprop_lif_step(
            &mut v,
            &mut e,
            &mut psi,
            &ii,
            &mut spikes,
            &pre,
            &mut refr,
            0.1,
            10.0,
            20.0,
            3.0,
            -65.0,
            -50.0,
            -70.0,
            1.0,
        );
    }
    assert!(total > 0, "LIF should fire with I=30");
    assert!(e.iter().any(|&x| x > 0.0), "eligibility should accumulate");
}

#[test]
fn eprop_izh_multi_accumulates_weights() {
    let n = 8;
    let n_pre = 4;
    let mut v = vec![-65.0f32; n];
    let mut u = vec![-13.0f32; n];
    let mut psi = vec![0.0f32; n];
    let mut e = vec![0.0f32; n_pre * n];
    let z_pre = vec![1i32; n_pre];
    let i_ext = vec![0.0f32; n];
    let weights = vec![0.5f32; n_pre * n];
    let mut spikes = vec![0i32; n];
    let mut total = 0usize;
    for _ in 0..100 {
        total += eprop_izhikevich_step_multi(
            &mut v,
            &mut u,
            &mut psi,
            &mut e,
            &z_pre,
            &i_ext,
            &weights,
            n_pre,
            0.5,
            20.0,
            0.02,
            0.2,
            -65.0,
            8.0,
            1.0,
            &mut spikes,
        );
    }
    assert!(total > 0, "strong weights should drive spikes");
}
