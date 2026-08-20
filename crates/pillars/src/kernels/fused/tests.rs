use super::*;

#[test]
fn hidden_izhikevich_accumulates_active_inputs() {
    let nh = 8;
    let ni = 8;
    let mut v = vec![-65.0f32; nh];
    let mut u = vec![-13.0f32; nh];
    let mut trace = vec![0.0f32; nh];
    let mut spikes = vec![0i32; nh];
    let mut refractory = vec![0.0f32; nh];
    let a = vec![0.02f32; nh];
    let b = vec![0.2f32; nh];
    let c = vec![-65.0f32; nh];
    let d = vec![8.0f32; nh];
    let weights: Vec<f32> = (0..ni * nh).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let mut input_spikes = vec![0i32; ni];
    input_spikes[0] = 1;
    input_spikes[3] = 1;
    let mut active_buf = Vec::with_capacity(ni);
    let mut total = 0i32;
    for _ in 0..200 {
        fused_hidden_izhikevich_step(
            &mut v,
            &mut u,
            &mut trace,
            &mut spikes,
            &mut refractory,
            &a,
            &b,
            &c,
            &d,
            &weights,
            &input_spikes,
            0.5,
            0.5,
            3.0,
            ni,
            &mut active_buf,
        );
        total += spikes.iter().sum::<i32>();
    }
    // I[j] = (weights[j] + weights[24+j])·10 ≈ 2.6–4.0 — лейны 5–7 (I ≥ 3.6)
    // раскачивают v до 30 и спайкают → total > 0.
    assert!(total > 0, "hidden layer should spike, got {total}");
}

#[test]
fn output_lif_integrates_hidden_spikes() {
    let na = 8;
    let nh = 8;
    let mut v = vec![-70.0f32; na];
    let mut spikes = vec![0i32; na];
    let mut refractory = vec![0.0f32; na];
    let tau_m = vec![10.0f32; na];
    let v_rest = vec![-70.0f32; na];
    let v_th = vec![-55.0f32; na];
    let v_reset = vec![-75.0f32; na];
    let weights: Vec<f32> = (0..nh * na).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let hidden_spikes = vec![1i32; nh];
    let mut total = 0i32;
    for _ in 0..200 {
        fused_output_lif_step(
            &mut v,
            &mut spikes,
            &mut refractory,
            &tau_m,
            &v_rest,
            &v_th,
            &v_reset,
            &weights,
            &hidden_spikes,
            1.0,
            3.0,
            nh,
        );
        total += spikes.iter().sum::<i32>();
    }
    assert!(total > 0, "output LIF should fire with strong drive");
}

#[test]
fn poisson_fused_is_deterministic() {
    let nh = 8;
    let ni = 8;
    let run_once = |v0: f32| {
        let mut v = vec![v0; nh];
        let mut u = vec![-13.0f32; nh];
        let mut trace = vec![0.0f32; nh];
        let mut spikes = vec![0i32; nh];
        let mut refractory = vec![0.0f32; nh];
        let a = vec![0.02f32; nh];
        let b = vec![0.2f32; nh];
        let c = vec![-65.0f32; nh];
        let d = vec![8.0f32; nh];
        let weights: Vec<f32> = (0..ni * nh).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut rng = vec![42u32; ni];
        let rates = vec![100.0f32; ni];
        let mut in_sp = vec![0i32; ni];
        let mut active_buf = Vec::with_capacity(ni);
        let mut total = 0i32;
        for _ in 0..50 {
            poisson_fused_hidden_step(
                &mut v,
                &mut u,
                &mut trace,
                &mut spikes,
                &mut refractory,
                &a,
                &b,
                &c,
                &d,
                &weights,
                &mut rng,
                &rates,
                &mut in_sp,
                0.5,
                0.5,
                3.0,
                ni,
                &mut active_buf,
            );
            total += spikes.iter().sum::<i32>();
        }
        (total, v, u)
    };
    let (t1, v1, u1) = run_once(-65.0);
    let (t2, v2, u2) = run_once(-65.0);
    assert_eq!(t1, t2, "LCG must be deterministic");
    assert_eq!(v1, v2, "v must be deterministic");
    assert_eq!(u1, u2, "u must be deterministic");
}
