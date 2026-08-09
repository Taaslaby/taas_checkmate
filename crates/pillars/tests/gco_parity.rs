//! # GCO parity: `pillars::kernels::phase::gco_step` vs inline legacy
//!
//! Транш 3a: мост `safe::gco_step` удалён (все потребители переведены на
//! pillars напрямую). Тест замораживает **бит-в-бит** поведение pillars-канона
//! против независимого inline-эталона, извлечённого из git HEAD
//! (`crates/kernels/src/neuron/saw.rs` — легаси `gco_step`, тот же кламп ±100,
//! та же нормализация на единичную окружность, то же группирование операций).

use pillars::kernels::phase::gco_step as pillars_gco;

const N: usize = 8;
const DT: f32 = 0.01;
const STEPS: usize = 2000;

/// Inline-эталон из git HEAD (crates/kernels/src/neuron/saw.rs, скалярный эквивалент).
fn legacy_gco_step(
    v: &mut [f32],
    u: &mut [f32],
    h_real: &[f32],
    h_imag: &[f32],
    i_ext: &[f32],
    speed: &[f32],
    dt: f32,
) {
    let n = v.len();
    for i in 0..n {
        let dv = h_real[i] - h_imag[i] * speed[i] + i_ext[i].clamp(-100.0, 100.0);
        let du = h_imag[i] + h_real[i] * speed[i];
        v[i] += dt * dv;
        u[i] += dt * du;
        let nrm = (v[i] * v[i] + u[i] * u[i] + 1e-9).sqrt();
        let inv = 1.0 / nrm;
        v[i] *= inv;
        u[i] *= inv;
    }
}

#[test]
fn gco_bit_identical_with_legacy() {
    let mut pv: Vec<f32> = (0..N).map(|i| 0.1 + i as f32 * 0.2).collect();
    let mut pu: Vec<f32> = (0..N).map(|i| -0.3 + i as f32 * 0.1).collect();
    let phr: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 0.1).collect();
    let phi: Vec<f32> = (0..N).map(|i| 0.5 - i as f32 * 0.05).collect();
    let pie: Vec<f32> = (0..N).map(|i| (i as f32 - 4.0) * 10.0).collect(); // разброс ±40
    let psp: Vec<f32> = (0..N).map(|i| 0.2 + i as f32 * 0.15).collect();

    let mut lv: Vec<f32> = pv.clone();
    let mut lu: Vec<f32> = pu.clone();

    for step in 0..STEPS {
        pillars_gco(&mut pv, &mut pu, &phr, &phi, &pie, &psp, DT);
        legacy_gco_step(&mut lv, &mut lu, &phr, &phi, &pie, &psp, DT);
        for i in 0..N {
            assert_eq!(
                pv[i].to_bits(),
                lv[i].to_bits(),
                "v[{i}] diverged at step {step}: pillars {} vs legacy {}",
                pv[i],
                lv[i]
            );
            assert_eq!(
                pu[i].to_bits(),
                lu[i].to_bits(),
                "u[{i}] diverged at step {step}: pillars {} vs legacy {}",
                pu[i],
                lu[i]
            );
        }
    }
    eprintln!("GCO: {STEPS} steps × {N} neurons bit-identical ✓");
}

#[test]
fn gco_tail_semantics_match_scalar() {
    // n=13 → SIMD-хвост; SIMD-путь и tail обязаны совпасть с эталоном
    let n = 13;
    let mut pv: Vec<f32> = (0..n).map(|i| 0.05 * i as f32 - 0.3).collect();
    let mut pu: Vec<f32> = (0..n).map(|i| -0.2 + 0.03 * i as f32).collect();
    let phr: Vec<f32> = (0..n).map(|i| 1.0 + 0.07 * i as f32).collect();
    let phi: Vec<f32> = (0..n).map(|i| 0.4 - 0.02 * i as f32).collect();
    let pie: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 25.0).collect();
    let psp: Vec<f32> = (0..n).map(|i| 0.15 + 0.1 * (i % 3) as f32).collect();

    let mut lv = pv.clone();
    let mut lu = pu.clone();

    for _ in 0..500 {
        pillars_gco(&mut pv, &mut pu, &phr, &phi, &pie, &psp, DT);
        legacy_gco_step(&mut lv, &mut lu, &phr, &phi, &pie, &psp, DT);
    }
    for i in 0..n {
        assert_eq!(pv[i].to_bits(), lv[i].to_bits(), "tail v[{i}] diverged");
        assert_eq!(pu[i].to_bits(), lu[i].to_bits(), "tail u[{i}] diverged");
    }
    eprintln!("GCO tail n={n}: 500 steps bit-identical ✓");
}
