//! 🧠 HH GPU benchmark — тот же ворклоад, что `bench_hh_200k.rs`, но на CUDA.
//!
//! Прогон: cargo run --example bench_hh_gpu --release --features cuda
//!
//! 200 000 нейронов HH, dt = 0.01 мс, 5000 шагов, половина под током
//! 50 µA/cm². Состояния живут в VRAM (persistent path, CUDA Graph).

use std::time::Instant;

use pillars::prelude::*;
use pillars_cuda::{CompiledModel, HH_CUDA_SOURCE};

const N: usize = 200_000;
const DT: f32 = 0.01; // мс
const STEPS: usize = 5_000;

/// Заполнить колонку `col` SoA-буфера значением `val`.
fn fill_col(buf: &mut [f32], n: usize, col: usize, val: f32) {
    for v in buf[col * n..(col + 1) * n].iter_mut() {
        *v = val;
    }
}

fn main() {
    // ── SoA-параметры: колонка 0 = dt, как в bench_hh_200k ──
    let mut params = vec![0.0f32; HHModel::N_PARAMS * N];
    fill_col(&mut params, N, 0, DT); // dt
    fill_col(&mut params, N, 1, 120.0); // g_Na
    fill_col(&mut params, N, 2, 50.0); // E_Na
    fill_col(&mut params, N, 3, 36.0); // g_K
    fill_col(&mut params, N, 4, -77.0); // E_K
    fill_col(&mut params, N, 5, 0.3); // g_Leak
    fill_col(&mut params, N, 6, -54.3); // E_Leak
    fill_col(&mut params, N, 7, q10_from_celsius_s(6.3)); // q10 = 1.0
    fill_col(&mut params, N, 8, 0.0); // v_thresh
    fill_col(&mut params, N, 9, 1.0); // inv_cm
    fill_col(&mut params, N, 10, 0.0); // g_CaT
    fill_col(&mut params, N, 11, -80.0); // E_CaT

    // ── Начальные состояния (steady-state при V = -65 mV) ──
    let (m_inf, h_inf, n_inf) = hh_steady_state(-65.0);
    let mut states = vec![0.0f32; HHModel::N_STATE * N];
    fill_col(&mut states, N, 0, -65.0); // v
    fill_col(&mut states, N, 1, m_inf);
    fill_col(&mut states, N, 2, h_inf);
    fill_col(&mut states, N, 3, n_inf);
    fill_col(&mut states, N, 4, 0.19); // c
    fill_col(&mut states, N, 5, 1.0); // was_below

    // Половина нейронов под током 50 µA/cm², половина в тишине.
    let i_in: Vec<f32> = (0..N).map(|i| if i < N / 2 { 50.0 } else { 0.0 }).collect();

    let model = CompiledModel::new(
        HH_CUDA_SOURCE,
        "hh_kernel",
        HHModel::N_STATE,
        HHModel::N_PARAMS,
        N,
        N,
    )
    .expect("NVRTC не смог собрать hh_kernel — нужен CUDA-тулкит и GPU");

    // Каждый замер стартует с одного и того же начального состояния,
    // чтобы спайки сравнивались честно (как в bench_hh_200k: шаги 0..5000).
    let mut spikes = vec![0i32; N];

    // ── Голый compute (CUDA Graph replay, без чтения спайков) ──
    model.upload(&states, &params, &i_in).expect("upload");
    for _ in 0..10 {
        model.step_device(DT).expect("step"); // разогрев + захват графа
    }
    let t0 = Instant::now();
    for _ in 0..STEPS {
        model.step_device(DT).expect("step");
    }
    let bare_secs = t0.elapsed().as_secs_f64();

    // ── Реалистичный цикл: спайки читаем каждый шаг (D2H) ──
    model.upload(&states, &params, &i_in).expect("upload");
    let t1 = Instant::now();
    let mut total: u64 = 0;
    for _ in 0..STEPS {
        model.step_device(DT).expect("step");
        model.download_spikes(&mut spikes).expect("d2h");
        total += spikes.iter().map(|&s| s as u64).sum::<u64>();
    }
    let loop_secs = t1.elapsed().as_secs_f64();

    // ── Отчёт ──
    let neuron_steps = N as f64 * STEPS as f64;
    let sim_ms = DT as f64 * STEPS as f64; // 50 биол. мс
    let rt = |secs: f64| sim_ms / (secs * 1000.0);
    let mps = |secs: f64| neuron_steps / secs / 1e6;

    // Замечание: `step_device` делает cu_ctx_synchronize на каждом шаге, так
    // что «голый compute» = replay CUDA Graph + host-sync за шаг (не чистый
    // device-time). «Цикл с чтением» дополнительно гоняет 800КБ D2H за шаг.
    println!("🧠 HH × {N} × {STEPS} (dt = {DT} мс), GPU (RTX 4060 Ti)");
    println!("  graph replay + sync:    {bare_secs:.3} с | {:.1} млн нер-шагов/с | {:.4}x реалтайма", mps(bare_secs), rt(bare_secs));
    println!("  цикл с чтением спайков: {loop_secs:.3} с | {:.1} млн/с | {:.4}x реалтайма", mps(loop_secs), rt(loop_secs));
    println!("  спайков: {total}");
}

/// Steady-state m∞, h∞, n∞ при 6.3°C (копия из bench_hh_200k).
fn hh_steady_state(v: f32) -> (f32, f32, f32) {
    let alpha_m = 0.1 * (25.0 - v) / (((25.0 - v) / 10.0).exp() - 1.0);
    let beta_m = 4.0 * (-v / 18.0).exp();
    let alpha_h = 0.07 * (-v / 20.0).exp();
    let beta_h = 1.0 / (((30.0 - v) / 10.0).exp() + 1.0);
    let alpha_n = 0.01 * (10.0 - v) / (((10.0 - v) / 10.0).exp() - 1.0);
    let beta_n = 0.125 * (-v / 80.0).exp();
    (
        alpha_m / (alpha_m + beta_m),
        alpha_h / (alpha_h + beta_h),
        alpha_n / (alpha_n + beta_n),
    )
}
