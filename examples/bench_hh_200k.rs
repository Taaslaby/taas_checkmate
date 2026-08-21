//! 🧠 Hodgkin-Huxley 200K Benchmark  
//! Запуск: cargo run --example bench_hh_200k --release

use pillars::prelude::*;
use std::time::Instant;

fn main() {
    let n: usize = 200_000;
    let dt: f32 = 0.01; // 0.01 ms = 100 kHz (как в bench_hh)
    let n_steps: usize = 5000; // 50 биол. мс — несколько спайков на нейрон

    println!("Создаю популяцию {n} нейронов HH...");
    let mut pop = NeuralPopulation::<HHModel>::new(n);

    // HHModel::N_PARAMS = 12 (col-major, индекс колонки):
    //   0:dt  1:g_Na  2:E_Na  3:g_K  4:E_K  5:g_Leak  6:E_Leak
    //   7:q10  8:v_thresh  9:inv_cm  10:g_CaT  11:E_CaT
    pop.fill_param(0, dt); // dt
    pop.fill_param(1, 120.0); // g_Na   (mS/cm² — так в bench_hh.rs)
    pop.fill_param(2, 50.0); // E_Na   (mV)
    pop.fill_param(3, 36.0); // g_K    (mS/cm²)
    pop.fill_param(4, -77.0); // E_K    (mV)
    pop.fill_param(5, 0.3); // g_Leak (mS/cm²)
    pop.fill_param(6, -54.3); // E_Leak (mV)
    pop.fill_param(7, q10_from_celsius_s(6.3)); // q10 = 1.0 при 6.3°C
    pop.fill_param(8, 0.0); // v_thresh (спайк при V > 0)
    pop.fill_param(9, 1.0); // inv_cm  = 1/Cm
    pop.fill_param(10, 0.0); // g_CaT   (0 = чистый HH, без пейсмейкера)
    pop.fill_param(11, -80.0); // E_CaT

    // Начальные состояния — steady-state при V = -65 mV
    // m∞≈0.053, h∞≈0.596, n∞≈0.317
    let v0 = -65.0;
    let (m_inf, h_inf, n_inf) = hh_steady_state(v0);
    pop.fill_state(0, v0); // v
    pop.fill_state(1, m_inf); // m
    pop.fill_state(2, h_inf); // h
    pop.fill_state(3, n_inf); // n
    pop.fill_state(4, 0.19); // c  (CaV-T gate, steady-state при -65 mV)
    pop.fill_state(5, 1.0); // was_below (для edge detection)

    // Входной ток: половина с I=0, половина с I=8 µA/cm²
    let i_amp = 50.0f32; // µA/cm² — надпороговый ток (как в bench_hh)
    let i_in: Vec<f32> = (0..n)
        .map(|i| if i < n / 2 { i_amp } else { 0.0 })
        .collect();
    let mut spikes = vec![0i32; n];

    // Разогрев
    println!("Разогрев (10 шагов)...");
    for _ in 0..10 {
        pop.step(&i_in, &mut spikes);
    }

    // Замер
    println!("Прогон {n_steps} шагов...");
    let mut total_spikes: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..n_steps {
        pop.step(&i_in, &mut spikes);
        total_spikes += spikes.iter().map(|&s| s as u64).sum::<u64>();
    }
    let elapsed = t0.elapsed();

    let ms_simulated = dt as f64 * n_steps as f64;
    let neuron_steps = n as f64 * n_steps as f64;
    let secs = elapsed.as_secs_f64();
    // Доля реального времени: симулированное время / стенное время.
    // (Было: ms_simulated / secs — это мс биологии за секунду стенки, и
    // печаталось как «быстрее реал.» с завышением в 1000×. Теперь честно.)
    let rt_factor = ms_simulated / (secs * 1000.0);
    let mps = neuron_steps / secs / 1e6;

    println!();
    println!("═══════════════════════════════════════════");
    println!("  🧠 HH × {n} нейронов × {n_steps} шагов");
    println!("═══════════════════════════════════════════");
    println!("  dt               = {dt} ms");
    println!("  симулировано     = {ms_simulated:.1} биол. мс");
    println!("  время стенки     = {secs:.3} с");
    println!("  доля реальн. вр. = {rt_factor:.4}×");
    println!("  пропускная спос. = {mps:.1} млн нейрон-шагов/с");
    println!("  спайков          = {total_spikes}");
    println!("═══════════════════════════════════════════");
}

/// Steady-state m∞, h∞, n∞ при температуре 6.3°C.
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
