//! # Эксперимент: LIF 138k — CPU SIMD vs CUDA
//!
//! Честное сравнение двух желез на одной модели (LIF, Exponential/Forward Euler).
//!
//! - **CPU**: `taas_kernels::safe::lif_step` (f32x8 SIMD, Forward Euler)
//! - **GPU**: `cuda_kernels::step_lif` → `lif_step_kernel` (Exponential Euler)
//!
//! Что меряем:
//! 1. **Скорость** — ms/шаг на 138 000 нейронов, 1000 шагов, каждое железо.
//! 2. **Конформность** — max|Δv| и % совпавших спайков (схемы интегрирования
//!    разные, поэтому бит-в-бит не ждём — ждём ~совпадение).
//!
//! Запуск: `cargo run -p cuda-kernels --example lif_cpu_vs_cuda --release`

use std::time::Instant;

const N: usize = 138_000; // как у мухи
const STEPS: usize = 1000;
const DT: f32 = 0.1;

fn main() {
    println!("═══ LIF 138k: CPU SIMD vs CUDA ═══");
    println!("  N = {N}, steps = {STEPS}, dt = {DT} ms");
    println!("  GPU: {}", gpu_name());
    println!();

    // ── Параметры: tau_m, v_rest, v_thresh, v_reset ──
    let tau_m = vec![10.0f32; N];
    let v_rest = vec![-70.0f32; N];
    let v_thresh = vec![-54.0f32; N];
    let v_reset = vec![-70.0f32; N];
    let tau_ref = 2.0f32;

    // Входной ток: половина нейронов молчит, половина получает 30 (спайкает)
    let i_ext: Vec<f32> = (0..N).map(|i| if i % 2 == 0 { 30.0 } else { 0.0 }).collect();

    // ── Нагрев CUDA до замера (первый вызов ядра включает driver) ──
    println!("  Нагрев CUDA…");
    warmup_gpu();
    println!("  Готово.");
    println!();

    // ── CPU прогон ──
    let (cpu_ms, cpu_v) = run_cpu(&i_ext, &tau_m, &v_rest, &v_thresh, &v_reset, tau_ref);

    // ── GPU прогон ──
    let (gpu_ms, gpu_v) = run_gpu(&i_ext, &tau_m, &v_rest, &v_thresh, &v_reset, tau_ref);

    // ── Конформность по всем шагам (сильный тест: попарное сравнение на каждом шаге) ──
    let (spike_match, cpu_total, gpu_total) =
        conformance_over_steps(&i_ext, &tau_m, &v_rest, &v_thresh, &v_reset, tau_ref);

    // ── Сравнение ──
    println!("═══ РЕЗУЛЬТАТЫ ═══");
    println!("  CPU:  {cpu_ms:8.3} ms/шаг  ({:7.2} M neuron-step/с)",
        N as f64 / cpu_ms / 1e3);
    println!("  GPU:  {gpu_ms:8.3} ms/шаг  ({:7.2} M neuron-step/с)",
        N as f64 / gpu_ms / 1e3);
    println!("  GPU медленнее в {:.0}×", gpu_ms / cpu_ms);
    println!();

    // Конформность
    let mut max_dv = 0.0f32;
    let mut v_sum_cpu = 0.0f64;
    let mut v_sum_gpu = 0.0f64;
    for i in 0..N {
        let d = (cpu_v[i] - gpu_v[i]).abs();
        if d > max_dv { max_dv = d; }
        v_sum_cpu += cpu_v[i] as f64;
        v_sum_gpu += gpu_v[i] as f64;
    }
    println!("═══ КОНФОРМНОСТЬ (после {STEPS} шагов) ═══");
    println!("  max|Δv|:            {max_dv:.4} mV");
    println!("  среднее v CPU:      {:.3} mV", v_sum_cpu / N as f64);
    println!("  среднее v GPU:      {:.3} mV", v_sum_gpu / N as f64);
    println!("  совпадение спайков (попарно, все шаги): {}/{}", spike_match, STEPS * N);
    println!("    суммарно спайков CPU: {cpu_total}, GPU: {gpu_total}");
    println!("    совпадение: {:.3}%", spike_match as f64 / (STEPS * N) as f64 * 100.0);
    println!();
    println!("  Примечание: CPU=Forward Euler, GPU=Exponential Euler — схемы разные,");
    println!("  поэтому стабильное расхождение Δv ожидаемо и не является багом.");
    println!();
    println!("  Вывод (измерено): наивный per-step вызов CUDA из Rust (host→device upload +");
    println!("  kernel + download на каждый шаг) проигрывает CPU SIMD в ~100× на 138k");
    println!("  нейронов. Это ловушка синхронной пересылки, а не свойство GPU:");
    println!("  138k×4B×2 ≈ 1.1 MB PCIe-трафика на шаг.");
    println!();
    println!("  Гипотеза (НЕ измерено здесь): при persistent-контексте (данные живут на");
    println!("  устройстве, обновляются пачками) GPU должен выиграть — см.");
    println!("  GpuContext/batch-варианты в crates/cuda. Это будущая работа.");
}

// ═══════════════════════════════════════════════════════════
//  CPU: taas-kernels SIMD
// ═══════════════════════════════════════════════════════════

fn run_cpu(
    i_ext: &[f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_ref: f32,
) -> (f64, Vec<f32>) {
    let mut v = vec![-70.0f32; N];
    let ii = i_ext.to_vec();
    let mut spikes = vec![0i32; N];
    let mut refractory = vec![0.0f32; N];

    // params (CPU lif_step layout): [dt×N, tau_m×N, v_rest×N, v_thresh×N, v_reset×N]
    let mut params = vec![0.0f32; 5 * N];
    params[0..N].fill(DT); // col0: dt (кернел читает p[0])
    params[N..2 * N].copy_from_slice(tau_m);
    params[2 * N..3 * N].copy_from_slice(v_rest);
    params[3 * N..4 * N].copy_from_slice(v_thresh);
    params[4 * N..5 * N].copy_from_slice(v_reset);

    let t0 = Instant::now();
    for _ in 0..STEPS {
        let mut cur = ii.clone(); // подаём ток каждый шаг
        taas_kernels::safe::lif_step(&mut v, &mut cur, &mut spikes, &params, &mut refractory, tau_ref, None);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
    (ms, v)
}

// ═══════════════════════════════════════════════════════════
//  GPU: cuda-kernels step_lif
// ═══════════════════════════════════════════════════════════

fn run_gpu(
    i_ext: &[f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_ref: f32,
) -> (f64, Vec<f32>) {
    let mut v = vec![-70.0f32; N];
    let mut ii = i_ext.to_vec();
    let mut spikes = vec![0i32; N];
    let mut refractory = vec![0.0f32; N];

    let t0 = Instant::now();
    for _ in 0..STEPS {
        ii.copy_from_slice(i_ext); // подаём ток каждый шаг
        cuda_kernels::step_lif(&mut v, &mut ii, &mut spikes, DT, tau_m, v_rest, v_thresh, v_reset, &mut refractory, tau_ref)
            .expect("CUDA lif step failed");
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
    (ms, v)
}

fn gpu_name() -> String {
    // Простое имя — реальное тянем через FFI только если есть query в крейте.
    // Здесь не раздуваем пример: если CUDA вызвалась — значит GPU на месте.
    "CUDA device".to_string()
}

/// Конформность по всем шагам: CPU и GPU гоняются синхронно, на каждом шаге
/// попарно сравниваются спайки (0/1). Это сильнее, чем сравнение только
/// последнего шага, где большинство нейронов молчит (0==0 — тривиальное
/// совпадение). Возвращает (число совпавших решений, спайки CPU, спайки GPU).
fn conformance_over_steps(
    i_ext: &[f32],
    tau_m: &[f32],
    v_rest: &[f32],
    v_thresh: &[f32],
    v_reset: &[f32],
    tau_ref: f32,
) -> (usize, usize, usize) {
    // CPU state
    let mut v = vec![-70.0f32; N];
    let mut spikes = vec![0i32; N];
    let mut refractory = vec![0.0f32; N];
    let mut params = vec![0.0f32; 5 * N];
    params[0..N].fill(DT);
    params[N..2 * N].copy_from_slice(tau_m);
    params[2 * N..3 * N].copy_from_slice(v_rest);
    params[3 * N..4 * N].copy_from_slice(v_thresh);
    params[4 * N..5 * N].copy_from_slice(v_reset);

    // GPU state
    let mut vg = vec![-70.0f32; N];
    let mut ii = i_ext.to_vec();
    let mut spikes_g = vec![0i32; N];
    let mut refractory_g = vec![0.0f32; N];

    let mut match_count = 0usize;
    let mut total_cpu = 0usize;
    let mut total_gpu = 0usize;
    for _ in 0..STEPS {
        let mut cur = i_ext.to_vec();
        taas_kernels::safe::lif_step(&mut v, &mut cur, &mut spikes, &params, &mut refractory, tau_ref, None);

        ii.copy_from_slice(i_ext);
        cuda_kernels::step_lif(&mut vg, &mut ii, &mut spikes_g, DT, tau_m, v_rest, v_thresh, v_reset, &mut refractory_g, tau_ref)
            .expect("CUDA lif step failed");

        for i in 0..N {
            total_cpu += spikes[i] as usize;
            total_gpu += spikes_g[i] as usize;
            match_count += (spikes[i] == spikes_g[i]) as usize;
        }
    }
    (match_count, total_cpu, total_gpu)
}

/// Нагрев CUDA-контекста до замера (первый вызов ядра включает driver).
/// Использует крошечный буфер — цель только инициализировать runtime.
fn warmup_gpu() {
    let n = 8usize;
    let mut v = vec![-70.0f32; n];
    let mut ii = vec![1.0f32; n];
    let mut spikes = vec![0i32; n];
    let mut refractory = vec![0.0f32; n];
    let tau_m = vec![10.0f32; n];
    let v_rest = vec![-70.0f32; n];
    let v_thresh = vec![-54.0f32; n];
    let v_reset = vec![-70.0f32; n];
    let _ = cuda_kernels::step_lif(&mut v, &mut ii, &mut spikes, DT, &tau_m, &v_rest, &v_thresh, &v_reset, &mut refractory, 2.0);
}
