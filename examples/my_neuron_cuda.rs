//! 🧬 Свой нейрон из ОДНОГО исходника → CPU SIMD + CUDA (NVRTC).
//!
//! Пишешь нейрон один раз с `#[dual_neuron]` — и получаешь сразу два
//! бэкенда из тех же самых уравнений:
//!   • CPU: f32x8 SIMD (`NeuralPopulation`) — как все модели зоопарка
//!   • GPU: автотранспиляция тела `step()` в CUDA C++ → NVRTC → PTX → ядро
//!
//! Прогон:
//! ```text
//! cargo run --example my_neuron_cuda --release --features cuda
//! ```
//!
//! Что делает:
//!   [1/3] паритет CPU ⟷ GPU — спайки должны совпасть по каждому нейрону,
//!         состояния — с точностью до нескольких ulp (FMA-контракция nvcc)
//!   [2/3] CPU SIMD: 262 144 нейрона × 2000 шагов (один поток)
//!   [3/3] GPU: персистентный путь (состояния живут в VRAM, CUDA Graph)

use std::time::Instant;

use pillars::dual_neuron;
use pillars::prelude::*;
use pillars_cuda::CompiledModel;

// ═══════════════════════════════════════════════════════════════
//  СВОЙ НЕЙРОН — всё, что нужно, это этот модуль.
//
//  MyLIF: LIF + адаптационный ток `u` (spike-frequency adaptation,
//  как у Izhikevich): на каждом спайке `u` подпрыгивает на `b` и
//  медленно затухает к a·(v − v_rest) — частота спадает сама.
// ═══════════════════════════════════════════════════════════════
#[dual_neuron]
pub mod my_lif {
    use pillars::core::compute::{ComputeNeuron, StepContext};
    use pillars::core::lane::SimdLane;
    use pillars::NeuronModel;

    /// Адаптивный LIF: медленный ток `u` тормозит генерацию при
    /// постоянном токе (адаптация частоты спайков).
    #[derive(NeuronModel)]
    pub struct MyLIFModel {
        #[state]
        pub v: f32,
        #[state]
        pub refr: f32,
        #[state]
        pub u: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub inv_tau_m: f32,
        #[param]
        pub v_rest: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub v_reset: f32,
        #[param]
        pub tau_ref: f32,
        #[param]
        pub a: f32,
        #[param]
        pub b: f32,
        #[param]
        pub inv_tau_u: f32,
    }

    impl ComputeNeuron for MyLIFModel {
        type State<S: SimdLane> = MyLIFState<S>;
        type Params<S: SimdLane> = MyLIFParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut MyLIFState<S>,
            params: &MyLIFParams<S>,
            ctx: &mut StepContext<S>,
        ) {
            let zero = S::zero();
            let is_refr = state.refr.cmp_gt(zero);

            // Forward Euler: dv = (v_rest − v + I − u) · inv_tau_m · dt
            let dv = (params.v_rest - state.v + ctx.i_syn - state.u) * params.inv_tau_m * ctx.dt();
            let v_new = state.v + dv;

            // Спайк — только вне рефрактерности
            let spike_raw = v_new.cmp_ge(params.v_thresh);
            let spike = is_refr.cond().then(zero).else_(spike_raw);
            ctx.emit_spike(spike);

            // Reset + подпрыг адаптационного тока на спайке
            let v_final = spike.cond().then(params.v_reset).else_(v_new);
            let u_inc = spike.cond().then(params.b).else_(zero);
            let du =
                (params.a * (state.v - params.v_rest) - state.u + u_inc) * params.inv_tau_u * ctx.dt();
            let u_new = state.u + du;

            state.v = is_refr.cond().then(state.v).else_(v_final);
            state.u = u_new;

            // Таймер рефрактерности
            let refr_dec = state.refr - ctx.dt();
            let refr_set = spike.cond().then(params.tau_ref).else_(refr_dec);
            state.refr = refr_set.max(zero);
        }
    }
}

pub use my_lif::*;

// ── Константы прогона ─────────────────────────────────────────────
const DT: f32 = 0.5; // мс — параметр [0], как во всех моделях зоопарка
const N_PARITY: usize = 64; // кратно 8 → capacity == n, stride == n
const N_BENCH: usize = 262_144;
const STEPS_PARITY: usize = 2_000;
const STEPS_BENCH: usize = 2_000;

fn main() {
    println!("╔══════════════════════════════════════════════════════");
    println!("║  MyLIF: свой нейрон — один исходник, два бэкенда");
    println!("║  {} состояний, {} параметров, ядро: `mylif_kernel`",
        MyLIFModel::N_STATE, MyLIFModel::N_PARAMS);
    println!("╚══════════════════════════════════════════════════════");
    println!();

    parity();
    bench();
}

fn build_pop(n: usize) -> NeuralPopulation<MyLIFModel> {
    let mut pop = NeuralPopulation::<MyLIFModel>::new(n);
    pop.fill_param(0, DT); // dt
    pop.fill_param(1, 1.0 / 10.0); // inv_tau_m
    pop.fill_param(2, -70.0); // v_rest
    pop.fill_param(3, -54.0); // v_thresh
    pop.fill_param(4, -70.0); // v_reset
    pop.fill_param(5, 2.0); // tau_ref
    pop.fill_param(6, 0.02); // a
    pop.fill_param(7, 0.2); // b
    pop.fill_param(8, 1.0 / 100.0); // inv_tau_u
    pop.fill_state(0, -70.0);
    pop.fill_state(1, 0.0);
    pop.fill_state(2, 0.0);
    pop
}

/// Половина нейронов под током 25, половина в тишине.
fn make_i_in(n: usize) -> Vec<f32> {
    (0..n).map(|i| if i % 2 == 0 { 25.0 } else { 0.0 }).collect()
}

/// SoA-снимок популяции (column-major) — формат, который ждёт CompiledModel.
fn snapshot(pop: &NeuralPopulation<MyLIFModel>) -> (Vec<f32>, Vec<f32>) {
    let states: Vec<f32> = (0..MyLIFModel::N_STATE)
        .flat_map(|c| pop.state_slice(c).to_vec())
        .collect();
    let params: Vec<f32> = (0..MyLIFModel::N_PARAMS)
        .flat_map(|c| pop.param_slice(c).to_vec())
        .collect();
    (states, params)
}

/// [1/3] Паритет: CPU (NeuralPopulation) ⟷ GPU (mylif_kernel) пошагово.
fn parity() {
    let mut pop = build_pop(N_PARITY);
    let i_in = make_i_in(N_PARITY);
    let (mut g_states, g_params) = snapshot(&pop);

    let model = CompiledModel::new(
        MY_LIF_CUDA_SOURCE,
        "mylif_kernel",
        MyLIFModel::N_STATE,
        MyLIFModel::N_PARAMS,
        N_PARITY,
        N_PARITY,
    )
    .expect("NVRTC не смог скомпилировать mylif_kernel — нужен CUDA-тулкит и GPU");

    let mut spikes = vec![0i32; N_PARITY];
    let mut g_spikes = vec![0i32; N_PARITY];
    let mut cpu_counts = vec![0i64; N_PARITY];
    let mut gpu_counts = vec![0i64; N_PARITY];
    let mut max_diff = 0.0f32;
    let mut max_bits = 0u32;

    for _ in 0..STEPS_PARITY {
        pop.step(&i_in, &mut spikes);
        for i in 0..N_PARITY {
            cpu_counts[i] += spikes[i] as i64;
        }
        model
            .step(&mut g_states, &g_params, &i_in, &mut g_spikes, DT)
            .expect("GPU шаг");
        for i in 0..N_PARITY {
            gpu_counts[i] += g_spikes[i] as i64;
        }
        for c in 0..MyLIFModel::N_STATE {
            for i in 0..N_PARITY {
                let cpu = pop.state_slice(c)[i];
                let gpu = g_states[c * N_PARITY + i];
                max_diff = max_diff.max((cpu - gpu).abs());
                max_bits = max_bits.max((cpu.to_bits() ^ gpu.to_bits()).count_ones());
            }
        }
    }

    let ok = cpu_counts == gpu_counts;
    let total: i64 = cpu_counts.iter().sum();
    println!("[1/3] Паритет CPU <-> GPU ({N_PARITY} нейронов x {STEPS_PARITY} шагов)");
    println!("      спайки совпали по каждому нейрону: {}", if ok { "ДА" } else { "НЕТ" });
    println!("      всего спайков: {total}");
    println!("      max |delta state| = {max_diff:.2e} | max diff bits = {max_bits}");
    if !ok {
        let bad = cpu_counts
            .iter()
            .zip(&gpu_counts)
            .filter(|(a, b)| a != b)
            .count();
        panic!("паритет нарушен: {bad} нейронов разошлись по спайкам");
    }
    println!();
}

/// [2/3] + [3/3] Пропускная способность CPU и GPU на 262K нейронов.
fn bench() {
    let mut pop = build_pop(N_BENCH);
    let i_in = make_i_in(N_BENCH);
    let mut spikes = vec![0i32; N_BENCH];

    // Разогрев, затем снимок стартового состояния для GPU (равный старт).
    for _ in 0..100 {
        pop.step(&i_in, &mut spikes);
    }
    let (snap_states, snap_params) = snapshot(&pop);

    // ── CPU SIMD (один поток) ──
    let t0 = Instant::now();
    let mut cpu_spikes: u64 = 0;
    for _ in 0..STEPS_BENCH {
        pop.step(&i_in, &mut spikes);
        cpu_spikes += spikes.iter().map(|&s| s as u64).sum::<u64>();
    }
    let cpu_secs = t0.elapsed().as_secs_f64();

    // ── GPU: персистентный путь (состояния и параметры в VRAM) ──
    let gpu = CompiledModel::new(
        MY_LIF_CUDA_SOURCE,
        "mylif_kernel",
        MyLIFModel::N_STATE,
        MyLIFModel::N_PARAMS,
        N_BENCH,
        N_BENCH,
    )
    .expect("NVRTC / CUDA");
    gpu.upload(&snap_states, &snap_params, &i_in).expect("upload");

    // Реалистичный цикл: спайки читаем на каждом шаге (D2H).
    let mut g_spikes = vec![0i32; N_BENCH];
    let t1 = Instant::now();
    let mut gpu_spikes: u64 = 0;
    for _ in 0..STEPS_BENCH {
        gpu.step_device(DT).expect("GPU шаг");
        gpu.download_spikes(&mut g_spikes).expect("D2H");
        gpu_spikes += g_spikes.iter().map(|&s| s as u64).sum::<u64>();
    }
    let gpu_loop = t1.elapsed().as_secs_f64();

    // Голый compute: без чтения спайков (только CUDA Graph replay).
    // Замер throughput, а не корректности — состояние устройства уже на
    // шаге 4000 (продолжение цикла выше), что на время не влияет.
    let t2 = Instant::now();
    for _ in 0..STEPS_BENCH {
        gpu.step_device(DT).expect("GPU шаг");
    }
    let gpu_core = t2.elapsed().as_secs_f64();

    // ── Отчёт ──
    let neuron_steps = N_BENCH as f64 * STEPS_BENCH as f64;
    let sim_ms = DT as f64 * STEPS_BENCH as f64; // 1000 мс = 1 биол. с
    let rt = |secs: f64| sim_ms / (secs * 1000.0);
    let mps = |secs: f64| neuron_steps / secs / 1e6;

    println!("[2/3] CPU SIMD (f32x8, 1 поток) — {N_BENCH} нейронов x {STEPS_BENCH} шагов");
    println!(
        "      {cpu_secs:.3} с | {:.1} млн нер-шагов/с | {:.2}x реалтайма | спайков: {cpu_spikes}",
        mps(cpu_secs),
        rt(cpu_secs)
    );
    println!();
    println!("[3/3] GPU (CUDA Graph, персистентный путь)");
    println!(
        "      цикл с чтением спайков: {gpu_loop:.3} с | {:.1} млн/с | {:.2}x реалтайма | спайков: {gpu_spikes}",
        mps(gpu_loop),
        rt(gpu_loop)
    );
    println!(
        "      голый compute:          {gpu_core:.3} с | {:.1} млн/с | {:.2}x реалтайма",
        mps(gpu_core),
        rt(gpu_core)
    );
    println!();
    println!("  скорость CPU/GPU (с чтением спайков): {:.1}x", cpu_secs / gpu_loop);
    let spike_diff = cpu_spikes.abs_diff(gpu_spikes);
    println!("  расхождение спайков: {spike_diff} (CPU {cpu_spikes} vs GPU {gpu_spikes})");
}
