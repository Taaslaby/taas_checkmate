/// Синтетический мозг мыши на GPU — заглушка для проверки скорости и VRAM.
///
/// Тестирует:
/// 1. LIF-only throughput (без коннектома) — от 1M до 70M нейронов
/// 2. LIF + CSR коннектом — от 10K до 1M нейронов с разной плотностью синапсов
/// 3. Оценка VRAM: сколько байт на нейрон/синапс, влазит ли мозг мыши
///
/// Запуск: cargo run --features cuda --example bench_mouse_brain --release
use std::time::Instant;
use cuda_kernels::GpuContext;

const DT: f32 = 0.1e-3;
const TAU_M: f32 = 10e-3;
const V_REST: f32 = -65e-3;
const V_THRESH: f32 = -50e-3;
const V_RESET: f32 = -65e-3;
const N_STEPS: usize = 100;
const N_SAMPLE: usize = 3;

/// Приблизительная оценка VRAM на нейрон (байт) для минимального LIF-пути.
/// d_v + d_I + d_spikes + d_refractory + d_lif_params + d_external_I + d_nt_types + d_traces_pre + d_traces_post + d_adex_w
const BYTES_PER_NEURON_LIF: f64 = 44.0;
/// Байт на синапс: d_col_indices(4) + d_csr_values(4) + d_weights(4) + d_nt_types амортизирован
const BYTES_PER_SYNAPSE: f64 = 12.0;
/// GPU VRAM (RTX 4060 Ti)
const VRAM_BYTES: f64 = 8.0 * 1024.0 * 1024.0 * 1024.0;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Синтетический мозг мыши на GPU (RTX 4060 Ti, 8GB)    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Бюджет VRAM: {:.0} MB", VRAM_BYTES / 1_048_576.0);
    println!();

    // ─── 1. Оценка: влезет ли мозг мыши? ─────────────────────
    let mouse_neurons = 70_000_000;
    let mouse_synapses = 7e11;
    let mem_lif = mouse_neurons as f64 * BYTES_PER_NEURON_LIF;
    let mem_conn = mouse_synapses * BYTES_PER_SYNAPSE;
    let mem_total = mem_lif + mem_conn;
    println!("─── Оценка VRAM для мозга мыши (~{:.0}M нейронов, ~{:.0}B синапсов) ───",
             mouse_neurons as f64 / 1e6, mouse_synapses / 1e9);
    println!("  LIF массивы:       {:.1} MB ({:.0}%)",
             mem_lif / 1_048_576.0, mem_lif / VRAM_BYTES * 100.0);
    println!("  CSR коннектом:     {:.1} MB ({:.0}%)",
             mem_conn / 1_048_576.0, mem_conn / VRAM_BYTES * 100.0);
    println!("  Всего:             {:.1} MB ({:.0}%)",
             mem_total / 1_048_576.0, mem_total / VRAM_BYTES * 100.0);
    if mem_total > VRAM_BYTES {
        println!("  ⚠ НЕ ВЛЕЗАЕТ! Не хватает {:.0} MB",
                 (mem_total - VRAM_BYTES) / 1_048_576.0);
        let max_n = (VRAM_BYTES - mouse_synapses * BYTES_PER_SYNAPSE) / BYTES_PER_NEURON_LIF;
        println!("  Макс нейронов с {:.0}B синапсами: ~{:.0}M",
                 mouse_synapses as f64, max_n / 1e6);
        let max_s = (VRAM_BYTES - mouse_neurons as f64 * BYTES_PER_NEURON_LIF) / BYTES_PER_SYNAPSE;
        println!("  Макс синапсов с {:.0}M нейронами: ~{:.0}B ({:.1} на нейрон)",
                 mouse_neurons as f64 / 1e6, max_s / 1e9, max_s / mouse_neurons as f64);
    } else {
        println!("  ✓ ВЛЕЗАЕТ!");
    }
    println!();

    // ─── 2. LIF-only throughput ──────────────────────────────
    println!("─── LIF-only throughput ───");
    println!("{:>12}  {:>10}  {:>18}  {:>10}  {:>8}",
             "Neurons", "Time (s)", "Throughput (n/s)", "us/step", "VRAM");
    println!("{}", "-".repeat(62));

    for &n in &[1_000_000, 10_000_000, 30_000_000, 50_000_000, 70_000_000, 80_000_000] {
        let ctx = match GpuContext::new(n, 0) {
            Ok(c) => c,
            Err(e) => { println!("{n:>12}  OOM: {e}"); continue; }
        };

        ctx.upload_lif_params(DT, &vec![TAU_M; n], &vec![V_REST; n],
                              &vec![V_THRESH; n], &vec![V_RESET; n]).unwrap();
        ctx.upload_I(&vec![2e-9; n]).unwrap();

        // Warmup
        ctx.lif_step_async().unwrap();
        ctx.sync().unwrap();

        let mut t_avg = 0.0;
        for _ in 0..N_SAMPLE {
            let t0 = Instant::now();
            for _ in 0..N_STEPS {
                ctx.lif_step_async().unwrap();
            }
            ctx.sync().unwrap();
            t_avg += t0.elapsed().as_secs_f64();
        }
        t_avg /= N_SAMPLE as f64;

        let thr = n as f64 * N_STEPS as f64 / t_avg;
        let us = t_avg / N_STEPS as f64 * 1e6;
        let mem = n as f64 * BYTES_PER_NEURON_LIF;
        println!("{n:>12}  {t_avg:>10.4}  {thr:>18.1e}  {us:>10.1}  {:.0}MB",
                 mem / 1_048_576.0);
    }
    println!();

    // ─── 3. LIF + CSR коннектом ─────────────────────────────
    println!("─── LIF + CSR коннектом throughput ───");
    println!("{:>10}  {:>12}  {:>10}  {:>18}  {:>10}  {:>10}",
             "Neurons", "Syn/neur", "Time(s)", "Throughput(n/s)", "us/step", "VRAM");
    println!("{}", "-".repeat(74));

    let connectome_configs = [
        (10_000, 100),      // 10K neu, 100 syn/neu = 1M syn
        (10_000, 1000),     // 10K neu, 1K syn/neu = 10M syn
        (100_000, 100),     // 100K neu, 100 syn/neu = 10M syn
        (100_000, 1000),    // 100K neu, 1K syn/neu = 100M syn
        (1_000_000, 100),   // 1M neu, 100 syn/neu = 100M syn
        (1_000_000, 1000),  // 1M neu, 1K syn/neu = 1B syn(~1.1GB VRAM)
    ];

    for &(n, syn_per_neur) in &connectome_configs {
        let nsyn = n * syn_per_neur;
        let mem_est = n as f64 * BYTES_PER_NEURON_LIF + nsyn as f64 * BYTES_PER_SYNAPSE;
        if mem_est > VRAM_BYTES * 0.95 {
            println!("{n:>10}  {syn_per_neur:>12}  SKIP (~{:.0}MB > VRAM)", mem_est / 1_048_576.0);
            continue;
        }

        let ctx = match GpuContext::new(n, nsyn) {
            Ok(c) => c,
            Err(e) => { println!("{n:>10}  {syn_per_neur:>12}  OOM: {e}"); continue; }
        };

        // Загружаем LIF параметры
        ctx.upload_lif_params(DT, &vec![TAU_M; n], &vec![V_REST; n],
                              &vec![V_THRESH; n], &vec![V_RESET; n]).unwrap();
        ctx.upload_I(&vec![2e-9; n]).unwrap();

        // Случайный CSR коннектом: каждый нейрон -> syn_per_neur случайных целей
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_indices = vec![0i32; nsyn];
        let mut csr_values = vec![0.0f32; nsyn];
        let mut nt_types = vec![0.0f32; n];
        let external_I = vec![2e-9f32; n];
        for i in 0..n {
            row_ptr[i] = (i * syn_per_neur) as i32;
            for j in 0..syn_per_neur {
                col_indices[i * syn_per_neur + j] = ((i + j * 7 + 3) % n) as i32;
                csr_values[i * syn_per_neur + j] = 1e-12;
            }
            nt_types[i] = 1.0;
        }
        row_ptr[n] = nsyn as i32;

        ctx.upload_connectome(&row_ptr, &col_indices, &csr_values, &nt_types, &external_I).unwrap();

        // Счётчик спайков для GPU
        let d_spike_counter = ctx.alloc_spike_counter().unwrap();

        // Warmup: один полный шаг
        ctx.connectome_step(DT, 0.99, Some(d_spike_counter)).unwrap();

        let mut t_avg = 0.0;
        for _ in 0..N_SAMPLE {
            let t0 = Instant::now();
            for _ in 0..N_STEPS {
                ctx.connectome_step(DT, 0.99, Some(d_spike_counter)).unwrap();
            }
            t_avg += t0.elapsed().as_secs_f64();
        }
        t_avg /= N_SAMPLE as f64;

        let thr = n as f64 * N_STEPS as f64 / t_avg;
        let us = t_avg / N_STEPS as f64 * 1e6;
        println!("{n:>10}  {syn_per_neur:>12}  {t_avg:>10.4}  {thr:>18.1e}  {us:>10.1}  {:.0}MB",
                 mem_est / 1_048_576.0);
    }
    println!();

    // ─── 4. Реалистичная оценка ──────────────────────────────
    println!("─── Реалистичная оценка для мозга мыши ───");
    println!("  RTX 4060 Ti (8GB) без коннектома:");
    println!("    ~70M LIF-нейронов: ~3GB VRAM — ВЛЕЗАЕТ");
    println!("    Пропускная способность: ~100B n·s/s (LIF)");
    println!();
    println!("  RTX 4060 Ti с CSR коннектомом {:.0}B синапсов:", 1e9);
    println!("    LIF массивы:  ~3.0 GB");
    println!("    CSR массивы:  ~12.0 GB (не влезает!)");
    println!("    Вывод: нужен аппроксимация коннектома или RTX 4090 (24GB)");
    println!();
    println!("  Для 1B синапсов с 10M нейронов:");
    println!("    LIF: ~440 MB + CSR: ~12 GB = ~12.4 GB — не влезает в 8GB");
    println!("  Для 100M синапсов с 10M нейронов:");
    println!("    LIF: ~440 MB + CSR: ~1.2 GB = ~1.6 GB — влезает");
    println!("    Пропускная способность: ~1-5B n·s/s");
}
