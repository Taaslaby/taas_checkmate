/// Time Arrow — локальная необратимость HH динамики.
///
/// ## Идея
///
/// Запускаем HH FORWARD, записываем V(t), gates(t) на каждом шаге.
/// Для каждого шага i вычисляем backward ошибку:
///   V_bwd_pred = V[i+1] − dV(V[i+1], gates[i+1])·dt
///   ε_i = |V[i] − V_bwd_pred|
///
/// Если HH линейны → ε = 0. Если нет → ε > 0 (мера необратимости).
/// ε / dt² ≈ |V''(t)| — кривизна динамики.
///
/// Запуск: cargo run --package karmac --example time_arrow --release

use std::time::Instant;
use taas_kernels::neuron::hh;

const DT: f32 = 0.01;
const N_STEPS: usize = 20_000; // 200ms
const I_EXT: f32 = 15.0;
const N_GATES: usize = 3; // Na-m, Na-h, K-n

/// Создать state для одного нейрона: v, gates, канальные параметры.
fn make_state() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<hh::ChannelMeta>) {
    let channels = hh::standard_channels();
    let meta = hh::build_channel_meta(&channels);
    let init_gates = hh::steady_state_gates(-65.0, &channels);

    let mut gates = vec![0.0f32; N_GATES];
    for gi in 0..N_GATES {
        gates[gi] = init_gates[gi];
    }

    (
        vec![-65.0f32],      // v [1]
        gates,               // gates [N_GATES]
        vec![I_EXT],         // Ii [1]
        vec![0.3f32],        // g_leak [1]
        vec![-65.0f32],      // E_leak [1]
        vec![1.0f32],        // C_m [1]
        meta,
    )
}

/// Один шаг HH без spike detection (использует hh_poly_step).
/// Не выделяет память внутри.
fn step_hh_no_spikes(
    v: &mut [f32], gates: &mut [f32],
    Ii: &[f32], g_leak: &[f32], E_leak: &[f32], C_m: &[f32],
    meta: &[hh::ChannelMeta], dt: f32,
) {
    unsafe {
        hh::hh_poly_step(
            v.as_mut_ptr(),
            gates.as_mut_ptr(),
            Ii.as_ptr(),
            g_leak.as_ptr(),
            E_leak.as_ptr(),
            C_m.as_ptr(),
            1, meta, dt,
            6.3, // celsius (squid axon @ H&H 1952)
        );
    }
}

fn main() {
    let t0 = Instant::now();
    println!("═══ Time Arrow: Backward Error of HH ═══");
    println!();
    println!("Проверяем: локальная необратимость HH dynamics.");
    println!("  dt = {DT}ms, steps = {N_STEPS} ({:.1}ms)", N_STEPS as f32 * DT);
    println!("  I_ext = {I_EXT} µA/cm²");
    println!();

    // ─── Буферы ───
    let (mut v, mut gates, Ii, g_leak, E_leak, C_m, meta) = make_state();
    let mut v_trace = Vec::with_capacity(N_STEPS);
    let mut gate_traces: Vec<Vec<f32>> = (0..N_GATES).map(|_| Vec::with_capacity(N_STEPS)).collect();

    // ─── 1. Forward ───
    println!("─── 1. Forward ───");
    for _ in 0..N_STEPS {
        v_trace.push(v[0]);
        for gi in 0..N_GATES { gate_traces[gi].push(gates[gi]); }
        step_hh_no_spikes(&mut v, &mut gates, &Ii, &g_leak, &E_leak, &C_m, &meta, DT);
    }

    // Count spikes via forward trace
    let mut spike_count = 0u32;
    for i in 1..v_trace.len() {
        if v_trace[i - 1] >= 0.0 && v_trace[i] < 0.0 { spike_count += 1; }
    }
    let rate = spike_count as f32 / (N_STEPS as f32 * DT * 0.001);
    println!("  {spike_count} spikes ({rate:.0} Hz)");
    println!("  V range: {:.1} to {:.1} mV",
             v_trace.iter().cloned().fold(f32::MAX, f32::min),
             v_trace.iter().cloned().fold(f32::MIN, f32::max));

    // ─── 2. Backward error ───
    //
    // Для каждого i (1..N-1):
    //   1. Берём V[i] и gates[i]
    //   2. Запускаем HH_step — получаем V_next = V[i] + dV·dt
    //   3. Сравниваем V_next с V[i+1] — это forward error
    //   4. Вычисляем backward_prediction: V_bwd = V[i] − dV·dt = 2*V[i] − V_next
    //   5. Сравниваем V_bwd с V[i-1] — это backward error

    println!();
    println!("─── 2. Одношаговая ошибка ───");

    let mut v_scratch = vec![0.0f32];
    let mut g_scratch = vec![0.0f32; N_GATES];
    let mut fwd_errors = Vec::with_capacity(N_STEPS - 2);
    let mut bwd_errors = Vec::with_capacity(N_STEPS - 2);

    for i in 1..N_STEPS - 1 {
        // Forward: from V[i], gates[i] → V_next
        v_scratch[0] = v_trace[i];
        for gi in 0..N_GATES { g_scratch[gi] = gate_traces[gi][i]; }
        step_hh_no_spikes(&mut v_scratch, &mut g_scratch,
                          &Ii, &g_leak, &E_leak, &C_m, &meta, DT);
        let v_next = v_scratch[0];
        let fwd_eps = (v_trace[i + 1] - v_next).abs();
        fwd_errors.push(fwd_eps);

        // Backward: V_bwd = V[i] — dV·dt = 2*V[i] — V_next
        let v_bwd = 2.0 * v_trace[i] - v_next;
        let bwd_eps = (v_trace[i - 1] - v_bwd).abs();
        bwd_errors.push(bwd_eps);
    }

    let mean_fwd = fwd_errors.iter().sum::<f32>() / fwd_errors.len() as f32;
    let max_fwd = fwd_errors.iter().cloned().fold(0.0f32, f32::max);
    let mean_bwd = bwd_errors.iter().sum::<f32>() / bwd_errors.len() as f32;
    let max_bwd = bwd_errors.iter().cloned().fold(0.0f32, f32::max);

    println!("  Forward error:   mean = {:.8} mV, max = {:.8} mV", mean_fwd, max_fwd);
    println!("  Backward error:  mean = {:.8} mV, max = {:.8} mV", mean_bwd, max_bwd);
    println!("  Ratio (bwd/fwd): {:.1}×", mean_bwd / mean_fwd.max(1e-12));

    // ─── 3. Error by voltage regime ───
    println!();
    println!("─── 3. Ошибка по V-режимам ───");

    let mut e_spike = Vec::new();
    let mut e_upstroke = Vec::new();
    let mut e_sub = Vec::new();
    let mut e_ahp = Vec::new();

    for (i, &eps) in bwd_errors.iter().enumerate() {
        let vi = v_trace[i + 1]; // the state from which we step
        if vi > 0.0 { e_spike.push(eps); }
        else if vi > -40.0 { e_upstroke.push(eps); }
        else if vi > -60.0 { e_sub.push(eps); }
        else { e_ahp.push(eps); }
    }

    let report = |name: &str, data: &[f32]| {
        if data.is_empty() { println!("  {name:>15}: —"); return; }
        let m = data.iter().sum::<f32>() / data.len() as f32;
        println!("  {name:>15}: mean = {:.8} mV ({} pts)", m, data.len());
    };

    report("V > 0 (spike)", &e_spike);
    report("[-40, 0] (upstroke)", &e_upstroke);
    report("[-60, -40] (subthresh)", &e_sub);
    report("V < -60 (AHP)", &e_ahp);

    // ─── 4. Error distribution ───
    println!();
    println!("─── 4. Гистограмма backward ошибки ───");

    let max_err = bwd_errors.iter().cloned().fold(0.0f32, f32::max).max(1e-10);
    let n_bins = 20usize;
    let mut hist = vec![0usize; n_bins];
    for &e in &bwd_errors {
        let bi = ((e / max_err * n_bins as f32) as usize).min(n_bins - 1);
        hist[bi] += 1;
    }
    let peak = *hist.iter().max().unwrap_or(&1).max(&1);
    for (bi, &c) in hist.iter().enumerate() {
        let bar = "█".repeat(c * 40 / peak);
        let lo = bi as f32 / n_bins as f32 * max_err;
        println!("  {:>8.6} : {:>6} {}", lo, c, bar);
    }

    // ─── 5. Analysis ───
    println!();
    println!("─── 5. Анализ ───");

    let information_loss_rate = mean_bwd / DT; // mV/ms
    let time_to_forget = 1.0 / information_loss_rate.max(1e-10); // ms

    println!("  Mean backward error: {mean_bwd:.8} mV");
    println!("  dt = {DT} ms");
    println!("  Information loss: {information_loss_rate:.6} mV/ms");
    println!("  Memory time: {time_to_forget:.2} ms");

    if mean_bwd < 1e-6 {
        println!();
        println!("  ✅ HH is NEARLY REVERSIBLE (error < 1e-6 mV)");
        println!("  → Neural dynamics preserve information locally.");
    } else if mean_bwd < 1e-3 {
        println!();
        println!("  ⚡ WEAK irreversibility (error {:.4} mV)", mean_bwd);
        println!("  → Information decays slowly.");
    } else {
        println!();
        println!("  ❌ STRONG irreversibility (error {:.4} mV)", mean_bwd);
        println!("  → Neural dynamics have a clear arrow of time.");
    }

    let ratio = mean_bwd / mean_fwd.max(1e-12);
    if ratio > 100.0 {
        println!("  → Backward error is {:.0}× larger than forward — STRONG TIME ARROW.", ratio);
    } else if ratio > 10.0 {
        println!("  → Backward error is {:.0}× larger than forward — MODERATE TIME ARROW.", ratio);
    } else {
        println!("  → Forward ≈ backward — WEAK TIME ARROW (quasi-reversible).");
    }

    println!();
    println!("═══ Done ({:.2}s) ═══", t0.elapsed().as_secs_f64());
}
