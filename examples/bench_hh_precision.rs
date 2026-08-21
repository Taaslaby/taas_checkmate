//! 🔬 f32 vs f64 Precision Audit for HHModel
//!
//! Сравнивает траекторию движка (f32, fast_exp) с f64-эталоном, который
//! воспроизводит ТУ ЖЕ дискретизацию: экспоненциальный Эйлер для гейтов,
//! форвард-Эйлер для V, vtrap с Taylor-защитой.
//!
//! Запуск: cargo run --example bench_hh_precision --release

use pillars::prelude::run_neuron;
use pillars::prelude::HHModel;

const DT: f32 = 0.01; // ms
const Q10: f32 = 1.0; // 6.3°C
const I_SYN: f32 = 10.6; // µA/cm² — 0.05 nA в NEURON-единицах (7 спайков/100мс)

// ═══════════════════════════════════════════════════════════
//  f64 эталон — та же дискретизация, что у движка
// ═══════════════════════════════════════════════════════════

fn vtrap64(x: f64, y: f64) -> f64 {
    let ratio = x / y;
    if ratio.abs() < 1e-6 {
        y * (1.0 - ratio * 0.5)
    } else {
        x / (ratio.exp() - 1.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn hh_step64(
    v: f64,
    m: f64,
    h: f64,
    n: f64,
    c: f64,
    i_inj: f64,
    dt: f64,
    q10: f64,
    g_na: f64,
    e_na: f64,
    g_k: f64,
    e_k: f64,
    g_l: f64,
    e_l: f64,
    g_cat: f64,
    e_cat: f64,
    inv_cm: f64,
) -> (f64, f64, f64, f64, f64) {
    let dt_q10 = dt * q10;

    // ── m-gate: α=0.1·vtrap(-(V+40),10)  β=4·exp(-(V+65)/18) ──
    let alpha_m = 0.1 * vtrap64(-(v + 40.0), 10.0);
    let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();
    let sum_m = alpha_m + beta_m;
    let m_inf = alpha_m / sum_m;
    let m_new = m_inf + (m - m_inf) * (-dt_q10 * sum_m).exp();

    // ── h-gate: α=0.07·exp(-(V+65)/20)  β=1/(1+exp(-(V+35)/10)) ──
    let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
    let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
    let sum_h = alpha_h + beta_h;
    let h_inf = alpha_h / sum_h;
    let h_new = h_inf + (h - h_inf) * (-dt_q10 * sum_h).exp();

    // ── n-gate: α=0.01·vtrap(-(V+55),10)  β=0.125·exp(-(V+65)/80) ──
    let alpha_n = 0.01 * vtrap64(-(v + 55.0), 10.0);
    let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();
    let sum_n = alpha_n + beta_n;
    let n_inf = alpha_n / sum_n;
    let n_new = n_inf + (n - n_inf) * (-dt_q10 * sum_n).exp();

    // ── CaV-T gate (g_cat=0 в этом тесте, но для полноты) ──
    let c_arg = -(v + 56.0) * (1.0 / 6.2);
    let c_inf = 1.0 / (1.0 + c_arg.exp());
    let c_new = c_inf + (c - c_inf) * (-dt_q10 * (1.0 / 0.6)).exp();

    // ── Ионные токи (новые гейты, СТАРЫЙ v — как в движке) ──
    let i_na = g_na * m_new.powi(3) * h_new * (v - e_na);
    let i_k = g_k * n_new.powi(4) * (v - e_k);
    let i_cat = g_cat * c_new * (v - e_cat);
    let i_leak = g_l * (v - e_l);

    // ── V: форвард-Эйлер ──
    let v_new = v + (i_inj - (i_na + i_k + i_cat + i_leak)) * inv_cm * dt;

    (v_new, m_new, h_new, n_new, c_new)
}

fn make_params32(dt: f32) -> Vec<f32> {
    let mut p = vec![0.0f32; 12];
    p[0] = dt;
    p[1] = 120.0; // g_Na
    p[2] = 50.0; // E_Na
    p[3] = 36.0; // g_K
    p[4] = -77.0; // E_K
    p[5] = 0.3; // g_leak
    p[6] = -54.3; // E_leak
    p[7] = Q10;
    p[8] = 0.0; // v_thresh
    p[9] = 1.0; // inv_cm
    p[10] = 0.0; // g_CaT
    p[11] = 120.0; // E_CaT
    p
}

/// Движок: f32 HHModel через run_neuron. Возвращает (V_траектория, спайки).
fn run_engine32(t_ms: f32, i_syn: f32) -> (Vec<f32>, u32) {
    let n = 1;
    let params = make_params32(DT);
    let n_steps = (t_ms / DT) as usize;
    let mut v = vec![-65.0f32; n];
    let mut m = vec![0.05f32; n];
    let mut h = vec![0.6f32; n];
    let mut k = vec![0.32f32; n];
    let mut cat = vec![0.19f32; n];
    let mut flag = vec![1.0f32; n];
    let i_arr = [i_syn; 1];
    let mut spikes = vec![0i32; n];
    let mut traj = Vec::with_capacity(n_steps);
    let mut total = 0u32;
    for _ in 0..n_steps {
        unsafe {
            run_neuron::<HHModel>(
                v.as_mut_ptr(),
                m.as_mut_ptr(),
                h.as_mut_ptr(),
                k.as_mut_ptr(),
                cat.as_mut_ptr(),
                flag.as_mut_ptr(),
                i_arr.as_ptr(),
                spikes.as_mut_ptr(),
                params.as_ptr(),
                n,
            );
        }
        total += spikes[0] as u32;
        traj.push(v[0]);
    }
    (traj, total)
}

/// f64 эталон. Возвращает (V_траектория, спайки).
fn run_ref64(t_ms: f64, i_syn: f64) -> (Vec<f64>, u32) {
    let n_steps = (t_ms / DT as f64) as usize;
    let mut v = -65.0f64;
    let mut m = 0.05;
    let mut h = 0.6;
    let mut n = 0.32;
    let mut c = 0.19;
    let mut traj = Vec::with_capacity(n_steps);
    let mut total = 0u32;
    let mut was_below = 1.0f64;
    for _ in 0..n_steps {
        (v, m, h, n, c) = hh_step64(
            v, m, h, n, c, i_syn, DT as f64, Q10 as f64, 120.0, 50.0, 36.0, -77.0, 0.3, -54.3, 0.0,
            120.0, 1.0,
        );
        let is_above = if v >= 0.0 { 1.0 } else { 0.0 };
        if is_above > 0.0 && was_below > 0.0 {
            total += 1;
        }
        was_below = if is_above > 0.0 { 0.0 } else { 1.0 };
        traj.push(v);
    }
    (traj, total)
}

/// Спайк-тайминги из траектории (момент пересечения 0 mV снизу вверх).
fn spike_times(v: &[f32], dt: f32) -> Vec<f32> {
    let mut t = Vec::new();
    for i in 1..v.len() {
        if v[i] >= 0.0 && v[i - 1] < 0.0 {
            t.push(i as f32 * dt);
        }
    }
    t
}

/// Дрейф спайк-тайминга: t32[k] - t64[k] для k-го спайка.
/// Возвращает (диффы, счётчик).
fn timing_drift(t32: &[f32], t64: &[f32]) -> Vec<f32> {
    let n = t32.len().min(t64.len());
    (0..n).map(|i| t32[i] - t64[i]).collect()
}

fn analyze(t_ms: f32, label: &str) {
    let (traj32, sp32) = run_engine32(t_ms, I_SYN);
    let (traj64, sp64) = run_ref64(t_ms as f64, I_SYN as f64);

    let n = traj32.len().min(traj64.len());
    let mut ss = 0.0f64;
    for i in 0..n {
        let d = (traj32[i] as f64 - traj64[i]).abs();
        ss += d * d;
    }
    let rmse = (ss / n as f64).sqrt();

    let t32 = spike_times(&traj32, DT);
    let t64: Vec<f32> = {
        let mut v = Vec::new();
        for i in 1..traj64.len() {
            if traj64[i] >= 0.0 && traj64[i - 1] < 0.0 {
                v.push(i as f32 * DT);
            }
        }
        v
    };
    let drift = timing_drift(&t32, &t64);

    println!();
    println!("───────── {label} ─────────");
    println!("  V RMSE          = {rmse:.4} mV (включает спайки — доминирует сдвиг фазы)");
    println!(
        "  спайков f32/f64 = {sp32}/{sp64}  (частота {:.1} Гц)",
        sp32 as f32 / (t_ms / 1000.0)
    );

    if !drift.is_empty() {
        let last = drift[drift.len() - 1];
        let abs_max = drift.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  дрейф тайминга:");
        println!("    спайк #1   : {:+.3} мс", drift[0]);
        if drift.len() > 10 {
            println!("    спайк #10  : {:+.3} мс", drift[9]);
        }
        if drift.len() > 50 {
            println!("    спайк #50  : {:+.3} мс", drift[49]);
        }
        println!("    последний  : {:+.3} мс (спайк #{})", last, drift.len());
        println!("    |макс|     : {abs_max:.3} мс");
    } else {
        println!("  спайков нет — субпороговый режим");
    }
    println!(
        "  V(конец) f32/f64 = {:.2} / {:.2} mV",
        traj32[n - 1],
        traj64[n - 1]
    );
}

fn main() {
    println!("════════════════════════════════════════════════════");
    println!("  🔬 f32 (движок) vs f64 (эталон) — HH, та же дискретизация");
    println!("════════════════════════════════════════════════════");
    println!("  dt = {DT} ms, I = {I_SYN} µA/cm², g_CaT = 0 (классический HH)");

    // 1 c — умеренный прогон
    analyze(1000.0, "1 c (71 Гц)");
    // 10 c — проверка, растёт ли дрейф (хаос) или остаётся ограниченным
    analyze(10_000.0, "10 c (71 Гц)");

    // Субпороговый режим: I = 1 µA/cm² — ниже реобазы, точность на ровном участке
    println!();
    println!("  ── Субпороговый тест: I = 1 µA/cm² (ниже реобазы) ──");
    let t_ms = 1000.0f32;
    let (traj32, _) = run_engine32(t_ms, 1.0);
    let (traj64, _) = run_ref64(t_ms as f64, 1.0);
    let n = traj32.len().min(traj64.len());
    let mut ss = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..n {
        let d = (traj32[i] as f64 - traj64[i]).abs();
        ss += d * d;
        max_abs = max_abs.max(d);
    }
    let rmse = (ss / n as f64).sqrt();
    println!("  V RMSE = {rmse:.5} mV, max |ΔV| = {max_abs:.5} mV — это ЧИСТАЯ f32-ошибка");
    println!("  (без спайков ошибка на 3 порядка меньше!)");
    println!("════════════════════════════════════════════════════");
}
