//! Parity: pillars-канон synapse-family vs inline legacy-эталоны.
//!
//! Волна 8, транш 1a: `delay`, `nmda`, `tsodyks_markram` переведены с
//! легаси-кернелов `kernels/neuron/*` на pillars-канон
//! (`pillars::kernels::delay`, `pillars::zoo::synapses::*`).
//! Транш 3e: `safe.rs` удалён — тест замораживает pillars-канон напрямую
//! (delay — обычный вызов, nmda/tsodyks — unsafe-вызов pillars).

#[test]
fn delay_circular_roundtrip() {
    let buf: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // pos % n — канон (легаси паниковал при pos >= n).
    for pos in 0..16 {
        let mut out = vec![0.0f32; 16];
        pillars::kernels::delay::delay_buffer_circular(&mut out, &buf, pos);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, buf[(pos + i) % 16], "circular pos={pos} i={i}");
        }
    }
}

#[test]
fn delay_multi_tap_roundtrip() {
    let buf: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let delays = vec![0i32, 3, 7, 9, 100, -2];
    let mut out = vec![0.0f32; delays.len()];
    pillars::kernels::delay::delay_buffer_multi_tap(&mut out, &buf, &delays);
    // Канон (pillars::kernels::delay): каждый tap клампится к [0, n-1];
    // отрицательный → usize-каст, ≥ n → n-1, ноль → текущий сэмпл.
    assert_eq!(out[0], buf[0], "delay 0 → current");
    assert_eq!(out[1], buf[3], "delay 3");
    assert_eq!(out[4], buf[9], "delay 100 → clamp n-1");
    assert_eq!(out[5], buf[9], "delay -2 → usize-cast огромен → clamp n-1");
}

#[test]
fn nmda_pillars_roundtrip() {
    let n = 13; // SIMD chunks + scalar tail
    let v: Vec<f32> = (0..n).map(|i| -70.0 + i as f32 * 10.0).collect();
    let g_act = vec![0.5f32; n];
    let mut params = vec![0.0f32; n * 4];
    for i in 0..n {
        params[i * 4] = 0.5; // g_nmda
        params[i * 4 + 1] = 0.0; // E_syn
        params[i * 4 + 2] = 1.0; // mg_conc
    }

    let mut out = vec![0.0f32; n];
    unsafe {
        pillars::zoo::synapses::nmda::nmda_mg_block_step(
            v.as_ptr(),
            g_act.as_ptr(),
            out.as_mut_ptr(),
            params.as_ptr(),
            n,
        );
    }
    // Mg²⁺-блок: при v ≈ -70 (mg действует) ток мал; при v > 0 (блок снят) — большой.
    assert!(out[0].abs() < out[n - 1].abs(), "NMDA Mg-block: v↑ → I↑");
    assert!(
        out[n - 1] > 0.0,
        "NMDA: ток при v=+50 должен быть положительным"
    );
}

/// tsodyks: pillars-канон (Exp Euler) — качественная проверка депрессии/фасилитации.
/// Развилка «Exp Euler vs легаси Forward Euler» задокументирована в волне 8.
#[test]
fn tsodyks_pillars_roundtrip() {
    let n = 13;
    let spikes = vec![1i32; n];
    let mut params = vec![0.0f32; n * 4];
    for i in 0..n {
        params[i * 4] = 0.5; // U
        params[i * 4 + 1] = 100.0; // tau_d
        params[i * 4 + 2] = 50.0; // tau_f
        params[i * 4 + 3] = 1.0; // W_base
    }
    let dt = 0.1;

    let mut x = vec![0.8f32; n];
    let mut u = vec![0.3f32; n];
    let mut weff = vec![0.0f32; n];
    unsafe {
        pillars::zoo::synapses::tsodyks_markram::tsodyks_markram_step(
            x.as_mut_ptr(),
            u.as_mut_ptr(),
            spikes.as_ptr(),
            weff.as_mut_ptr(),
            params.as_ptr(),
            n,
            dt,
        );
    }
    // Депрессия: x падает после спайка; w_eff > 0.
    assert!(
        x[0] < 0.8,
        "tsodyks: x должен упасть после спайка: {:.4}",
        x[0]
    );
    assert!(weff[0] > 0.0, "tsodyks: w_eff должен быть > 0");
}
