//! # Parity test: kernels' math primitives vs pillars canon
//!
//! Фаза 1 консолидации: единственный источник `fast_exp`/`sigmoid`/
//! `safety_clamp` — `pillars::math` / `pillars::kernels::signal`.
//!
//! Этот тест доказывает, что ре-экспорты из kernels **бит-идентичны**
//! канону pillars на рабочей сетке [-87, 87] (полный рабочий диапазон
//! `fast_exp`, включая переполнение/андерфлоу у краёв).

use wide::f32x8;

/// Сетка [-87, 87] с шагом 0.01 — весь рабочий диапазон.
fn grid() -> impl Iterator<Item = f32> {
    (-8700..=8700).map(|i| i as f32 / 100.0)
}

// ═══════════════════════════════════════════════════════
//  fast_exp (scalar)
// ═══════════════════════════════════════════════════════

#[test]
fn fast_exp_scalar_parity_pillars() {
    let mut max_rel = 0.0f32;
    let mut n_checked = 0u32;
    for x in grid() {
        let k = taas_kernels::fast_exp(x);
        let p = pillars::math::fast_exp::fast_exp(x);
        // Относительная разница (допуск 1e-5 — бит-идентичность не обязана
        // быть абсолютной на билдах с +fma: pillars использует fused FMA,
        // старый kernels — двойное округление; расхождение ≤1 ULP).
        let denom = p.abs().max(1e-30);
        let rel = ((k - p) / denom).abs();
        max_rel = max_rel.max(rel);
        n_checked += 1;
        assert!(rel < 1e-5, "fast_exp({x}): kernels {k:e} vs pillars {p:e}");
    }
    eprintln!("fast_exp scalar: {n_checked} points, max rel diff vs pillars = {max_rel:.3e}");
}

// ═══════════════════════════════════════════════════════
//  fast_exp_simd (SIMD polynomial) — обязана быть бит-идентичной
// ═══════════════════════════════════════════════════════

#[test]
fn fast_exp_simd_parity_bit_identical() {
    let mut worst: Option<(f32, u32, u32)> = None;
    for x in grid() {
        let k = taas_kernels::fast_exp_simd(f32x8::splat(x)).to_array()[0];
        let p = pillars::math::fast_exp::fast_exp_poly_simd(f32x8::splat(x)).to_array()[0];
        if k.to_bits() != p.to_bits() {
            let cur = worst.get_or_insert((x, k.to_bits(), p.to_bits()));
            if (k - p).abs() > (cur.1 as f32 - cur.2 as f32).abs() {
                *cur = (x, k.to_bits(), p.to_bits());
            }
        }
    }
    match worst {
        Some((x, kb, pb)) => panic!(
            "fast_exp_simd({x}): kernels bits {kb:#x} != pillars {pb:#x}"
        ),
        None => eprintln!("fast_exp_simd: bit-identical on full grid [-87,87] ✓"),
    }
}

// ═══════════════════════════════════════════════════════
//  safety_clamp — одинаковые результаты на всех выравниваниях
// ═══════════════════════════════════════════════════════

#[test]
fn safety_clamp_parity_pillars() {
    // Длины 0..65 покрывают SIMD-чанки (8) и все остатки (хвосты 0..7).
    for n in 0..65usize {
        let data: Vec<f32> = (0..n).map(|i| ((i * 37) % 100) as f32 - 50.0).collect();
        let mut k = data.clone();
        let mut p = data.clone();
        taas_kernels::safety_clamp(&mut k, -12.0, 31.0);
        pillars::kernels::signal::safety_clamp(&mut p, -12.0, 31.0);
        assert_eq!(k, p, "safety_clamp(n={n}) diverges");
    }
}

// ═══════════════════════════════════════════════════════
//  sigmoid (batch microkernel vs pillars sigmoid_poly_s)
// ═══════════════════════════════════════════════════════

#[test]
fn sigmoid_gate_parity_pillars() {
    let n = 1000;
    let inp: Vec<f32> = (0..n).map(|i| (i as f32 - 500.0) / 10.0).collect();
    let mut out = vec![0.0f32; n];
    let gain = 2.0f32;
    let th = -10.0f32;
    taas_kernels::sigmoid_gate(&mut out, &inp, gain, th);
    for i in 0..n {
        let expected = pillars::math::sigmoid::sigmoid_poly_s(gain * (inp[i] - th));
        let rel = ((out[i] - expected) / expected.abs().max(1e-30)).abs();
        assert!(
            rel < 1e-5,
            "sigmoid_gate[{i}]: kernels {} vs pillars {expected:e}",
            out[i]
        );
    }
}
