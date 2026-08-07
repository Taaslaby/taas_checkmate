//! Golden/regression anchor for the **pillars decode layer**.
//!
//! Фаза 2 консолидации: единственный источник декодеров — `pillars::kernels::decode`.
//! Этот тест **замораживает** семантику pillars-декоder'ов против НЕЗАВИСИМЫХ скалярных
//! эталонных циклов (написаны вручную прямо здесь), а НЕ layer-to-layer — иначе после
//! того как legacy `decode.rs` начнёт делегировать в pillars, сравнение стало бы
//! тавтологией. Тест переживает удаление legacy и меняется только когда осознанно
//! меняют формулу канона.
//!
//! Размеры выбраны немножественными 8 где возможно — чтобы гонять и SIMD-хвост.

use pillars::kernels::decode;

/// Детерминированный псевдослучайный массив с обоими знаками, в стабильном диапазоне.
fn data(n: usize, seed: i64) -> Vec<f32> {
    // LCG без внешнего rand: x_{k+1} = (a·x_k + c) mod m, отображение в [-2, 2].
    let mut s = seed.max(1);
    let a = 6364136223846793005i64;
    let c = 1442695040888963407i64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(a).wrapping_add(c);
            // Старшие 32 бита -> u32 -> [-2, 2)
            ((s >> 33) as u32 as f32 / (u32::MAX as f32 / 4.0)) - 2.0
        })
        .collect()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for i in 0..got.len() {
        let d = (got[i] - want[i]).abs();
        let scale = want[i].abs().max(1e-6);
        assert!(
            d <= atol * scale.max(1.0),
            "{name}[{i}]: got {} want {} (diff {:.2e} > atol {:.2})",
            got[i],
            want[i],
            d,
            atol
        );
    }
}

// ─────────────────────────────────────────────
// 1. t3_complex_decoder: out[o] = bias[o] + Σ_j amps[j]·cos[j]·W[o·nin+j]
// ─────────────────────────────────────────────
#[test]
fn t3_complex_decoder_anchor() {
    let n_in = 13; // не кратно 8 → SIMD-хвост
    let n_out = 4;
    let amps = data(n_in, 1);
    let cos = data(n_in, 2);
    let w = data(n_in * n_out, 3);
    let b = data(n_out, 4);

    let mut out = vec![0.0f32; n_out];
    decode::t3_complex_decoder(&mut out, &amps, &cos, &w, &b);

    let mut want = b.clone();
    for o in 0..n_out {
        let mut acc = b[o];
        for j in 0..n_in {
            acc += amps[j] * cos[j] * w[o * n_in + j];
        }
        want[o] = acc;
    }
    assert_close("t3_complex_decoder", &out, &want, 1e-4);
}

// ─────────────────────────────────────────────────
// 2. t3_pst_features: os = sin(p1[i]-p2[j]), oc = cos(...)
// ─────────────────────────────────────────────────
#[test]
fn t3_pst_features_anchor() {
    let n1 = 5;
    let n2 = 3;
    let p1 = data(n1, 5);
    let p2 = data(n2, 6);

    let mut os = vec![0.0f32; n1 * n2];
    let mut oc = vec![0.0f32; n1 * n2];
    decode::t3_pst_features(&mut os, &mut oc, &p1, &p2);

    let mut ws = vec![0.0f32; n1 * n2];
    let mut wc = vec![0.0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let d = p1[i] - p2[j];
            ws[i * n2 + j] = d.sin();
            wc[i * n2 + j] = d.cos();
        }
    }
    assert_close("t3_pst_features sin", &os, &ws, 1e-4);
    assert_close("t3_pst_features cos", &oc, &wc, 1e-4);
}

// ────────────────────────────────────────────────
// 3. t3_lms_train: w += lr·error·s; bias[0] += lr·error
// ────────────────────────────────────────────────
#[test]
fn t3_lms_train_anchor() {
    let n = 15;
    let mut w = data(n, 7);
    let mut bias = vec![0.3f32];
    let s = data(n, 8);
    let (err, lr) = (0.7f32, 0.03f32);

    decode::t3_lms_train(&mut w, &mut bias, &s, err, lr);

    let mut want_w = data(n, 7);
    for i in 0..n {
        want_w[i] += lr * err * s[i];
    }
    let mut want_bias = 0.3f32;
    want_bias += lr * err;
    assert_close("t3_lms_train w", &w, &want_w, 1e-5);
    assert_close("t3_lms_train bias", &bias, &[want_bias], 1e-5);
}

// ────────────────────────────────────────────────
// 4. nlms_train: w += lr·error·s/(‖s‖²+ε); bias += lr·error/(‖s‖²+ε)
// ────────────────────────────────────────────────
#[test]
fn nlms_train_anchor() {
    let n = 17;
    let mut w = data(n, 9);
    let mut bias = vec![-0.5f32];
    let s = data(n, 10);
    let (err, lr, eps) = (0.4f32, 0.02f32, 1e-4f32);

    decode::nlms_train(&mut w, &mut bias, &s, err, lr, eps);

    let norm_sq: f32 = s.iter().map(|x| x * x).sum();
    let mu = lr * err / (norm_sq + eps);
    let mut want_w = data(n, 9);
    for i in 0..n {
        want_w[i] += mu * s[i];
    }
    let want_bias = -0.5f32 + lr * err / (norm_sq + eps);
    assert_close("nlms_train w", &w, &want_w, 1e-5);
    assert_close("nlms_train bias", &bias, &[want_bias], 1e-5);
}

// ────────────────────────────────────────────────
// 5. kalman_update: k=c·st/(S+1e-8); w+=k·err; c=max(c−k·st·c, 1e-12)
// ────────────────────────────────────────────────
#[test]
fn kalman_update_anchor() {
    let n = 20;
    let mut w = data(n, 11);
    let mut c = data(n, 12).into_iter().map(|x| x.abs() + 0.1).collect::<Vec<_>>();
    let st = data(n, 13);
    let (err, s) = (0.6f32, 2.0f32);

    decode::kalman_update(&mut w, &mut c, &st, err, s);

    let s2 = s + 1e-8;
    let mut want_w = data(n, 11);
    let mut want_c = data(n, 12).into_iter().map(|x| x.abs() + 0.1).collect::<Vec<_>>();
    for i in 0..n {
        let k = want_c[i] * st[i] / s2;
        want_w[i] += k * err;
        want_c[i] = (want_c[i] - k * st[i] * want_c[i]).max(1e-12);
    }
    assert_close("kalman_update w", &w, &want_w, 1e-4);
    assert_close("kalman_update cov", &c, &want_c, 1e-4);
}

// ────────────────────────────────────────────────
// 6. adam_train: g=err·s; m=β1m+(1-β1)g; v=β2v+(1-β2)g²;
//    w -= lr·m/(√(v/bc2)+ε)/bc1
// ────────────────────────────────────────────────
#[test]
fn adam_train_anchor() {
    let n = 21;
    let w0 = data(n, 14);
    let m0 = data(n, 15);
    let v0 = data(n, 16).into_iter().map(|x| x.abs()).collect::<Vec<_>>();
    let st = data(n, 17);
    let (lr, beta1, beta2) = (0.01f32, 0.9f32, 0.999f32);
    let (b1c, b2c, eps) = (
        1.0f32 - beta1.powi(2),
        1.0f32 - beta2.powi(2),
        1e-8f32,
    );
    let err = 0.9f32;

    let mut w = w0.clone();
    let mut m = m0.clone();
    let mut v = v0.clone();
    decode::adam_train(&mut w, &mut m, &mut v, &st, err, lr, beta1, beta2, b1c, b2c, eps);

    let mut want_w = w0;
    let mut want_m = m0;
    let mut want_v = v0;
    for i in 0..n {
        let g = err * st[i];
        want_m[i] = beta1 * want_m[i] + (1.0 - beta1) * g;
        want_v[i] = beta2 * want_v[i] + (1.0 - beta2) * g * g;
        want_w[i] -= lr * want_m[i] / ((want_v[i] / b2c).sqrt() + eps) / b1c;
    }
    assert_close("adam_train w", &w, &want_w, 1e-4);
    assert_close("adam_train m", &m, &want_m, 1e-5);
    assert_close("adam_train v", &v, &want_v, 1e-5);
}

// ────────────────────────────────────────────────
// 7. pst_decode: out[f] = Σ_t W[t·F+f]·buf[(pos+t)%l][f]
// ────────────────────────────────────────────────
#[test]
fn pst_decode_anchor() {
    let feat = 5;
    let l = 4;
    let pos = 2;
    let buf = data(l * feat, 18);
    let wg = data(l * feat, 19);
    let mut out = vec![0.0f32; feat];
    decode::pst_decode(&mut out, &buf, &wg, l, pos);

    let mut want = vec![0.0f32; feat];
    for t in 0..l {
        let src = (pos + t) % l;
        for f in 0..feat {
            want[f] += wg[t * feat + f] * buf[src * feat + f];
        }
    }
    assert_close("pst_decode", &out, &want, 1e-4);
}

// ────────────────────────────────────────────────
// 8. ring_synthesis: out[i] = w[i]·a[i]·t[i]
// ────────────────────────────────────────────────
#[test]
fn ring_synthesis_anchor() {
    let n = 23;
    let a = data(n, 20);
    let t = data(n, 21);
    let w = data(n, 22);
    let mut out = vec![0.0f32; n];
    decode::ring_synthesis(&mut out, &a, &t, &w);

    let want = (0..n).map(|i| w[i] * a[i] * t[i]).collect::<Vec<f32>>();
    assert_close("ring_synthesis", &out, &want, 1e-5);
}