//! 🧠 MNIST 10 цифр: СУПЕР-мультиградиент — BPTT сквозь транс-нейроны.
//!
//! Вместо локального STDP (слабый, учитель-ток) — **полный градиентный спуск**:
//! функция потерь (softmax-cross-entropy на readout) распространяется назад
//! через 28 шагов внимания в ключи w и значения v КАЖДОГО нейрона.
//! Внутренности нейрона гладкие (softmax + leaky-интегратор acc) — вся
//! цепочка дифференцируема, surrogate gradient не нужен.
//!
//! Архитектура (как в mnist_10_digits.rs, но без STDP):
//!   20 нейронов × 8 слотов (2 полосы колонок на класс, data-driven).
//!   Признак нейрона d = Σ_t acc_d(t). Readout: softmax 10×22.
//!   Обучаются: 320 нейроновых параметров (w/v) + 230 readout (W, b).
//!
//! Прогон: cargo run --example mnist_grad --release

use std::process::Command;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const ROWS: usize = 28;
const COLS: usize = 28;
const N_CLASSES: usize = 10;
const N_NEURONS: usize = 20; // 2 полосы на класс
const N_SLOTS: usize = 8;
const FEAT: usize = N_NEURONS + 2; // acc×20 + spread + ink
const BAND_W: usize = 8;

// Модельные константы (совпадают с mnist_10_digits.rs)
const DT: f32 = 0.5;
const DECAY: f32 = 0.9;
const GAIN: f32 = 3.0;
const INV_TAU: f32 = 0.2;
const K_SCALE: f32 = 2.0;
const SCORE_CLAMP: f32 = 50.0;

// Гиперпараметры обучения
const BATCH: usize = 256;
const EPOCHS: usize = 120;
const LR_WV: f32 = 0.02;   // нейроновые w/v (×0.5 на 40 и 80 эпохах)
const LR_RO: f32 = 0.05;   // readout
const TRAIN_LIMIT: usize = 20000; // подвыборка для скорости

const PIXEL_SCALE: f32 = 1.0 / 255.0;

fn fetch_gz(name: &str, url: &str) -> Vec<u8> {
    let dir = "target/mnist";
    let cache = format!("{dir}/{name}");
    std::fs::create_dir_all(dir).ok();
    if std::path::Path::new(&cache).exists() {
        return std::fs::read(&cache).expect("read cache");
    }
    let out = Command::new("bash")
        .args(["-c", &format!("curl -sL '{url}' | gunzip > '{cache}' && echo OK")])
        .output()
        .expect("curl+gunzip");
    assert!(out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
        "download failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::read(&cache).expect("read downloaded")
}

fn be_i32(b: &[u8], off: usize) -> i32 {
    i32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn load_images(raw: &[u8]) -> Vec<Vec<f32>> {
    let n = be_i32(raw, 4) as usize;
    let mut imgs = Vec::with_capacity(n);
    let base = 16;
    for i in 0..n {
        let start = base + i * ROWS * COLS;
        imgs.push(raw[start..start + ROWS * COLS].iter().map(|&b| b as f32 * PIXEL_SCALE).collect());
    }
    imgs
}

fn load_labels(raw: &[u8]) -> Vec<u8> {
    let n = be_i32(raw, 4) as usize;
    raw[8..8 + n].to_vec()
}

fn band_density(img: &[f32], row: usize, lo: usize) -> f32 {
    let start = row * COLS;
    let hi = (lo + BAND_W).min(COLS);
    img[start + lo..start + hi].iter().sum::<f32>() / (hi - lo) as f32
}

fn row_spread(img: &[f32], row: usize) -> f32 {
    let start = row * COLS;
    let mut sw = 0.0f32;
    let mut s = 0.0f32;
    for c in 0..COLS {
        let v = img[start + c];
        sw += v * (c as f32);
        s += v;
    }
    if s < 0.01 { return 0.0; }
    let mean = sw / s;
    let mut var = 0.0f32;
    for c in 0..COLS {
        let v = img[start + c];
        var += v * (c as f32 - mean) * (c as f32 - mean);
    }
    (var / s).sqrt() / COLS as f32
}

/// Глобальные признаки: суммарный разброс строк + общие чернила.
fn global_feats(img: &[f32]) -> [f32; 2] {
    let spread: f32 = (0..ROWS).map(|r| row_spread(img, r)).sum();
    let ink: f32 = img.iter().sum();
    [spread, ink]
}

// ═══════════════ Модель и BPTT ═══════════════

/// Обучаемые параметры: w/v каждого нейрона + readout (W, b).
struct Net {
    w: [[f32; N_SLOTS]; N_NEURONS],
    v: [[f32; N_SLOTS]; N_NEURONS],
    rw: [[f32; FEAT]; N_CLASSES],
    rb: [f32; N_CLASSES],
}

impl Default for Net {
    fn default() -> Self {
        Net {
            w: [[1.0; N_SLOTS]; N_NEURONS],
            v: [[1.0; N_SLOTS]; N_NEURONS],
            rw: [[0.0; FEAT]; N_CLASSES],
            rb: [0.0; N_CLASSES],
        }
    }
}

#[derive(Clone, Copy, Default)]
struct StepRec {
    q: f32,
    a: [f32; N_SLOTS],
    mem: [f32; N_SLOTS],
    pre: f32,
    clamp: [bool; N_SLOTS],
}

#[derive(Default)]
struct Grads {
    dw: [[f32; N_SLOTS]; N_NEURONS],
    dv: [[f32; N_SLOTS]; N_NEURONS],
    drw: [[f32; FEAT]; N_CLASSES],
    drb: [f32; N_CLASSES],
}

impl Net {
    /// Forward по изображению: признаки нейронов + записи шагов для backward.
    fn forward(&self, img: &[f32], bands: &[usize; N_NEURONS]) -> ([f32; N_NEURONS], [[StepRec; ROWS]; N_NEURONS]) {
        let mut feats = [0.0f32; N_NEURONS];
        let mut all = [[StepRec::default(); ROWS]; N_NEURONS];
        for d in 0..N_NEURONS {
            let mut mem = [0.0f32; N_SLOTS];
            let mut acc = 0.0f32;
            for (t, slot) in all[d].iter_mut().enumerate() {
                let q = band_density(img, t, bands[d]);
                let mut e = [0.0f32; N_SLOTS];
                let mut clamp = [false; N_SLOTS];
                let mut esum = 0.0f32;
                for i in 0..N_SLOTS {
                    let raw = q * self.w[d][i] * K_SCALE;
                    if raw > SCORE_CLAMP {
                        e[i] = SCORE_CLAMP.exp();
                        clamp[i] = true;
                    } else if raw < -SCORE_CLAMP {
                        e[i] = (-SCORE_CLAMP).exp();
                        clamp[i] = true;
                    } else {
                        e[i] = raw.exp();
                    }
                    esum += e[i];
                }
                let mut a = [0.0f32; N_SLOTS];
                for i in 0..N_SLOTS { a[i] = e[i] / esum; }

                let mut out = 0.0f32;
                for (i, &ai) in a.iter().enumerate() { out += ai * self.v[d][i] * mem[i]; }
                let pre = acc + DT * (out - acc * INV_TAU);
                acc = if pre > 0.0 { pre } else { 0.0 };
                feats[d] += acc;

                *slot = StepRec { q, a, mem, pre, clamp };

                for i in (1..N_SLOTS).rev() { mem[i] = mem[i - 1] * DECAY; }
                mem[0] = q * GAIN;
            }
        }
        (feats, all)
    }

    /// Backward: dL/dw, dL/dv для нейронов. `df` = dL/df (из readout).
    fn backward(&self, all: &[[StepRec; ROWS]; N_NEURONS], df: &[f32; N_NEURONS]) -> Grads {
        let mut g = Grads::default();
        for d in 0..N_NEURONS {
            let mut dacc_next = 0.0f32;
            for t in (0..ROWS).rev() {
                let r = &all[d][t];
                let dacc = df[d] + dacc_next;
                let dpre = if r.pre > 0.0 { dacc } else { 0.0 };
                dacc_next = dpre * (1.0 - DT * INV_TAU);
                let dout = dpre * DT;
                let mut da = [0.0f32; N_SLOTS];
                for (i, da_i) in da.iter_mut().enumerate() {
                    g.dv[d][i] += dout * r.a[i] * r.mem[i];
                    *da_i = dout * self.v[d][i] * r.mem[i];
                }
                let mut dot = 0.0f32;
                for (i, &ai) in r.a.iter().enumerate() { dot += ai * da[i]; }
                for (i, &ai) in r.a.iter().enumerate() {
                    let ds = ai * (da[i] - dot);
                    if !r.clamp[i] {
                        g.dw[d][i] += ds * r.q * K_SCALE;
                    }
                }
            }
        }
        g
    }
}

/// Полный вектор признаков: нейроновые acc + глобальные.
fn feat_vec(feats: &[f32; N_NEURONS], glob: &[f32; 2]) -> [f32; FEAT] {
    let mut x = [0.0f32; FEAT];
    x[..N_NEURONS].copy_from_slice(feats);
    x[N_NEURONS] = glob[0];
    x[N_NEURONS + 1] = glob[1];
    x
}

/// Readout forward + backward. Возвращает (loss, dL/dfeats, readout-градиенты).
fn readout_bwd(net: &Net, x: &[f32; FEAT], label: usize) -> (f32, [f32; N_NEURONS], Grads) {
    let mut z = [0.0f32; N_CLASSES];
    for (c, zc) in z.iter_mut().enumerate() {
        *zc = net.rb[c] + (0..FEAT).map(|j| net.rw[c][j] * x[j]).sum::<f32>();
    }
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; N_CLASSES];
    let mut esum = 0.0f32;
    for (c, ec) in e.iter_mut().enumerate() { *ec = (z[c] - m).exp(); esum += *ec; }
    let mut p = [0.0f32; N_CLASSES];
    for (c, pc) in p.iter_mut().enumerate() { *pc = e[c] / esum; }

    let loss = -p[label].max(1e-7).ln();
    let mut dz = [0.0f32; N_CLASSES];
    for (c, dzc) in dz.iter_mut().enumerate() { *dzc = p[c] - if c == label { 1.0 } else { 0.0 }; }

    let mut df = [0.0f32; FEAT];
    let mut g = Grads::default();
    for (c, &dzc) in dz.iter().enumerate() {
        g.drb[c] = dzc;
        for j in 0..FEAT {
            g.drw[c][j] = dzc * x[j];
            df[j] += dzc * net.rw[c][j];
        }
    }
    let mut dfeat = [0.0f32; N_NEURONS];
    dfeat[..N_NEURONS].copy_from_slice(&df[..N_NEURONS]);
    (loss, dfeat, g)
}

/// Предсказание (argmax logits).
fn predict(net: &Net, x: &[f32; FEAT]) -> usize {
    let mut best = 0usize;
    let mut best_z = f32::NEG_INFINITY;
    for c in 0..N_CLASSES {
        let z = net.rb[c] + (0..FEAT).map(|j| net.rw[c][j] * x[j]).sum::<f32>();
        if z > best_z { best_z = z; best = c; }
    }
    best
}

// ═══════════════ Flatten-хелперы для Adam ═══════════════

fn wv_to_vec(net: &Net) -> Vec<f32> {
    let mut p = Vec::with_capacity(N_NEURONS * N_SLOTS * 2);
    for d in 0..N_NEURONS { p.extend_from_slice(&net.w[d]); }
    for d in 0..N_NEURONS { p.extend_from_slice(&net.v[d]); }
    p
}

fn wv_from_vec(net: &mut Net, p: &[f32]) {
    let mut k = 0;
    for d in 0..N_NEURONS {
        for i in 0..N_SLOTS { net.w[d][i] = p[k]; k += 1; }
    }
    for d in 0..N_NEURONS {
        for i in 0..N_SLOTS { net.v[d][i] = p[k]; k += 1; }
    }
}

fn wv_grads_vec(g: &Grads) -> Vec<f32> {
    let mut p = Vec::with_capacity(N_NEURONS * N_SLOTS * 2);
    for d in 0..N_NEURONS { p.extend_from_slice(&g.dw[d]); }
    for d in 0..N_NEURONS { p.extend_from_slice(&g.dv[d]); }
    p
}

fn ro_to_vec(net: &Net) -> Vec<f32> {
    let mut p = Vec::with_capacity(N_CLASSES * (FEAT + 1));
    for c in 0..N_CLASSES { p.extend_from_slice(&net.rw[c]); p.push(net.rb[c]); }
    p
}

fn ro_from_vec(net: &mut Net, p: &[f32]) {
    let mut k = 0;
    for c in 0..N_CLASSES {
        for j in 0..FEAT { net.rw[c][j] = p[k]; k += 1; }
        net.rb[c] = p[k];
        k += 1;
    }
}

fn ro_grads_vec(g: &Grads) -> Vec<f32> {
    let mut p = Vec::with_capacity(N_CLASSES * (FEAT + 1));
    for c in 0..N_CLASSES { p.extend_from_slice(&g.drw[c]); p.push(g.drb[c]); }
    p
}

// ═══════════════ Adam ═══════════════

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl Adam {
    fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    fn step(&mut self, p: &mut [f32], g: &[f32], lr: f32) {
        self.t += 1;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for i in 0..p.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            let mh = self.m[i] / bc1;
            let vh = self.v[i] / bc2;
            p[i] -= lr * mh / (vh.sqrt() + eps);
        }
    }
}

/// Loss одного образца (для finite-difference проверки градиентов).
fn loss_of(net: &Net, img: &[f32], bands: &[usize; N_NEURONS], label: usize) -> f32 {
    let (feats, _) = net.forward(img, bands);
    let x = feat_vec(&feats, &global_feats(img));
    let (loss, _, _) = readout_bwd(net, &x, label);
    loss
}

/// Проверка рукописного BPTT конечными разностями (на случайном малом образце).
fn grad_check() {
    println!("🔬 Gradient-check (аналитика vs конечные разности)...");
    // Невырожденный кейс: w=v=1, readout КРОШЕЧНЫЙ с разными весами по классам
    // (иначе Σ dz = 0 обнуляет градиенты фич; большие веса насыщают softmax и
    // зажимают loss на клампе). Структурированный вход.
    let mut net = Net::default();
    for c in 0..N_CLASSES {
        for j in 0..FEAT {
            net.rw[c][j] = 0.001 * (1.0 + 0.1 * (c as f32) + 0.05 * (j as f32));
        }
    }
    // Рядовая структура: средние строки ярче (как у цифры)
    let img: Vec<f32> = (0..ROWS * COLS)
        .map(|k| {
            let row = k / COLS;
            let col = k % COLS;
            if (12..20).contains(&row) && (8..22).contains(&col) { 0.6 } else { 0.05 }
        })
        .collect();
    let bands = [4usize; N_NEURONS];
    let label = 3usize;

    let (feats, all) = net.forward(&img, &bands);
    let x = feat_vec(&feats, &global_feats(&img));
    let (_, dfeat, _) = readout_bwd(&net, &x, label);
    let g = net.backward(&all, &dfeat);

    let eps = 1e-2f32;
    let mut max_rel = 0.0f32;
    let mut nonzero = 0usize;
    let base_loss = loss_of(&net, &img, &bands, label);
    println!("   base loss = {base_loss:.4} (должен быть ~2.3, не насыщен)");
    for &(d, i) in &[(0usize, 0usize), (5, 3), (2, 7), (9, 4)] {
        let analytic = g.dw[d][i];
        let orig = net.w[d][i];
        net.w[d][i] = orig + eps;
        let lp = loss_of(&net, &img, &bands, label);
        net.w[d][i] = orig - eps;
        let lm = loss_of(&net, &img, &bands, label);
        net.w[d][i] = orig;
        let numeric = (lp - lm) / (2.0 * eps);
        let rel = (analytic - numeric).abs() / (analytic.abs() + 1e-4); // abs-пол против деления нуля
        if analytic.abs() > 1e-4 { max_rel = max_rel.max(rel); } // релятивно — только значимые
        if analytic.abs() > 1e-6 { nonzero += 1; }
        println!("   dw[{d}][{i}]: analytic={analytic:.5} numeric={numeric:.5} err={rel:.3}");
    }
    for &(d, i) in &[(0usize, 1usize), (4, 6)] {
        let analytic = g.dv[d][i];
        let orig = net.v[d][i];
        net.v[d][i] = orig + eps;
        let lp = loss_of(&net, &img, &bands, label);
        net.v[d][i] = orig - eps;
        let lm = loss_of(&net, &img, &bands, label);
        net.v[d][i] = orig;
        let numeric = (lp - lm) / (2.0 * eps);
        let rel = (analytic - numeric).abs() / (analytic.abs() + 1e-4);
        if analytic.abs() > 1e-4 { max_rel = max_rel.max(rel); }
        if analytic.abs() > 1e-6 { nonzero += 1; }
        println!("   dv[{d}][{i}]: analytic={analytic:.5} numeric={numeric:.5} err={rel:.3}");
    }
    assert!(nonzero >= 4, "gradient check degenerate: only {nonzero}/6 params nonzero");
    assert!(max_rel < 0.05, "gradient check FAILED: max rel err {max_rel:.4}");
    println!("   ✅ gradient check PASSED (max rel err {max_rel:.4})\n");
}

fn main() {
    grad_check();
    println!("📥 Загружаем MNIST...");
    let base = "https://ossci-datasets.s3.amazonaws.com/mnist";
    let train_img = load_images(&fetch_gz("train-images.idx3-ubyte", &format!("{base}/train-images-idx3-ubyte.gz")));
    let train_lbl = load_labels(&fetch_gz("train-labels.idx1-ubyte", &format!("{base}/train-labels-idx1-ubyte.gz")));
    let test_img  = load_images(&fetch_gz("t10k-images.idx3-ubyte", &format!("{base}/t10k-images-idx3-ubyte.gz")));
    let test_lbl  = load_labels(&fetch_gz("t10k-labels.idx1-ubyte", &format!("{base}/t10k-labels-idx1-ubyte.gz")));
    let n_train = TRAIN_LIMIT.min(train_img.len());
    println!("   тренировка: {} (лимит {} для скорости)", train_img.len(), n_train);
    println!("   тест:       {}\n", test_img.len());

    // ─── Data-driven полосы классов (только train) ──────────────
    let mut class_ink = [[0.0f64; COLS]; N_CLASSES];
    let mut class_n = [0usize; N_CLASSES];
    for (img, &label) in train_img.iter().zip(&train_lbl).take(n_train) {
        class_n[label as usize] += 1;
        for row in 0..ROWS {
            for c in 0..COLS {
                class_ink[label as usize][c] += img[row * COLS + c] as f64;
            }
        }
    }
    let mut profile = [[0.0f64; COLS]; N_CLASSES];
    for d in 0..N_CLASSES {
        for c in 0..COLS {
            profile[d][c] = class_ink[d][c] / (class_n[d].max(1) as f64 * ROWS as f64);
        }
    }
    let global: Vec<f64> = (0..COLS)
        .map(|c| (0..N_CLASSES).map(|d| profile[d][c]).sum::<f64>() / N_CLASSES as f64)
        .collect();
    // Две самые различительные полосы на класс (по одному нейрону на каждую),
    // со штрафом за пересечение с уже занятыми.
    let mut band_of = [0usize; N_NEURONS];
    let mut used: Vec<usize> = Vec::new();
    for d in 0..N_CLASSES {
        for k in 0..2 {
            let mut best = (f64::NEG_INFINITY, 0usize);
            for lo in 0..=(COLS - BAND_W) {
                let overlap_pen: f64 = used
                    .iter()
                    .map(|&u| ((BAND_W as i64 - (lo as i64 - u as i64).abs()).max(0)) as f64 * 0.5)
                    .sum();
                let s: f64 = (lo..lo + BAND_W).map(|c| profile[d][c] - global[c]).sum::<f64>() - overlap_pen;
                if s > best.0 { best = (s, lo); }
            }
            band_of[d * 2 + k] = best.1;
            used.push(best.1);
        }
    }
    println!("🔍 Полосы классов (2/класс, data-driven): {band_of:?}\n");

    let mut net = Net::default();
    let mut adam_wv = Adam::new(N_NEURONS * N_SLOTS * 2);
    let mut adam_ro = Adam::new(N_CLASSES * (FEAT + 1));

    // ─── Обучение ───────────────────────────────────────────────
    println!("⚡ Обучение BPTT: {} образцов × {EPOCHS} эпох, batch={BATCH}", n_train);
    let t0 = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..n_train).collect();

    let mut lr_wv = LR_WV;
    let mut lr_ro = LR_RO;
    for epoch in 0..EPOCHS {
        if epoch == 40 || epoch == 80 {
            lr_wv *= 0.5;
            lr_ro *= 0.5;
        }
        indices.shuffle(&mut rng);
        let mut loss_sum = 0.0f64;
        let mut n_seen = 0usize;
        for chunk in indices.chunks(BATCH) {
            let mut g = Grads::default();
            for &idx in chunk {
                let (feats, all) = net.forward(&train_img[idx], &band_of);
                let x = feat_vec(&feats, &global_feats(&train_img[idx]));
                let (loss, dfeat, g_ro) = readout_bwd(&net, &x, train_lbl[idx] as usize);
                loss_sum += loss as f64;
                n_seen += 1;
                let g_n = net.backward(&all, &dfeat);
                for d in 0..N_NEURONS {
                    for i in 0..N_SLOTS {
                        g.dw[d][i] += g_n.dw[d][i];
                        g.dv[d][i] += g_n.dv[d][i];
                    }
                }
                for c in 0..N_CLASSES {
                    g.drb[c] += g_ro.drb[c];
                    for j in 0..FEAT { g.drw[c][j] += g_ro.drw[c][j]; }
                }
            }
            let s = chunk.len() as f32;
            for d in 0..N_NEURONS {
                for i in 0..N_SLOTS { g.dw[d][i] /= s; g.dv[d][i] /= s; }
            }
            for c in 0..N_CLASSES {
                g.drb[c] /= s;
                for j in 0..FEAT { g.drw[c][j] /= s; }
            }
            let mut pw = wv_to_vec(&net);
            let gw = wv_grads_vec(&g);
            adam_wv.step(&mut pw, &gw, lr_wv);
            wv_from_vec(&mut net, &pw);
            let mut pr = ro_to_vec(&net);
            let gr = ro_grads_vec(&g);
            adam_ro.step(&mut pr, &gr, lr_ro);
            ro_from_vec(&mut net, &pr);
        }
        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            println!("   эпоха {epoch:>3}: loss={:.4}  ({:.0}с)",
                     loss_sum / n_seen as f64, t0.elapsed().as_secs());
        }
    }
    println!("   обучено за {:.1}с\n", t0.elapsed().as_secs_f32());

    // ─── Тест ───────────────────────────────────────────────────
    println!("🧪 Тестируем...");
    let n_test = test_img.len();
    let mut cm = [[0usize; N_CLASSES]; N_CLASSES];
    let mut acc_count = 0usize;
    for (img, &label) in test_img.iter().zip(&test_lbl) {
        let (feats, _) = net.forward(img, &band_of);
        let x = feat_vec(&feats, &global_feats(img));
        let best = predict(&net, &x);
        cm[label as usize][best] += 1;
        if best == label as usize { acc_count += 1; }
    }

    let w0: Vec<f32> = net.w[0].to_vec();
    let v0: Vec<f32> = net.v[0].to_vec();

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  🏆 Точность: {:.1}%  ({}/{})      ║",
             acc_count as f64 * 100.0 / n_test as f64, acc_count, n_test);
    println!("║  🔑 w₀: {:.2?}  ║", &w0[..4]);
    println!("║  🔑 v₀: {:.2?}  ║", &v0[..4]);
    println!("╚══════════════════════════════════════════╝");

    println!("\n🧮 Матрица ошибок (строки — истина, столбцы — предсказание)");
    print!("    ");
    for p in 0..N_CLASSES { print!("  {p:>3}"); }
    println!("   |  точн.");
    for (a, row) in cm.iter().enumerate() {
        print!("  {a} |");
        for (p, &v) in row.iter().enumerate() {
            if a == p {
                print!(" \x1b[32m{v:>3}\x1b[0m");
            } else if v > 0 {
                print!(" \x1b[31m{v:>3}\x1b[0m");
            } else {
                print!(" {v:>3}");
            }
        }
        let total: usize = row.iter().sum();
        println!("   | {:5.1}%", row[a] as f64 * 100.0 / total.max(1) as f64);
    }
}
