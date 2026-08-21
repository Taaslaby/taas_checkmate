//! 🧠 MNIST: все 10 цифр на 10 Transformer-нейронах + обученный readout.
//!
//! Итог трёх итераций (все измерены, все ~10%):
//!   argmax по нейронам не работает для 10 классов — нейроны коррелированы
//!   (общий признак), а у каждого класса нет своей уникальной «полосы».
//!
//! Правильная архитектура: 10 нейронов = извлекатели признаков (временное
//! внимание + STDP подстраивают, КАК читать полосу), а решение принимает
//! **обученный softmax-классификатор** поверх 10-мерного вектора acc-признаков.
//! Это ровно то, как устроены readout-слои в спайковых сетях.
//!
//! Сравниваются:
//!     1. argmax по нейронам (базовый — должен быть ~10%)
//!     2. softmax над СЫРЫМИ полосовыми признаками (нет нейронов)
//!     3. softmax над ПРИЗНАКАМИ НЕЙРОНОВ (acc после внимания+STDP)
//!     4. softmax над склейкой (нейроновые + сырые + глобальные)
//! Если (3) > (2) — нейроны реально добавляют ценность, а не только полосы.
//!
//! Прогон: cargo run --example mnist_10_digits --release

use std::process::Command;

use pillars::core::population::NeuralPopulation;
use pillars::zoo::neurons::{TransformerModel, TransformerModelParamIdx};

const ROWS: usize = 28;
const COLS: usize = 28;
const N_CLASSES: usize = 10;
const BAND_W: usize = 8;
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

/// Средняя плотность чернил в строке `row` для полосы колонок [lo, lo+BAND_W).
fn band_density(img: &[f32], row: usize, lo: usize) -> f32 {
    let start = row * COLS;
    let hi = (lo + BAND_W).min(COLS);
    img[start + lo..start + hi].iter().sum::<f32>() / (hi - lo) as f32
}

fn make_neuron(thresh: f32, eta: f32, class: usize) -> NeuralPopulation<TransformerModel> {
    let mut pop = NeuralPopulation::<TransformerModel>::new(1);
    let mut fp = |idx: TransformerModelParamIdx, v: f32| pop.fill_param(idx as usize, v);
    fp(TransformerModelParamIdx::dt, 0.5);
    fp(TransformerModelParamIdx::decay, 0.9);
    fp(TransformerModelParamIdx::gain, 3.0);
    fp(TransformerModelParamIdx::inv_tau_acc, 0.2);
    fp(TransformerModelParamIdx::thresh, thresh);
    fp(TransformerModelParamIdx::k_scale, 2.0);
    fp(TransformerModelParamIdx::eta_k, eta);
    fp(TransformerModelParamIdx::eta_v, eta * 0.5);
    fp(TransformerModelParamIdx::w_base, 1.0);
    fp(TransformerModelParamIdx::w_span, 2.0);
    fp(TransformerModelParamIdx::v_base, 1.0);
    fp(TransformerModelParamIdx::v_span, 2.0);
    let peak = (class % 8) as f32;
    for i in 0..8 {
        let d = (i as f32 - peak).abs();
        let w = 1.0 + 1.5 * (-d * d / 4.5).exp();
        pop.fill_state(9 + i, w);
        pop.fill_state(17 + i, 1.0);
    }
    pop
}

fn reset_temporal(neuron: &mut NeuralPopulation<TransformerModel>) {
    for col in 0..9 {
        neuron.fill_state(col, 0.0);
    }
}

/// Прогнать нейрон на изображении через его полосу. Возвращает (спайки, acc_sum).
fn run_digit(
    neuron: &mut NeuralPopulation<TransformerModel>,
    img: &[f32],
    teach: f32,
    band: usize,
) -> (i32, f32) {
    reset_temporal(neuron);
    let mut spikes = vec![0i32; 1];
    let mut total = 0i32;
    let mut acc_sum = 0.0f32;
    for row in 0..ROWS {
        let val = band_density(img, row, band) + teach;
        neuron.step(&[val], &mut spikes);
        total += spikes[0];
        acc_sum += neuron.state_slice(8)[0];
    }
    (total, acc_sum)
}

// ═══════════════ Softmax-регрессия (без зависимостей) ═══════════════

struct Softmax {
    mean: Vec<f32>,
    std: Vec<f32>,
    w: Vec<Vec<f32>>, // [class][feature]
    b: Vec<f32>,
}

impl Softmax {
    fn train(x: &[Vec<f32>], y: &[usize], epochs: usize, lr: f32) -> Self {
        let (n, d) = (x.len(), x[0].len());
        let k = N_CLASSES;
        let mean: Vec<f32> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f32>() / n as f32).collect();
        let std: Vec<f32> = (0..d)
            .map(|j| (x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f32>() / n as f32).sqrt().max(1e-6))
            .collect();
        let xn: Vec<Vec<f32>> = x.iter()
            .map(|r| r.iter().enumerate().map(|(j, &v)| (v - mean[j]) / std[j]).collect())
            .collect();
        let mut w = vec![vec![0.0f32; d]; k];
        let mut b = vec![0.0f32; k];
        for _ in 0..epochs {
            let mut gw = vec![vec![0.0f32; d]; k];
            let mut gb = vec![0.0f32; k];
            for (xi, &yi) in xn.iter().zip(y) {
                let scores: Vec<f32> = (0..k)
                    .map(|c| b[c] + (0..d).map(|j| w[c][j] * xi[j]).sum::<f32>())
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = ex.iter().sum();
                for c in 0..k {
                    let p = ex[c] / sum;
                    let grad = (p - if c == yi { 1.0 } else { 0.0 }) / n as f32;
                    gb[c] += grad;
                    for j in 0..d { gw[c][j] += grad * xi[j]; }
                }
            }
            for c in 0..k {
                b[c] -= lr * gb[c];
                for j in 0..d { w[c][j] -= lr * gw[c][j]; }
            }
        }
        Softmax { mean, std, w, b }
    }

    fn predict(&self, x: &[f32]) -> usize {
        let z: Vec<f32> = x.iter().enumerate().map(|(j, &v)| (v - self.mean[j]) / self.std[j]).collect();
        let mut best = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (c, wc) in self.w.iter().enumerate() {
            let s = self.b[c] + (0..z.len()).map(|j| wc[j] * z[j]).sum::<f32>();
            if s > best_s { best_s = s; best = c; }
        }
        best
    }
}

fn print_cm(title: &str, cm: &[[usize; N_CLASSES]; N_CLASSES]) {
    println!("\n{title}");
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

fn main() {
    println!("📥 Загружаем MNIST (все 10 цифр)...");
    let base = "https://ossci-datasets.s3.amazonaws.com/mnist";
    let train_img = load_images(&fetch_gz("train-images.idx3-ubyte", &format!("{base}/train-images-idx3-ubyte.gz")));
    let train_lbl = load_labels(&fetch_gz("train-labels.idx1-ubyte", &format!("{base}/train-labels-idx1-ubyte.gz")));
    let test_img  = load_images(&fetch_gz("t10k-images.idx3-ubyte", &format!("{base}/t10k-images-idx3-ubyte.gz")));
    let test_lbl  = load_labels(&fetch_gz("t10k-labels.idx1-ubyte", &format!("{base}/t10k-labels-idx1-ubyte.gz")));
    println!("   тренировка: {}", train_img.len());
    println!("   тест:       {}\n", test_img.len());

    const THRESH: f32 = 0.3;
    const TEACH: f32 = 0.8;

    // ─── Data-driven полосы классов ─────────────────────────────
    let mut class_ink = [[0.0f64; COLS]; N_CLASSES];
    let mut class_n = [0usize; N_CLASSES];
    for (img, &label) in train_img.iter().zip(&train_lbl) {
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
            profile[d][c] = class_ink[d][c] / (class_n[d] as f64 * ROWS as f64);
        }
    }
    let global: Vec<f64> = (0..COLS)
        .map(|c| (0..N_CLASSES).map(|d| profile[d][c]).sum::<f64>() / N_CLASSES as f64)
        .collect();
    // Жадный выбор: разные полосы по возможности (штраф за пересечение
    // с уже занятыми — 10 полос по 8 колонок не помещаются в 28 без перекрытий).
    let mut band_of = [0usize; N_CLASSES];
    let mut used: Vec<usize> = Vec::new();
    for d in 0..N_CLASSES {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for lo in 0..=(COLS - BAND_W) {
            let overlap_pen: f64 = used
                .iter()
                .map(|&u| ((BAND_W as i64 - (lo as i64 - u as i64).abs()).max(0)) as f64 * 0.5)
                .sum();
            let s: f64 = (lo..lo + BAND_W).map(|c| profile[d][c] - global[c]).sum::<f64>() - overlap_pen;
            if s > best.0 { best = (s, lo); }
        }
        band_of[d] = best.1;
        used.push(best.1);
    }
    println!("🔍 Полосы классов (data-driven, width={BAND_W}): {band_of:?}\n");

    // ─── Тренировка 10 нейронов ─────────────────────────────────
    let mut learned: Vec<NeuralPopulation<TransformerModel>> =
        (0..N_CLASSES).map(|d| make_neuron(THRESH, 0.01, d)).collect();
    let n_train = train_img.len();
    println!("🎓 Тренируем {} образцов (учитель на свою цифру)...", n_train);
    let t0 = std::time::Instant::now();
    for (i, (img, &label)) in train_img.iter().zip(&train_lbl).enumerate() {
        let d = label as usize;
        for c in 0..N_CLASSES {
            let teach = if c == d { TEACH } else { 0.0 };
            let _ = run_digit(&mut learned[c], img, teach, band_of[c]);
        }
        if (i + 1) % 15000 == 0 {
            println!("   ... {}/{}  ({:.0}с)", i + 1, n_train, t0.elapsed().as_secs());
        }
    }
    println!("   обучено за {:.1}с\n", t0.elapsed().as_secs_f32());

    // Замораживаем STDP
    for n in learned.iter_mut() {
        n.fill_param(TransformerModelParamIdx::eta_k as usize, 0.0);
        n.fill_param(TransformerModelParamIdx::eta_v as usize, 0.0);
    }

    // ─── Признаки: нейроновые (acc) + глобальные, и сырые (полосы) ──
    // Глобальные: сумма разброса строк (форма) + общее кол-во чернил.
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

    let mut feats = |xs: &[Vec<f32>]| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        // (neuron+global, raw+global); оба 12-мерные, склейка даст 24-мерный
        let mut acc_f = Vec::with_capacity(xs.len());
        let mut raw_f = Vec::with_capacity(xs.len());
        for img in xs {
            let mut a = vec![0.0f32; N_CLASSES + 2];
            let mut r = vec![0.0f32; N_CLASSES + 2];
            for d in 0..N_CLASSES {
                let (_, acc) = run_digit(&mut learned[d], img, 0.0, band_of[d]);
                a[d] = acc;
                r[d] = (0..ROWS).map(|row| band_density(img, row, band_of[d])).sum::<f32>();
            }
            let spread: f32 = (0..ROWS).map(|row| row_spread(img, row)).sum();
            let ink: f32 = img.iter().sum();
            a[N_CLASSES] = spread;
            a[N_CLASSES + 1] = ink;
            r[N_CLASSES] = spread;
            r[N_CLASSES + 1] = ink;
            acc_f.push(a);
            raw_f.push(r);
        }
        (acc_f, raw_f)
    };

    println!("🧪 Собираем признаки...");
    let (train_acc, train_raw) = feats(&train_img);
    let (test_acc, test_raw) = feats(&test_img);
    let train_y: Vec<usize> = train_lbl.iter().map(|&l| l as usize).collect();
    let test_y: Vec<usize> = test_lbl.iter().map(|&l| l as usize).collect();
    let n_test = test_y.len();

    // ─── 1. argmax по нейронам (базовый) ────────────────────────
    let mut cm_arg = [[0usize; N_CLASSES]; N_CLASSES];
    for (f, &l) in test_acc.iter().zip(&test_y) {
        let mut best = 0usize;
        for d in 1..N_CLASSES { if f[d] > f[best] { best = d; } }
        cm_arg[l][best] += 1;
    }
    let acc_arg = diag(&cm_arg) as f64 * 100.0 / n_test as f64;

    // ─── 2. softmax над сырыми полосами ─────────────────────────
    let sm_raw = Softmax::train(&train_raw, &train_y, 300, 0.1);
    let mut cm_raw = [[0usize; N_CLASSES]; N_CLASSES];
    for (f, &l) in test_raw.iter().zip(&test_y) {
        cm_raw[l][sm_raw.predict(f)] += 1;
    }
    let acc_raw = diag(&cm_raw) as f64 * 100.0 / n_test as f64;

    // ─── 3. softmax над признаками нейронов ─────────────────────
    let sm_n = Softmax::train(&train_acc, &train_y, 300, 0.1);
    let mut cm_n = [[0usize; N_CLASSES]; N_CLASSES];
    for (f, &l) in test_acc.iter().zip(&test_y) {
        cm_n[l][sm_n.predict(f)] += 1;
    }
    let acc_n = diag(&cm_n) as f64 * 100.0 / n_test as f64;

    // ─── 4. softmax над склейкой (нейроновые + сырые + глобальные) ──
    let mut train_cat = Vec::with_capacity(train_acc.len());
    let mut test_cat = Vec::with_capacity(test_acc.len());
    for i in 0..train_acc.len() {
        let mut v = train_acc[i].clone();
        v.extend_from_slice(&train_raw[i]);
        train_cat.push(v);
    }
    for i in 0..test_acc.len() {
        let mut v = test_acc[i].clone();
        v.extend_from_slice(&test_raw[i]);
        test_cat.push(v);
    }
    let sm_cat = Softmax::train(&train_cat, &train_y, 400, 0.1);
    let mut cm_cat = [[0usize; N_CLASSES]; N_CLASSES];
    for (f, &l) in test_cat.iter().zip(&test_y) {
        cm_cat[l][sm_cat.predict(f)] += 1;
    }
    let acc_cat = diag(&cm_cat) as f64 * 100.0 / n_test as f64;

    // ─── Отчёт ──────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  1. argmax по нейронам (база):           {:5.1}%    ║", acc_arg);
    println!("║  2. softmax над сырыми полосами:         {:5.1}%    ║", acc_raw);
    println!("║  3. softmax над признаками нейронов:     {:5.1}%    ║", acc_n);
    println!("║  4. softmax над склейкой (нейроны+сырые): {:5.1}%    ║", acc_cat);
    println!("╚══════════════════════════════════════════════════════╝");
    print_cm("🧮 Матрица ошибок: лучшая конфигурация — склейка признаков (STDP + сырые)",
             &cm_cat);
}

fn diag(cm: &[[usize; N_CLASSES]; N_CLASSES]) -> usize {
    cm.iter().enumerate().map(|(i, row)| row[i]).sum()
}
