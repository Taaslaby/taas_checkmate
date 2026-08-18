//! # Научные примитивы плоттеров (`plots::sci`)
//!
//! Чистые математические функции без UI — их можно тестировать в отрыве от
//! графиков и переиспользовать в других анализах (не только в плоттерах):
//!
//! * [`fft`] — радикс-2 БПФ (Кули–Туки), эталон для валидации спектра;
//! * [`power_spectrum`] — периодограмма + доминирующая частота;
//! * [`isis_ms`] — межспайковые интервалы из времён спайков;
//! * [`hist_linear`] — линейная гистограмма с последней «корзиной-стоком»;
//! * [`antiphase_stats`] — второй круговой момент фаз (детектор антифазы).

/// Радикс-2 БПФ (Кули–Туки), in-place, `re/im` длины — степень двойки.
/// Классическая реализация — эталон для валидации спектра.
pub(super) fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && re.len() == im.len());
    let shift = n.trailing_zeros();
    for i in 1..n {
        let j = i.reverse_bits() >> (usize::BITS - shift);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wre, wim) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0usize;
        while i < n {
            let (mut wr, mut wi) = (1.0, 0.0);
            for j in 0..half {
                let u = i + j;
                let v = u + half;
                let tr = re[v] * wr - im[v] * wi;
                let ti = re[v] * wi + im[v] * wr;
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] += tr;
                im[u] += ti;
                let nwr = wr * wre - wi * wim;
                wi = wr * wim + wi * wre;
                wr = nwr;
            }
            i += len;
        }
        len *= 2;
    }
}

/// Периодограмма: относительная мощность по частотам (Гц) + доминирующая
/// частота. Сигнал: вычитание среднего, окно Ханна, дополнение нулями до
/// степени двойки (разрешение ~fs/1024). Нормировка P = |X|²/n² — единичная
/// мощность у синусоиды с амплитудой 1 (стандарт для «отн. мощности»).
pub(super) fn power_spectrum(sig: &[f32], fs: f64) -> (Vec<[f64; 2]>, f64, f64) {
    let m = sig.len();
    if m < 4 {
        return (Vec::new(), 0.0, 0.0);
    }
    let mut n = 1024usize;
    while n < m {
        n *= 2;
    }
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    let mean: f64 = sig.iter().map(|&v| v as f64).sum::<f64>() / m as f64;
    for (t, v) in sig.iter().enumerate() {
        // окно Ханна: подавляет утечки спектра по краям окна
        let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * t as f64 / (m - 1) as f64).cos());
        re[t] = (*v as f64 - mean) * w;
    }
    fft(&mut re, &mut im);
    let mut freqs = Vec::with_capacity(n / 2);
    let mut dom_f = 0.0;
    let mut dom_p = 0.0;
    for k in 1..n / 2 {
        let f = k as f64 * fs / n as f64;
        if f > 300.0 {
            break; // выше 300 Гц — вне интереса (CPG/локомоция 5–30 Гц)
        }
        let p = (re[k] * re[k] + im[k] * im[k]) / (n as f64 * n as f64);
        freqs.push([f, p]);
        if p > dom_p {
            dom_p = p;
            dom_f = f;
        }
    }
    (freqs, dom_f, dom_p)
}

/// Все межспайковые интервалы (мс) из времён спайков (шаги × dt_ms):
/// для каждого нейрона с ≥2 спайками берём разности соседних времён.
pub(super) fn isis_ms(spike_times: &[Vec<u32>], dt_ms: f32) -> Vec<f32> {
    let mut isis = Vec::new();
    for tr in spike_times {
        for w in tr.windows(2) {
            isis.push((w[1] - w[0]) as f32 * dt_ms);
        }
    }
    isis
}

/// Линейная гистограмма: счётчики по `bins` корзинам на [0, cap].
/// Значения выше `cap` (выбросы) не отбрасываются — уходят в последнюю корзину.
pub(super) fn hist_linear(values: &[f32], bins: usize, cap: f32) -> Vec<f64> {
    let mut counts = vec![0.0f64; bins];
    if values.is_empty() || bins == 0 || cap <= 0.0 {
        return counts;
    }
    for &v in values {
        let b = ((v / cap).clamp(0.0, 1.0) * (bins as f32 - 1e-6)) as usize;
        counts[b.min(bins - 1)] += 1.0;
    }
    counts
}

/// Второй круговой момент фаз — детектор **антифазы** (r2 ∈ `[0,1]`).
/// Фазы в [0,1) (единица = один цикл сети). Если спайки идут точно в
/// противофазе (φ и φ+0.5), второй момент схлопывается в 1; если фазы
/// равномерны — ≈0. Возвращает (r2, ось в долях цикла [0, 0.5)).
pub(super) fn antiphase_stats(phases: &[f64]) -> (f64, f64) {
    if phases.is_empty() {
        return (0.0, 0.0);
    }
    let (mut c, mut s) = (0.0f64, 0.0f64);
    for &ph in phases {
        let th = 2.0 * std::f64::consts::PI * ph; // угол на окружности
        c += (2.0 * th).cos();
        s += (2.0 * th).sin();
    }
    let n = phases.len() as f64;
    c /= n;
    s /= n;
    let r2 = (c * c + s * s).sqrt();
    let axis = (0.5 * s.atan2(c) / (2.0 * std::f64::consts::PI)).rem_euclid(0.5);
    (r2, axis)
}

// ── научная валидация ────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::{antiphase_stats, hist_linear, isis_ms, power_spectrum};

    /// Валидация спектра: синусоида 20 Гц (fs = 1000 Гц) должна дать
    /// доминирующую частоту ≈ 20 Гц — эталон, по которому проверяем БПФ.
    #[test]
    fn fft_detects_known_frequency() {
        let fs = 1000.0;
        let sig: Vec<f32> = (0..1000)
            .map(|t| (2.0 * std::f64::consts::PI * 20.0 * t as f64 / fs).sin() as f32)
            .collect();
        let (_, dom_f, dom_p) = power_spectrum(&sig, fs);
        assert!(dom_p > 0.0, "у синусоиды должна быть мощность");
        assert!(
            (dom_f - 20.0).abs() < 1.5,
            "ожидали ~20 Гц, получили {dom_f}"
        );
    }

    /// Валидация вычитания среднего: константа не должна давать пиков.
    #[test]
    fn fft_flat_signal_has_no_dominant_frequency() {
        let sig = vec![5.0f32; 256];
        let (_, _dom_f, dom_p) = power_spectrum(&sig, 1000.0);
        assert!(dom_p < 1e-6, "у константы не должно быть пиков, p={dom_p}");
    }

    /// ISI: разности соседних времён спайков × dt, по каждому нейрону.
    #[test]
    fn isi_from_spike_times() {
        let spike_times = vec![vec![0u32, 10, 20], vec![5, 15], Vec::new(), vec![3]];
        let isis = isis_ms(&spike_times, 1.0);
        assert_eq!(isis, vec![10.0, 10.0, 10.0]);
        // dt влияет на масштаб
        let isis2 = isis_ms(&spike_times, 0.5);
        assert_eq!(isis2, vec![5.0, 5.0, 5.0]);
    }

    /// Антифаза: спайки ровно на фазах 0 и 0.5 (пол-цикла) — r2 ≈ 1,
    /// ось на половинном шаге (0 или ~0.5 в долях цикла, тут 0).
    #[test]
    fn antiphase_detected() {
        // мотор A бьёт в начале цикла, мотор B — ровно через пол-цикла.
        let mut phases = Vec::new();
        phases.extend(std::iter::repeat_n(0.0, 100));
        phases.extend(std::iter::repeat_n(0.5, 100));
        let (r2, _) = antiphase_stats(&phases);
        assert!(r2 > 0.95, "чистая антифаза должна дать r2≈1, получили {r2}");
    }

    /// Равномерные фазы (нет антифазы) — r2 ≈ 0.
    #[test]
    fn uniform_phases_have_low_r2() {
        let phases: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.377).fract()).collect();
        let (r2, _) = antiphase_stats(&phases);
        assert!(
            r2 < 0.15,
            "равномерные фазы должны дать r2≈0, получили {r2}"
        );
    }

    /// Гистограмма: значения внутри [0, cap] попадают в свою корзину,
    /// сумма счётчиков равна числу значений. Формула использует
    /// `bins − 1e-6`, поэтому 0.1 → корзина 0, а 1.0 → последняя (9).
    #[test]
    fn hist_linear_bins_values() {
        let values = vec![0.1, 0.5, 0.9, 1.0];
        let counts = hist_linear(&values, 10, 1.0);
        assert_eq!(counts.len(), 10);
        assert_eq!(counts.iter().sum::<f64>(), values.len() as f64);
        assert_eq!(counts[0], 1.0, "0.1 → корзина 0");
        assert_eq!(counts[4], 1.0, "0.5 → корзина 4");
        assert_eq!(counts[8], 1.0, "0.9 → корзина 8");
        assert_eq!(counts[9], 1.0, "1.0 → последняя корзина");
    }

    /// Выбросы выше `cap` не теряются — уходят в последнюю корзину.
    #[test]
    fn hist_linear_clamps_outliers_to_last_bin() {
        let values = vec![0.0, 2.0, 5.0, 10.0]; // cap = 1.0
        let counts = hist_linear(&values, 4, 1.0);
        assert_eq!(counts[0], 1.0, "0.0 → первая корзина");
        assert_eq!(counts[3], 3.0, "все выбросы → последняя корзина");
    }

    /// Пустой ввод и невалидный `cap` — нулевые счётчики, без паники.
    #[test]
    fn hist_linear_handles_empty_and_bad_cap() {
        assert!(hist_linear(&[], 4, 1.0).iter().all(|&c| c == 0.0));
        assert!(hist_linear(&[1.0, 2.0], 4, 0.0).iter().all(|&c| c == 0.0));
        assert!(hist_linear(&[1.0, 2.0], 0, 1.0).is_empty());
    }
}
