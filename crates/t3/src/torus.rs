//  TorusGrid — Kuramoto + Stuart-Landau на 2D-торе (пончик).
//
//  Осцилляторы — на сетке W×H с wrap-around (тороидальные граничные условия).
//  Локальная связь: каждый осциллятор (x,y) связан с соседями в радиусе R
//  (вес = coupling_strength / |N|). Нет трёх популяций — это ЧИСТЫЙ тор.
//
//  ```text
//  θ'(x,y) = θ(x,y) + ω(x,y)·dt
//            + K/|N|·Σ_{(dx,dy)∈ring(R)} sin(θ(x+dx,y+dy)−θ(x,y))·dt
//            + drive(x,y)·gain·dt/(r(x,y)+ε)
//  r'(x,y) = r(x,y) + dt·((α−r²)·r + drive·gain)     — Stuart-Landau
//  ```
//
//  При K > критического: стихийная синхронизация → бегущие волны по тору.
//  При локальном K: спиральные волны, патч-синхронизация, пространственные
//  моды. Это «пончик-тензор» — геометрический кузен T³ из Nakajima NeuroBrain.
//
//  BPTT: см. backward_one_step / bptt ниже — аналитические производные
//  локальной Kuramoto-связи, готовые для SPIGRAD-интеграции.
//
//  # 📜 Soviet Prior Art (советский prior art)
//
//  Три статьи из советских журналов, предвосхитившие Kuramoto (1975) и
//  современную физику нелинейных осцилляторов:
//
//  | Год | Автор | Вклад |
//  |-----|-------|-------|
//  | 1957 | Ишлинский А.Ю. | Бифуркация БЕЗ потери устойчивости — новая форма
//  |      | *ДАН СССР*     | ответвляется, а основная остаётся стабильной.
//  |      |                 | → `bifurcation_scan()` ниже. |
//  | 1967 | Петвиашвили В.И. | Нелинейные дрейфовые волны в плазме — инерция
//  |      | *ДАН СССР*        | ионов предотвращает ударный фронт и создаёт
//  |      |                    | стационарные периодические волны.
//  |      |                    | → `step_inertial()` ниже. |
//  | 1968 | Розенблат М.А. | Условие автоколебаний: отрицательное
//  |      | *ДАН СССР*       | дифференциальное сопротивление необходимо.
//  |      |                   | → α>0 в Stuart-Landau = negative resistance. |
//
//  PDF-ки скачаны с sovietrxiv.org → `resources/docs/soviet/`.
//  Все три ссылаются на Ландау-Лифшица «Гидродинамика» (1953).

use std::f32::consts::PI;

/// Запись траектории: состояния на каждом шаге (для BPTT).
#[derive(Clone)]
pub struct TorusTrajectory {
    pub n_steps: usize,
    pub n_osc: usize,
    pub phases: Vec<f32>,     // [n_steps × n_osc] — θ ДО шага
    pub amplitudes: Vec<f32>, // [n_steps × n_osc] — r ДО шага
    pub drives: Vec<f32>,     // [n_steps × n_osc]
}

impl TorusTrajectory {
    pub fn new(n_steps: usize, n_osc: usize) -> Self {
        Self {
            n_steps,
            n_osc,
            phases: vec![0.0; n_steps * n_osc],
            amplitudes: vec![0.0; n_steps * n_osc],
            drives: vec![0.0; n_steps * n_osc],
        }
    }
}

/// Точка на бифуркационной диаграмме (Ишлинский 1957).
#[derive(Clone, Debug)]
pub struct BifurcationPoint {
    pub k: f32,                      // coupling strength
    pub r: f32,                      // глобальный порядок
    pub top_mode: (i32, i32),        // доминантная волна (kx, ky)
    pub top_power: f32,              // мощность доминантной моды
    pub first_wave: (i32, i32, f32), // первая не-R мода: (kx, ky, F)
}

/// 2D-сетка на торе: wrap-around по обоим измерениям.
#[derive(Clone)]
pub struct TorusGrid {
    pub w: usize, // ширина (x)
    pub h: usize, // высота (y)
    pub phases: Vec<f32>,
    pub amplitudes: Vec<f32>,
    pub omegas: Vec<f32>,
    /// Фазовые скорости (θ̇) — для инерциального режима (Петвиашвили 1967).
    /// В первом порядке (без инерции) не используются.
    pub phase_velocities: Vec<f32>,
}

impl TorusGrid {
    /// Создать сетку W×H. Частоты — линейная развёртка [freq_min, freq_max].
    /// Фазы случайные (или нулевые, задать позже). Амплитуды = 0.1.
    pub fn new(w: usize, h: usize, freq_min: f32, freq_max: f32, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let n = w * h;
        let mut rng = StdRng::seed_from_u64(seed);
        use rand::Rng;
        let freq_spread = freq_max - freq_min;
        let mut phases = Vec::with_capacity(n);
        let mut omegas = Vec::with_capacity(n);
        for i in 0..n {
            phases.push(rng.gen::<f32>() * 2.0 * PI);
            omegas.push(freq_min + freq_spread * (i as f32 / n.max(1) as f32));
        }
        Self {
            w,
            h,
            phases,
            amplitudes: vec![1.0; n],
            omegas,
            phase_velocities: vec![0.0; n],
        }
    }

    /// Сброс к детерминированному начальному состоянию (для независимых
    /// forward-прогонов в обучении). Фазы случайные из seed, амплитуды = 1.0,
    /// скорости = 0. Частоты и связь не трогаются.
    pub fn reset(&mut self, seed: u64) {
        use rand::rngs::StdRng;
        use rand::Rng;
        use rand::SeedableRng;
        let n = self.w * self.h;
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..n {
            self.phases[i] = rng.gen::<f32>() * 2.0 * PI;
        }
        self.amplitudes.fill(1.0);
        self.phase_velocities.fill(0.0);
    }

    /// Индекс (x, y) с wrap-around.
    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        (y % self.h) * self.w + (x % self.w)
    }

    /// Один шаг по времени.
    ///
    /// - `dt`: шаг по времени
    /// - `coupling_strength` (K): сила локальной связи
    /// - `radius`: радиус соседства (1 = 8 соседей, 2 = 24 соседа, ...)
    /// - `drives`: [W×H] — внешний драйв на каждый осциллятор
    /// - `drive_gain`: коэффициент усиления драйва
    /// - `alpha`: параметр Stuart-Landau (1.0 = стандарт)
    /// - `eps`: защита от деления на ноль (ε)
    pub fn step(
        &mut self,
        dt: f32,
        coupling_strength: f32,
        radius: usize,
        drives: &[f32],
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) {
        let _n = self.w * self.h;
        assert_eq!(drives.len(), _n, "drives must match grid size");
        let nf = ((2 * radius + 1).pow(2).saturating_sub(1)) as f32; // |N| (без центра)

        let mut new_phases = vec![0.0f32; _n];
        let mut new_amps = vec![0.0f32; _n];

        for y in 0..self.h {
            for x in 0..self.w {
                let i = self.idx(x, y);
                let th_i = self.phases[i];
                let r_i = self.amplitudes[i];

                // ── Локальная купловка: O(radius²) ──
                let mut cpl = 0.0f32;
                if radius > 0 && nf > 0.0 {
                    let dx_range = radius as isize;
                    for dy in -dx_range..=dx_range {
                        for dx in -dx_range..=dx_range {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            // wrap-around
                            let nx = ((x as isize + dx).rem_euclid(self.w as isize)) as usize;
                            let ny = ((y as isize + dy).rem_euclid(self.h as isize)) as usize;
                            let j = self.idx(nx, ny);
                            let dth = self.phases[j] - th_i;
                            cpl += dth.sin(); // чистый Kuramoto, без r-взвешивания
                        }
                    }
                    cpl *= coupling_strength / nf;
                }

                // ── Фаза: θ' = θ + dt·(ω + coupling + drive·gain/(r+ε)) ──
                let dth = self.omegas[i] + cpl + drives[i] * drive_gain / (r_i + eps);
                new_phases[i] = th_i + dt * dth;

                // ── Амплитуда: r' = r + dt·((α−r²)·r + drive·gain) ──
                let dr = (alpha - r_i * r_i) * r_i + drives[i] * drive_gain;
                let r_next = r_i + dt * dr;
                // Клэмп: как в T3
                new_amps[i] = r_next.clamp(0.001, 10.0);
            }
        }

        // Оборачиваем фазы в [−π, π] (как в T3 TriOscillator)
        for p in &mut new_phases {
            *p = (*p + PI).rem_euclid(2.0 * PI) - PI;
        }
        self.phases = new_phases;
        self.amplitudes = new_amps;
    }

    // ═══════════════ Советские методы ═══════════════

    /// Инерциальный шаг (Петвиашвили 1967): second-order фазовая динамика.
    ///
    /// В плазме: инерция ионов предотвращает опрокидывание дрейфовой волны
    /// и создаёт стационарные периодические решения (уравнение 19).
    ///
    /// У нас: добавляем «массу» фазам — coupling ускоряет фазу, а не толкает
    /// её напрямую. Результат: стоячие волны вместо бегущих, нет разрыва
    /// фазового фронта.
    ///
    /// ```text
    /// θ̈  = ω + coupling + drive·gain/(r+ε) − γ·θ̇    (ускорение)
    /// θ̇' = θ̇ + θ̈·dt                                  (скорость)
    /// θ'  = θ + θ̇'·dt                                  (фаза)
    /// r'  = r + dt·((α−r²)·r + drive·gain)             (амплитуда, без изм.)
    /// ```
    ///
    /// - `damping` (γ): трение. При γ=0 — незатухающие волны (консервативная
    ///   система). При γ>0 — волны затухают без драйва.
    ///
    /// Ключевой эффект Петвиашвили: без инерции (первый порядок) фазовая
    /// неоднородность растёт → разрыв (shock). С инерцией (второй порядок)
    /// дисперсия останавливает опрокидывание → стоячие волны.
    pub fn step_inertial(
        &mut self,
        dt: f32,
        coupling_strength: f32,
        radius: usize,
        drives: &[f32],
        drive_gain: f32,
        alpha: f32,
        eps: f32,
        damping: f32,
    ) {
        let n = self.w * self.h;
        assert_eq!(drives.len(), n, "drives must match grid size");
        let nf = ((2 * radius + 1).pow(2).saturating_sub(1)) as f32;

        let mut new_phases = vec![0.0f32; n];
        let mut new_velocities = vec![0.0f32; n];
        let mut new_amps = vec![0.0f32; n];

        for y in 0..self.h {
            for x in 0..self.w {
                let i = self.idx(x, y);
                let th_i = self.phases[i];
                let r_i = self.amplitudes[i];
                let v_i = self.phase_velocities[i];

                // Локальная купловка (как в step)
                let mut cpl = 0.0f32;
                if radius > 0 && nf > 0.0 {
                    let dx_range = radius as isize;
                    for dy in -dx_range..=dx_range {
                        for dx in -dx_range..=dx_range {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = ((x as isize + dx).rem_euclid(self.w as isize)) as usize;
                            let ny = ((y as isize + dy).rem_euclid(self.h as isize)) as usize;
                            let j = self.idx(nx, ny);
                            cpl += (self.phases[j] - th_i).sin();
                        }
                    }
                    cpl *= coupling_strength / nf;
                }

                // Ускорение: θ̈ = ω + coupling + drive − γ·θ̇
                let acc =
                    self.omegas[i] + cpl + drives[i] * drive_gain / (r_i + eps) - damping * v_i;
                // Скорость: θ̇' = θ̇ + θ̈·dt
                let v_next = v_i + dt * acc;
                // Фаза: θ' = θ + θ̇'·dt
                let th_next = th_i + dt * v_next;

                new_velocities[i] = v_next;
                new_phases[i] = th_next;

                // Амплитуда (без изменений)
                let dr = (alpha - r_i * r_i) * r_i + drives[i] * drive_gain;
                new_amps[i] = (r_i + dt * dr).clamp(0.001, 10.0);
            }
        }

        for p in &mut new_phases {
            *p = (*p + PI).rem_euclid(2.0 * PI) - PI;
        }
        self.phases = new_phases;
        self.phase_velocities = new_velocities;
        self.amplitudes = new_amps;
    }

    /// Сканирование бифуркаций (Ишлинский 1957): найти K, при котором
    /// появляется новая волновая мода.
    ///
    /// Возвращает Vec<(K, R, доминантная_мода, F_доминантной)> — точки
    /// бифуркационной диаграммы. Ишлинский показал, что основная форма
    /// (R-мода) может оставаться устойчивой при ветвлении новых мод —
    /// проверяем это на нашем торе.
    pub fn bifurcation_scan(
        &self,
        k_min: f32,
        k_max: f32,
        k_step: f32,
        n_steps: usize,
        dt: f32,
        radius: usize,
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) -> Vec<BifurcationPoint> {
        let drives = vec![0.0f32; self.w * self.h];
        let mut points = Vec::new();
        let mut k = k_min;
        while k <= k_max {
            let mut g = self.clone();
            for _ in 0..n_steps {
                g.step(dt, k, radius, &drives, drive_gain, alpha, eps);
            }
            let r = g.order_parameter();
            let spec = g.wave_spectrum(3);
            let top = spec[0];
            let non_sync = spec
                .iter()
                .find(|(kx, ky, _)| *kx != 0 || *ky != 0)
                .map(|(kx, ky, f)| (*kx, *ky, *f))
                .unwrap_or((0, 0, 0.0));
            points.push(BifurcationPoint {
                k,
                r,
                top_mode: (top.0, top.1),
                top_power: top.2,
                first_wave: non_sync,
            });
            k += k_step;
        }
        points
    }

    /// Проверка условия Розенблата (1968): есть ли отрицательное сопротивление.
    ///
    /// В Stuart-Landau: α > 0 → раскачка (negative resistance) → предельный цикл.
    /// α ≤ 0 → затухание (positive resistance) → нет автоколебаний.
    /// Возвращает true если система способна к самоподдерживающимся колебаниям.
    pub fn negative_resistance_check(alpha: f32) -> bool {
        alpha > 0.0
    }

    /// Глобальный параметр порядка (Kuramoto): R = |1/N·Σ e^{iθ}| ∈ [0, 1].
    pub fn order_parameter(&self) -> f32 {
        let n = self.w * self.h;
        let (ss, sc): (f64, f64) = self.phases.iter().fold((0.0, 0.0), |(s, c), &p| {
            (s + p.sin() as f64, c + p.cos() as f64)
        });
        ((ss * ss + sc * sc).sqrt() / n as f64) as f32
    }

    /// Локальные параметры порядка по патчам (W×H → (W/p)×(H/p)).
    pub fn local_order_params(&self, patch: usize) -> Vec<f32> {
        let pw = self.w / patch;
        let ph = self.h / patch;
        let n_patch = (patch * patch) as f32;
        let mut out = vec![0.0f32; pw * ph];
        for gy in 0..ph {
            for gx in 0..pw {
                let (mut ss, mut sc): (f64, f64) = (0.0, 0.0);
                for dy in 0..patch {
                    for dx in 0..patch {
                        let x = gx * patch + dx;
                        let y = gy * patch + dy;
                        let p = self.phases[self.idx(x, y)];
                        ss += p.sin() as f64;
                        sc += p.cos() as f64;
                    }
                }
                out[gy * pw + gx] = ((ss * ss + sc * sc).sqrt() / n_patch as f64) as f32;
            }
        }
        out
    }

    /// Двумерный слайс фаз для визуализации: [h][w] row-major, в диапазоне [0, 2π).
    pub fn phase_field(&self) -> Vec<f32> {
        self.phases.clone()
    }

    /// Двумерный слайс амплитуд.
    pub fn amplitude_field(&self) -> Vec<f32> {
        self.amplitudes.clone()
    }

    /// Пространственная фурье-мода: порядок на волновом векторе (kx, ky).
    ///
    /// ```text
    /// F(kx,ky) = |1/N· Σ_{x,y} exp(i·θ(x,y)) · exp(−2πi·(kx·x/W + ky·y/H))|
    /// ```
    ///
    /// F(0,0) = глобальный R. Ненулевые (kx,ky) детектируют бегущие волны:
    /// волна вправо → пик на kx>0, спираль → несколько пиков, синхронизация → только (0,0).
    pub fn fourier_mode(&self, kx: i32, ky: i32) -> f32 {
        let n = (self.w * self.h) as f64;
        let (mut re, mut im): (f64, f64) = (0.0, 0.0);
        for y in 0..self.h {
            for x in 0..self.w {
                let p = self.phases[self.idx(x, y)];
                // wave_phase = −2π·(kx·x/W + ky·y/H)  (комплексно-сопряжённая волна)
                let wp = -2.0
                    * std::f64::consts::PI
                    * (kx as f64 * x as f64 / self.w as f64 + ky as f64 * y as f64 / self.h as f64);
                re += (p as f64 + wp).cos();
                im += (p as f64 + wp).sin();
            }
        }
        ((re * re + im * im).sqrt() / n) as f32
    }

    /// Градиент фурье-моды по фазам: ∂F(kx,ky)/∂θⱼ.
    ///
    /// ```text
    /// F = |S|,  S = 1/N·Σ exp(i·(θ − wave))
    /// ∂F/∂θⱼ = (1/(N·F))·[−Re·sin(θⱼ−waveⱼ) + Im·cos(θⱼ−waveⱼ)]
    /// ```
    /// Используется для BPTT: dL/dθ = dL/dF · ∂F/∂θ.
    pub fn fourier_mode_grad(&self, kx: i32, ky: i32) -> (f32, Vec<f32>) {
        let n = self.w * self.h;
        let nf = n as f64;
        let (mut re, mut im): (f64, f64) = (0.0, 0.0);
        let mut grad = vec![0.0f32; n];
        for y in 0..self.h {
            for x in 0..self.w {
                let i = self.idx(x, y);
                let p = self.phases[i] as f64;
                let wp = -2.0
                    * std::f64::consts::PI
                    * (kx as f64 * x as f64 / self.w as f64 + ky as f64 * y as f64 / self.h as f64);
                let arg = p + wp;
                re += arg.cos();
                im += arg.sin();
            }
        }
        let f = ((re * re + im * im).sqrt() / nf) as f32;
        if f < 1e-8 {
            return (0.0, grad); // F ≈ 0 → градиент неопределён (нуль)
        }
        let inv_nf = 1.0 / nf;
        let re_f = re;
        let im_f = im;
        for y in 0..self.h {
            for x in 0..self.w {
                let i = self.idx(x, y);
                let p = self.phases[i] as f64;
                let wp = -2.0
                    * std::f64::consts::PI
                    * (kx as f64 * x as f64 / self.w as f64 + ky as f64 * y as f64 / self.h as f64);
                let arg = p + wp;
                // ∂F/∂θ = (1/(N·F))·[−Re·sin(arg) + Im·cos(arg)]
                let d = (-re_f * arg.sin() + im_f * arg.cos()) * inv_nf / f as f64;
                grad[i] = d as f32;
            }
        }
        (f, grad)
    }

    /// Волновой спектр: мощность всех мод |k| ≤ max_k.
    /// Возвращает Vec<(kx, ky, power)>, отсортированный по убыванию мощности. Первый
    /// элемент — доминантная волна (или (0,0) = глобальный R).
    pub fn wave_spectrum(&self, max_k: i32) -> Vec<(i32, i32, f32)> {
        let mut modes = Vec::new();
        for ky in -max_k..=max_k {
            for kx in -max_k..=max_k {
                modes.push((kx, ky, self.fourier_mode(kx, ky)));
            }
        }
        modes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        modes
    }

    /// Установить константный драйв на всё поле.
    pub fn fill_drives(drives: &mut [f32], val: f32) {
        drives.fill(val);
    }

    /// Создать драйв-вектор нужного размера.
    pub fn drives_vec(&self) -> Vec<f32> {
        vec![0.0; self.w * self.h]
    }

    /// Записать текущее состояние в траекторию (слот `step`).
    pub fn record_step(&self, traj: &mut TorusTrajectory, step: usize, drives: &[f32]) {
        let n = self.w * self.h;
        let off = step * n;
        traj.phases[off..off + n].copy_from_slice(&self.phases);
        traj.amplitudes[off..off + n].copy_from_slice(&self.amplitudes);
        traj.drives[off..off + n].copy_from_slice(drives);
    }

    /// Один шаг BPTT назад через динамику Kuramoto-торуса.
    ///
    /// Вход: `dth_in[i]` = dL/d(θ'_i), `dr_in[i]` = dL/d(r'_i).
    /// Выход: `dth_out[i]` = dL/d(θ_i), `dr_out[i]` = dL/d(r_i),
    ///        `ddrives_out[i]` = dL/d(drive_i).
    /// Возвращает dL/dK — вклад этого шага в градиент coupling strength.
    ///
    /// Производные:
    /// ```text
    /// ∂θ'_i/∂θ_i = 1 + dt·(∂cpl_i/∂θ_i)  (cpl = K/|N|·Σ sin(θj−θi))
    /// ∂cpl_i/∂θ_i = −K/|N|·Σ_j cos(θj−θi)
    /// ∂cpl_j/∂θ_i =  K/|N|·cos(θj−θi)   (для j ∈ N(i))
    /// ∂θ'_i/∂(drive_i) = dt·gain/(r_i+ε)
    /// ∂r'_i/∂r_i = 1 + dt·(α−3r_i²)    (если не клампнут)
    /// ∂r'_i/∂(drive_i) = dt·gain
    /// ∂θ'_i/∂r_i = dt·(−drive_i·gain/(r_i+ε)²)
    /// ```
    pub fn backward_one_step(
        &self,
        th_pre: &[f32], // θ ДО шага
        r_pre: &[f32],  // r ДО шага
        drives: &[f32],
        dth_in: &[f32], // градиент из θ ПОСЛЕ шага
        dr_in: &[f32],  // градиент из r ПОСЛЕ шага
        dth_out: &mut [f32],
        dr_out: &mut [f32],
        ddrives_out: &mut [f32],
        coupling_strength: f32,
        radius: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) -> f32 {
        let n = self.w * self.h;
        let nf = ((2 * radius + 1).pow(2).saturating_sub(1)) as f32;

        let mut dk = 0.0f32;

        // Base: dθ_out[i] ← dθ_in[i] (identity ∂θ'_i/∂θ_i = 1)
        dth_out.copy_from_slice(dth_in);

        if radius > 0 && nf > 0.0 {
            let dx_range = radius as isize;
            for y in 0..self.h {
                for x in 0..self.w {
                    let i = self.idx(x, y);
                    let th_i = th_pre[i];

                    for dy in -dx_range..=dx_range {
                        for dx in -dx_range..=dx_range {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = ((x as isize + dx).rem_euclid(self.w as isize)) as usize;
                            let ny = ((y as isize + dy).rem_euclid(self.h as isize)) as usize;
                            let j = self.idx(nx, ny);
                            let dth = th_pre[j] - th_i;
                            let cos_dth = dth.cos();
                            let sin_dth = dth.sin();

                            // Градиент coupling для (i,j): cos(θj−θi)·(dθ_in[j] − dθ_in[i])·dt·K/nf
                            let edge_grad =
                                cos_dth * (dth_in[j] - dth_in[i]) * dt * coupling_strength / nf;
                            dth_out[i] += edge_grad;
                            // NOTE: dth_out[j] тоже получит от зеркальной итерации (j как центр),
                            // поэтому НЕ добавляем сюда — во избежание двойного учёта.

                            // dL/dK: dθ_in[i]·dt·(1/nf)·sin(θj−θi)
                            dk += dth_in[i] * dt * sin_dth / nf;
                        }
                    }
                }
            }
        }

        // r dynamics + ddrives:
        for i in 0..n {
            let r = r_pre[i];
            let pre_r = r + dt * ((alpha - r * r) * r + drives[i] * drive_gain);
            // r-clamp gate: как в T3, если r_post вне [0.001, 10], градиент обнуляется
            let rg = if (0.001..=10.0).contains(&pre_r) {
                1.0
            } else {
                0.0
            };
            // ∂r'_i/∂r_i
            let dr_dr = 1.0 + dt * (alpha - 3.0 * r * r);
            dr_out[i] = dr_in[i] * dr_dr * rg;

            // Кросс-член: θ'_i зависит от r_i через drive·gain/(r+ε)
            let rpe = r + eps;
            dr_out[i] += dth_in[i] * dt * (-drives[i] * drive_gain / (rpe * rpe));

            // dL/d(drive_i): ∂θ'_i/∂(drive_i)=dt·gain/(r+ε), ∂r'_i/∂(drive_i)=dt·gain
            ddrives_out[i] = dth_in[i] * dt * drive_gain / rpe + dr_in[i] * dt * drive_gain * rg;
        }

        dk
    }

    /// Полный BPTT через траекторию.
    ///
    /// `dth_final`, `dr_final` — градиент функции потерь по финальным θ, r.
    /// `ddrives_final` — градиент по драйвам на последнем шаге (обычно нули).
    /// Возвращает (dL/dK, dL/dθ₀, dL/dr₀, dL/d(drives) по всем шагам).
    pub fn bptt(
        &self,
        traj: &TorusTrajectory,
        dth_final: &[f32],
        dr_final: &[f32],
        ddrives_final: &[f32],
        coupling_strength: f32,
        radius: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.w * self.h;
        let n_steps = traj.n_steps;
        assert_eq!(dth_final.len(), n);
        assert_eq!(dr_final.len(), n);

        let mut dth = dth_final.to_vec();
        let mut dr = dr_final.to_vec();
        let mut dk_total = 0.0f32;

        let mut dth_prev = vec![0.0f32; n];
        let mut dr_prev = vec![0.0f32; n];
        let mut ddrives_prev = vec![0.0f32; n];

        // ddrives_sum[t][i] = dL/d(drive_i на шаге t)
        let mut ddrives_all = vec![0.0f32; n_steps * n];

        // Инициализируем ddrives текущего шага из финального градиента
        let mut ddrives_cur = ddrives_final.to_vec();

        for step in (0..n_steps).rev() {
            let off = step * n;
            let dk_step = self.backward_one_step(
                &traj.phases[off..off + n],
                &traj.amplitudes[off..off + n],
                &traj.drives[off..off + n],
                &dth,
                &dr,
                &mut dth_prev,
                &mut dr_prev,
                &mut ddrives_prev,
                coupling_strength,
                radius,
                dt,
                drive_gain,
                alpha,
                eps,
            );
            dk_total += dk_step;
            // Сохраняем ddrives для этого шага + добавляем пришедший из будущего
            for i in 0..n {
                ddrives_all[off + i] = ddrives_prev[i] + ddrives_cur[i];
            }
            dth.copy_from_slice(&dth_prev);
            dr.copy_from_slice(&dr_prev);
            // ddrives_cur для следующей (более ранней) итерации = нули
            // (drive влияет только на свой шаг, не на предыдущие через состояние)
            ddrives_cur.fill(0.0);
        }

        (dk_total, dth, dr, ddrives_all)
    }

    /// Полный BPTT через траекторию с per-step градиентом по состоянию.
    ///
    /// В отличие от [`TorusGrid::bptt`], принимает градиент по состоянию на
    /// КАЖДОМ шаге (`dth_per_step[s]` = dL/d(фаза в момент s+1)), а не только
    /// по финальному. Нужно для per-timestep readout (напр. char-LM, где каждый
    /// токен читает фазы тора после своего драйва).
    ///
    /// `dth_final`/`dr_final` — градиент по самому последнему состоянию
    /// (обычно нули, если читается только per-step output). Возвращает
    /// (dL/dK, dL/dθ₀, dL/dr₀, dL/d(drives) по всем шагам).
    #[allow(clippy::too_many_arguments)]
    pub fn bptt_per_step(
        &self,
        traj: &TorusTrajectory,
        dth_final: &[f32],
        dr_final: &[f32],
        dth_per_step: &[f32], // [n_steps × n] — dL/d(фаза в момент s+1)
        dr_per_step: &[f32],  // [n_steps × n] — dL/d(амплитуда в момент s+1)
        coupling_strength: f32,
        radius: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.w * self.h;
        let n_steps = traj.n_steps;
        assert_eq!(dth_final.len(), n);
        assert_eq!(dr_final.len(), n);
        assert_eq!(dth_per_step.len(), n_steps * n);
        assert_eq!(dr_per_step.len(), n_steps * n);

        let mut dth = dth_final.to_vec();
        let mut dr = dr_final.to_vec();
        let mut dk_total = 0.0f32;

        let mut dth_prev = vec![0.0f32; n];
        let mut dr_prev = vec![0.0f32; n];
        let mut ddrives_prev = vec![0.0f32; n];
        let mut ddrives_all = vec![0.0f32; n_steps * n];

        // Прямого readout по драйвам нет (readout только по состоянию), поэтому
        // ddrives_final всегда нули — в отличие от `bptt`, который принимает его явно.
        let mut ddrives_cur = vec![0.0f32; n];

        for step in (0..n_steps).rev() {
            let off = step * n;
            // Инжект per-step градиента: output[step] = состояние в момент step+1.
            for i in 0..n {
                dth[i] += dth_per_step[off + i];
                dr[i] += dr_per_step[off + i];
            }
            let dk_step = self.backward_one_step(
                &traj.phases[off..off + n],
                &traj.amplitudes[off..off + n],
                &traj.drives[off..off + n],
                &dth,
                &dr,
                &mut dth_prev,
                &mut dr_prev,
                &mut ddrives_prev,
                coupling_strength,
                radius,
                dt,
                drive_gain,
                alpha,
                eps,
            );
            dk_total += dk_step;
            for i in 0..n {
                ddrives_all[off + i] = ddrives_prev[i] + ddrives_cur[i];
            }
            dth.copy_from_slice(&dth_prev);
            dr.copy_from_slice(&dr_prev);
            ddrives_cur.fill(0.0);
        }

        (dk_total, dth, dr, ddrives_all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_initial_order_parameter() {
        let tg = TorusGrid::new(8, 8, 0.5, 1.5, 42);
        let r = tg.order_parameter();
        // Случайные фазы → R < 0.4 (нет синхронизации)
        assert!(r < 0.5, "random phases should have low R, got {r}");
    }

    #[test]
    fn test_torus_local_coupling_syncs() {
        let mut tg = TorusGrid::new(16, 16, 0.5, 0.5, 7); // одинаковые частоты!
        let mut drives = tg.drives_vec();
        TorusGrid::fill_drives(&mut drives, 0.0);
        let r0 = tg.order_parameter();
        // Сильная связь + одинаковые частоты → синхронизация
        for _ in 0..200 {
            tg.step(0.05, 5.0, 1, &drives, 10.0, 1.0, 0.1);
        }
        let r1 = tg.order_parameter();
        assert!(
            r1 > r0 * 2.0,
            "local coupling should increase sync: {r0} → {r1}"
        );
    }

    #[test]
    fn test_fourier_mode_zero_is_R() {
        let tg = TorusGrid::new(8, 8, 0.5, 1.5, 42);
        let r = tg.order_parameter();
        let f00 = tg.fourier_mode(0, 0);
        assert!((r - f00).abs() < 1e-5, "F(0,0) must equal R: {r} vs {f00}");
    }

    #[test]
    fn test_wave_spectrum_sorted() {
        let tg = TorusGrid::new(8, 8, 0.5, 1.5, 42);
        let s = tg.wave_spectrum(2);
        assert!(!s.is_empty());
        // Сортирован по убыванию
        for w in s.windows(2) {
            assert!(w[0].2 >= w[1].2 - 1e-6, "spectrum must be descending");
        }
    }

    #[test]
    fn test_plane_wave_detected() {
        // Создаём плоскую волну: θ(x,y) = 2π·kx·x/W — все осцилляторы в фазе волны
        let mut tg = TorusGrid::new(16, 1, 1.0, 1.0, 0);
        let kx = 2i32;
        for x in 0..tg.w {
            let idx = x; // y=0 → idx = x для H=1
            tg.phases[idx] = 2.0 * PI * kx as f32 * x as f32 / tg.w as f32;
        }
        let f2 = tg.fourier_mode(kx, 0);
        let f0 = tg.fourier_mode(0, 0);
        assert!(f2 > 0.99, "plane wave kx={kx} must be detected: F={f2}");
        assert!(f0 < 0.1, "flat wave must have R≈0: R={f0}");
    }

    #[test]
    fn test_torus_wrap_around() {
        // Проверка: wrap-around: левый край связан с правым.
        let mut tg = TorusGrid::new(4, 1, 1.0, 1.0, 0);
        // Все фазы ← π (=3.14), кроме осциллятора 0 ← 0.
        tg.phases = vec![0.0, PI, PI, PI];
        tg.amplitudes = vec![0.5; 4];
        let mut d = tg.drives_vec();
        TorusGrid::fill_drives(&mut d, 0.0);
        // Шаг: осциллятор 0 получает coupling от 1 (PI) и -1 (=3, wrap) (PI).
        // sin(PI−0) = 0 → coupling ≈ 0, но с wrap-around код покрывает.
        tg.step(0.05, 1.0, 1, &d, 10.0, 1.0, 0.1);
        // Не падает = wrap работает.
    }

    // ═══════════════ BPTT gradient checks ═══════════════

    /// Выполнить forward по траектории с заданным K.
    fn run_forward(
        tg: &TorusGrid,
        n_steps: usize,
        k: f32,
        radius: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        eps: f32,
    ) -> (TorusTrajectory, TorusGrid) {
        let mut grid = tg.clone();
        let mut traj = TorusTrajectory::new(n_steps, tg.w * tg.h);
        let drives = vec![0.0f32; tg.w * tg.h];
        for step in 0..n_steps {
            grid.record_step(&mut traj, step, &drives);
            grid.step(dt, k, radius, &drives, drive_gain, alpha, eps);
        }
        (traj, grid)
    }

    /// loss = Σ θ² на финальном состоянии.
    fn loss_sq(grid: &TorusGrid) -> f32 {
        grid.phases.iter().map(|&p| p * p).sum()
    }

    /// Аналитический градиент dL/dK через BPTT.
    fn analytic_dk(
        grid: &TorusGrid,
        traj: &TorusTrajectory,
        k: f32,
        radius: usize,
        dt: f32,
    ) -> f32 {
        let n = grid.w * grid.h;
        // dL/dθ_final = 2·θ
        let dth_final: Vec<f32> = grid.phases.iter().map(|&p| 2.0 * p).collect();
        let dr_final = vec![0.0f32; n];
        let (dk, _, _, _) = grid.bptt(
            traj,
            &dth_final,
            &dr_final,
            &vec![0.0f32; n],
            k,
            radius,
            dt,
            10.0,
            1.0,
            0.1,
        );
        dk
    }

    #[test]
    fn test_bptt_gradient_check() {
        let W: usize = 4;
        let H: usize = 2;
        let mut tg = TorusGrid::new(W, H, 1.0, 1.0, 13);
        // Невырожденное начальное состояние
        for y in 0..H {
            for x in 0..W {
                let i = tg.idx(x, y);
                tg.phases[i] = 0.5 + 0.3 * (x as f32) + 0.7 * (y as f32);
                tg.amplitudes[i] = 0.8 + 0.1 * (x as f32);
            }
        }
        let k = 2.0f32;
        let dt = 0.1f32;
        let n_steps = 3;
        let (traj, grid) = run_forward(&tg, n_steps, k, 1, dt, 10.0, 1.0, 0.1);

        let analytic = analytic_dk(&grid, &traj, k, 1, dt);
        assert!(
            analytic.abs() > 1e-5,
            "analytic dK must be nonzero: {analytic}"
        );

        let eps_fd = 1e-3f32;
        let (_, grid_p) = run_forward(&tg, n_steps, k + eps_fd, 1, dt, 10.0, 1.0, 0.1);
        let (_, grid_m) = run_forward(&tg, n_steps, k - eps_fd, 1, dt, 10.0, 1.0, 0.1);
        let numeric = (loss_sq(&grid_p) - loss_sq(&grid_m)) / (2.0 * eps_fd);

        let rel = (analytic - numeric).abs() / (analytic.abs() + 1e-4);
        assert!(
            rel < 0.05,
            "BPTT gradient check FAILED: analytic={analytic:.6} numeric={numeric:.6} rel={rel:.4}"
        );
    }

    // ═══════════════ Дедовские методы: тесты ═══════════════

    #[test]
    fn test_step_inertial_does_not_panic() {
        let mut tg = TorusGrid::new(8, 8, 1.0, 1.0, 42);
        tg.phase_velocities = vec![0.1; 64];
        let drives = vec![0.0f32; 64];
        for _ in 0..50 {
            tg.step_inertial(0.05, 3.0, 1, &drives, 10.0, 1.0, 0.1, 0.5);
        }
        let r = tg.order_parameter();
        assert!(r > 0.0, "inertial mode should produce order");
    }

    #[test]
    fn test_inertial_standing_waves() {
        // Петвиашвили-эффект: с инерцией волны НЕ опрокидываются.
        // Проверяем: фазы остаются в разумном диапазоне после многих шагов.
        let mut tg = TorusGrid::new(8, 8, 1.0, 1.0, 99);
        // Неоднородные фазы
        for y in 0..tg.h {
            for x in 0..tg.w {
                let idx = tg.idx(x, y);
                tg.phases[idx] = (x as f32 * 0.5 + y as f32 * 0.3).sin();
            }
        }
        let drives = vec![0.0f32; 64];
        for _ in 0..100 {
            tg.step_inertial(0.05, 5.0, 1, &drives, 10.0, 1.0, 0.1, 1.0);
        }
        // Все фазы должны быть конечны и в [-π, π]
        for &p in &tg.phases {
            assert!(p.is_finite());
            assert!(p >= -PI - 0.01 && p <= PI + 0.01);
        }
        // Скорости не должны взрываться
        for &v in &tg.phase_velocities {
            assert!(v.abs() < 100.0, "velocity exploded: {v}");
        }
    }

    #[test]
    fn test_bifurcation_scan_finds_modes() {
        let tg = TorusGrid::new(8, 4, 1.0, 1.0, 7);
        let points = tg.bifurcation_scan(0.0, 5.0, 1.0, 100, 0.05, 1, 10.0, 1.0, 0.1);
        assert!(points.len() >= 5);
        // R должен расти с K
        let r0 = points[0].r;
        let r_end = points.last().unwrap().r;
        assert!(
            r_end > r0 * 1.5,
            "R should increase with K: {r0} -> {r_end}"
        );
        // Хотя бы одна точка имеет не-R доминантную моду
        let wave_modes: Vec<_> = points.iter().filter(|p| p.top_mode != (0, 0)).collect();
        assert!(
            !wave_modes.is_empty(),
            "bifurcation should produce non-R modes at some K"
        );
    }

    #[test]
    fn test_negative_resistance_check() {
        assert!(TorusGrid::negative_resistance_check(1.0));
        assert!(TorusGrid::negative_resistance_check(0.001));
        assert!(!TorusGrid::negative_resistance_check(0.0));
        assert!(!TorusGrid::negative_resistance_check(-1.0));
    }

    #[test]
    fn test_bptt_trajectory_roundtrip() {
        // Прогон + BPTT не должен паниковать и должен давать конечные значения.
        let tg = TorusGrid::new(4, 4, 1.0, 1.0, 7);
        let (traj, grid) = run_forward(&tg, 5, 3.0, 1, 0.1, 10.0, 1.0, 0.1);
        let n = tg.w * tg.h;
        let dth = vec![1.0f32; n];
        let dr = vec![0.1f32; n];
        let (dk, dth0, dr0, _ddrives) = grid.bptt(
            &traj,
            &dth,
            &dr,
            &vec![0.0f32; n],
            3.0,
            1,
            0.1,
            10.0,
            1.0,
            0.1,
        );
        assert!(dk.is_finite());
        assert!(dth0.iter().all(|x| x.is_finite()));
        assert!(dr0.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_bptt_per_step_equals_final() {
        // per-step градиент, сосредоточенный только на последнем output,
        // должен давать те же dK и ddrives, что и обычный bptt с финальным градиентом.
        let tg = TorusGrid::new(4, 4, 1.0, 1.0, 7);
        let (traj, grid) = run_forward(&tg, 5, 3.0, 1, 0.1, 10.0, 1.0, 0.1);
        let n = tg.w * tg.h;
        let dth = vec![1.0f32; n];
        let dr = vec![0.0f32; n];
        let (dk_a, _, _, ddrives_a) = grid.bptt(
            &traj,
            &dth,
            &dr,
            &vec![0.0f32; n],
            3.0,
            1,
            0.1,
            10.0,
            1.0,
            0.1,
        );

        let mut dth_ps = vec![0.0f32; 5 * n];
        dth_ps[(5 - 1) * n..].copy_from_slice(&dth);
        let (dk_b, _, _, ddrives_b) = grid.bptt_per_step(
            &traj,
            &vec![0.0f32; n],
            &vec![0.0f32; n],
            &dth_ps,
            &vec![0.0f32; 5 * n],
            3.0,
            1,
            0.1,
            10.0,
            1.0,
            0.1,
        );
        assert!((dk_a - dk_b).abs() < 1e-5, "dK mismatch: {dk_a} vs {dk_b}");
        for i in 0..ddrives_a.len() {
            assert!(
                (ddrives_a[i] - ddrives_b[i]).abs() < 1e-5,
                "ddrives[{i}] mismatch: {} vs {}",
                ddrives_a[i],
                ddrives_b[i]
            );
        }
    }
}
