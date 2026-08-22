// ═══════════════════════════════════════════════════════════
//  Markov kinetic-scheme gates (NeuroML2 `gateKS`)
// ═══════════════════════════════════════════════════════════
//  В отличие от HH-гейтов (одна переменная q с inf/tau), кинетическая схема —
//  это N состояний с направленными переходами, интегрируемых как master
//  equation: d p_i/dt = Σ_j (p_j·k_{j→i}) − p_i·Σ_j k_{i→j}. Открытая доля —
//  сумма занятостей открытых состояний, возведённая в `instances`.
//
//  Интегратор — точная (при замороженном v) матричная экспонента exp(Q·dt),
//  реализованная scaling-and-squaring + ряд Тейлора степени 8 (Higham). Для
//  2-состояний это совпадает с аналитикой `q_inf + (q−q_inf)·exp(−(α+β)·dt)`.

use super::spec::{ClosedForm, GateKinetics, RateForm, TauForm, VarForm};

/// Максимальное число состояний одного кинетического гейта. Плотная матричная
/// экспонента — O(N³) на шаг; 16 покрывает любые реальные схемы (NaV 8,
/// CaV 12, HCN 4 состояния) с запасом.
pub const MAX_MARKOV_STATES: usize = 16;

/// Один направленный переход в марковской схеме.
///
/// * [`MarkovRate::Directed`] — `forwardTransition`/`reverseTransition`: сырая
///   per-time скорость `r(v)` (в ms⁻¹);
/// * [`MarkovRate::TauInfForward`]/[`MarkovRate::TauInfReverse`] — ноги
///   `tauInfTransition`: forward `inf/tau`, reverse `(1−inf)/tau`.
#[derive(Debug, Clone, PartialEq)]
pub enum MarkovRate {
    /// `r(v)` — скорость перехода в данном направлении.
    Directed { rate: ClosedForm },
    /// Forward-нога `tauInfTransition`: `steady_state(v) / tau`.
    TauInfForward { steady_state: VarForm, tau: TauForm },
    /// Reverse-нога `tauInfTransition`: `(1 − steady_state(v)) / tau`.
    TauInfReverse { steady_state: VarForm, tau: TauForm },
}

/// Скорость (ms⁻¹) перехода `from → to` при напряжении `v`.
#[inline(always)]
pub fn markov_rate_eval(rate: &MarkovRate, v: f32) -> f32 {
    match rate {
        MarkovRate::Directed { rate } => rate.eval(v),
        MarkovRate::TauInfForward { steady_state, tau } => steady_state.eval(v) / tau.eval(v),
        MarkovRate::TauInfReverse { steady_state, tau } => {
            (1.0 - steady_state.eval(v)) / tau.eval(v)
        }
    }
}

/// Направленное ребро `from → to` (индексы состояний) со скоростью перехода.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovTransition {
    pub from: usize,
    pub to: usize,
    pub rate: MarkovRate,
}

/// Кинетическая схема (NeuroML2 `gateKS`): состояния (open/closed) + набор
/// направленных переходов, интегрируемых как master equation.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovGateSpec {
    /// Вклад состояния `i` в открытую долю (вес). Для обычных KS-гейтов это
    /// 0 (`closedState`) / 1 (`openState`); для `gateFractional` — сумма
    /// `fractionalConductance` субгейтов, открытых в этом состоянии.
    pub open: Vec<f32>,
    /// Направленные переходы (индексы состояний резолвятся при парсинге).
    pub transitions: Vec<MarkovTransition>,
    /// Экспонента `instances`: `fcond = open_fraction^power`.
    pub power: u32,
}

impl MarkovGateSpec {
    #[inline(always)]
    pub fn n_states(&self) -> usize {
        self.open.len()
    }

    /// Validate this gate specification for the master-equation integrator.
    ///
    /// Returns `Ok(())` if the spec is well-formed, otherwise a descriptive
    /// error. Guards against the indexing panics a malformed **public** spec
    /// would otherwise trigger inside [`markov_step`]/`markov_rate_matrix`:
    /// out-of-range transition endpoints, empty/non-finite state weights, and
    /// non-finite or negative rates/`tau`.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_states();
        if n < 2 {
            return Err(format!("gate needs ≥2 states, got {n}"));
        }
        if n > MAX_MARKOV_STATES {
            return Err(format!("gate has {n} states (max {MAX_MARKOV_STATES})"));
        }
        for (i, &w) in self.open.iter().enumerate() {
            if !w.is_finite() {
                return Err(format!("open weight {i} is non-finite: {w}"));
            }
        }
        for t in &self.transitions {
            if t.from >= n || t.to >= n {
                return Err(format!(
                    "transition endpoint out of range: {} -> {} (n_states={n})",
                    t.from, t.to
                ));
            }
            if t.from == t.to {
                return Err(format!(
                    "self-transition on state {} is not allowed",
                    t.from
                ));
            }
            validate_markov_rate(&t.rate)?;
        }
        Ok(())
    }
}

/// Validate the numeric parameters of a single closed-form rate.
fn validate_closed_form(cf: ClosedForm, what: &str) -> Result<(), String> {
    let (rate, midpoint, scale) = match cf {
        ClosedForm::Exp {
            rate,
            midpoint,
            scale,
        }
        | ClosedForm::Sigmoid {
            rate,
            midpoint,
            scale,
        }
        | ClosedForm::ExpLinear {
            rate,
            midpoint,
            scale,
        } => (rate, midpoint, scale),
    };
    if !rate.is_finite() || !midpoint.is_finite() || !scale.is_finite() {
        return Err(format!(
            "{what}: non-finite parameters (rate={rate}, midpoint={midpoint}, scale={scale})"
        ));
    }
    if rate < 0.0 {
        return Err(format!("{what}: negative rate {rate}"));
    }
    if scale == 0.0 {
        return Err(format!("{what}: zero scale (division by zero)"));
    }
    Ok(())
}

/// Validate a transition rate: closed forms must be finite/non-negative, and
/// `tauInfTransition` `tau` must be finite and strictly positive (a divisor).
fn validate_markov_rate(rate: &MarkovRate) -> Result<(), String> {
    match rate {
        MarkovRate::Directed { rate } => validate_closed_form(*rate, "directed transition rate"),
        MarkovRate::TauInfForward { steady_state, tau }
        | MarkovRate::TauInfReverse { steady_state, tau } => {
            match steady_state {
                VarForm::Closed(cf) => validate_closed_form(*cf, "tauInf steady_state")?,
                // Кастомное LEMS-выражение — валидируем конечность на выборке V.
                VarForm::Expr(e) => {
                    for v in [-100.0, -50.0, 0.0, 50.0] {
                        let x = e.eval(v);
                        if !x.is_finite() {
                            return Err(format!(
                                "tauInf steady_state expression non-finite at V={v} mV: {x}"
                            ));
                        }
                    }
                }
            }
            match tau {
                TauForm::Fixed(t) => {
                    if !t.is_finite() || *t <= 0.0 {
                        return Err(format!("tauInf tau must be finite and > 0, got {t}"));
                    }
                }
                TauForm::Expr(e) => {
                    for v in [-100.0, -50.0, 0.0, 50.0] {
                        let t = e.eval(v);
                        if !t.is_finite() || t <= 0.0 {
                            return Err(format!(
                                "tauInf tau expression non-finite/≤0 at V={v} mV: {t}"
                            ));
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

/// Опустить HH-гейт ([`GateKinetics`] + экспонента) до эквивалентной
/// 2-состоянийной марковской схемы: состояние 0 = closed, 1 = open;
/// forward = α(v), reverse = β(v). Позволяет `MarkovPopulation` гонять
/// произвольную смесь HH- и KS-каналов одним (скалярным) интегратором.
pub fn gate_kinetics_to_markov(kinetics: GateKinetics, power: u32) -> MarkovGateSpec {
    let (fwd, rev) = match kinetics {
        GateKinetics::Rates { forward, reverse } => (
            MarkovRate::Directed {
                rate: match forward {
                    RateForm::Closed(cf) => cf,
                    // Expr rates (Ca-dependent) never lower to Markov — the
                    // markov path runs on gateFractional/ks channels only.
                    RateForm::Expr(_) => unreachable!("custom rate has no Markov lowering"),
                },
            },
            MarkovRate::Directed {
                rate: match reverse {
                    RateForm::Closed(cf) => cf,
                    RateForm::Expr(_) => unreachable!("custom rate has no Markov lowering"),
                },
            },
        ),
        GateKinetics::TauInf { steady_state, tau } => (
            MarkovRate::TauInfForward {
                steady_state: steady_state.clone(),
                tau: tau.clone(),
            },
            MarkovRate::TauInfReverse { steady_state, tau },
        ),
        // Algebraic gate — no ODE, cannot lower to a 2-state Markov scheme.
        // `resolve_markov_channel` only runs on gateFractional/ks channels,
        // which never contain `gateHHInstantaneous`.
        GateKinetics::Instantaneous { .. } => {
            unreachable!("gateHHInstantaneous has no kinetics to lower to Markov")
        }
    };
    MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: fwd,
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: rev,
            },
        ],
        power,
    }
}

/// Один субгейт `gateFractional` (NeuroML2 `subGate`): независимый tau-inf
/// процесс `dq/dt = (inf − q)/tau`, взвешенный `fractional_conductance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractionalSubgate {
    /// `inf(V)` — steady-state вола открытия (NeuroML `steadyState`).
    pub steady_state: ClosedForm,
    /// Фиксированный time course (ms) — NeuroML `timeCourse`.
    pub tau: f32,
    /// Вес субгейта в суммарной проводимости (`fractionalConductance`).
    pub fractional_conductance: f32,
}

/// Гейт `gateFractional` (NeuroML2): `q = Σ frac_i·q_i`, `fcond = q^instances`.
/// Каждый субгейт — независимая 2-состоянийная tau-inf переменная; итоговая
/// открытая доля — взвешенная сумма их занятостей.
#[derive(Debug, Clone, PartialEq)]
pub struct FractionalGateSpec {
    /// Субгейты в порядке объявления.
    pub subgates: Vec<FractionalSubgate>,
    /// Экспонента `instances`: `fcond = (Σ frac_i·q_i)^power`.
    pub power: u32,
}

/// Опустить `gateFractional` до эквивалентной марковской схемы над
/// **декартовым произведением** состояний субгейтов. Каждый субгейт —
/// независимый 2-состоянийный tau-inf процесс, поэтому совместное
/// распределение живёт на 2^K состояниях; вес состояния = сумма
/// `fractional_conductance` открытых в нём субгейтов. Матрица переходов —
/// кронекеровская сумма (переходы переворачивают ровно один бит), так что
/// master equation факторизуется и бит-в-бит воспроизводит K независимых
/// субгейтов.
pub fn fractional_gate_to_markov(gate: &FractionalGateSpec) -> MarkovGateSpec {
    let k = gate.subgates.len();
    assert!(
        k <= 4,
        "gateFractional: {k} subgates → {}^{k} states > {MAX_MARKOV_STATES}",
        2
    );
    let n = 1usize << k;
    let mut open = vec![0.0f32; n];
    for s in 0..n {
        let mut w = 0.0f32;
        for (i, sg) in gate.subgates.iter().enumerate() {
            if s & (1 << i) != 0 {
                w += sg.fractional_conductance;
            }
        }
        open[s] = w;
    }
    let mut transitions = Vec::new();
    for (i, sg) in gate.subgates.iter().enumerate() {
        for s in 0..n {
            if s & (1 << i) == 0 {
                let t = s | (1 << i);
                transitions.push(MarkovTransition {
                    from: s,
                    to: t,
                    rate: MarkovRate::TauInfForward {
                        steady_state: VarForm::Closed(sg.steady_state),
                        tau: TauForm::Fixed(sg.tau),
                    },
                });
                transitions.push(MarkovTransition {
                    from: t,
                    to: s,
                    rate: MarkovRate::TauInfReverse {
                        steady_state: VarForm::Closed(sg.steady_state),
                        tau: TauForm::Fixed(sg.tau),
                    },
                });
            }
        }
    }
    MarkovGateSpec {
        open,
        transitions,
        power: gate.power,
    }
}

/// Открытая доля гейта: взвешенная сумма занятостей состояний (`Σ w_i·p_i`).
/// Для обычных KS-гейтов веса 0/1 → это сумма занятостей открытых состояний;
/// для `gateFractional` — взвешенная сумма субгейтовых занятостей.
#[inline(always)]
pub fn markov_open_fraction(gate: &MarkovGateSpec, p: &[f32]) -> f32 {
    gate.open.iter().zip(p).map(|(w, &pi)| w * pi).sum()
}

/// Матрица скоростей переходов `Q` (n×n, row-major) при напряжении `v`,
/// масштабированная на `q10`: `Q[i][j] = k_{j→i}` для i≠j, а диагональ
/// `Q[i][i] = −Σ_{j≠i} k_{i→j}`. Колонки Q в сумме дают ноль (сохранение Σp).
fn markov_rate_matrix(gate: &MarkovGateSpec, v: f32, q10: f32) -> Vec<f32> {
    let n = gate.n_states();
    let mut q = vec![0.0f32; n * n];
    for t in &gate.transitions {
        let rate = markov_rate_eval(&t.rate, v) * q10;
        q[t.to * n + t.from] += rate; // приток в `to` из `from`
        q[t.from * n + t.from] -= rate; // отток из `from`
    }
    q
}

/// Умножение плотных матриц n×n (row-major).
fn mat_mul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik != 0.0 {
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
    }
    c
}

/// Единичная матрица n×n (row-major).
fn identity(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Решение плотной линейной системы A·x = b (n×n) методом Гаусса с частичным
/// выбором ведущего элемента. Мутирует `a` и `b`, возвращает решение в `b`.
fn solve_linear(a: &mut [f32], b: &mut [f32], n: usize) {
    for col in 0..n {
        // Частичный выбор ведущего элемента (устойчивость на жёстких матрицах Q).
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let cand = a[r * n + col].abs();
            if cand > best {
                best = cand;
                piv = r;
            }
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        if d == 0.0 {
            continue; // сингулярно (недостижимое состояние) — оставляем как есть
        }
        for r in (col + 1)..n {
            let factor = a[r * n + col] / d;
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= factor * a[col * n + j];
            }
            b[r] -= factor * b[col];
        }
    }
    // Обратная подстановка.
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        let d = a[i * n + i];
        b[i] = if d != 0.0 { s / d } else { 0.0 };
    }
}

/// Стационарное распределение гейта при напряжении `v`: единственный вектор
/// `p` с Σp = 1 и `Q·p = 0` (решается замена последней строки Q на Σp=1).
/// Используется для инициализации состояний в покое.
pub fn markov_steady_state(gate: &MarkovGateSpec, v: f32) -> Vec<f32> {
    let n = gate.n_states();
    let q = markov_rate_matrix(gate, v, 1.0); // q10 не влияет на стационар
    let mut m = q;
    for j in 0..n {
        m[(n - 1) * n + j] = 1.0; // замена последней строки на Σp = 1
    }
    let mut rhs = vec![0.0f32; n];
    rhs[n - 1] = 1.0;
    solve_linear(&mut m, &mut rhs, n);
    rhs
}

/// Продвинуть занятость `p` (Σp = 1) на один шаг `dt` при замороженном `v`:
/// `p ← exp(Q·dt)·p` (scaling-and-squaring, ряд Тейлора степени 8).
/// `q10` — температурное масштабирование скоростей (как в HH-движке).
pub fn markov_step(gate: &MarkovGateSpec, p: &mut [f32], v: f32, dt: f32, q10: f32) {
    const TAYLOR_DEGREE: usize = 8;
    const SCALE_TARGET: f32 = 0.5;

    let n = gate.n_states();
    assert_eq!(
        p.len(),
        n,
        "markov_step: occupancy length {} != gate state count {n}",
        p.len()
    );
    let q = markov_rate_matrix(gate, v, q10);
    let mut a: Vec<f32> = q.iter().map(|x| x * dt).collect(); // A = Q·dt

    // Масштабирование: делим пополам, пока max-норма строки ≤ 0.5. Без
    // фиксированного капа: для конечных входов норма убывает вдвое за шаг,
    // так что s ≈ log2(norm/0.5) (≤ ~127 для f32). Прежний кап
    // (MAX_SQUARINGS=12) оставлял огромную матрицу, чьи 8-е степени Тейлора
    // переполнялись до NaN/Inf.
    let mut s = 0u32;
    loop {
        let mut norm = 0.0f32;
        for i in 0..n {
            let row_sum: f32 = (0..n).map(|j| a[i * n + j].abs()).sum();
            norm = norm.max(row_sum);
        }
        if norm <= SCALE_TARGET || norm == 0.0 || !norm.is_finite() {
            break;
        }
        for x in &mut a {
            *x *= 0.5;
        }
        s += 1;
    }

    // Ряд Тейлора: E = Σ_{k=0}^{8} A^k / k!.
    let mut e = identity(n);
    let mut term = identity(n);
    let mut factorial = 1.0f32;
    for k in 1..=TAYLOR_DEGREE {
        term = mat_mul(&term, &a, n);
        factorial *= k as f32;
        for idx in 0..(n * n) {
            e[idx] += term[idx] / factorial;
        }
    }
    // Квадратирование: E ← E^(2^s).
    for _ in 0..s {
        e = mat_mul(&e, &e, n);
    }

    // p ← E·p.
    let p_old = p.to_vec();
    for i in 0..n {
        p[i] = (0..n).map(|j| e[i * n + j] * p_old[j]).sum();
    }
    // exp(Q·dt) — стохастическая матрица; клампим ничтожные отрицательные
    // значения и перенормируем Σp=1 (f32-дрейф). При не-конечном результате
    // (патологические скорости, которые `validate()` отсекает на входе)
    // возвращаем стационарное распределение — валидный вероятностный вектор.
    let sum: f32 = p.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for pi in p.iter_mut() {
            *pi = (*pi / sum).max(0.0);
        }
    } else {
        p.copy_from_slice(&markov_steady_state(gate, v));
    }
}
