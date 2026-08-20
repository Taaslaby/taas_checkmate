#![allow(non_snake_case)]

use super::*;
use crate::math::fast_exp::fast_exp;

/// Scalar exponential-Euler HH step over arbitrary channels.
/// Returns `true` if a spike (reverse zero-crossing through 0 mV) occurred.
#[allow(clippy::too_many_arguments)]
fn hh_step_scalar(
    v: &mut f32,
    gates: &mut [f32],
    i_ext: f32,
    channels: &[ChannelSpec],
    g_leak: f32,
    e_leak: f32,
    c_m: f32,
    dt: f32,
) -> bool {
    let v_old = *v;
    let mut i_ion = 0.0f32;
    let mut gi = 0usize;
    for ch in channels {
        let mut g_ch = ch.g_max;
        for gate in &ch.gates {
            let (a, b) = gate.kinetics.alpha_beta(*v);
            let sum = a + b;
            let m_inf = if sum > 1e-12 { a / sum } else { 0.5 };
            let decay = fast_exp(-dt * sum.max(1e-12));
            let gv = &mut gates[gi];
            *gv = m_inf + (*gv - m_inf) * decay;
            g_ch *= gv.powi(gate.power as i32);
            gi += 1;
        }
        i_ion += g_ch * (*v - ch.E_rev);
    }
    let i_leak = g_leak * (*v - e_leak);
    let dv = (i_ext - i_ion - i_leak) / c_m;
    let v_new = *v + dv * dt;
    *v = v_new;
    v_old >= 0.0 && v_new < 0.0
}

/// Standard squid axon: must NOT fire spontaneously with zero input.
#[test]
fn standard_channels_quiescent_without_input() {
    let channels = standard_channels();
    let dt = 0.05;
    let mut v = -65.0f32;
    let mut gates = steady_state_gates(v, &channels);
    let mut spikes = 0u32;
    for _ in 0..4000 {
        // 200 ms
        if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
            spikes += 1;
        }
    }
    assert_eq!(spikes, 0, "standard HH should not self-spike, got {spikes}");
    // Resting potential of the squid axon is ≈ −65 mV (NOT the leak
    // reversal −54.387); the membrane must sit near rest, not drift.
    assert!(
        v > -75.0 && v < -45.0,
        "v should stay near resting potential, got {v}"
    );
}

/// Pacemaking (CaV-T): must fire spontaneously with zero input.
#[test]
fn pacemaking_fires_spontaneously() {
    let channels = pacemaking_channels();
    let dt = 0.05;
    let mut v = -65.0f32;
    let mut gates = steady_state_gates(v, &channels);
    let mut spikes = 0u32;
    for _ in 0..4000 {
        // 200 ms
        if hh_step_scalar(&mut v, &mut gates, 0.0, &channels, 0.3, -54.387, 1.0, dt) {
            spikes += 1;
        }
        assert!(v.is_finite() && v > -120.0 && v < 80.0, "v runaway: {v}");
    }
    assert!(
        spikes >= 3,
        "pacemaking HH should fire ≥3 spikes in 200 ms, got {spikes}"
    );
}

/// Gate ordering: 4 gates for pacemaking (m, h, n, c), steady states in [0,1].
#[test]
fn gate_order_and_steady_state() {
    let ch = pacemaking_channels();
    assert_eq!(ch.len(), 3);
    let n_gates: usize = ch.iter().map(|c| c.gates.len()).sum();
    assert_eq!(n_gates, 4);
    let g = steady_state_gates(-65.0, &ch);
    assert_eq!(g.len(), 4);
    for (i, &x) in g.iter().enumerate() {
        assert!((0.0..=1.0).contains(&x), "gate {i} = {x} out of [0,1]");
    }
    // CaV-T is the last gate (index 3)
    assert_eq!(ch[2].name, "CaT");
    assert_eq!(ch[2].gates[0].label, "c");
    assert_eq!(ch[2].E_rev, 120.0);
}

/// Rate-function sanity: steady-state m_inf saturates with depolarization.
#[test]
fn na_m_inf_saturates() {
    let channels = standard_channels();
    let g_hyper = steady_state_gates(-80.0, &channels);
    let g_depol = steady_state_gates(0.0, &channels);
    assert!(
        g_depol[0] > g_hyper[0],
        "m_inf should rise with depolarization"
    );
}

/// Замкнутые формы (`ClosedForm`) должны бит-в-бит воспроизводить канонические
/// скалярные rate-функции (`na_m_alpha` и т.д.) — иначе data-driven движок
/// разойдётся с NEURON-валидированными функциями.
#[test]
fn closed_forms_match_canonical_rate_fns_bit_exact() {
    let vs = [
        -120.0f32, -100.0, -80.0, -65.0, -55.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0,
    ];
    for &v in &vs {
        let cases = [
            (
                "na_m_alpha",
                ClosedForm::ExpLinear {
                    rate: 0.1,
                    midpoint: 40.0,
                    scale: 10.0,
                }
                .eval(v),
                na_m_alpha(v),
            ),
            (
                "na_m_beta",
                ClosedForm::Exp {
                    rate: 4.0,
                    midpoint: 65.0,
                    scale: 18.0,
                }
                .eval(v),
                na_m_beta(v),
            ),
            (
                "na_h_alpha",
                ClosedForm::Exp {
                    rate: 0.07,
                    midpoint: 65.0,
                    scale: 20.0,
                }
                .eval(v),
                na_h_alpha(v),
            ),
            (
                "na_h_beta",
                ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 35.0,
                    scale: 10.0,
                }
                .eval(v),
                na_h_beta(v),
            ),
            (
                "k_n_alpha",
                ClosedForm::ExpLinear {
                    rate: 0.01,
                    midpoint: 55.0,
                    scale: 10.0,
                }
                .eval(v),
                k_n_alpha(v),
            ),
            (
                "k_n_beta",
                ClosedForm::Exp {
                    rate: 0.125,
                    midpoint: 65.0,
                    scale: 80.0,
                }
                .eval(v),
                k_n_beta(v),
            ),
            (
                "cat_c_alpha",
                GateKinetics::TauInf {
                    steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                        rate: 1.0,
                        midpoint: 56.0,
                        scale: 6.2,
                    }),
                    tau: TauForm::Fixed(0.6),
                }
                .alpha_beta(v)
                .0,
                cat_c_alpha(v),
            ),
            (
                "cat_c_beta",
                GateKinetics::TauInf {
                    steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                        rate: 1.0,
                        midpoint: 56.0,
                        scale: 6.2,
                    }),
                    tau: TauForm::Fixed(0.6),
                }
                .alpha_beta(v)
                .1,
                cat_c_beta(v),
            ),
        ];
        for (name, got, want) in cases {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name} @ {v}: data-driven {got} != canonical {want}"
            );
        }
    }
}

/// Двухсостоянийная марковская схема (closed ⇌ open с постоянными
/// скоростями α/β) обязана совпадать с аналитическим решением master
/// equation: `q(t+dt) = q_inf + (q−q_inf)·exp(−(α+β)·dt)`. Это якорь,
/// доказывающий, что матричная экспонента реализована корректно.
#[test]
fn markov_two_state_matches_analytic_exponential() {
    // Постоянные скорости: Sigmoid с rate=α/β и «плоской» V-зависимостью
    // (midpoint=0, scale → ∞ не годится для f32; берём огромный scale,
    // при v=−65 аргумент ≈ 0 → inf ≈ rate/2). Проще: закрыть форму в
    // почти-константу через очень пологий sigmoid.
    let alpha = 0.3f32; // per ms
    let beta = 0.1f32;
    let v = -65.0f32;
    let dt = 0.05f32;

    let gate = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: const_form(alpha),
                },
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: MarkovRate::Directed {
                    rate: const_form(beta),
                },
            },
        ],
        power: 1,
    };

    // Старт из closed (p=[1,0]).
    let mut p = vec![1.0f32, 0.0];
    let q0 = markov_open_fraction(&gate, &p);
    assert_eq!(q0, 0.0);

    // Аналитический эталон: точная (std) экспонента. Скорости берём теми же
    // формами, что и движок (markov_rate_eval), чтобы сравнивались только
    // интеграторы, а не кодировка скоростей.
    let (a, b) = (
        markov_rate_eval(&gate.transitions[0].rate, v),
        markov_rate_eval(&gate.transitions[1].rate, v),
    );
    let q_inf = a / (a + b);
    let expected = q_inf + (q0 - q_inf) * (-(a + b) * dt).exp();

    markov_step(&gate, &mut p, v, dt, 1.0);
    let q1 = markov_open_fraction(&gate, &p);

    // Матричная экспонента vs точная аналитика — до ~1e-7 (ряд Тейлора
    // степени 8 при max-норме ≤ 0.5).
    assert!(
        (q1 - expected).abs() < 1e-6,
        "2-state Markov: q1={q1} != analytic {expected}"
    );
    // Сохранение вероятности.
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Σp = {sum} != 1");
}

/// Почти-постоянная замкнутая форма: сигмоид с очень большим scale — при
/// физиологических v аргумент ≈ 0, поэтому eval(v) ≈ rate/2. Хелпер для
/// тестов со скоростями, не зависящими от напряжения.
fn const_form(rate: f32) -> ClosedForm {
    // rate/2 при scale → ∞; хотим rate → задаём rate' = 2·rate.
    ClosedForm::Sigmoid {
        rate: 2.0 * rate,
        midpoint: 0.0,
        scale: 1.0e9,
    }
}

/// Стационарное распределение двухсостоянийной схемы обязано сходиться к
/// `q_inf = α/(α+β)` и сохранять Σp = 1.
#[test]
fn markov_steady_state_matches_q_inf() {
    let alpha = 0.3f32;
    let beta = 0.1f32;
    let gate = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: const_form(alpha),
                },
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: MarkovRate::Directed {
                    rate: const_form(beta),
                },
            },
        ],
        power: 1,
    };
    let p = markov_steady_state(&gate, -65.0);
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Σp = {sum} != 1");
    // Стационарная открытая доля через реальные скорости движка (а не α/β
    // «сырые»), чтобы сравнение не зависело от кодировки const_form.
    let a = markov_rate_eval(&gate.transitions[0].rate, -65.0);
    let b = markov_rate_eval(&gate.transitions[1].rate, -65.0);
    let q_inf = a / (a + b);
    assert!(
        (p[1] - q_inf).abs() < 1e-5,
        "open occupancy {} != q_inf {q_inf}",
        p[1]
    );
}

/// HH-гейт, опущенный до 2-состоянийного Маркова, должен давать ту же
/// открытую долю, что и канонический экспоненциальный Эйлер HH-движка.
#[test]
fn gate_kinetics_lowered_to_markov_matches_hh() {
    let kinetics = standard_channels()[0].gates[0].kinetics.clone(); // NaV-m
    let v = -40.0f32;
    let dt = 0.05f32;
    let (a, b) = kinetics.alpha_beta(v);
    let q_inf = a / (a + b);

    // HH-путь: q = q_inf + (q0 − q_inf)·exp(−(a+b)·dt).
    let q0 = 0.05f32;
    let hh = q_inf + (q0 - q_inf) * fast_exp(-(a + b) * dt);

    // Markov-путь: опустить gate и шагнуть из occupancy [1−q0, q0].
    let gate = gate_kinetics_to_markov(kinetics, 1);
    let mut p = vec![1.0 - q0, q0];
    markov_step(&gate, &mut p, v, dt, 1.0);
    let mkv = markov_open_fraction(&gate, &p);

    assert!(
        (mkv - hh).abs() < 1e-5,
        "Markov-lowered gate {mkv} != HH {hh}"
    );
}

/// Трёхсостоянийная обратимая цепь C0 ⇌ C1(open) ⇌ C2 — якорь на N>2
/// состояний: стационар обязан удовлетворять детальному балансу
/// (`p1/p0 = a/b`, `p2/p1 = c/d`), а шаг — сохранять вероятность и
/// сходиться к нему (матричная экспонента уже не 2-состоянийная аналитика).
#[test]
fn markov_three_state_chain_steady_state_and_conservation() {
    // Номинальные скорости (per ms): 0→1=a, 1→0=b, 1→2=c, 2→1=d.
    let (a, b, c, d) = (0.3f32, 0.1f32, 0.2f32, 0.05f32);
    let gate = MarkovGateSpec {
        open: vec![0.0, 1.0, 0.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: const_form(a),
                },
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: MarkovRate::Directed {
                    rate: const_form(b),
                },
            },
            MarkovTransition {
                from: 1,
                to: 2,
                rate: MarkovRate::Directed {
                    rate: const_form(c),
                },
            },
            MarkovTransition {
                from: 2,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: const_form(d),
                },
            },
        ],
        power: 1,
    };

    // Стационар через детальный баланс по **фактическим** скоростям движка.
    let v = -65.0f32;
    let r: Vec<f32> = gate
        .transitions
        .iter()
        .map(|t| markov_rate_eval(&t.rate, v))
        .collect();
    let (a, b, c, d) = (r[0], r[1], r[2], r[3]);
    let p0 = 1.0 / (1.0 + a / b + a * c / (b * d));
    let p1 = (a / b) * p0;
    let p2 = (a * c / (b * d)) * p0;

    let ss = markov_steady_state(&gate, v);
    let sum: f32 = ss.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Σp = {sum} != 1");
    assert!((ss[0] - p0).abs() < 1e-5, "p0 {} != {p0}", ss[0]);
    assert!((ss[1] - p1).abs() < 1e-5, "p1 {} != {p1}", ss[1]);
    assert!((ss[2] - p2).abs() < 1e-5, "p2 {} != {p2}", ss[2]);

    // Из [1,0,0] за 100 шагов: вероятность сохраняется, открытая доля растёт.
    let mut p = vec![1.0f32, 0.0, 0.0];
    for _ in 0..100 {
        markov_step(&gate, &mut p, v, 0.05, 1.0);
    }
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Σp = {sum} != 1 после 100 шагов");
    assert!(
        markov_open_fraction(&gate, &p) > 0.0,
        "open fraction должна расти"
    );
}

/// `gateFractional` из двух субгейтов опускается в марковскую схему на
/// 4 состояниях (декартово произведение), чья открытая доля — взвешенная
/// сумма субгейтовых занятостей: `Σ frac_i·q_i`.
#[test]
fn fractional_gate_to_markov_matches_weighted_sum() {
    let frac = FractionalGateSpec {
        subgates: vec![
            FractionalSubgate {
                steady_state: ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 20.0,
                    scale: 5.0,
                },
                tau: 10.0,
                fractional_conductance: 0.7,
            },
            FractionalSubgate {
                steady_state: ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 40.0,
                    scale: 5.0,
                },
                tau: 20.0,
                fractional_conductance: 0.3,
            },
        ],
        power: 1,
    };
    let gate = fractional_gate_to_markov(&frac);
    assert_eq!(gate.n_states(), 4);
    // Веса: s=00→0, s=01(bit0=субгейт0)→0.7, s=10→0.3, s=11→1.0.
    assert_eq!(gate.open, vec![0.0, 0.7, 0.3, 1.0]);
    // 2 субгейта × 2 состояния (бит не выставлен) × 2 ноги = 8 переходов.
    assert_eq!(gate.transitions.len(), 8);

    // Стационарная открытая доля обязана совпадать с прямым взвешенным
    // стационаром независимых субгейтов.
    let v = -30.0f32;
    let ss = markov_steady_state(&gate, v);
    let sum: f32 = ss.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Σp = {sum} != 1");
    let got = markov_open_fraction(&gate, &ss);
    let want: f32 = frac
        .subgates
        .iter()
        .map(|sg| sg.fractional_conductance * sg.steady_state.eval(v))
        .sum();
    assert!((got - want).abs() < 1e-5, "fractional open {got} != {want}");
}

/// Прямые public-спеки `MarkovGateSpec` обязаны валидироваться, а не
/// приводить к панике индексации в `markov_step`/`markov_rate_matrix`:
/// пустые состояния, выход за границы, отрицательные/нулевые скорости,
/// недопустимый tau и self-переходы.
#[test]
fn markov_gate_spec_validate_rejects_malformed() {
    let directed = |rate: ClosedForm| MarkovTransition {
        from: 0,
        to: 1,
        rate: MarkovRate::Directed { rate },
    };

    // Пустой список состояний (0 состояний).
    let empty = MarkovGateSpec {
        open: vec![],
        transitions: vec![],
        power: 1,
    };
    assert!(empty.validate().unwrap_err().contains("≥2 states"));

    // Выход за границы: to = 2 при n = 2.
    let oob = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![MarkovTransition {
            from: 0,
            to: 2,
            rate: MarkovRate::Directed {
                rate: const_form(0.1),
            },
        }],
        power: 1,
    };
    assert!(oob.validate().unwrap_err().contains("out of range"));

    // Self-переход.
    let self_t = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![MarkovTransition {
            from: 0,
            to: 0,
            rate: MarkovRate::Directed {
                rate: const_form(0.1),
            },
        }],
        power: 1,
    };
    assert!(self_t.validate().unwrap_err().contains("self-transition"));

    // Отрицательная скорость.
    let neg = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![directed(ClosedForm::Exp {
            rate: -1.0,
            midpoint: 0.0,
            scale: 10.0,
        })],
        power: 1,
    };
    assert!(neg.validate().unwrap_err().contains("negative rate"));

    // Нулевой scale (деление на ноль).
    let zscale = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![directed(ClosedForm::Exp {
            rate: 1.0,
            midpoint: 0.0,
            scale: 0.0,
        })],
        power: 1,
    };
    assert!(zscale.validate().unwrap_err().contains("zero scale"));

    // Недопустимый (нулевой) tau для tauInfTransition.
    let bad_tau = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![MarkovTransition {
            from: 0,
            to: 1,
            rate: MarkovRate::TauInfForward {
                steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                    rate: 1.0,
                    midpoint: 0.0,
                    scale: 5.0,
                }),
                tau: TauForm::Fixed(0.0),
            },
        }],
        power: 1,
    };
    assert!(bad_tau.validate().unwrap_err().contains("tau"));

    // Корректный спек (опущенный NaV-m) проходит валидацию.
    let good = gate_kinetics_to_markov(standard_channels()[0].gates[0].kinetics.clone(), 3);
    assert!(good.validate().is_ok());
}

/// `markov_step` обязан отклонить occupancy-вектор неправильной длины.
#[test]
#[should_panic(expected = "occupancy length")]
fn markov_step_panics_on_short_occupancy() {
    let gate = gate_kinetics_to_markov(standard_channels()[0].gates[0].kinetics.clone(), 1);
    let mut p = vec![1.0f32]; // 1 элемент против 2 состояний
    markov_step(&gate, &mut p, -65.0, 0.05, 1.0);
}

/// Экстремальные rate×dt (1e10) раньше оставляли матрицу с нормой ~2.4e6
/// после жёсткого капа (12 halvings), и ряд Тейлора 8-й степени переполнялся
/// до NaN/Inf. Теперь масштабирование полное: результат конечен,
/// неотрицателен и нормирован.
#[test]
fn markov_step_extreme_rate_stays_finite_and_normalized() {
    let big = 1e10f32;
    let gate = MarkovGateSpec {
        open: vec![0.0, 1.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: const_form(big),
                },
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: MarkovRate::Directed {
                    rate: const_form(big),
                },
            },
        ],
        power: 1,
    };
    let mut p = vec![1.0f32, 0.0];
    markov_step(&gate, &mut p, -65.0, 1.0, 1.0);
    for &pi in &p {
        assert!(pi.is_finite() && pi >= 0.0, "occupancy {pi}");
    }
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Σp = {sum} != 1");
    // Симметричная схема → стационар 0.5/0.5.
    assert!(
        (p[0] - 0.5).abs() < 1e-4 && (p[1] - 0.5).abs() < 1e-4,
        "p = {p:?}"
    );
}
