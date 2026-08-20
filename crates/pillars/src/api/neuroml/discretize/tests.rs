// NEURON reference tables are f64 output rounded to 6 decimals; the
// f32 comparison target is identical either way, so the extra digits are
// intentional (kept verbatim from the generator for auditability).
#![allow(clippy::excessive_precision)]
use super::*;
use crate::api::neuroml::parse::{
    BiophysicalProperties, IntracellularProperties, IrChannelDensity, MembraneProperties, MorphoIR,
    Point3d, Segment, SegmentParent,
};
use crate::api::Population;
use crate::kernels::hh_channels::{
    gate_kinetics_to_markov, standard_channels, ChannelSpec, ClosedForm, GateKinetics,
    MarkovGateSpec, MarkovRate, MarkovTransition, TauForm, VarForm,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn seg(
    id: usize,
    parent: Option<usize>,
    px: f32,
    py: f32,
    pz: f32,
    pd: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    dd: f32,
) -> Segment {
    Segment {
        id,
        name: None,
        proximal: Some(Point3d {
            x: px,
            y: py,
            z: pz,
            diameter: pd,
        }),
        distal: Point3d {
            x: dx,
            y: dy,
            z: dz,
            diameter: dd,
        },
        parent: parent.map(|p| SegmentParent { segment: p }),
    }
}

/// Сома (корень, короткая) + дендрит длиной 1000 µm, d = 2 µm.
fn ball_and_stick() -> MorphoIR {
    MorphoIR {
        id: "bs".to_string(),
        segments: vec![
            // Сома: (0,0,0)→(0,10,0), d = 20 µm → L = 10 µm.
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 10.0, 0.0, 20.0),
            // Дендрит: (0,10,0)→(0,1010,0), d = 2 µm → L = 1000 µm.
            seg(1, Some(0), 0.0, 10.0, 0.0, 2.0, 0.0, 1010.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    }
}

fn default_params() -> CableParams {
    CableParams::default()
}

/// Сома — один компартмент; дендрит 1000 µm при λ=1000 µm и d_λ=0.1 →
/// ровно 10 компартментов. Топология: корень → 10 детей цепочкой.
#[test]
fn d_lambda_splits_long_segment_into_10() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();

    assert_eq!(tree.len(), 11);
    assert_eq!(tree.root, 0);

    let soma = &tree.compartments[0];
    assert_eq!(soma.parent, None);
    assert_eq!(soma.segment_id, 0);
    assert_eq!(soma.length_um, 10.0);
    assert_eq!(soma.diameter_um, 20.0);

    let first = &tree.compartments[1];
    assert_eq!(first.parent, Some(0));
    assert_eq!(first.segment_id, 1);
    assert_eq!(first.length_um, 100.0);
    assert_eq!(first.diameter_um, 2.0);
    for i in 2..11 {
        assert_eq!(tree.compartments[i].parent, Some(i - 1), "comp {i}");
        assert_eq!(tree.compartments[i].segment_id, 1);
    }
}

/// Позиции и path-дистанции: сома (центр y=5, path=5), дендритные
/// компартменты — центры 60/160/…/960, path шагает ровно на длину
/// компартмента (100 µm).
#[test]
fn compartment_positions_and_path() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();

    let soma = &tree.compartments[0];
    assert_eq!((soma.x, soma.y, soma.z), (0.0, 5.0, 0.0));
    assert!((soma.path_um - 5.0).abs() < 1e-5);

    // Дендрит (0,10)→(0,1010) → 10 компартментов по 100 µm: центры
    // 60..960, path шагает на 100 µm от первого центра 60.
    for (i, y_expected, path_expected) in [
        (1, 60.0, 60.0),
        (2, 160.0, 160.0),
        (9, 860.0, 860.0),
        (10, 960.0, 960.0),
    ] {
        let c = &tree.compartments[i];
        assert!(
            (c.y - y_expected).abs() < 1e-3,
            "comp {i}: y = {} vs {y_expected}",
            c.y
        );
        assert!(
            (c.path_um - path_expected).abs() < 1e-3,
            "comp {i}: path = {} vs {path_expected}",
            c.path_um
        );
        assert_eq!(c.x, 0.0);
        assert_eq!(c.z, 0.0);
    }
}

/// Аксиальное сопротивление 100 µm-компартмента (d=2 µm, Ra=100 Ω·cm) —
/// аналитически 4·Ra·L/(π·d²)·1e-2 ≈ 31.83 MΩ.
#[test]
fn axial_resistance_matches_analytic() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();
    let first = &tree.compartments[1];
    let expected = 4.0 * 100.0 * 100.0 / (PI * 2.0 * 2.0) * 1e-2;
    assert!(
        (first.r_axial_mohm - expected).abs() / expected < 1e-4,
        "R_a: {} vs {}",
        first.r_axial_mohm,
        expected
    );
}

/// Мембранные параметры: C = cm·π·d·L·1e-5 [nF], g = π·d·L·1e-2/Rm [µS].
#[test]
fn capacitance_and_leak_match_analytic() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();
    let first = &tree.compartments[1]; // L=100 µm, d=2 µm
    let area_cm2 = PI * 2.0 * 100.0 * 1e-8;
    let expected_cm = 1.0 * area_cm2 * 1e3;
    let expected_g = area_cm2 / 20_000.0 * 1e6;
    assert!((first.cm_nf - expected_cm).abs() / expected_cm < 1e-4);
    assert!((first.g_leak_us - expected_g).abs() / expected_g < 1e-4);
    assert!((first.area_um2 - PI * 2.0 * 100.0).abs() < 1e-3);
}

/// Короткий сегмент (L ≪ 0.1λ) → ровно один компартмент (не ноль).
#[test]
fn short_segment_is_single_compartment() {
    let m = MorphoIR {
        id: "s".to_string(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 2.0, 0.0, 10.0, 0.0, 2.0)],
        segment_groups: vec![],
    };
    let tree = discretize(&m, &default_params()).unwrap();
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.compartments[0].parent, None);
}

/// Несколько корней — ошибка (нейрон — одно дерево с одной сомой).
#[test]
fn multiple_roots_error() {
    let m = MorphoIR {
        id: "multi".to_string(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
            seg(1, None, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
        ],
        segment_groups: vec![],
    };
    assert!(discretize(&m, &default_params()).is_err());
}

// ─── Cable solver ─────────────────────────────────────────

/// Один компартмент (сома d=20, L=20): τ = C/g = 20 мс. Покой — точная
/// неподвижная точка: без входа V остаётся ровно E_leak.
#[test]
fn cable_solver_rest_is_fixed_point() {
    let m = MorphoIR {
        id: "s".to_string(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    let tree = discretize(&m, &default_params()).unwrap();
    let mut solver = CableSolver::new(tree, -65.0, 0.05);
    let i_inj = [0.0f32; 1];
    for _ in 0..100 {
        solver.step(&i_inj);
    }
    assert!((solver.v(0) - (-65.0)).abs() < 1e-6);
}

/// Один компартмент: под током садится на steady state V = E + I/g, а при
/// t = τ значение ≈ E + 0.632·ΔV (экспонента, τ = 20 мс).
#[test]
fn cable_solver_single_compartment_time_constant() {
    let m = MorphoIR {
        id: "s".to_string(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    let tree = discretize(&m, &default_params()).unwrap();
    let g = tree.compartments[0].g_leak_us;
    let i_inj = [0.006_283_185f32]; // ≈ g → ΔV ≈ 1 mV
    let dt = 0.05;
    let mut solver = CableSolver::new(tree, -65.0, dt);

    // t = τ = 20 мс (400 шагов × 0.05).
    for _ in 0..400 {
        solver.step(&i_inj);
    }
    let dv_ss = i_inj[0] / g; // mV
    let expected = -65.0 + dv_ss * (1.0 - (-1.0f32).exp());
    assert!(
        (solver.v(0) - expected).abs() < 0.05,
        "V(τ) = {} vs {}",
        solver.v(0),
        expected
    );

    // t → ∞ (ещё 4000 шагов = 200 мс = 10τ).
    for _ in 0..4_000 {
        solver.step(&i_inj);
    }
    let v_ss = -65.0 + dv_ss;
    assert!(
        (solver.v(0) - v_ss).abs() < 0.01,
        "V_ss = {} vs {}",
        solver.v(0),
        v_ss
    );
}

/// Ball-and-stick: ток в сому → деполяризация затухает монотонно с
/// расстоянием (сома сильнее всего, кончик дендрита — меньше всего).
#[test]
fn cable_solver_ball_and_stick_attenuates() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();
    let mut solver = CableSolver::new(tree, -65.0, 0.05);

    let n = solver.len();
    let mut i_inj = vec![0.0f32; n];
    i_inj[0] = 0.01; // ток в сому (nA)

    // 4000 шагов × 0.05 = 200 мс ≫ τ=20 мс и кабельное уравновешивание.
    for _ in 0..4_000 {
        solver.step(&i_inj);
    }

    assert!(solver.v(0) > -65.0, "сома должна деполяризоваться");
    for i in 0..n - 1 {
        assert!(
            solver.v(i) > solver.v(i + 1),
            "затухание: V[{}]={:.4} <= V[{}]={:.4}",
            i,
            solver.v(i),
            i + 1,
            solver.v(i + 1)
        );
    }
    // Кончик дендрита существенно ближе к покою, чем сома.
    assert!(
        solver.v(n - 1) > -65.0,
        "кончик тоже деполяризован, но слабо: {}",
        solver.v(n - 1)
    );
    assert!(
        (solver.v(0) - (-65.0)) > (solver.v(n - 1) - (-65.0)),
        "сома должна быть сильнее кончика"
    );
}

/// `cable_voltage` зеркалит покомпартментный V: длина = n·comps,
/// сома-записи совпадают с `state_slice(0)`, дендрит затухает.
#[test]
fn cable_population_exposes_per_compartment_voltage() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();
    let mut pop = CablePopulation::new(&tree, 2, -65.0, 0.05, 6.3);
    let n_comps = tree.len();

    // До шагов: покой везде.
    let cv = pop.cable_voltage().expect("кабельная популяция даёт V");
    assert_eq!((cv.n_neurons, cv.n_comps), (2, n_comps));
    assert_eq!(cv.v.len(), 2 * n_comps);
    assert!(cv.v.iter().all(|&v| (v + 65.0).abs() < 1e-4));

    // Шагаем с током в сомы обоих нейронов.
    let mut i_in = vec![0.01f32; 2];
    let mut spikes = vec![0i32; 2];
    for _ in 0..4_000 {
        pop.step(&i_in, &mut spikes);
    }
    i_in.fill(0.0);

    let cv = pop.cable_voltage().unwrap();
    let soma = tree.root;
    for neuron in 0..2 {
        let base = neuron * n_comps;
        // Сома-запись == state_slice(0) (публичная колонка).
        assert_eq!(cv.v[base + soma], pop.state_slice(0)[neuron]);
        // Дендрит затухает: кончик ближе к покою, чем сома.
        assert!(cv.v[base + soma] > cv.v[base + n_comps - 1]);
        assert!(cv.v[base + n_comps - 1] > -65.0);
    }
}

/// Implicit Euler безусловно устойчив: один компартмент при dt=100 мс
/// (5τ) под током монотонно движется к steady state, не пересекая его.
#[test]
fn cable_solver_implicit_stable_at_large_dt() {
    let m = MorphoIR {
        id: "s".to_string(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    let tree = discretize(&m, &default_params()).unwrap();
    let g = tree.compartments[0].g_leak_us;
    let i_inj = [0.006_283_185f32];
    let v_ss = -65.0 + i_inj[0] / g;

    let mut solver = CableSolver::new(tree, -65.0, 100.0); // dt = 5τ
    let mut v = -65.0f32;
    for _ in 0..20 {
        solver.step(&i_inj);
        v = solver.v(0);
        assert!(
            v <= v_ss + 1e-4,
            "overshoot past steady state: {v} vs {v_ss}"
        );
        assert!(v.is_finite());
    }
    assert!((v - v_ss).abs() < 0.01, "converged: {v} vs {v_ss}");
}

// ─── Active channels ─────────────────────────────────────

/// Преобразовать `ChannelSpec` (g_max в mS/cm²) в `ActiveChannel` с
/// абсолютной проводимостью (µS) для компартмента площадью `area_cm2`.
fn active_from_spec(spec: &ChannelSpec, area_cm2: f32, v_rest: f32) -> ActiveChannel {
    let gates: Vec<(GateKinetics, u32)> = spec
        .gates
        .iter()
        .map(|g| (g.kinetics.clone(), g.power))
        .collect();
    let g_max_us = spec.g_max * area_cm2 * 1e3; // mS/cm² · cm² · 1e3 = µS
    let q10s = vec![1.0; gates.len()];
    ActiveChannel::new(gates, q10s, spec.E_rev, g_max_us, v_rest)
}

/// Один компартмент со стандартными NaV/KV + HH-лик должен спайкать под
/// надпороговым током — активные каналы реально входят в кабельный солвер.
#[test]
fn active_cable_single_compartment_hh_fires() {
    let std = standard_channels(); // [0] = NaV, [1] = KV
    let m = MorphoIR {
        id: "s".to_string(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² → HH-лик (0.3 mS/cm², E=-54.3).
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
    let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
    solver.set_channels(
        0,
        vec![
            active_from_spec(&std[0], area_cm2, -65.0),
            active_from_spec(&std[1], area_cm2, -65.0),
        ],
    );
    solver.set_v(0, -65.0);

    let mut i_inj = vec![0.0f32; solver.len()];
    let mut spikes = 0;
    let mut was_below = true;
    for _ in 0..8_000 {
        i_inj[0] = 0.3; // nA, надпороговый
        solver.step(&i_inj);
        let v = solver.v(0);
        if v >= 0.0 && was_below {
            spikes += 1;
        }
        was_below = v < 0.0;
    }
    assert!(
        spikes >= 1,
        "HH-компартмент должен спайкать, получено {spikes} спайков"
    );
}

// ─── Biophysics → CableParams bridge ─────────────────────

/// `resistivity`→Ra, `specificCapacitance`→cm, пассивный лик→rm, и
/// `erev` лика как e_leak. condDensity 3e-4 S/cm² → rm = 3333 Ω·cm².
#[test]
fn cable_params_from_biophysics_maps_units() {
    let bp = BiophysicalProperties {
        membrane: MembraneProperties {
            channel_densities: vec![IrChannelDensity {
                id: "pas".into(),
                ion_channel: "pas".into(),
                cond_density: 3e-4, // S/cm²
                erev: -70.0,
                ion: "non_specific".into(),
                segment_group: "soma_group".into(),
            }],
            non_uniform_densities: vec![],
            channel_populations: vec![],
            nernst_populations: vec![],
            nernst_densities: vec![],
            ghk_densities: vec![],
            init_memb_potential: -65.0,
            specific_capacitance: 1.5,
        },
        intracellular: IntracellularProperties { resistivity: 150.0 },
        species: vec![],
    };

    let params = CableParams::from_biophysics(&bp);
    assert_eq!(params.ra_ohm_cm, 150.0);
    assert_eq!(params.cm_uf_per_cm2, 1.5);
    assert!(
        (params.rm_ohm_cm2 - 3333.333).abs() < 1.0,
        "rm: {}",
        params.rm_ohm_cm2
    );
    assert_eq!(passive_leak_reversal(&bp), -70.0);
}

/// Без пассивного канала: rm остаётся дефолтом, e_leak = initMembPotential.
#[test]
fn cable_params_fallback_without_passive_channel() {
    let bp = BiophysicalProperties {
        membrane: MembraneProperties {
            channel_densities: vec![],
            non_uniform_densities: vec![],
            channel_populations: vec![],
            nernst_populations: vec![],
            nernst_densities: vec![],
            ghk_densities: vec![],
            init_memb_potential: -58.0,
            specific_capacitance: 1.0,
        },
        intracellular: IntracellularProperties { resistivity: 100.0 },
        species: vec![],
    };
    let params = CableParams::from_biophysics(&bp);
    assert_eq!(params.rm_ohm_cm2, CableParams::default().rm_ohm_cm2);
    assert_eq!(passive_leak_reversal(&bp), -58.0);
}

// ─── SegmentGroup → compartments ─────────────────────────

/// `group_compartments` мапит id сегментов на индексы компартментов:
/// сома (segment 0) → [0], дендрит (segment 1) → 10 компартментов.
#[test]
fn group_compartments_maps_segments() {
    let m = ball_and_stick();
    let tree = discretize(&m, &default_params()).unwrap();
    assert_eq!(tree.group_compartments(&[0]), vec![0]);
    assert_eq!(tree.group_compartments(&[1]), (1..11).collect::<Vec<_>>());
    assert_eq!(
        tree.group_compartments(&[0, 1]),
        (0..11).collect::<Vec<_>>()
    );
}

// ─── Markov channels in the cable ────────────────────────

/// Стандартный NaV (m³h), опущенный до 2-состоянийного Маркова, работает
/// как `MarkovActiveChannel`: открытая доля конечна и в [0, 1].
#[test]
fn markov_active_channel_runs() {
    let std = standard_channels();
    let gates: Vec<MarkovGateSpec> = std[0]
        .gates
        .iter()
        .map(|g| gate_kinetics_to_markov(g.kinetics.clone(), g.power))
        .collect();
    let mut ch = MarkovActiveChannel::new(
        gates.clone(),
        vec![1.0; gates.len()],
        std[0].E_rev,
        0.001,
        -65.0,
    );
    for _ in 0..100 {
        let f = ch.step_open(-65.0, 0.05);
        assert!(
            f.is_finite() && (0.0..=1.0).contains(&f),
            "open fraction {f}"
        );
    }
}

// ─── NEURON parity (passive cable) ───────────────────────

/// Эталон NEURON 9.0.2 для пассивного ball-and-stick (сома + дендрит),
/// сгенерированный `gen_cable_ref.py`: Ra=100, cm=1, Rm=20000, E=-65,
/// I=+0.01 nA в сому (10–60 мс), dt=0.025, secondorder=0 (backward Euler).
const REF_V_SOMA: [f32; 21] = [
    -65.0, -65.0, -65.0, -63.824051, -63.304875, -62.940604, -62.662314, -62.446288, -62.278122,
    -62.147147, -62.045129, -61.965666, -61.903770, -63.031507, -63.513130, -63.848150, -64.103655,
    -64.301933, -64.456276, -64.576483, -64.670114,
];
const REF_V_TIP: [f32; 21] = [
    -65.0, -65.0, -65.0, -64.836370, -64.449930, -64.103704, -63.827867, -63.612175, -63.444054,
    -63.313085, -63.211068, -63.131605, -63.069709, -63.185127, -63.534014, -63.850988, -64.104041,
    -64.301986, -64.456283, -64.576484, -64.670114,
];

#[test]
fn passive_ball_and_stick_matches_neuron() {
    // Сома L=20/d=20 + дендрит L=1000/d=2 — ровно как в gen_cable_ref.py.
    let m = MorphoIR {
        id: "bs".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    let params = CableParams {
        ra_ohm_cm: 100.0,
        cm_uf_per_cm2: 1.0,
        rm_ohm_cm2: 20_000.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    assert_eq!(tree.len(), 11, "1 сома + 10 дендритных компартментов");
    let mut solver = CableSolver::new(tree, -65.0, 0.025);

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 2400usize; // t = 60 мс
    let mut i_inj = vec![0.0f32; solver.len()];
    let mut v_soma = Vec::with_capacity(n_steps + 1);
    let mut v_tip = Vec::with_capacity(n_steps + 1);
    v_soma.push(solver.v(0));
    v_tip.push(solver.v(10));
    for step in 0..n_steps {
        i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
        solver.step(&i_inj);
        v_soma.push(solver.v(0));
        v_tip.push(solver.v(10));
    }

    let grid = 200usize; // 5 мс / 0.025 мс
    let mut d_soma = 0.0f32;
    let mut d_tip = 0.0f32;
    for (k, (&rs, &rt)) in REF_V_SOMA.iter().zip(REF_V_TIP.iter()).enumerate() {
        d_soma = d_soma.max((v_soma[k * grid] - rs).abs());
        d_tip = d_tip.max((v_tip[k * grid] - rt).abs());
    }
    println!("cable vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV");
    // Оба — backward Euler (secondorder=0), один Hines-солвер; остаток —
    // f32 (pillars) против f64 (NEURON) за 4000 шагов (~4e-4 мВ).
    assert!(
        d_soma < 0.01 && d_tip < 0.01,
        "soma={d_soma:.6} tip={d_tip:.6} mV"
    );
}

/// Эталон NEURON 9.0.2 для **ветвящегося** пассивного кабеля (сома →
/// ствол → два асимметричных дендрита), сгенерированный
/// `gen_cable_branch_ref.py`: Ra=100, cm=1, Rm=20000, E=-65, I=+0.01 nA в
/// сому (10–60 мс), dt=0.025, secondorder=0 (backward Euler). Проверяет
/// Hines-солвер на нетривиальной топологии (одна ветвь → две дочери).
const REF_BRANCH_SOMA: [f32; 21] = [
    -65.000000, -65.000000, -65.000000, -63.825576, -63.333325, -63.023504, -62.804708, -62.641781,
    -62.517441, -62.421476, -62.347035, -62.289158, -62.244113, -63.383464, -63.848399, -64.136945,
    -64.339171, -64.489190, -64.603477, -64.691610, -64.759951,
];
const REF_BRANCH_BRANCH: [f32; 21] = [
    -65.000000, -65.000000, -65.000000, -64.916861, -64.686765, -64.458712, -64.267275, -64.113731,
    -63.992635, -63.897795, -63.823744, -63.766002, -63.721004, -63.769086, -63.971872, -64.178653,
    -64.353520, -64.494157, -64.605199, -64.692208, -64.760159,
];
const REF_BRANCH_TIP_A: [f32; 21] = [
    -65.000000, -65.000000, -65.000000, -64.989517, -64.887070, -64.720039, -64.551670, -64.406343,
    -64.288123, -64.194284, -64.120580, -64.062959, -64.018003, -63.993444, -64.068586, -64.214346,
    -64.366145, -64.498565, -64.606732, -64.692740, -64.760344,
];
const REF_BRANCH_TIP_B: [f32; 21] = [
    -65.000000, -65.000000, -65.000000, -64.932539, -64.711562, -64.485878, -64.295165, -64.141862,
    -64.020848, -63.926037, -63.851996, -63.794257, -63.749261, -63.781665, -63.975333, -64.179744,
    -64.353886, -64.494282, -64.605242, -64.692223, -64.760164,
];

// ─── NEURON parity (1000-compartment cable) ─────────────────

/// Эталон NEURON 9.0.2 для **1001-компартментного** пассивного кабеля —
/// сгенерирован `gen_cable_long_ref.py`: сома L=20/d=20 (nseg=1) +
/// дендрит L=100000/d=2 (nseg=1000, d_λ=0.1 → сегменты 100 мкм), Ra=100,
/// cm=1, Rm=20000, E=-65, I=+0.01 nA в сому (10–60 мс), dt=0.025,
/// secondorder=0. Проверяет, что f32-элиминация не деградирует на длинной
/// цепочке: сома должна совпасть (~4e-4), а середина/кончик (50–100 λ,
/// куда импульс не доходит) — не дрейфовать от −65.
///
/// Замечание: слишком мелкая сетка (d_λ=0.001 → сегменты 1 мкм) численно
/// патологична — осевая проводимость в ~550 раз больше мембранной, и
/// f32-элиминация дрейфует (~2e-3 мВ/шаг). Сетка λ/10 — хорошо
/// обусловлена.
const REF_V_LONG_SOMA: [f32; 21] = [
    -65.0, -65.0, -65.0, -63.825486, -63.335048, -63.036762, -62.839418, -62.703421, -62.607393,
    -62.538478, -62.488445, -62.451803, -62.424785, -63.579269, -64.054791, -64.341927, -64.530912,
    -64.660623, -64.751915, -64.817253, -64.864580,
];
const REF_V_LONG_MID: [f32; 21] = [-65.0; 21];
const REF_V_LONG_TIP: [f32; 21] = [-65.0; 21];

#[test]
fn passive_long_cable_matches_neuron() {
    let m = MorphoIR {
        id: "bs".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            // Дендрит L=100000 мкм: d_λ=0.1 → 1000 сегментов по 100 мкм.
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 100020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    let params = CableParams {
        ra_ohm_cm: 100.0,
        cm_uf_per_cm2: 1.0,
        rm_ohm_cm2: 20_000.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let n = tree.len();
    assert_eq!(n, 1001, "сома 1 + дендрит 1000");
    let mid = 501; // dend(0.5): сегмент 500 (0-indexed), глобально 1+500
    let tip = n - 1;
    let mut solver = CableSolver::new(tree, -65.0, 0.025);

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 2400usize; // t = 60 мс
    let mut i_inj = vec![0.0f32; n];
    // Сэмплинг на границах шага (как NEURON Vector.record).
    let mut v_soma = vec![solver.v(0)];
    let mut v_mid = vec![solver.v(mid)];
    let mut v_tip = vec![solver.v(tip)];
    for step in 0..n_steps {
        i_inj.fill(0.0);
        i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
        solver.step(&i_inj);
        if (step + 1) % 200 == 0 {
            v_soma.push(solver.v(0));
            v_mid.push(solver.v(mid));
            v_tip.push(solver.v(tip));
        }
    }
    let mut d_soma = 0.0f32;
    let mut d_mid = 0.0f32;
    let mut d_tip = 0.0f32;
    for k in 0..21 {
        d_soma = d_soma.max((v_soma[k] - REF_V_LONG_SOMA[k]).abs());
        d_mid = d_mid.max((v_mid[k] - REF_V_LONG_MID[k]).abs());
        d_tip = d_tip.max((v_tip[k] - REF_V_LONG_TIP[k]).abs());
    }
    println!("long cable vs NEURON: max |ΔV| soma={d_soma:.6} mid={d_mid:.6} tip={d_tip:.6} mV");
    assert!(d_soma < 0.005, "soma: max |ΔV| = {d_soma:.4} mV");
    assert!(d_mid < 0.001, "mid: max |ΔV| = {d_mid:.4} mV");
    assert!(d_tip < 0.001, "tip: max |ΔV| = {d_tip:.4} mV");
}

#[test]
fn passive_branching_cable_matches_neuron() {
    // Сома L=20/d=20, ствол L=1000/d=2, дочери da L=500/d=1 и db L=200/d=2
    // (обе от кончика ствола) — ровно как в gen_cable_branch_ref.py.
    let m = MorphoIR {
        id: "branch".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            seg(2, Some(1), 0.0, 1020.0, 0.0, 1.0, 0.0, 1520.0, 0.0, 1.0),
            seg(3, Some(1), 0.0, 1020.0, 0.0, 2.0, 200.0, 1020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    let params = CableParams {
        ra_ohm_cm: 100.0,
        cm_uf_per_cm2: 1.0,
        rm_ohm_cm2: 20_000.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    // 1 сома + 10 ствол + 1 branch-узел + 8 da + 2 db.
    assert_eq!(tree.len(), 22, "1+10+1+8+2 компартмента");
    let mut solver = CableSolver::new(tree, -65.0, 0.025);

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 2400usize; // t = 60 мс
    let mut i_inj = vec![0.0f32; solver.len()];
    // Компартменты: сома=0, кончик ствола (центр)=10, branch-узел=11,
    // кончик da=19, кончик db=21. NEURON `trunk(1.0)` = branch-узел (x=1.0).
    let (i_soma, i_branch, i_tip_a, i_tip_b) = (0usize, 11usize, 19usize, 21usize);
    let mut v_soma = Vec::with_capacity(n_steps + 1);
    let mut v_branch = Vec::with_capacity(n_steps + 1);
    let mut v_tip_a = Vec::with_capacity(n_steps + 1);
    let mut v_tip_b = Vec::with_capacity(n_steps + 1);
    v_soma.push(solver.v(i_soma));
    v_branch.push(solver.v(i_branch));
    v_tip_a.push(solver.v(i_tip_a));
    v_tip_b.push(solver.v(i_tip_b));
    for step in 0..n_steps {
        i_inj[0] = if step >= on && step < off { 0.01 } else { 0.0 };
        solver.step(&i_inj);
        v_soma.push(solver.v(i_soma));
        v_branch.push(solver.v(i_branch));
        v_tip_a.push(solver.v(i_tip_a));
        v_tip_b.push(solver.v(i_tip_b));
    }

    let grid = 200usize; // 5 мс / 0.025 мс
    let (mut d_soma, mut d_branch, mut d_a, mut d_b) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..REF_BRANCH_SOMA.len() {
        d_soma = d_soma.max((v_soma[k * grid] - REF_BRANCH_SOMA[k]).abs());
        d_branch = d_branch.max((v_branch[k * grid] - REF_BRANCH_BRANCH[k]).abs());
        d_a = d_a.max((v_tip_a[k * grid] - REF_BRANCH_TIP_A[k]).abs());
        d_b = d_b.max((v_tip_b[k * grid] - REF_BRANCH_TIP_B[k]).abs());
    }
    println!(
            "branching cable vs NEURON: max |ΔV| soma={d_soma:.6} branch={d_branch:.6} tipA={d_a:.6} tipB={d_b:.6} mV"
        );
    // Один backward Euler (secondorder=0) Hines-солвер; остаток — f32 vs f64.
    assert!(
        d_soma < 0.01 && d_branch < 0.01 && d_a < 0.01 && d_b < 0.01,
        "soma={d_soma:.6} branch={d_branch:.6} tipA={d_a:.6} tipB={d_b:.6} mV"
    );
}

/// Активный HH на соме кабеля против NEURON `hh` (эталон `gen_cable_hh_ref.py`:
/// сома L=20/d=20, NaV/KV + лик, I=+0.3 nA, dt=0.025, t=10–90 мс).
/// Гейты pillars — exponential-Euler, NEURON — backward-Euler, поэтому
/// сверяется **число спайков** (не трасса): 8 ± 2.
#[test]
fn active_hh_soma_matches_neuron_spike_count() {
    let std = standard_channels();
    let m = MorphoIR {
        id: "s".into(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² → HH-лик (0.3 mS/cm², E=-54.3).
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
    let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
    solver.set_channels(
        0,
        vec![
            active_from_spec(&std[0], area_cm2, -65.0),
            active_from_spec(&std[1], area_cm2, -65.0),
        ],
    );
    solver.set_v(0, -65.0);

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 3600usize; // t = 90 мс
    let mut i_inj = vec![0.0f32; solver.len()];
    let mut spikes = 0u32;
    let mut was_below = true;
    for step in 0..n_steps {
        i_inj[0] = if step >= on && step < off { 0.3 } else { 0.0 };
        solver.step(&i_inj);
        let v = solver.v(0);
        if v >= 0.0 && was_below {
            spikes += 1;
        }
        was_below = v < 0.0;
    }
    println!("active cable vs NEURON: spikes={spikes} (NEURON=8)");
    assert!((spikes as i32 - 8).abs() <= 2, "spikes={spikes} (NEURON=8)");
}

// ─── NEURON parity (Markov/kschan in the cable) ──────────

/// Эталон NEURON 9.0.2 для ball-and-stick с 3-состоянием ks3
/// (Na-m/Na-h rates) на соме — сгенерирован `gen_cable_ks_ref.py`.
/// pas 0.3 mS/cm² / E=-54.3 везде, ks3 gbar=36 mS/cm² / ek=-77 на соме,
/// V0=-65, I=+0.125664 nA (10 µA/cm²) в сому (10–60 мс), dt=0.025,
/// secondorder=0. K-шунт сомы держит её у покоя, пассивный дендрит
/// (e_pas=-54.3) заряжается — канал реально работает.
const REF_V_KS3_SOMA: [f32; 21] = [
    -65.0, -67.796423, -67.671610, -65.322792, -65.312031, -65.311104, -65.311307, -65.311586,
    -65.311819, -65.312004, -65.312149, -65.312263, -65.312351, -67.635761, -67.649498, -67.650045,
    -67.649160, -67.648254, -67.647486, -67.646852, -67.646331,
];
const REF_V_KS3_TIP: [f32; 21] = [
    -65.0, -57.698422, -56.749023, -56.339287, -56.216609, -56.200489, -56.198458, -56.198246,
    -56.198261, -56.198295, -56.198325, -56.198349, -56.198368, -56.474851, -56.584485, -56.599277,
    -56.601068, -56.601149, -56.601029, -56.600905, -56.600800,
];

/// 3-state gate (closed ⇌ open ⇌ inactivated), rates = Na-m (0↔1) + Na-h
/// (1↔2) — зеркало `ks3.mod`, как в `tests/ks_vs_neuron.rs::ks3_gate`.
fn ks3_gate() -> MarkovGateSpec {
    MarkovGateSpec {
        open: vec![0.0, 1.0, 0.0],
        transitions: vec![
            MarkovTransition {
                from: 0,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: ClosedForm::ExpLinear {
                        rate: 0.1,
                        midpoint: 40.0,
                        scale: 10.0,
                    },
                },
            },
            MarkovTransition {
                from: 1,
                to: 0,
                rate: MarkovRate::Directed {
                    rate: ClosedForm::Exp {
                        rate: 4.0,
                        midpoint: 65.0,
                        scale: 18.0,
                    },
                },
            },
            MarkovTransition {
                from: 1,
                to: 2,
                rate: MarkovRate::Directed {
                    rate: ClosedForm::Exp {
                        rate: 0.07,
                        midpoint: 65.0,
                        scale: 20.0,
                    },
                },
            },
            MarkovTransition {
                from: 2,
                to: 1,
                rate: MarkovRate::Directed {
                    rate: ClosedForm::Sigmoid {
                        rate: 1.0,
                        midpoint: 35.0,
                        scale: 10.0,
                    },
                },
            },
        ],
        power: 1,
    }
}

/// Марков-канал (kschan) в кабеле против NEURON: 3-состояние ks3 на соме
/// ball-and-stick, субпороговый протокол, сверка V(сома)/V(кончик) на
/// сетке 5 мс. NEURON интегрирует KINETIC backward-Euler (METHOD sparse),
/// pillars — точной матричной экспонентой → согласие с точностью
/// интегратора: max |ΔV| ≈ 0.014 мВ (замер; допуск 0.03), против ~6e-4 у
/// точечного `ks_vs_neuron.rs` — добавка от аксиального пути 1000 мкм.
#[test]
fn markov_cable_matches_neuron_ks3() {
    let m = MorphoIR {
        id: "bs".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    // Rm = 1/0.3 mS/cm² = 3333 Ω·cm² (pas, как в ks_vs_neuron).
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let n = tree.len();
    let soma = tree.root;
    let tip = n - 1;
    let area_cm2 = tree.compartments[soma].area_um2 * 1e-8;
    let mut solver = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
    // V0 = -65 (покой), хотя e_leak (pas reversal) = -54.3 — как в
    // NEURON `finitialize(-65)` при e_pas=-54.3.
    for i in 0..n {
        solver.set_v(i, -65.0);
    }
    // gbar 36 mS/cm² → абсолютная µS по площади сомы.
    let g_max_us = 0.036 * area_cm2 * 1e6;
    solver.set_markov_channels(
        soma,
        vec![MarkovActiveChannel::new(
            vec![ks3_gate()],
            vec![1.0],
            -77.0,
            g_max_us,
            -65.0,
        )],
    );

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 2400usize; // t = 60 мс
    let mut i_inj = vec![0.0f32; n];
    // Сэмплинг на границах шага (как NEURON Vector.record и ks_vs_neuron):
    // t=0 до первого шага, далее после шага `s` → t = (s+1)·dt.
    let mut v_soma = vec![solver.v(soma)];
    let mut v_tip = vec![solver.v(tip)];
    for step in 0..n_steps {
        i_inj.fill(0.0);
        i_inj[soma] = if step >= on && step < off {
            0.125664
        } else {
            0.0
        };
        solver.step(&i_inj);
        if (step + 1) % 200 == 0 {
            v_soma.push(solver.v(soma));
            v_tip.push(solver.v(tip));
        }
    }
    let mut d_soma = 0.0f32;
    let mut d_tip = 0.0f32;
    for (k, (&rs, &rt)) in REF_V_KS3_SOMA.iter().zip(REF_V_KS3_TIP.iter()).enumerate() {
        d_soma = d_soma.max((v_soma[k] - rs).abs());
        d_tip = d_tip.max((v_tip[k] - rt).abs());
    }
    println!("markov cable vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV");
    assert!(d_soma < 0.03, "soma: max |ΔV| = {d_soma:.4} mV");
    assert!(d_tip < 0.03, "tip: max |ΔV| = {d_tip:.4} mV");
}

/// Стохастический `channelPopulation` против NEURON: **среднее по ансамблю**
/// сходится к детерминистическому следу ks3.
///
/// Детерминистический предел дискретной популяции совпадает с `channelDensity`:
/// та же master equation (тот же `p_open`), та же суммарная проводимость
/// `single_cond·N = gbar·area`. Поэтому эталон — те же `REF_V_KS3_*`, что и в
/// `markov_cable_matches_neuron_ks3` (NEURON `KINETIC` backward-Euler). Канальный
/// шум `∝ 1/√N` усредняется по `N_SEEDS` независимым seed'ам: матожидание
/// V(t) сходится к детерминистическому следу с точностью `≈ σ_ансамбля/√N_SEEDS`.
///
/// jNeuroML экспортирует `channelPopulation` **детерминистически** (mean-field,
/// общая проводимость `number × single_cond`); биномиальная выборка `n_open` —
/// наше расширение, валидированное здесь по среднему и отдельно по дисперсии
/// (`sample_binomial` mean/var-тест).
#[test]
fn channel_population_mean_matches_neuron() {
    // Быстрый режим: биномиальная выборка стоит ~N·p RNG-вызовов/шаг, а
    // шум среднего убывает как 1/√(N·seed'ов) — работа на заданную точность
    // инвариантна к разбиению N/seed. Умеренное N и несколько seed'ов дают
    // σ_mean ~ 0.02 мВ при <2 с (не 8192/128 → 70 с).
    const N_CHANNELS: usize = 2048;
    const N_SEEDS: usize = 16;

    let m = MorphoIR {
        id: "bs".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let n = tree.len();
    let soma = tree.root;
    let tip = n - 1;
    let area_cm2 = tree.compartments[soma].area_um2 * 1e-8;
    // Суммарная проводимость: `single_cond · N = 0.036 S/cm² · area` — тот
    // же gbar, что и в детерминистическом ks3-тесте, т.е. один и тот же
    // mean-field ток. Разложение на single_cond × N живёт в compile.rs.
    let g_max_us = 0.036 * area_cm2 * 1e6;

    let n_steps = 4000usize;
    let on = 400usize; // t = 10 мс
    let off = 2400usize; // t = 60 мс
    let grid = 200usize; // семплинг каждые 5 мс, как в REF_V_KS3_*
    let n_samples = 1 + n_steps / grid;

    let base = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
    let base_seed = 0x5EED_CAFE_0BAD_F00Du64;
    let mut acc_soma = vec![0.0f64; n_samples];
    let mut acc_tip = vec![0.0f64; n_samples];

    for seed in 0..N_SEEDS {
        let mut solver = base.clone();
        // V0 = -65 (покой), e_leak = -54.3 — как в NEURON `finitialize(-65)`.
        for i in 0..n {
            solver.set_v(i, -65.0);
        }
        // Свежий канал: occupancy в steady-state при v_rest=-65, независимый
        // seed (потоки независимы → усреднение сходится по ЦПТ).
        solver.set_stochastic_channels(
            soma,
            vec![StochasticMarkovActiveChannel::new(
                vec![ks3_gate()],
                vec![1.0],
                -77.0,
                g_max_us,
                N_CHANNELS,
                -65.0,
                base_seed.wrapping_add(seed as u64),
            )],
        );

        let mut i_inj = vec![0.0f32; n];
        acc_soma[0] += solver.v(soma) as f64;
        acc_tip[0] += solver.v(tip) as f64;
        let mut k = 0usize;
        for step in 0..n_steps {
            i_inj.fill(0.0);
            i_inj[soma] = if step >= on && step < off {
                0.125664
            } else {
                0.0
            };
            solver.step(&i_inj);
            if (step + 1) % grid == 0 {
                k += 1;
                acc_soma[k] += solver.v(soma) as f64;
                acc_tip[k] += solver.v(tip) as f64;
            }
        }
    }

    let mut d_soma = 0.0f32;
    let mut d_tip = 0.0f32;
    for (idx, (&rs, &rt)) in REF_V_KS3_SOMA.iter().zip(REF_V_KS3_TIP.iter()).enumerate() {
        let mean_soma = (acc_soma[idx] / N_SEEDS as f64) as f32;
        let mean_tip = (acc_tip[idx] / N_SEEDS as f64) as f32;
        d_soma = d_soma.max((mean_soma - rs).abs());
        d_tip = d_tip.max((mean_tip - rt).abs());
    }
    println!(
            "channelPopulation mean vs NEURON: max |ΔV| soma={d_soma:.6} tip={d_tip:.6} mV (N={N_CHANNELS}, seeds={N_SEEDS})"
        );
    // Замер: soma≈0.041, tip≈0.015 мВ (детерминированный по фиксированным
    // seed'ам — не флакает). Допуск 0.1 = 2.4× запаса над замером.
    assert!(d_soma < 0.1, "soma: max |ΔV| = {d_soma:.4} mV");
    assert!(d_tip < 0.1, "tip: max |ΔV| = {d_tip:.4} mV");
}

// ─── Adaptive step (CVODE-analogue) ──────────────────────

/// Прогнать `AdaptiveCableSolver` до `tstop`, подавая импульс в сому.
/// Возвращает (min_dt, max_dt, число шагов).
fn run_adaptive(
    solver: &mut AdaptiveCableSolver,
    pulse_on: f32,
    pulse_off: f32,
    i_amp: f32,
    tstop: f32,
) -> (f32, f32, u32) {
    let soma = solver.root();
    let n = solver.len();
    let mut i_inj = vec![0.0f32; n];
    let mut t = 0.0f32;
    let mut min_dt = f32::INFINITY;
    let mut max_dt = 0.0f32;
    let mut steps = 0u32;
    while t < tstop - 1e-6 {
        i_inj.fill(0.0);
        i_inj[soma] = if t >= pulse_on && t < pulse_off {
            i_amp
        } else {
            0.0
        };
        let dt = solver.step(&i_inj);
        t += dt;
        min_dt = min_dt.min(dt);
        max_dt = max_dt.max(dt);
        steps += 1;
    }
    (min_dt, max_dt, steps)
}

/// Пассивный ball-and-stick с адаптивным шагом: финальные V(сома)/V(кончик)
/// совпадают с фиксированным эталоном NEURON (t=100 → -64.6701 мВ), а dt
/// реально меняется (сжатие на фронте импульса, рост на пологих участках).
#[test]
fn adaptive_passive_cable_matches_neuron() {
    let m = MorphoIR {
        id: "bs".into(),
        segments: vec![
            seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
            seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
        ],
        segment_groups: vec![],
    };
    let params = CableParams {
        ra_ohm_cm: 100.0,
        cm_uf_per_cm2: 1.0,
        rm_ohm_cm2: 20_000.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let base = ActiveCableSolver::new(tree, -65.0, 0.025, 6.3);
    let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 1.0, 1e-3, 1e-3);

    let (min_dt, max_dt, steps) = run_adaptive(&mut solver, 10.0, 60.0, 0.01, 100.0);
    let v_soma = solver.v(0);
    let v_tip = solver.v(10);
    println!(
            "adaptive passive: min_dt={min_dt:.4} max_dt={max_dt:.4} steps={steps} v_soma={v_soma:.6} v_tip={v_tip:.6}"
        );
    assert!((v_soma + 64.670114).abs() < 0.01, "v_soma={v_soma}");
    assert!((v_tip + 64.670114).abs() < 0.01, "v_tip={v_tip}");
    assert!(
        min_dt < max_dt,
        "dt должен адаптироваться: {min_dt} vs {max_dt}"
    );
}

/// Переменный порядок (CVODE-стиль): солвер стартует на BDF1, после
/// принятых шагов с запасом поднимается до BDF2 (Richardson-экстраполяция),
/// а на фронте импульса (отказ шага) сбрасывается обратно на BDF1.
#[test]
fn adaptive_variable_order_transitions() {
    // Прогон протокола с фиксированным допуском; возвращает (v_soma,
    // min_order, max_order, была ли демоция порядка после отказа шага).
    fn run(tol: f32) -> (f32, u8, u8, bool) {
        let m = MorphoIR {
            id: "bs".into(),
            segments: vec![
                seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0),
                seg(1, Some(0), 0.0, 20.0, 0.0, 2.0, 0.0, 1020.0, 0.0, 2.0),
            ],
            segment_groups: vec![],
        };
        let params = CableParams {
            ra_ohm_cm: 100.0,
            cm_uf_per_cm2: 1.0,
            rm_ohm_cm2: 20_000.0,
            d_lambda: 0.1,
        };
        let tree = discretize(&m, &params).unwrap();
        let base = ActiveCableSolver::new(tree, -65.0, 0.025, 6.3);
        let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 1.0, tol, tol);

        let soma = solver.root();
        let n = solver.len();
        let mut i_inj = vec![0.0f32; n];
        let mut t = 0.0f32;
        let mut min_order = u8::MAX;
        let mut max_order = 0u8;
        let mut saw_rejection_restart = false;
        let mut prev_order = 1u8;
        while t < 100.0 - 1e-6 {
            i_inj.fill(0.0);
            // Резкий скачок 0 → 0.5 nA: жёсткий фронт, который должен
            // отклонить BDF2-шаг и сбросить порядок на BDF1.
            i_inj[soma] = if (10.0..60.0).contains(&t) { 0.5 } else { 0.0 };
            let dt = solver.step(&i_inj);
            t += dt;
            let order = solver.order();
            // Демоция: порядок упал 2 → 1 после шага (отказ на фронте).
            if prev_order == 2 && order == 1 {
                saw_rejection_restart = true;
            }
            prev_order = order;
            min_order = min_order.min(order);
            max_order = max_order.max(order);
        }
        (solver.v(0), min_order, max_order, saw_rejection_restart)
    }

    // Основной прогон: рабочий допуск 1e-3.
    let (v_soma, min_order, max_order, saw_rejection_restart) = run(1e-3);
    // Эталон: жёсткий допуск 1e-5 — практически сошёлся.
    let (v_ref, _, _, _) = run(1e-5);
    println!(
            "adaptive order: min={min_order} max={max_order} rejection_restart={saw_rejection_restart} v_soma={v_soma:.6} v_ref={v_ref:.6}"
        );
    // Порядок реально поднимался до 2 (и стартовал с 1 — до первого
    // принятого шага промоция невозможна).
    assert_eq!(max_order, 2, "солвер должен достигать BDF2");
    // На фронте импульса (жёсткий скачок) шаг отклоняется → рестарт BDF1.
    assert!(
        saw_rejection_restart,
        "фронт импульса должен вызвать отказ шага и рестарт порядка"
    );
    // Переключение порядка не теряет точность: на сильном импульсе
    // (размах ~100 мВ) локальный контроль ошибки даёт накопление
    // ~tol-масштаба; против сошедшегося эталона (1e-5) — < 2 мВ.
    assert!(
        (v_soma - v_ref).abs() < 2.0,
        "v_soma={v_soma} vs v_ref={v_ref}"
    );
}

/// Активный HH с адаптивным шагом: то же число спайков (8), что у фикс.
/// dt, и dt сжимается на спайках (min_dt < max_dt).
#[test]
fn adaptive_active_hh_spike_count() {
    let std = standard_channels();
    let m = MorphoIR {
        id: "s".into(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let area_cm2 = tree.compartments[0].area_um2 * 1e-8;
    let mut base = ActiveCableSolver::new(tree, -54.3, 0.025, 6.3);
    base.set_channels(
        0,
        vec![
            active_from_spec(&std[0], area_cm2, -65.0),
            active_from_spec(&std[1], area_cm2, -65.0),
        ],
    );
    base.set_v(0, -65.0);
    let mut solver = AdaptiveCableSolver::new(base, 0.025, 0.001, 0.1, 1e-3, 1e-3);

    let soma = solver.root();
    let n = solver.len();
    let mut i_inj = vec![0.0f32; n];
    let mut t = 0.0f32;
    let mut min_dt = f32::INFINITY;
    let mut max_dt = 0.0f32;
    let mut spikes = 0u32;
    let mut was_below = true;
    while t < 100.0 - 1e-6 {
        i_inj.fill(0.0);
        i_inj[soma] = if (10.0..90.0).contains(&t) { 0.3 } else { 0.0 };
        let dt = solver.step(&i_inj);
        t += dt;
        min_dt = min_dt.min(dt);
        max_dt = max_dt.max(dt);
        let v = solver.v(soma);
        if v >= 0.0 && was_below {
            spikes += 1;
        }
        was_below = v < 0.0;
    }
    println!("adaptive active: spikes={spikes} min_dt={min_dt:.4} max_dt={max_dt:.4}");
    assert!((spikes as i32 - 8).abs() <= 2, "spikes={spikes}");
    assert!(
        min_dt < max_dt,
        "dt должен адаптироваться: {min_dt} vs {max_dt}"
    );
}

// ─── Stochastic channels (`channelPopulation`) ────────────

/// `sample_binomial` — точный сэмплер: среднее ≈ n·p, дисперсия ≈ n·p·(1−p),
/// и детерминирован по сиду (тот же сид → та же последовательность).
#[test]
fn sample_binomial_mean_variance_and_determinism() {
    let n = 64usize;
    let p = 0.5f32;
    let draws = 100_000usize;

    let mut rng = StdRng::seed_from_u64(0xB00B5);
    let mut sum = 0u64;
    let mut sum_sq = 0u64;
    for _ in 0..draws {
        let k = sample_binomial(&mut rng, n, p) as u64;
        sum += k;
        sum_sq += k * k;
    }
    let mean = sum as f64 / draws as f64;
    let var = sum_sq as f64 / draws as f64 - mean * mean;
    let expect_mean = n as f64 * f64::from(p);
    let expect_var = n as f64 * f64::from(p) * (1.0 - f64::from(p));
    assert!((mean - expect_mean).abs() < 0.2, "mean = {mean}");
    assert!((var - expect_var).abs() < 0.5, "var = {var}");

    // Детерминизм: два генератора с одним сидом дают одну последовательность.
    let mut a = StdRng::seed_from_u64(42);
    let mut b = StdRng::seed_from_u64(42);
    let seq_a: Vec<usize> = (0..64).map(|_| sample_binomial(&mut a, n, p)).collect();
    let seq_b: Vec<usize> = (0..64).map(|_| sample_binomial(&mut b, n, p)).collect();
    assert_eq!(seq_a, seq_b);
}

/// Stochastic-канал с p_open=0.5 и N=64: среднее открытой доли ≈ 0.5 (шум
/// реально сэмплируется), разные сиды дают разные трейны.
#[test]
fn stochastic_channel_sampling_is_noisy_and_reproducible() {
    // 2-состоянийный gate с steadyState=0.5 в v_rest. Конвенция движка —
    // аргумент `-(v + midpoint)/scale`, поэтому half-activation при v=-65
    // даёт midpoint = +65 (НЕ -65).
    let gate = gate_kinetics_to_markov(
        GateKinetics::TauInf {
            steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: 65.0,
                scale: 10.0,
            }),
            tau: TauForm::Fixed(1.0),
        },
        1,
    );
    // v_rest = -65 → inf = 1/(1+exp(0)) = 0.5.
    let mut ch = StochasticMarkovActiveChannel::new(
        vec![gate.clone()],
        vec![1.0],
        0.0,
        64.0,
        64,
        -65.0,
        1234,
    );
    let steps = 100_000usize;
    let mut sum = 0.0f64;
    for _ in 0..steps {
        sum += f64::from(ch.step_open(-65.0, 0.01));
    }
    let mean = sum / steps as f64;
    assert!((mean - 0.5).abs() < 0.01, "mean open fraction = {mean}");

    // Два канала с разными сидами — разные последовательности.
    let mut c1 =
        StochasticMarkovActiveChannel::new(vec![gate.clone()], vec![1.0], 0.0, 64.0, 64, -65.0, 1);
    let mut c2 = StochasticMarkovActiveChannel::new(vec![gate], vec![1.0], 0.0, 64.0, 64, -65.0, 2);
    let mut diff = 0;
    for _ in 0..64 {
        let f1 = c1.step_open(-65.0, 0.01);
        let f2 = c2.step_open(-65.0, 0.01);
        if (f1 - f2).abs() > 1e-6 {
            diff += 1;
        }
    }
    assert!(diff > 0, "разные сиды должны давать разный шум");
}

/// Nernst-обращение: [Ca]int=0.1 µM / [Ca]ext=2 mM при 6.3°C → ≈ +119 мВ
/// (канонический Ca-реверс). `ca_ext <= 0` → 0 (LEMS guard).
#[test]
fn nernst_reversal_sanity() {
    let e = nernst_reversal_mv(1e-4, 2.0, 6.3 + 273.15);
    assert!((e - 119.2).abs() < 1.0, "nernst e_rev = {e}");
    assert_eq!(nernst_reversal_mv(1e-4, 0.0, 300.0), 0.0);
}

/// decayingPool: flux-масштаб сферической оболочки сходится к аналитике, а
/// ODE выходит на steady state `rest + iCa·flux·tau` (постоянный 1 nA).
#[test]
fn decaying_pool_steady_state_and_relaxation() {
    // Сферическая сома R=10 мкм (площадь 4πR²), оболочка 0.1 мкм.
    let area_um2 = 4.0 * PI * 10.0 * 10.0;
    let flux = decaying_pool_flux_scale(area_um2, 0.1);
    // V_shell = (4π/3)(R³−(R−d)³) → flux ≈ 0.0417 мМ/мс на нА (аналитика).
    assert!((flux - 0.0417).abs() < 1e-3, "flux = {flux}");

    let pool = CaPool {
        resting_mm: 1e-4,
        tau_ms: 100.0,
        flux_scale: flux,
    };
    let i_ca = 1.0f32; // нА
    let steady = 1e-4 + flux * 100.0 * i_ca;
    let mut ca = 1e-4f32;
    for _ in 0..5000 {
        ca = pool.step(ca, i_ca, 0.1); // 500 мс суммарно = 5τ
    }
    // 5τ → e^-5 ≈ 0.007, т.е. в пределах ~0.03 мМ от steady.
    assert!((ca - steady).abs() < 0.05, "ca = {ca}, steady = {steady}");

    // Без тока [Ca] возвращается к покою (релаксация), и не уходит < 0.
    let mut ca2 = 4.0f32;
    for _ in 0..5000 {
        ca2 = pool.step(ca2, 0.0, 0.1);
    }
    assert!((ca2 - 1e-4).abs() < 0.05, "decay to rest: ca = {ca2}");
}

/// Ca²⁺-подсистема (Nernst + decaying pool) против NEURON `cacum.mod` +
/// ohmic Ca-канал (`cachan_simple.mod`).
///
/// Единственная сферическая сома (diam = L = 20 µm, area = 1256.6 µm²) с
/// пассивной утечкой (0.3 mS/cm² → −65 mV), всегда-открытым ohmic Ca-каналом
/// (`gbar = 1 mS/cm²`, `e_rev = Nernst(ca_ext/ca_int)`) и тонкой оболочкой
/// (`decayingPool`, depth = 0.1 µm, tau = 100 ms, cai0 = 1e-4 mM).
///
/// Эталон `REF_V_CA`/`REF_CAI_CA` генерится `gen_ca_ref.py` (NEURON
/// `secondorder=0` — backward Euler, как Hines-солвер). Единственная
/// осознанная расходимость — геометрия оболочки: NEURON `cacum.mod`
/// считает объём как `area·depth` (тонкая оболочка), а LEMS
/// `decayingPoolConcentrationModel` — точную сферическую оболочку
/// `(4π/3)(R³−(R−d)³)`. При depth/R = 0.1/10 = 1e-2 объёмы различаются на
/// ~1% (flux-масштаб pillars на ~1% больше) — отсюда допуск по [Ca] 1.5%
/// и по V 0.5 мВ (измерено: max rel |ΔCa| ≈ 0.9%, max |ΔV| ≈ 0.08 мВ).
#[test]
fn ca_subsystem_matches_neuron_cacum() {
    const REF_V_CA: [f32; 8] = [
        -65.000_000,
        0.379_119,
        15.061_532,
        14.028_261,
        9.739_405,
        5.033_044,
        1.313_686,
        -0.889_314,
    ];
    const REF_CAI_CA: [f32; 8] = [
        0.000_100, 0.042_215, 0.067_031, 0.095_532, 0.146_739, 0.236_989, 0.349_230, 0.440_467,
    ];

    let m = MorphoIR {
        id: "ca_soma".into(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    // pas: 0.3 mS/cm² → Rm = 3333.33 Ω·cm², e_pas = −65 мВ.
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let soma = tree.root;
    let area_um2 = tree.compartments[soma].area_um2;
    let area_cm2 = area_um2 * 1e-8;

    let mut solver = ActiveCableSolver::new(tree, -65.0, 0.05, 6.3);
    solver.set_v(soma, -65.0);
    // Всегда-открытый ohmic Ca-канал (пустые gates → fopen = 1).
    let g_max_us = 1.0e-3 * area_cm2 * 1e6;
    solver.set_nernst_channels(
        soma,
        vec![NernstActiveChannel::new(
            vec![],
            vec![],
            2.0,
            g_max_us,
            -65.0,
        )],
    );
    let pool = CaPool {
        resting_mm: 1e-4,
        tau_ms: 100.0,
        flux_scale: decaying_pool_flux_scale(area_um2, 0.1),
    };
    solver.set_ca_pool(soma, pool, 1e-4);

    let dt = 0.05f32;
    let times = [0.0f32, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0];
    let steps_at: Vec<usize> = times.iter().map(|t| (t / dt).round() as usize).collect();
    let mut v_ours = vec![solver.v(soma)];
    let mut ca_ours = vec![solver.ca(soma)];
    let zero = [0.0f32; 1];
    for step in 1..=1000usize {
        solver.step(&zero);
        if steps_at.contains(&step) {
            v_ours.push(solver.v(soma));
            ca_ours.push(solver.ca(soma));
        }
    }

    let mut dv = 0.0f32;
    let mut dca_rel = 0.0f32;
    for k in 0..8 {
        dv = dv.max((v_ours[k] - REF_V_CA[k]).abs());
        dca_rel = dca_rel.max((ca_ours[k] - REF_CAI_CA[k]).abs() / REF_CAI_CA[k].max(1e-6));
    }
    println!("ca subsystem vs NEURON: max |ΔV| = {dv:.6} mV, max rel |ΔCa| = {dca_rel:.4}");
    assert!(dv < 0.5, "V: max |ΔV| = {dv:.4} mV");
    assert!(dca_rel < 0.015, "[Ca]: max rel err = {dca_rel:.4}");
}

/// `channelPopulationNernst` mean-field: среднее по ансамблю стохастических
/// Ca-каналов сходится к детерминистическому `NernstActiveChannel` (тот же
/// p_open, та же суммарная проводимость `single_cond·N`). Транзитивно это
/// и NEURON-parity: детерминистический предел — тот самый
/// `channelDensityNernst`, что сверен в `ca_subsystem_matches_neuron_cacum`.
/// Канальный шум ∝ 1/√N усредняется по N_SEEDS независимым seed'ам.
#[test]
fn channel_population_nernst_mean_matches_density() {
    const N_CHANNELS: usize = 2048;
    const N_SEEDS: usize = 16;

    let m = MorphoIR {
        id: "ca_soma".into(),
        segments: vec![seg(0, None, 0.0, 0.0, 0.0, 20.0, 0.0, 20.0, 0.0, 20.0)],
        segment_groups: vec![],
    };
    let params = CableParams {
        rm_ohm_cm2: 3333.333,
        cm_uf_per_cm2: 1.0,
        ra_ohm_cm: 100.0,
        d_lambda: 0.1,
    };
    let tree = discretize(&m, &params).unwrap();
    let soma = tree.root;
    let area_um2 = tree.compartments[soma].area_um2;
    let area_cm2 = area_um2 * 1e-8;
    let g_max_us = 1.0e-3 * area_cm2 * 1e6;

    // Один и тот же сигмоидный gate в двух представлениях (HH-кинетика для
    // детерминистического канала, Markov-схема для стохастического).
    let kinetics = GateKinetics::TauInf {
        steady_state: VarForm::Closed(ClosedForm::Sigmoid {
            rate: 1.0,
            midpoint: -100.0,
            scale: 10.0,
        }),
        tau: TauForm::Fixed(0.1),
    };
    let markov_gate = gate_kinetics_to_markov(kinetics.clone(), 1);
    let pool = CaPool {
        resting_mm: 1e-4,
        tau_ms: 100.0,
        flux_scale: decaying_pool_flux_scale(area_um2, 0.1),
    };

    let dt = 0.05f32;
    let times = [0.0f32, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0];
    let steps_at: Vec<usize> = times.iter().map(|t| (t / dt).round() as usize).collect();
    let zero = [0.0f32; 1];

    // Детерминистический эталон (NernstActiveChannel, тот же gate).
    let mut det = ActiveCableSolver::new(tree.clone(), -65.0, 0.05, 6.3);
    det.set_v(soma, -65.0);
    det.set_nernst_channels(
        soma,
        vec![NernstActiveChannel::new(
            vec![(kinetics, 1)],
            vec![1.0],
            2.0,
            g_max_us,
            -65.0,
        )],
    );
    det.set_ca_pool(soma, pool, 1e-4);
    let mut det_v = vec![det.v(soma)];
    let mut det_ca = vec![det.ca(soma)];
    for step in 1..=1000usize {
        det.step(&zero);
        if steps_at.contains(&step) {
            det_v.push(det.v(soma));
            det_ca.push(det.ca(soma));
        }
    }

    // Среднее по независимым seed'ам стохастического канала.
    let base_seed = 0xCA11_CAF0_0DBA_D00Du64;
    let mut acc_v = [0.0f64; 8];
    let mut acc_ca = [0.0f64; 8];
    for seed in 0..N_SEEDS {
        let mut solver = ActiveCableSolver::new(tree.clone(), -65.0, 0.05, 6.3);
        solver.set_v(soma, -65.0);
        solver.set_stochastic_nernst_channels(
            soma,
            vec![StochasticNernstActiveChannel::new(
                vec![markov_gate.clone()],
                vec![1.0],
                2.0,
                g_max_us,
                N_CHANNELS,
                -65.0,
                base_seed.wrapping_add(seed as u64),
            )],
        );
        solver.set_ca_pool(soma, pool, 1e-4);
        acc_v[0] += solver.v(soma) as f64;
        acc_ca[0] += solver.ca(soma) as f64;
        let mut k = 0usize;
        for step in 1..=1000usize {
            solver.step(&zero);
            if steps_at.contains(&step) {
                k += 1;
                acc_v[k] += solver.v(soma) as f64;
                acc_ca[k] += solver.ca(soma) as f64;
            }
        }
    }

    let nf = N_SEEDS as f64;
    let mut dv = 0.0f64;
    let mut dca_rel = 0.0f64;
    for k in 0..8 {
        dv = dv.max((acc_v[k] / nf - det_v[k] as f64).abs());
        dca_rel =
            dca_rel.max((acc_ca[k] / nf - det_ca[k] as f64).abs() / det_ca[k].max(1e-6) as f64);
    }
    println!(
            "channelPopulationNernst mean vs density: max |ΔV| = {dv:.6} mV, max rel |ΔCa| = {dca_rel:.4}"
        );
    assert!(dv < 0.1, "V: mean-field max |ΔV| = {dv:.4} mV");
    assert!(
        dca_rel < 0.01,
        "[Ca]: mean-field max rel err = {dca_rel:.4}"
    );
}
