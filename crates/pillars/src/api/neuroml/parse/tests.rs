use super::*;
use crate::kernels::hh_channels::{
    k_n_alpha, k_n_beta, na_h_alpha, na_h_beta, na_m_alpha, na_m_beta, ClosedForm, GateKinetics,
    MarkovRate, TauForm, VarForm,
};

/// Реальный Ih-канал Poirazi 2003 (`h.channel.nml` из OSB): кастомные
/// LEMS-`ComponentType` `Poirazi_h_tau` (piecewise double-exponential tau)
/// и `Poirazi_h_ss` (Boltzmann-подобный) — нижются в [`LemsExpr`].
const POIRAZI_H_CHANNEL: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="NeuroML_ionChannel">
  <ionChannel id="hs" conductance="1pS" type="ionChannelHH" species="na">
    <gateHHtauInf id="n" instances="1">
      <timeCourse type="Poirazi_h_tau"/>
      <steadyState type="Poirazi_h_ss" midpoint="-82mV" scale="8.8mV"/>
    </gateHHtauInf>
  </ionChannel>
  <ComponentType name="Poirazi_h_tau" extends="baseVoltageDepTime">
    <Constant name="TIME_SCALE" dimension="time" value="1 ms"/>
    <Constant name="VOLT_SCALE" dimension="voltage" value="1 mV"/>
    <Dynamics>
      <DerivedVariable name="V" dimension="none" value="v / VOLT_SCALE"/>
      <DerivedVariable name="tmp" value="2 * (1 / (exp((V+145)/-17.5) + exp((V+16.8)/16.5)) + 5)" dimension="none"/>
      <ConditionalDerivedVariable name="t" dimension="time" exposure="t" >
        <Case condition="V .lt. -30" value="tmp * TIME_SCALE"/>
        <Case value="1 * TIME_SCALE"/>
      </ConditionalDerivedVariable>
    </Dynamics>
  </ComponentType>
  <ComponentType name="Poirazi_h_ss" extends="baseVoltageDepVariable">
    <Parameter name="midpoint" dimension="voltage"/>
    <Parameter name="scale" dimension="voltage"/>
    <Constant name="VOLT_SCALE" dimension="voltage" value="1 mV"/>
    <Dynamics>
      <DerivedVariable name="V" dimension="none" value="v / VOLT_SCALE"/>
      <DerivedVariable name="x" exposure="x" dimension="none" value="1 - (1 / (1 + exp(0 - (V-midpoint/VOLT_SCALE)/(scale/VOLT_SCALE))))"/>
    </Dynamics>
  </ComponentType>
</neuroml>
"#;

/// Кастомный LEMS-`ComponentType` в `gateHHtauInf` (реальный Ih Poirazi
/// 2003): `timeCourse`/`steadyState` с нестандартным `type` резолвятся в
/// [`TauForm::Expr`]/[`VarForm::Expr`], и выражения совпадают с аналитикой
/// (в т.ч. piecewise-ветка V≥−30 → tau = 1).
#[test]
fn custom_component_type_gate_lowers_to_lems_expr() {
    let ir = parse(POIRAZI_H_CHANNEL).unwrap();
    assert!(ir.warnings.is_empty(), "warnings: {:?}", ir.warnings);
    let ch = ir.channels.iter().find(|c| c.id == "hs").unwrap();
    let g = ch.gates[0].kinetics.clone().unwrap();
    let GateKinetics::TauInf { steady_state, tau } = g else {
        panic!("hs gate должен быть TauInf")
    };

    // steadyState: Poirazi_h_ss = 1/(1+exp((v-midpoint)/scale)), midpoint -82, scale 8.8.
    let VarForm::Expr(ss) = steady_state else {
        panic!("Poirazi_h_ss должен быть кастомным выражением")
    };
    let ss90 = ss.eval(-90.0);
    let want = 1.0f32 / (1.0f32 + ((-90.0f32 + 82.0) / 8.8).exp());
    assert!((ss90 - want).abs() < 1e-5, "ss(-90) = {ss90}, want {want}");

    // timeCourse: Poirazi_h_tau — piecewise: V < -30 → double-exp, else 1.
    let TauForm::Expr(tau) = tau else {
        panic!("Poirazi_h_tau должен быть кастомным выражением")
    };
    let tau90 = tau.eval(-90.0);
    let want_tau = 2.0f32
        * (1.0f32 / (((-90.0f32 + 145.0) / -17.5).exp() + ((-90.0f32 + 16.8) / 16.5).exp())
            + 5.0f32);
    assert!(
        (tau90 - want_tau).abs() < 1e-4,
        "tau(-90) = {tau90}, want {want_tau}"
    );
    assert!(
        (tau.eval(-20.0) - 1.0).abs() < 1e-6,
        "tau(-20) должен быть 1"
    );
}

/// Канонический кальмаровый аксон в NeuroML2: Na (m³h) + K (n⁴) через
/// `ionChannelHH`/`gateHHrates`. Значения rate/midpoint/scale — точная
/// NeuroML-кодировка `na_m_alpha` и т.д. (якорь round-trip из gap-анализа).
const SQUID_CHANNELS: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="squid_channels">
  <ionChannelHH id="naChans" conductance="10pS" species="na">
    <gateHHrates id="m" instances="3">
      <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
    </gateHHrates>
    <gateHHrates id="h" instances="1">
      <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
      <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
    </gateHHrates>
  </ionChannelHH>
  <ionChannelHH id="kChans" conductance="10pS" species="k">
    <gateHHrates id="n" instances="4">
      <forwardRate type="HHExpLinearRate" rate="0.1per_ms" midpoint="-55mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="0.125per_ms" midpoint="-65mV" scale="-80mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;

fn channel<'a>(ir: &'a Ir, id: &str) -> &'a IrChannel {
    ir.channels
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("channel `{id}` not parsed"))
}

fn gate_kinetics(ch: &IrChannel, id: &str) -> GateKinetics {
    ch.gates
        .iter()
        .find(|g| g.id == id)
        .and_then(|g| g.kinetics.clone())
        .unwrap_or_else(|| panic!("gate `{id}` not parsed (or not compilable)"))
}

/// m/h/n rate-функции round-trip бит-в-бит против канонических скалярных
/// функций движка (na_m_alpha и т.д.) — якорь из gap-анализа §4.4.
#[test]
fn squid_ion_channel_rates_roundtrip_bit_exact() {
    let ir = parse(SQUID_CHANNELS).unwrap();
    assert_eq!(ir.channels.len(), 2);
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );

    let m = gate_kinetics(channel(&ir, "naChans"), "m");
    let h = gate_kinetics(channel(&ir, "naChans"), "h");
    let n = gate_kinetics(channel(&ir, "kChans"), "n");

    let vs = [
        -120.0f32, -100.0, -80.0, -65.0, -55.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0,
    ];
    for &v in &vs {
        let (a, b) = m.alpha_beta(v);
        assert_eq!(a.to_bits(), na_m_alpha(v).to_bits(), "m alpha @ {v}");
        assert_eq!(b.to_bits(), na_m_beta(v).to_bits(), "m beta @ {v}");
        let (a, b) = h.alpha_beta(v);
        assert_eq!(a.to_bits(), na_h_alpha(v).to_bits(), "h alpha @ {v}");
        assert_eq!(b.to_bits(), na_h_beta(v).to_bits(), "h beta @ {v}");
        let (a, b) = n.alpha_beta(v);
        assert_eq!(a.to_bits(), k_n_alpha(v).to_bits(), "n alpha @ {v}");
        assert_eq!(b.to_bits(), k_n_beta(v).to_bits(), "n beta @ {v}");
    }
}

/// `gateHHtauInf` (km/HCN-стиль) парсится без warning в каноническое
/// `GateKinetics::TauInf` (steadyState + фиксированный tau).
#[test]
fn gate_hh_tau_inf_parses_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hcn_test">
  <ionChannelHH id="hcn" conductance="10pS" species="hcn">
    <gateHHtauInf id="m" instances="1">
      <timeCourse type="fixedTimeCourse" tau="10ms"/>
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
    </gateHHtauInf>
  </ionChannelHH>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    assert_eq!(ir.channels.len(), 1);
    let g = gate_kinetics(&ir.channels[0], "m");
    assert_eq!(
        g,
        GateKinetics::TauInf {
            steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: 80.0,
                scale: 10.0
            }),
            tau: TauForm::Fixed(10.0),
        }
    );
}

/// `gateHHInstantaneous` распознаётся (не «молча скипается»), но его
/// кинетика ещё не компилируется движком → `kinetics = None` + честный kind.
#[test]
fn gate_hh_instantaneous_recognised_but_not_compilable() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="inst_test">
  <ionChannelHH id="ca" conductance="10pS" species="ca">
    <gateHHInstantaneous id="m" instances="2">
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-20mV" scale="5mV"/>
    </gateHHInstantaneous>
  </ionChannelHH>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    let ch = &ir.channels[0];
    assert_eq!(ch.gates[0].kind, "gateHHInstantaneous");
    // Algebraic gate — compiles to `Instantaneous` kinetics (no ODE).
    let kinetics = ch.gates[0].kinetics.clone().expect("compiles now");
    assert!(matches!(kinetics, GateKinetics::Instantaneous { .. }));
}

/// q10Settings (q10ExpTemp) парсится без ошибок.
#[test]
fn q10_settings_parse() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="q10_test">
  <ionChannelHH id="hcn" conductance="10pS" species="hcn">
    <gateHHrates id="m" instances="1">
      <q10Settings type="q10ExpTemp" q10Factor="3" experimentalTemp="6.3degC"/>
      <forwardRate type="HHExpRate" rate="0.01per_ms" midpoint="-70mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="0.01per_ms" midpoint="-70mV" scale="-10mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    let ch = &ir.channels[0];
    assert_eq!(ch.gates[0].kind, "gateHHrates");
    assert!(matches!(
        ch.gates[0].kinetics,
        Some(GateKinetics::Rates { .. })
    ));
    // q10 хранится на гейте (парсится, но движком пока не применяется).
    assert!(matches!(ch.gates[0].q10, Some(IrQ10::ExpTemp { .. })));
}

/// Гварды: нулевой `scale` и неположительные `instances` должны падать с
/// понятной ошибкой, а не молча порождать inf/NaN или обёрнутый экспонент.
#[test]
fn rejects_zero_scale_and_bad_instances() {
    // scale = 0 → деление на ноль в замкнутой форме.
    let zero_scale = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">
  <ionChannelHH id="ch" conductance="10pS" species="na">
    <gateHHrates id="m" instances="3">
      <forwardRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="0mV"/>
      <reverseRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-1mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
    assert!(parse(zero_scale).is_err(), "scale=0 должен давать ошибку");

    // instances = 0 → не PositiveInteger.
    let zero_instances = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">
  <ionChannelHH id="ch" conductance="10pS" species="na">
    <gateHHrates id="m" instances="0">
      <forwardRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      <reverseRate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-10mV"/>
    </gateHHrates>
  </ionChannelHH>
</neuroml>
"#;
    assert!(
        parse(zero_instances).is_err(),
        "instances=0 должен давать ошибку"
    );
}

/// `ionChannelKS`/`gateKS` (2-состояния closed/open + forward/reverse
/// transitions) парсится в марковскую схему без warning, со скоростями
/// `Directed`, совпадающими по битам с HH-формами.
#[test]
fn gate_ks_two_state_parses_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_test">
  <ionChannelKS id="na_ks" conductance="10pS" species="na">
    <gateKS id="m" instances="3">
      <closedState id="c1"/>
      <openState id="o1"/>
      <forwardTransition from="c1" to="o1">
        <rate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    assert_eq!(ir.ks_channels.len(), 1);
    let gate = &ir.ks_channels[0].gates[0];
    assert_eq!(gate.n_states(), 2);
    assert_eq!(gate.open, vec![0.0, 1.0]);
    assert_eq!(gate.power, 3);
    assert_eq!(gate.transitions.len(), 2);
    // forward: closed(0) → open(1) со скоростью, бит-в-бит равной
    // `ClosedForm::ExpLinear { rate: 0.1, midpoint: 40, scale: 10 }`
    // (та же кодировка, что у `forwardRate` в `gateHHrates`).
    let fwd = &gate.transitions[0];
    assert_eq!(fwd.from, 0);
    assert_eq!(fwd.to, 1);
    assert_eq!(
        fwd.rate,
        MarkovRate::Directed {
            rate: ClosedForm::ExpLinear {
                rate: 0.1,
                midpoint: 40.0,
                scale: 10.0
            }
        }
    );
    // reverse: open(1) → closed(0).
    let rev = &gate.transitions[1];
    assert_eq!(rev.from, 1);
    assert_eq!(rev.to, 0);
    assert_eq!(
        rev.rate,
        MarkovRate::Directed {
            rate: ClosedForm::Exp {
                rate: 4.0,
                midpoint: 65.0,
                scale: 18.0
            }
        }
    );
}

/// `tauInfTransition` раскрывается в две направленные ноги
/// (forward `inf/tau`, reverse `(1−inf)/tau`).
#[test]
fn gate_ks_tau_inf_expands_to_forward_reverse() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_tau">
  <ionChannelKS id="hcn_ks" conductance="10pS" species="hcn">
    <gateKS id="h" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="10ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    let gate = &ir.ks_channels[0].gates[0];
    assert_eq!(gate.transitions.len(), 2);
    let fwd = &gate.transitions[0];
    assert_eq!((fwd.from, fwd.to), (0, 1));
    assert_eq!(
        fwd.rate,
        MarkovRate::TauInfForward {
            steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: 80.0,
                scale: 10.0
            }),
            tau: TauForm::Fixed(10.0),
        }
    );
    let rev = &gate.transitions[1];
    assert_eq!((rev.from, rev.to), (1, 0));
    assert_eq!(
        rev.rate,
        MarkovRate::TauInfReverse {
            steady_state: VarForm::Closed(ClosedForm::Sigmoid {
                rate: 1.0,
                midpoint: 80.0,
                scale: 10.0
            }),
            tau: TauForm::Fixed(10.0),
        }
    );
}

/// Переход на неизвестное состояние — честная ошибка, а не молчаливый
/// дроп ребра (который тихо сломал бы master equation).
#[test]
fn gate_ks_unknown_state_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks_bad">
  <ionChannelKS id="ch" conductance="10pS" species="na">
    <gateKS id="m" instances="1">
      <closedState id="c1"/>
      <openState id="o1"/>
      <forwardTransition from="c1" to="nope">
        <rate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="1per_ms" midpoint="-40mV" scale="-10mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
    assert!(
        parse(xml).is_err(),
        "неизвестное состояние должно давать ошибку"
    );
}

/// `gateFractional` с двумя субгейтами парсится без warning в
/// [`FractionalGateSpec`] (веса + tau-inf субгейты), не как HH-гейт.
#[test]
fn gate_fractional_parses_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="frac_test">
  <ionChannelHH id="ca" conductance="10pS" species="ca">
    <gateFractional id="m" instances="2">
      <subGate id="sg1" fractionalConductance="0.6">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-20mV" scale="5mV"/>
        <timeCourse type="fixedTimeCourse" tau="10ms"/>
      </subGate>
      <subGate id="sg2" fractionalConductance="0.4">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-40mV" scale="5mV"/>
        <timeCourse type="fixedTimeCourse" tau="20ms"/>
      </subGate>
    </gateFractional>
  </ionChannelHH>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    let ch = &ir.channels[0];
    assert!(
        ch.gates.is_empty(),
        "gateFractional не должен быть HH-гейтом"
    );
    assert_eq!(ch.fractional_gates.len(), 1);
    let fg = &ch.fractional_gates[0];
    assert_eq!(fg.power, 2);
    assert_eq!(fg.subgates.len(), 2);
    assert_eq!(fg.subgates[0].fractional_conductance, 0.6);
    assert_eq!(fg.subgates[0].tau, 10.0);
    assert_eq!(
        fg.subgates[0].steady_state,
        ClosedForm::Sigmoid {
            rate: 1.0,
            midpoint: 20.0,
            scale: 5.0
        }
    );
    assert_eq!(fg.subgates[1].fractional_conductance, 0.4);
    assert_eq!(fg.subgates[1].tau, 20.0);
}

/// Трёхсостоянийная кинетическая схема (closed ⇌ open ⇌ inactivated)
/// парсится без warning: 3 состояния, open-веса [0,1,0], 4 перехода.
#[test]
fn gate_ks_three_state_parses_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ks3_test">
  <ionChannelKS id="na3" conductance="10pS" species="na">
    <gateKS id="m" instances="3">
      <closedState id="c1"/>
      <openState id="o1"/>
      <closedState id="i1"/>
      <forwardTransition from="c1" to="o1">
        <rate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
      </forwardTransition>
      <reverseTransition from="c1" to="o1">
        <rate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
      </reverseTransition>
      <forwardTransition from="o1" to="i1">
        <rate type="HHExpRate" rate="0.2per_ms" midpoint="-65mV" scale="-20mV"/>
      </forwardTransition>
      <reverseTransition from="o1" to="i1">
        <rate type="HHExpRate" rate="0.01per_ms" midpoint="-35mV" scale="-10mV"/>
      </reverseTransition>
    </gateKS>
  </ionChannelKS>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    let gate = &ir.ks_channels[0].gates[0];
    assert_eq!(gate.n_states(), 3);
    assert_eq!(gate.open, vec![0.0, 1.0, 0.0]);
    assert_eq!(gate.power, 3);
    assert_eq!(gate.transitions.len(), 4);
    // c1(0) ⇌ o1(1) ⇌ i1(2).
    assert_eq!((gate.transitions[0].from, gate.transitions[0].to), (0, 1));
    assert_eq!((gate.transitions[1].from, gate.transitions[1].to), (1, 0));
    assert_eq!((gate.transitions[2].from, gate.transitions[2].to), (1, 2));
    assert_eq!((gate.transitions[3].from, gate.transitions[3].to), (2, 1));
}

// ─── Фаза 5a: морфология ─────────────────────────────────

/// Минимальная морфология: сома (корень) + дендрит (дочка), группы.
/// Парсится без warning; segmentParent резолвится в индекс родителя.
#[test]
fn morphology_segments_and_groups_parse_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morph_test">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="10" z="0" diameter="2"/>
      <distal x="0" y="100" z="0" diameter="2"/>
    </segment>
    <segmentGroup id="soma_group">
      <member segment="0"/>
    </segmentGroup>
    <segmentGroup id="dend_group">
      <member segment="1"/>
      <include segmentGroup="soma_group"/>
    </segmentGroup>
  </morphology>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    assert_eq!(ir.morphologies.len(), 1);
    let m = &ir.morphologies[0];
    assert_eq!(m.id, "m1");
    assert_eq!(m.segments.len(), 2);
    assert_eq!(m.segment_groups.len(), 2);

    // Сома — корень (нет родителя).
    let soma = &m.segments[0];
    assert_eq!(soma.id, 0);
    assert_eq!(soma.name.as_deref(), Some("soma"));
    assert!(soma.parent.is_none(), "сома должна быть корнем");
    assert_eq!(soma.distal.diameter, 20.0);

    // Дендрит — дочка сомы (fractionalLength по умолчанию 1.0).
    let dend = &m.segments[1];
    assert_eq!(dend.id, 1);
    let parent = dend.parent.as_ref().expect("дендрит должен иметь родителя");
    assert_eq!(parent.segment, 0);

    // Группы: soma_group = {0}, dend_group = {1} ∪ include soma_group.
    let soma_group = &m.segment_groups[0];
    assert_eq!(soma_group.id, "soma_group");
    assert_eq!(soma_group.members, vec![0]);
    let dend_group = &m.segment_groups[1];
    assert_eq!(dend_group.id, "dend_group");
    assert_eq!(dend_group.members, vec![1]);
    assert_eq!(dend_group.includes, vec!["soma_group"]);
}

/// `<segment>` без `<proximal>` наследует точку от родительского distal —
/// этот случай должен парситься (proximal = None), а не падать.
#[test]
fn segment_without_proximal_parses_with_none() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morph_prox">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <distal x="0" y="100" z="0" diameter="2"/>
    </segment>
  </morphology>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    let m = &ir.morphologies[0];
    assert!(m.segments[0].proximal.is_some(), "сома имеет proximal");
    assert!(
        m.segments[1].proximal.is_none(),
        "дендрит без proximal → None"
    );
}

/// `<cell>` + `<biophysicalProperties>` резолвятся по id: cell ссылается на
/// морфологию и биофизику; channelDensity/initMembPotential/resistivity
/// парсятся в IR без warning.
#[test]
fn cell_with_biophysical_properties_parses() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="cell_test">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="10" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group">
      <member segment="0"/>
    </segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific" segmentGroup="soma_group"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    assert!(
        ir.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ir.warnings
    );
    assert_eq!(ir.morpho_cells.len(), 1);
    let cell = &ir.morpho_cells[0];
    assert_eq!(cell.id, "c1");
    assert_eq!(cell.morphology, "m1");
    assert_eq!(cell.biophysics.as_deref(), Some("bp1"));

    let bp = ir.biophysics.get("bp1").expect("bp1 должен быть распарсен");
    assert_eq!(bp.membrane.channel_densities.len(), 1);
    let cd = &bp.membrane.channel_densities[0];
    assert_eq!(cd.ion_channel, "pas");
    assert_eq!(cd.erev, -70.0);
    assert_eq!(cd.segment_group, "soma_group");
    // 3.0 S_per_m2 → 3e-4 S/cm² (unit-aware, не raw strip).
    assert!(
        (cd.cond_density - 3e-4).abs() < 1e-9,
        "condDensity: {}",
        cd.cond_density
    );
    assert_eq!(bp.membrane.init_memb_potential, -65.0);
    assert_eq!(bp.membrane.specific_capacitance, 1.0);
    assert_eq!(bp.intracellular.resistivity, 100.0);
    // pas (ion non_specific) распознаётся как пассивный лик.
    assert_eq!(bp.passive_leak().map(|l| l.id.as_str()), Some("pasChans"));
}

/// `condDensity`/`erev` — обязательные атрибуты `channelDensity`;
/// отсутствие любого из них должно давать ошибку, а не NaN.
#[test]
fn channel_density_missing_erev_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad_cd">
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" ion="non_specific"/>
    </membraneProperties>
    <intracellularProperties/>
  </biophysicalProperties>
</neuroml>
"#;
    assert!(
        parse(xml).is_err(),
        "channelDensity без erev должен давать ошибку"
    );
}

/// `channelPopulation` и `channelPopulationNernst` парсятся
/// (number/ionChannel; у обычного — ещё `erev`, у Nernst — динамическое
/// обращение); single-channel conductance pS → µS.
#[test]
fn channel_population_and_nernst_parse() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="cp_test">
  <ionChannelKS id="kchan" conductance="625pS" species="k">
    <gateKS id="g" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-65mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="1ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelPopulation id="noisyK" ionChannel="kchan" number="64" erev="-40mV" ion="k"/>
      <channelPopulationNernst id="caPop" ionChannel="kchan" number="100" ion="ca"/>
    </membraneProperties>
    <intracellularProperties/>
  </biophysicalProperties>
</neuroml>
"#;
    let ir = parse(xml).unwrap();
    let bp = ir.biophysics.get("bp1").unwrap();
    assert_eq!(bp.membrane.channel_populations.len(), 1);
    let cp = &bp.membrane.channel_populations[0];
    assert_eq!(cp.id, "noisyK");
    assert_eq!(cp.ion_channel, "kchan");
    assert_eq!(cp.number, 64);
    assert_eq!(cp.erev, -40.0);
    // Nernst-популяция парсится отдельно (без `erev`, динамическое обращение).
    assert_eq!(bp.membrane.nernst_populations.len(), 1);
    let cpn = &bp.membrane.nernst_populations[0];
    assert_eq!(cpn.id, "caPop");
    assert_eq!(cpn.ion_channel, "kchan");
    assert_eq!(cpn.number, 100);
    // 625pS → 0.625 nS → 6.25e-4 µS.
    let ks = ir
        .ks_channels
        .iter()
        .find(|c| c.id == "kchan")
        .expect("kchan должен быть распарсен");
    assert!((ks.conductance.unwrap() - 6.25e-4).abs() < 1e-12);
    // Nernst-популяция больше НЕ warning (теперь поддерживается).
    assert!(
        !ir.warnings
            .iter()
            .any(|w| w.contains("channelPopulationNernst")),
        "channelPopulationNernst не должен warning: {:?}",
        ir.warnings
    );
}

/// `MorphoIR::resolve_segments`: `include`-цепи (dend → apical), `all`,
/// и ошибка на неизвестную группу.
#[test]
fn resolve_segments_resolves_includes_and_all() {
    let m = MorphoIR {
        id: "m".into(),
        segments: (0..3)
            .map(|i| Segment {
                id: i,
                name: None,
                proximal: None,
                distal: Point3d {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    diameter: 2.0,
                },
                parent: None,
            })
            .collect(),
        segment_groups: vec![
            SegmentGroup {
                id: "soma".into(),
                members: vec![0],
                includes: vec![],
            },
            SegmentGroup {
                id: "dend".into(),
                members: vec![1],
                includes: vec!["apical".into()],
            },
            SegmentGroup {
                id: "apical".into(),
                members: vec![2],
                includes: vec![],
            },
        ],
    };
    assert_eq!(m.resolve_segments("soma").unwrap(), vec![0]);
    assert_eq!(m.resolve_segments("dend").unwrap(), vec![1, 2]);
    assert_eq!(m.resolve_segments("all").unwrap(), vec![0, 1, 2]);
    assert!(m.resolve_segments("nope").is_err());
}

/// Конвенция Q10 (jNeuroML-семантика): `q10Fixed` якорится в 6.3 °C,
/// `q10ExpTemp` — в свой `experimentalTemp`; оба дают 1.0 в якоре, а
/// множитель — `q10^((T−T0)/10)`. Это ровно то, что NEURON-экспорт
/// jNeuroML вставляет в .mod (множитель в `rates()`), поэтому сверка
/// держится при любой температуре.
#[test]
fn ir_q10_factor_at_convention() {
    // q10Fixed = 2 при 6.3 °C → 1.0 (якорь); при 16.3 °C → 2.0.
    assert!((IrQ10::Fixed(2.0).factor_at(6.3) - 1.0).abs() < 1e-6);
    assert!((IrQ10::Fixed(2.0).factor_at(16.3) - 2.0).abs() < 1e-6);
    assert!((IrQ10::Fixed(2.0).factor_at(35.0) - 2.0f32.powf((35.0 - 6.3) / 10.0)).abs() < 1e-6);

    // q10ExpTemp: factor 2.3 @ 23 °C → 1.0 при 23 °C; при 35 °C →
    // 2.3^((35−23)/10) ≈ 2.71 (это M-ток Poirazi при физиологической
    // температуре, применяется в cable-пути).
    let km = IrQ10::ExpTemp {
        factor: 2.3,
        experimental_temp: 23.0,
    };
    assert!((km.factor_at(23.0) - 1.0).abs() < 1e-6);
    assert!((km.factor_at(35.0) - 2.3f32.powf(1.2)).abs() < 1e-5);
    assert!((km.factor_at(6.3) - 2.3f32.powf((6.3 - 23.0) / 10.0)).abs() < 1e-6);

    // q10Fixed = 1 (na Poirazi) — температурно-независимый канал.
    assert!((IrQ10::Fixed(1.0).factor_at(35.0) - 1.0).abs() < 1e-6);
    assert!((IrQ10::Fixed(1.0).factor_at(6.3) - 1.0).abs() < 1e-6);
}
