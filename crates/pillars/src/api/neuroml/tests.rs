// Хардкоженные NEURON-эталоны несут полную измеренную точность — лишних
// знаков нет, это точные значения сэмплов (excessive_precision не применим).
#![allow(clippy::excessive_precision)]
use super::*;

/// A minimal NeuroML2 doc: one regular-spiking 2003 Izhikevich cell driven
/// by a pulse generator.
const RS: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="test">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <pulseGenerator id="pg" delay="0ms" duration="200ms" amplitude="10"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

/// Синтаксическая XML-ошибка несёт **структурированную** позицию
/// `(row, col)` (для редактора/IDE), а не только отформатированную строку:
/// незакрытая кавычка в атрибуте на строке 3 → `pos = (3, 1)`.
#[test]
fn xml_error_reports_line_and_column() {
    let xml = "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\">\n  <izhikevichCell id=\"rs\" v0=\"-70mV\" a=\"0.02/>\n</neuroml>";
    let err = match load(xml) {
        Err(e) => e,
        Ok(_) => panic!("ожидалась ошибка парсинга"),
    };
    assert_eq!(err.pos, Some((3, 1)), "позиция должна быть структурирована");
    assert!(
        err.msg.contains("expected"),
        "сообщение без позиционного префикса, got: {}",
        err.msg
    );
    // Display сохраняет стабильный формат «XML: R:C: …» для существующих
    // потребителей строкового вывода (логи, статусы).
    assert!(
        err.to_string().contains("XML: 3:1:"),
        "Display должен сохранить формат, got: {err}"
    );
}

/// Точечные популяции не имеют кабельной геометрии: `cable_info`/`cable_voltage`
/// честно возвращают `None`.
#[test]
fn point_neuron_population_has_no_cable() {
    let mut ln = load(RS).unwrap();
    assert!(ln.cable_info(0).is_none());
    assert!(ln.network.cable_voltage(0).is_none());
    ln.run();
    assert!(ln.network.cable_voltage(0).is_none());
}

#[test]
fn single_rs_neuron_spikes_under_pulse() {
    let mut ln = load(RS).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    let res = ln.run();
    assert_eq!(res.steps, (200.0 / 0.5) as usize);
    // Regular-spiking Izhikevich with I=10 fires repeatedly (~22 Hz).
    assert!(
        res.spike_counts[0] > 0,
        "RS neuron should spike under the pulse, got {:?}",
        res.spike_counts
    );
}

#[test]
fn no_pulse_means_no_spikes() {
    // The pulse starts at 500 ms — after the 200 ms simulation — so the
    // resting neuron (v0 = -70, u = b·v0) must stay silent.
    let xml = RS.replace(
        "delay=\"0ms\" duration=\"200ms\"",
        "delay=\"500ms\" duration=\"10ms\"",
    );
    let mut ln = load(&xml).unwrap();
    let res = ln.run();
    assert_eq!(res.spike_counts[0], 0);
}

#[test]
fn rejects_unknown_cell_reference() {
    let xml = RS.replace("component=\"rs\"", "component=\"nope\"");
    assert!(load(&xml).is_err());
}

/// A minimal NeuroML2 doc: one tonic-spiking 2007 Izhikevich cell driven
/// by a pulse generator.
const IZH2007: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="test2007">
  <izhikevich2007Cell id="rs07" v0="-60mV" C="100pF" k="0.7" vr="-60mV" vt="-40mV" vpeak="35mV" a="0.03" b="-2" c="-50mV" d="100"/>
  <pulseGenerator id="pg" delay="0ms" duration="500ms" amplitude="100"/>
  <network id="net">
    <population id="pop" component="rs07" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn single_izh2007_neuron_spikes_under_pulse() {
    let mut ln = load_with(IZH2007, 1.0, 500.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "tonic-spiking 2007 neuron should spike under I=100, got {:?}",
        res.spike_counts
    );
}

#[test]
fn strips_unit_suffixes() {
    // "10mV" → 10.0, "0.02" → 0.02, "0.07 nA" → 0.07.
    assert_eq!(parse::to_mv("10mV").unwrap(), 10.0);
    assert_eq!(parse::raw("0.02").unwrap(), 0.02);
    assert_eq!(parse::raw("0.07 nA").unwrap(), 0.07);
    assert!(parse::raw("mV").is_err());
}

/// Классический Hodgkin–Huxley: один `hhCell` под импульсом 50 nA
/// должен генерировать спайки (стандартный кальмаровый аксон).
const HH: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_test">
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
  </hhCell>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="50"/>
  <network id="net">
    <population id="pop" component="squid" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn hh_cell_spikes_under_pulse() {
    let mut ln = load_with(HH, 0.01, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "классический HH под импульсом должен спайкать, got {:?}",
        res.spike_counts
    );
}

#[test]
fn hh_cell_requires_na_and_k_channels() {
    // Без K-канала — понятная ошибка, а не тихое молчание.
    let xml = HH.replace(
            "<channelDensity id=\"k\" ionChannel=\"kChans\" condDensity=\"36 mS_per_cm2\" erev=\"-77mV\"/>",
            "",
        );
    assert!(load_with(&xml, 0.01, 100.0).is_err());
}

/// Канонический кальмаровый аксон, но каналы описаны **полным** NeuroML2
/// (`ionChannelHH` + `gateHHrates` m³h/n⁴), а не подстроковой эвристикой.
/// Должен спайкать ровно как `hh_cell_spikes_under_pulse` (тот же HHModel).
const HH_SQUID: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_squid">
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
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
  </hhCell>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="50"/>
  <network id="net">
    <population id="pop" component="squid" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn hh_squid_nml_gatehh_rates_spikes_under_pulse() {
    let mut ln = load_with(HH_SQUID, 0.01, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    // Каналы распознаны — никаких «пропущено» warnings.
    assert!(
        ln.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ln.warnings
    );
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "классический HH (gateHHrates) должен спайкать, got {:?}",
        res.spike_counts
    );
}

/// Нестандартный канал (gateHHtauInf HCN) рядом со стандартными Na/K —
/// клетка запускается generic-движком (`ConductancePopulation`): без
/// warning, без NaN, и спайкает под импульсом (Na/K доминируют).
#[test]
fn hh_cell_mixed_channels_run_on_generic_engine() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_hcn">
  <ionChannelHH id="hcnChans" conductance="10pS" species="hcn">
    <gateHHtauInf id="m" instances="1">
      <timeCourse type="fixedTimeCourse" tau="10ms"/>
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
    </gateHHtauInf>
  </ionChannelHH>
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
    <channelDensity id="hcn" ionChannel="hcnChans" condDensity="0.1 mS_per_cm2" erev="-40mV"/>
  </hhCell>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="50"/>
  <network id="net">
    <population id="pop" component="squid" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.01, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    assert!(
        ln.warnings.is_empty(),
        "смешанная клетка не должна давать warning, got {:?}",
        ln.warnings
    );
    let res = ln.run();
    let v = ln.network.state(0, 0).to_vec();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "v должен быть конечным, got {v:?}"
    );
    assert!(
        res.spike_counts[0] > 0,
        "смешанный HH под импульсом должен спайкать, got {:?}",
        res.spike_counts
    );
}

/// `hhCell` с каналом, описанным **кинетической схемой** (`ionChannelKS` +
/// `gateKS` 2-состояния), рядом со стандартным Na/K — клетка запускается
/// скалярным master-equation движком (`MarkovPopulation`): без warning, без
/// NaN, спайкает под импульсом (Na/K доминируют, KS-канал — малая добавка).
#[test]
fn hh_cell_kinetic_scheme_channel_runs_on_markov_engine() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_ks">
  <ionChannelKS id="hcn_ks" conductance="10pS" species="hcn">
    <gateKS id="m" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-80mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="10ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
    <channelDensity id="hcn" ionChannel="hcn_ks" condDensity="0.1 mS_per_cm2" erev="-40mV"/>
  </hhCell>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="50"/>
  <network id="net">
    <population id="pop" component="squid" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.01, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    assert!(
        ln.warnings.is_empty(),
        "KS-канал не должен давать warning, got {:?}",
        ln.warnings
    );
    let res = ln.run();
    let v = ln.network.state(0, 0).to_vec();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "v должен быть конечным, got {v:?}"
    );
    assert!(
        res.spike_counts[0] > 0,
        "смешанный HH+KS под импульсом должен спайкать, got {:?}",
        res.spike_counts
    );
}

/// `hhCell` с каналом, описанным **дробной проводимостью** (`gateFractional`
/// из двух tau-inf субгейтов), рядом со стандартным Na/K — клетка должна
/// запуститься скалярным master-equation движком (fractional → произведение
/// субгейтов): без warning, без NaN, спайкает под импульсом.
#[test]
fn hh_cell_fractional_channel_runs_on_markov_engine() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_frac">
  <ionChannelHH id="caFrac" conductance="10pS" species="ca">
    <gateFractional id="m" instances="1">
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
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
    <channelDensity id="ca" ionChannel="caFrac" condDensity="0.1 mS_per_cm2" erev="120mV"/>
  </hhCell>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="50"/>
  <network id="net">
    <population id="pop" component="squid" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.01, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    assert!(
        ln.warnings.is_empty(),
        "gateFractional не должен давать warning, got {:?}",
        ln.warnings
    );
    let res = ln.run();
    let v = ln.network.state(0, 0).to_vec();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "v должен быть конечным, got {v:?}"
    );
    assert!(
        res.spike_counts[0] > 0,
        "смешанный HH+fractional под импульсом должен спайкать, got {:?}",
        res.spike_counts
    );
}

/// Канал с неподдерживаемым гейтом (gateHHratesTau — распознаётся, но не
/// компилируется), прицепленный к hhCell, — честная ошибка загрузки
/// (не молча, не пропуск). `gateHHInstantaneous` теперь компилируется
/// (алгебраический гейт), поэтому как «неподдерживаемый» не годится.
#[test]
fn hh_cell_unsupported_gate_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="hh_inst">
  <ionChannelHH id="caChans" conductance="10pS" species="ca">
    <gateHHratesTau id="m" instances="2">
      <forwardRate type="HHExpRate" rate="0.01per_ms" midpoint="-20mV" scale="5mV"/>
      <timeCourse type="fixedTimeCourse" tau="1ms"/>
    </gateHHratesTau>
  </ionChannelHH>
  <hhCell id="squid" v0="-65mV" C="1uF" leakConductance="0.3uS" leakReversal="-54.3mV">
    <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV"/>
    <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV"/>
    <channelDensity id="ca" ionChannel="caChans" condDensity="1 mS_per_cm2" erev="120mV"/>
  </hhCell>
  <network id="net"><population id="pop" component="squid" size="1"/></network>
</neuroml>
"#;
    assert!(load_with(xml, 0.01, 100.0).is_err());
}

/// Каноническая фигура Brette & Gerstner 2005 (Fig. 4, regular-spiking +
/// bursting AdEx) — **регрессия на нормализацию единиц**. Pillars' AdEx
/// voltage-normalised (÷gL): a→a/gL, b→b/gL·1000 (мВ), ток nA→мВ (·1000/gL).
/// Без этой нормализации обе клетки молчат; с ней — 3 + 6 = 9 спайков
/// за 200 мс при dt = 0.5 (сверено с численным эталоном).
const ADEX_FIG4: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Figure4_AdEx">
  <adExIaFCell id="adEx2005" C="281pF" gL="30nS" EL="-70.6mV" reset="-70.6mV" VT="-50.4mV" thresh="-20mV" delT="2mV" tauw="144ms" a="4nS" b="0.0805nA" refract="5ms"/>
  <adExIaFCell id="adExBurst" C="281pF" gL="30nS" EL="-65mV" reset="-47.2mV" VT="-50.4mV" thresh="-20mV" delT="2mV" tauw="40ms" a="4nS" b="0.08nA" refract="0ms"/>
  <pulseGenerator id="pulseGen1" delay="50ms" duration="400ms" amplitude="0.75 nA"/>
  <pulseGenerator id="pulseGen2" delay="50ms" duration="400ms" amplitude="0.5 nA"/>
  <network id="net1">
    <population id="adexpop1" component="adEx2005" size="1"/>
    <population id="adexpop2" component="adExBurst" size="1"/>
    <explicitInput target="adexpop1[0]" input="pulseGen1" destination="synapses"/>
    <explicitInput target="adexpop2[0]" input="pulseGen2" destination="synapses"/>
  </network>
</neuroml>
"#;

#[test]
fn adex_figure4_units_normalised() {
    let mut ln = load_with(ADEX_FIG4, 0.5, 200.0).unwrap();
    assert_eq!(ln.network.num_populations(), 2);
    let res = ln.run();
    // adexpop1 (0.75 nA) → 3 спайка, adexpop2 (0.5 nA, bursting) → 6.
    assert_eq!(res.spike_counts, vec![3, 6], "AdEx должен дать 3+6 спайков");
}

/// iafCell (LIF с C/gL) под импульсом: регрессия на нормализацию тока.
/// Без `current_scale = 1000/gL` ток 0.0032 nA слишком мал и клетка молчит;
/// с ней → 16 мВ и спайки (как в NML2_GapJunctions.nml).
#[test]
fn iaf_cell_current_is_scaled_by_gl() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="iaf_test">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <pulseGenerator id="pg" delay="50ms" duration="200ms" amplitude="0.0032nA"/>
  <network id="net"><population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="pg"/></network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 300.0).unwrap();
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "iafCell под 0.0032 nA должен спайкать после нормализации, got {:?}",
        res.spike_counts
    );
}

/// iafRefCell = iafCell + refract — должен парситься и спайкать.
#[test]
fn iaf_ref_cell_spikes() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="iaf_ref">
  <iafRefCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF" refract="5ms"/>
  <pulseGenerator id="pg" delay="0ms" duration="200ms" amplitude="0.0032nA"/>
  <network id="net"><population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="pg"/></network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 300.0).unwrap();
    let res = ln.run();
    assert!(res.spike_counts[0] > 0, "iafRefCell должен спайкать");
}

/// Классический fitzHughNagumoCell: единственный параметр I (встроенный
/// ток) → limit cycle → спайки (edge-detection при v_thresh = 0.5).
#[test]
fn fhn_cell_spikes_under_bias() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="fhn">
  <fitzHughNagumoCell id="fn1" I="0.8"/>
  <network id="net"><population id="pop" component="fn1" size="1"/></network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 500.0).unwrap();
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "классический FHN (I=0.8) должен осциллировать, got {:?}",
        res.spike_counts
    );
}

/// fitzHughNagumo1969Cell: явные a, b, phi, I.
#[test]
fn fhn1969_cell_spikes_under_bias() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="fhn69">
  <fitzHughNagumo1969Cell id="fn" a="0.7" b="0.08" I="1.0" phi="0.08" V0="0.0" W0="0.0"/>
  <network id="net"><population id="pop" component="fn" size="1"/></network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 500.0).unwrap();
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "FHN 1969 (I=1.0) должен осциллировать, got {:?}",
        res.spike_counts
    );
}

/// `gapJunction` больше не «пропущенный» элемент — парсится в IR (даже без
/// `electricalProjection` это просто объявленный компонент, не warning).
#[test]
fn gap_junction_parses_without_warning() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="gj">
  <gapJunction id="gj1" conductance="10pS"/>
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <network id="net"><population id="pop" component="iaf" size="1"/></network>
</neuroml>
"#;
    let ln = load(xml).unwrap();
    assert!(
        !ln.warnings.iter().any(|w| w.contains("gapJunction")),
        "gap junction не должен быть в warnings, got {:?}",
        ln.warnings
    );
}

/// LEMS `<include>`: главный файл подключает фрагмент с клеткой, резолв
/// идёт относительно пути файла. Пишем во временную директорию.
#[test]
fn load_file_resolves_includes() {
    let dir = std::env::temp_dir().join(format!("nml_inc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cells = "<izhikevichCell id=\"rs\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65.0mV\" d=\"8\"/>";
    // Кириллический комментарий ПЕРЕД <include> — регрессия на байтовые
    // индексы: to_lowercase() меняет длину строки, find() по нижнему
    // регистру давал сдвиг, и include не находился/резал XML.
    let main = "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"inc\">\n\
             <!-- Йокай: включаем фрагмент с клеткой -->\n\
             <include href=\"cells.xml\"/>\n\
             <pulseGenerator id=\"pg\" delay=\"0ms\" duration=\"200ms\" amplitude=\"10\"/>\n\
             <network id=\"net\"><population id=\"pop\" component=\"rs\" size=\"1\"/>\n\
             <explicitInput target=\"pop[0]\" input=\"pg\"/></network>\n</neuroml>";
    std::fs::write(dir.join("cells.xml"), cells).unwrap();
    let main_path = dir.join("main.nml");
    std::fs::write(&main_path, main).unwrap();
    let mut ln = load_file(&main_path).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "включённая клетка должна спайкать, got {:?}",
        res.spike_counts
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Морфологическая клетка (сома + дендрит) реально запускается через
/// `load()` → кабельная популяция: пассивный кабель, импульс в сому →
/// деполяризация выше покоя. Проверяет весь конвейер Фазы 5.
const MORPHO_CELL: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morpho_test">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="20" z="0" diameter="2"/>
      <distal x="0" y="220" z="0" diameter="2"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
    <segmentGroup id="dendrite_group"><member segment="1"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific" segmentGroup="all"/>
      <initMembPotential value="-70mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <pulseGenerator id="pg" delay="10ms" duration="90ms" amplitude="0.1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn morpho_cell_runs_as_cable_population() {
    let mut ln = load_with(MORPHO_CELL, 0.05, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    // Кабельная популяция — CPU-only (нет #[dual_neuron] CUDA-источника).
    assert!(ln.gpu_pops[0].source.is_empty());
    let pid = 0;
    // В покое сома сидит на e_leak = -70 мВ (из pas-канала).
    assert!((ln.network.state(pid, 0)[0] + 70.0).abs() < 1e-3);

    let res = ln.run();
    assert_eq!(res.steps, 2000);
    // Импульс 0.1 nA в сому (10–100 мс, τ ≈ 3 мс) → заметная
    // деполяризация выше покоя. Пассивный кабель не спайкает (порог 0 мВ).
    let v_soma = ln.network.state(pid, 0)[0];
    assert!(
        v_soma > -68.0,
        "сома должна деполяризоваться под током, got {v_soma}"
    );
    assert_eq!(res.spike_counts[0], 0, "пассивный кабель не спайкает");

    // Мост в GUI: публичная геометрия + покомпартментный V после прогона.
    let info = ln.cable_info(pid).expect("морфо-популяция даёт CableInfo");
    assert_eq!(info.root, 0);
    // pas 3 S/m² → rm = 3333 Ω·cm² → λ=408 µm (d=2), d_λ=0.1 → target 41 µm
    // → дендрит 200 µm = 5 компартментов; сома — 1. Итого 6.
    assert_eq!(info.compartments.len(), 6);
    assert_eq!(info.compartments[0].parent, None);
    assert_eq!(info.compartments[1].parent, Some(0));
    // Имена сегментов из NeuroML `name`.
    let names: std::collections::HashMap<usize, &str> = info
        .segment_names
        .iter()
        .map(|(id, n)| (*id, n.as_str()))
        .collect();
    assert_eq!(names.get(&0).copied(), Some("soma"));
    assert_eq!(names.get(&1).copied(), Some("dend"));
    // Позиции: сома центр (0,10,0)/path 10; дендритные центры 40..200,
    // path шагает по 40 µm (длина компартмента).
    assert_eq!(info.compartments[0].y, 10.0);
    assert!((info.compartments[0].path_um - 10.0).abs() < 1e-4);
    assert_eq!(info.compartments[1].y, 40.0);
    assert!((info.compartments[1].path_um - 40.0).abs() < 1e-4);
    let last = info.compartments.last().unwrap();
    assert!(
        (last.path_um - 200.0).abs() < 1e-3,
        "path = {}",
        last.path_um
    );
    assert_eq!(last.y, 200.0);

    // Покомпартментный V: длина = 1 нейрон × 6 компартментов, сома-запись
    // совпадает с публичной колонкой состояния.
    let cv = ln.network.cable_voltage(pid).expect("cable voltage");
    assert_eq!((cv.n_neurons, cv.n_comps), (1, 6));
    assert_eq!(cv.v.len(), 6);
    assert_eq!(cv.v[0], ln.network.state(pid, 0)[0]);
    // Дендрит затухает — кончик ближе к покою, чем сома.
    assert!(cv.v[0] > cv.v[5]);
}

/// Ссылка клетки на неизвестную морфологию/биофизику — честная ошибка.
#[test]
fn morpho_cell_unknown_references_error() {
    let xml = MORPHO_CELL.replace("morphology=\"m1\"", "morphology=\"nope\"");
    assert!(load_with(&xml, 0.05, 100.0).is_err());
    let xml2 = MORPHO_CELL.replace(
        "biophysicalProperties=\"bp1\"",
        "biophysicalProperties=\"nope\"",
    );
    assert!(load_with(&xml2, 0.05, 100.0).is_err());
}

/// Морфо-клетка с активными NaV/KV (HH) каналами на соме по segmentGroup:
/// под импульсом спайкает — активная динамика реально входит в кабель.
const MORPHO_HH: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morpho_hh">
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
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-54.3mV" ion="non_specific" segmentGroup="soma_group"/>
      <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV" ion="na" segmentGroup="soma_group"/>
      <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV" ion="k" segmentGroup="soma_group"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="0.3"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn morpho_cell_hh_channels_spike() {
    let mut ln = load_with(MORPHO_HH, 0.025, 100.0).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    assert!(
        ln.warnings.is_empty(),
        "unexpected warnings: {:?}",
        ln.warnings
    );
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "морфо-клетка с NaV/KV должна спайкать, got {:?}",
        res.spike_counts
    );
}

/// Морфо-клетка с `channelDensityNonUniform` Na на дендрите: канал больше
/// не пропускается с warning — плотность линейно интерполируется вдоль
/// пути (proximal → distal) и реально входит в кабель (клетка спайкает).
const MORPHO_NONUNIFORM: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morpho_nu">
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
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="20" z="0" diameter="5"/>
      <distal x="0" y="120" z="0" diameter="5"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
    <segmentGroup id="dend_group"><member segment="1"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-54.3mV" ion="non_specific"/>
      <channelDensity id="na" ionChannel="naChans" condDensity="120 mS_per_cm2" erev="50mV" ion="na" segmentGroup="soma_group"/>
      <channelDensity id="k" ionChannel="kChans" condDensity="36 mS_per_cm2" erev="-77mV" ion="k" segmentGroup="soma_group"/>
      <channelDensityNonUniform id="naDend" ionChannel="naChans" erev="50mV" ion="na" variableParameter="naDens">
        <variableParameter parameter="naDens" segmentGroup="dend_group">
          <proximal value="120 S_per_m2"/>
          <distal value="0 S_per_m2"/>
        </variableParameter>
      </channelDensityNonUniform>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <pulseGenerator id="pg" delay="0ms" duration="100ms" amplitude="0.3"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#;

#[test]
fn morpho_cell_non_uniform_density_loads_and_spikes() {
    let mut ln = load_with(MORPHO_NONUNIFORM, 0.025, 100.0).unwrap();
    assert!(
        ln.warnings.is_empty(),
        "channelDensityNonUniform не должен попадать в warnings: {:?}",
        ln.warnings
    );
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "неравномерный Na должен реально войти в кабель: {:?}",
        res.spike_counts
    );
    let v = ln.network.state(0, 0);
    assert!(v.iter().all(|x| x.is_finite()), "V должно быть конечным");
}

/// Направление градиента не-uniform плотности имеет значение: proximal-у-
/// корня vs proximal-у-кончика дают разные детерминированные траектории.
#[test]
fn morpho_cell_non_uniform_density_gradient_matters() {
    // Proximal=120 у корня / distal=0 у кончика — базовая конфигурация.
    let xml_dist = MORPHO_NONUNIFORM
        .replace(
            "<proximal value=\"120 S_per_m2\"/>",
            "<distal value=\"120 S_per_m2\"/>",
        )
        .replace(
            "<distal value=\"0 S_per_m2\"/>",
            "<proximal value=\"0 S_per_m2\"/>",
        );
    let mut a = load_with(MORPHO_NONUNIFORM, 0.025, 100.0).unwrap();
    let mut b = load_with(&xml_dist, 0.025, 100.0).unwrap();
    a.run();
    b.run();
    let va = a.network.state(0, 0).to_vec();
    let vb = b.network.state(0, 0).to_vec();
    let max_diff = va
        .iter()
        .zip(&vb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "смена proximal/distal должна менять динамику кабеля, max_diff = {max_diff}"
    );
}

/// Морфо-клетка с кинетическим (`ionChannelKS`) каналом HCN на соме:
/// Марков-канал подключается к кабелю и гоняется без NaN.
#[test]
fn morpho_cell_markov_channel_runs() {
    let xml = MORPHO_HH
            .replace(
                "<channelDensity id=\"na\" ionChannel=\"naChans\" condDensity=\"120 mS_per_cm2\" erev=\"50mV\" ion=\"na\" segmentGroup=\"soma_group\"/>",
                "",
            )
            .replace(
                "<channelDensity id=\"k\" ionChannel=\"kChans\" condDensity=\"36 mS_per_cm2\" erev=\"-77mV\" ion=\"k\" segmentGroup=\"soma_group\"/>",
                "<channelDensity id=\"hcn\" ionChannel=\"hcn_ks\" condDensity=\"0.1 mS_per_cm2\" erev=\"-40mV\" ion=\"hcn\" segmentGroup=\"soma_group\"/>",
            );
    // Вставить ionChannelKS перед <morphology>.
    let xml = xml.replace(
            "  <morphology id=\"m1\">",
            "  <ionChannelKS id=\"hcn_ks\" conductance=\"10pS\" species=\"hcn\">\n    <gateKS id=\"m\" instances=\"1\">\n      <closedState id=\"c\"/>\n      <openState id=\"o\"/>\n      <tauInfTransition from=\"c\" to=\"o\">\n        <steadyState type=\"HHSigmoidVariable\" rate=\"1\" midpoint=\"-80mV\" scale=\"10mV\"/>\n        <timeCourse type=\"fixedTimeCourse\" tau=\"10ms\"/>\n      </tauInfTransition>\n    </gateKS>\n  </ionChannelKS>\n  <morphology id=\"m1\">",
        );
    let mut ln = load_with(&xml, 0.025, 100.0).unwrap();
    ln.run();
    let v = ln.network.state(0, 0).to_vec();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "сома должна быть конечной, got {v:?}"
    );
    // HCN (e_rev -40) тянет сому из покоя к -40 — деполяризация, не NaN.
    assert!(
        v[0] > -65.0,
        "HCN должен деполяризовать сому от покоя -65, got {}",
        v[0]
    );
}

/// Морфо-клетка с `channelPopulation` (популяция из 64 дискретных
/// `ionChannelKS`-каналов на соме): дискретный канал входит в кабель как
/// **биномиальный** шум (n_open ~ Binomial(N, p_open)), а не как
/// детерминированный mean-field. Проверяем, что (а) не warning, (б) шум
/// реально разводит два нейрона с одинаковой инициализацией, (в) тот же
/// сид даёт побитово тот же результат.
const MORPHO_CHANNEL_POP: &str = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="morpho_pop">
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
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-65mV" ion="non_specific" segmentGroup="soma_group"/>
      <channelPopulation id="noisyK" ionChannel="kchan" number="64" erev="-40mV" ion="k"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <network id="net">
    <population id="pop" component="c1" size="2"/>
  </network>
</neuroml>
"#;

#[test]
fn morpho_cell_channel_population_stochastic_noise() {
    let mut ln = load_with(MORPHO_CHANNEL_POP, 0.05, 200.0).unwrap();
    assert!(
        ln.warnings.is_empty(),
        "channelPopulation не должен попадать в warnings: {:?}",
        ln.warnings
    );
    let res = ln.run();
    assert_eq!(res.steps, 4000);
    let v = ln.network.state(0, 0).to_vec();
    assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
    assert!(
        (v[0] - v[1]).abs() > 1e-3,
        "биномиальный шум должен развести два нейрона: {v:?}"
    );

    // Детерминизм: тот же сид (hash id) → побитово тот же результат.
    let mut ln2 = load_with(MORPHO_CHANNEL_POP, 0.05, 200.0).unwrap();
    ln2.run();
    assert_eq!(v, ln2.network.state(0, 0).to_vec());
}

/// `channelPopulation` с явным `segment` (id одиночного сегмента) ставится
/// на **этот** сегмент, а не на сому. Сильный почти-полностью-открытый
/// деполяризующий канал на `segment="2"` тянет дистальный компартмент к
/// `erev`, тогда как сома остаётся у покоя (связана только через тонкие
/// высокоомные дендриты). Регрессия на «всегда сома»-поведение.
#[test]
fn morpho_cell_channel_population_segment_placement() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="pop_seg">
  <ionChannelKS id="depol" conductance="100pS" species="non_specific">
    <gateKS id="g" instances="1">
      <closedState id="c"/>
      <openState id="o"/>
      <tauInfTransition from="c" to="o">
        <steadyState type="HHSigmoidVariable" rate="1" midpoint="-100mV" scale="10mV"/>
        <timeCourse type="fixedTimeCourse" tau="1ms"/>
      </tauInfTransition>
    </gateKS>
  </ionChannelKS>
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <!-- Длинные тонкие дендриты → высокое осевое сопротивление: сома
         электрически изолирована от дистального сегмента. -->
    <segment id="1" name="dend1">
      <parent segment="0"/>
      <distal x="0" y="120" z="0" diameter="1"/>
    </segment>
    <segment id="2" name="dend2">
      <parent segment="1"/>
      <distal x="0" y="220" z="0" diameter="1"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-65mV" ion="non_specific" segmentGroup="soma_group"/>
      <channelPopulation id="depolPop" ionChannel="depol" number="10000" erev="0mV" ion="non_specific" segment="2"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties><resistivity value="100 ohm_cm"/></intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.05, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    ln.run();

    let info = ln.cable_info(0).unwrap();
    let seg2_comp = info
        .compartments
        .iter()
        .position(|c| c.segment_id == 2)
        .expect("компартмент сегмента 2 существует");
    let v = ln.network.cable_voltage(0).unwrap();
    let v_root = v.v[info.root];
    let v_seg2 = v.v[seg2_comp];

    // Канал на сегменте 2: его компартмент — самый деполяризованный
    // (тянется к erev=0), сома — заметно ближе к покою. Если бы канал
    // падал на сому (старое поведение), сома была бы деполяризованнее
    // дистального сегмента, и условие бы не выполнилось.
    assert!(
        v_seg2 > v_root + 1.0,
        "сегмент 2 должен быть деполяризованнее сомы: soma={v_root}, seg2={v_seg2}"
    );
    assert!(
        v_seg2 < 0.0,
        "сегмент 2 приближается к erev=0 снизу: {v_seg2}"
    );
}

/// Реальный пример NeuroML2 (`NML2_FullCell.nml` из репозитория NeuroML2):
/// инлайн-морфология, `<include>`, пассивный `pas`, Na-канал и
/// `channelPopulation` с `segment="2"` — грузится с диска и запускается.
/// Сам файл — библиотека компонентов без `<network>`, поэтому тест
/// дописывает сеть во временную копию (и копирует включаемый файл).
#[test]
fn loads_neuroml_example_full_cell() {
    let src =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/examples/neuroml");
    let dir = std::env::temp_dir().join(format!("nml_fullcell_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(src.join("NML2_FullCell.nml"), dir.join("NML2_FullCell.nml")).unwrap();
    std::fs::copy(
        src.join("NML2_SimpleIonChannel.nml"),
        dir.join("NML2_SimpleIonChannel.nml"),
    )
    .unwrap();
    let full = dir.join("NML2_FullCell.nml");
    let xml = std::fs::read_to_string(&full).unwrap();
    let xml = xml.replace(
            "</neuroml>",
            "<network id=\"net\"><population id=\"pop\" component=\"SpikingCell\" size=\"1\"/></network>\n</neuroml>",
        );
    std::fs::write(&full, xml).unwrap();

    let mut ln = load_file(&full).unwrap();
    assert_eq!(ln.network.num_populations(), 1);
    let info = ln.cable_info(0).expect("FullCell — морфологическая клетка");
    // Все 4 сегмента (сома + 2 дендрита + шипик) представлены в
    // дискретизации.
    let seg_ids: std::collections::HashSet<usize> =
        info.compartments.iter().map(|c| c.segment_id).collect();
    assert_eq!(
        seg_ids.len(),
        4,
        "все 4 сегмента дискретизованы: {seg_ids:?}"
    );
    let res = ln.run();
    assert!(res.steps > 0);
    let v = ln.network.state(0, 0).to_vec();
    assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Реальная OSB-модель — редуцированная CA1-клетка Poirazi 2003
/// (`resources/validate/osb/poirazi_ca1_soma.nml`): реальная морфология
/// сомы + реальные каналы (na, kdr, leak, hs=Ih, km=M). Ih использует
/// кастомные LEMS-`ComponentType` (`Poirazi_h_tau` — piecewise
/// double-exponential tau, `Poirazi_h_ss`) — грузится без warning и
/// запускается.
#[test]
fn loads_poirazi_ca1_soma_osb_model() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_ca1_soma.nml");
    let mut ln = load_file(&path).unwrap();
    assert!(
        ln.warnings.is_empty(),
        "кастомные ComponentType не должны давать warnings: {:?}",
        ln.warnings
    );
    let info = ln.cable_info(0).expect("CA1 — морфологическая клетка");
    assert_eq!(info.compartments.len(), 1, "сома — один компартмент");
    let res = ln.run();
    assert!(res.steps > 0);
    let v = ln.network.state(0, 0).to_vec();
    assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
}

/// `inhomogeneousValue` condDensity expressions are LEMS in **S/m²** (the
/// `1e4 *` prefix in real Poirazi 2003 exports), so the engine must divide
/// by `1e4` to get S/cm² — and must evaluate `exp(...)` as the exponential,
/// not as the `H(...)` Heaviside step.
#[test]
fn inhomogeneous_density_units_and_exp() {
    // H(p-300) = 1 → 1e4 * 7e-3 S/m² = 7e-3 S/cm².
    let a = super::compile::eval_inhomogeneous_expr("1e4 * H(p-300) * 7e-3", 400.0);
    assert!((a - 7e-3).abs() < 1e-9, "step-on density = {a}");
    // H(p-300) = 0 → 0.
    let b = super::compile::eval_inhomogeneous_expr("1e4 * H(p-300) * 7e-3", 100.0);
    assert_eq!(b, 0.0, "step-off density = {b}");
    // Sigmoid (hdend): at p=280 exp(0)=1 → (1.872e-5 + 8·1.872e-5/2) S/cm².
    let c = super::compile::eval_inhomogeneous_expr(
        "1e4 * (1.872e-5 + (8*1.872e-5)/(1 + exp((280-p)/50)))",
        280.0,
    );
    assert!((c - 9.36e-5).abs() < 1e-9, "sigmoid density = {c}");
}

/// Регрессия: полная Poirazi 2003 CA1 пирамида (5160 морфо-сегментов →
/// 5249 компартментов, GHK2 Ca, capool) в покое **без внешнего тока**
/// садится на ≈ −76 мВ и **не спайкует** (NEURON-эталон
/// `gen_poirazi_full_ref.py`, secondorder=0, dt=0.025: t=0,5,10,15,20 мс
/// → −70.0, −75.07, −75.77, −75.98, −76.02).
///
/// Интеграционная защита трёх ранее найденных багов единиц:
/// 1. `inhomogeneousValue` плотности S/m² трактовались как S/cm² (×10000)
///    → артефактные 74 спайка/150 мс;
/// 2. `resistivity` kohm_cm парсился без суффикса (×1000, space-clamp)
///    → сома сидела на −73 вместо −76;
/// 3. Ca-зависимые гейты читали ca=0 вместо реального [Ca²⁺].
///
/// Остаток ~0.5 мВ к NEURON — численный (быстрый транзиент на жёсткой
/// системе), поэтому допуск 1.0 мВ, а не бит-точность сомного теста.
/// ~3.7 мин в debug — `#[ignore]`, вручную через `--ignored`.
#[test]
#[ignore]
fn poirazi_full_cell_rests_at_neuron_resting_potential() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_full/poirazi.net.nml");
    // 35 °C — тот же якорь, что у NEURON-эталона и у бит-точного soma-теста.
    let mut ln = load_file_at(&path, 35.0).expect("полная Poirazi грузится");
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);

    let info = ln.cable_info(0).expect("CA1 — морфологическая клетка");
    assert_eq!(info.compartments.len(), 5249, "d_λ-дискретизация 5160→5249");

    // 20 мс @ 0.025 мс (800 шагов); сома на сетке 5 мс.
    let steps = 800usize;
    let grid = 200usize;
    let mut soma = ln.network.state(0, 0)[0];
    let mut traj = vec![soma];
    for s in 0..steps {
        ln.network.step();
        soma = ln.network.state(0, 0)[0];
        if (s + 1) % grid == 0 {
            traj.push(soma);
        }
    }

    const REF: [f32; 5] = [-70.0, -75.07, -75.77, -75.98, -76.02];
    assert_eq!(traj.len(), REF.len());
    for (i, (&got, &want)) in traj.iter().zip(REF.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1.0,
            "t={} мс: сома {got:.4} vs NEURON {want:.4}",
            i * 5
        );
    }

    // Ноль спайков + кабельный градиент (Ra-фикс): весь транзиент глубоко
    // субпороговый, и разброс вдоль дендрита > 0.5 мВ (не space-clamp).
    let cv = ln.network.cable_voltage(0).unwrap();
    let n = cv.v.len();
    let mut global_max = -1e9f32;
    let mut global_min = 1e9f32;
    for k in 0..n {
        global_max = global_max.max(cv.v[k]);
        global_min = global_min.min(cv.v[k]);
    }
    assert!(global_max < -50.0, "нет спайков: max V = {global_max}");
    assert!(
        global_max - global_min > 0.5,
        "кабельный градиент: [{global_min}, {global_max}]"
    );
}

/// Быстрый smoke: полная Poirazi грузится (5249 компартментов) и за
/// первые ~2 мс не спайкует и не взрывается. Покрывает load/compile/
/// discretize-путь в default-сюите — без 205-секундного `#[ignore]`-теста.
#[test]
fn loads_poirazi_full_cell_and_stays_subthreshold() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_full/poirazi.net.nml");
    let mut ln = load_file_at(&path, 35.0).expect("полная Poirazi грузится");
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let info = ln.cable_info(0).expect("CA1 — морфологическая клетка");
    assert_eq!(info.compartments.len(), 5249, "d_λ-дискретизация 5160→5249");
    // 0.2 мс @ 0.025 мс — только проверка, что Hines-солвер стартует без
    // NaN/взрыва; полный покой — в `#[ignore]`-регрессии выше.
    for _ in 0..8 {
        ln.network.step();
    }
    let cv = ln.network.cable_voltage(0).unwrap();
    for (k, &v) in cv.v.iter().enumerate() {
        assert!(v.is_finite(), "V[{k}] конечен: {v}");
        assert!(v < -50.0, "V[{k}] субпорог: {v}");
    }
}

/// NEURON-parity редуцированной CA1-сомы Poirazi 2003 (реальная OSB-
/// модель), **при физиологической 35 °C с per-channel Q10** (km:
/// q10ExpTemp factor 2.3 @ 23 °C → Q10(35) ≈ 2.3^(12/10) ≈ 2.71; na:
/// q10Fixed 1). Эталон — `resources/validate/neuron_models/gen_poirazi_ref.py`
/// (4 кастомных мода с Q10-множителем из celsius + pas), значения
/// хардкодятся в REF_* ниже (CI не требует NEURON). M-ток (60 mS/cm²)
/// доминирует в покое → сома сидит на ≈−79 мВ, оба протокола
/// субпороговые (чистое V(t)-сравнение): деполяризация +2e-7 nA и
/// гиперполяризация −1e-7 nA (Ih sag). Измерено: max |ΔV| ≈ 0.0007 мВ
/// (машинная точность) — кастомный LEMS-tau, M-ток и Q10 сводятся к
/// NEURON cnexp бит-точно; допуск 0.05 мВ — запас на fast_exp (pillars)
/// vs exp (NEURON) за 6000 шагов.
const REF_V_DEPOLARIZING_SPIKES: [f32; 31] = [
    -70.000000, -79.010513, -79.173065, -78.982016, -78.795172, -78.652367, -78.552021, -78.482415,
    -78.432226, -78.393663, -78.362185, -77.978058, -77.898520, -77.886237, -77.881775, -77.875009,
    -77.866319, -77.857092, -77.848154, -77.839851, -77.832278, -78.171074, -78.217677, -78.206793,
    -78.190657, -78.178066, -78.168837, -78.161745, -78.155937, -78.150956, -78.146571,
];
const REF_V_HYPERPOLARIZING_IH: [f32; 31] = [
    -70.000000, -79.010513, -79.173065, -78.982016, -78.795172, -78.652367, -78.552021, -78.482415,
    -78.432226, -78.393663, -78.362185, -78.515226, -78.524175, -78.502677, -78.478574, -78.457299,
    -78.439086, -78.423319, -78.409464, -78.397160, -78.386164, -78.197312, -78.158430, -78.152816,
    -78.151590, -78.149506, -78.146500, -78.143141, -78.139806, -78.136670, -78.133791,
];

#[test]
fn poirazi_ca1_soma_matches_neuron() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_ca1_soma.nml");
    let base = std::fs::read_to_string(&path).unwrap();

    let cases: [(&str, &str, [f32; 31]); 2] = [
        ("hyperpolarizing_ih", "-1e-7", REF_V_HYPERPOLARIZING_IH),
        ("depolarizing_spikes", "+2e-7", REF_V_DEPOLARIZING_SPIKES),
    ];
    for (name, amp, ref_v) in cases {
        let xml = base.replace("amplitude=\"-1e-7\"", &format!("amplitude=\"{amp}\""));
        // `load_file` (не `load`) — резолвит <include href="channels/…">
        // относительно каталога файла, поэтому пишем во временную копию
        // рядом с исходным файлом.
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resources/validate/osb");
        let tmp_path = dir.join(format!("poirazi_tmp_{}_{}.nml", name, std::process::id()));
        std::fs::write(&tmp_path, &xml).unwrap();
        // Сверка при физиологической 35 °C: per-channel Q10 применяется
        // и в pillars, и в NEURON-эталоне (см. poirazi_km.mod Q10).
        let mut ln = load_file_at(&tmp_path, 35.0).unwrap();
        let _ = std::fs::remove_file(&tmp_path);
        assert!(
            ln.warnings.is_empty(),
            "{name}: неожиданные warnings: {:?}",
            ln.warnings
        );
        // Пошаговая трасса V(сома) — dt = 0.025 мс, length = 150 мс →
        // 6000 шагов + начальная точка (как NEURON Vector).
        let steps = 6000usize;
        let dt = 0.025f32;
        let mut v = Vec::with_capacity(steps + 1);
        v.push(ln.network.state(0, 0)[0]);
        for s in 0..steps {
            let t = s as f32 * dt;
            for (p, indices, amps) in ln.injections_at(t) {
                ln.network.inject_current(p, &indices, &amps);
            }
            for (p, indices, amps) in ln.clamp_currents(t) {
                ln.network.inject_current(p, &indices, &amps);
            }
            ln.network.step();
            v.push(ln.network.state(0, 0)[0]);
        }
        assert_eq!(v.len(), 6001, "{name}");
        let grid = 200usize; // 5 мс / 0.025 мс
        let mut max_d = 0.0f32;
        for (k, &rv) in ref_v.iter().enumerate() {
            let d = (v[k * grid] - rv).abs();
            max_d = max_d.max(d);
        }
        println!("poirazi CA1 soma vs NEURON [{name}]: max |ΔV| = {max_d:.4} mV");
        assert!(
            max_d < 0.05,
            "{name}: pillars vs NEURON max |ΔV| = {max_d} mV > 0.05"
        );
    }
}

/// Ca²⁺-подсистема end-to-end: `channelDensityNernst` + `species` +
/// `decayingPoolConcentrationModel` на морфо-клетке — парсится без warning,
/// [Ca²⁺] стартует с `initialConcentration` и растёт под Ca-током, V конечен.
#[test]
fn morpho_cell_ca_nernst_and_decaying_pool() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ca_cell">
  <ionChannelHH id="caChan" conductance="10pS" species="ca">
    <gateHHtauInf id="m" instances="1">
      <!-- half-activation −100 mV → ~97% open at rest (−65 mV) -->
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-100mV" scale="10mV"/>
      <timeCourse type="fixedTimeCourse" tau="0.1ms"/>
    </gateHHtauInf>
  </ionChannelHH>
  <decayingPoolConcentrationModel id="CaPool" restingConc="1e-4 mM" decayConstant="100 ms" shellThickness="0.1 um"/>
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-65mV" ion="non_specific" segmentGroup="soma_group"/>
      <channelDensityNernst id="caDens" ionChannel="caChan" condDensity="1.0 mS_per_cm2" ion="ca" segmentGroup="soma_group"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
    <species id="ca" ion="ca" concentrationModel="CaPool" initialConcentration="1e-4 mM" initialExtConcentration="2 mM"/>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.05, 200.0).unwrap();
    assert!(
        ln.warnings.is_empty(),
        "Ca-подсистема не должна давать warning: {:?}",
        ln.warnings
    );

    // [Ca] стартует с initialConcentration (1e-4 мМ = 0.1 µM).
    let ca0 = {
        let pop = ln.network.pop_mut(0).unwrap();
        let cable = pop
            .as_any_mut()
            .downcast_mut::<super::discretize::CablePopulation>()
            .unwrap();
        cable.ca_soma(0)
    };
    assert!((ca0 - 1e-4).abs() < 1e-6, "initial [Ca] = {ca0}");

    // Публичное зеркало `cable_ca` уже отражает начальную концентрацию
    // (до первого шага), а не только после `step`.
    let root = ln.cable_info(0).unwrap().root;
    let ca_init = ln.network.cable_ca(0).unwrap();
    assert!(
        (ca_init.ca[root] - ca0).abs() < 1e-6,
        "cable_ca до шага = {} != {ca0}",
        ca_init.ca[root]
    );

    let res = ln.run();
    assert!(res.steps > 0);

    // Ca-канал открыт при покое → [Ca] растёт от restingConc.
    let ca1 = {
        let pop = ln.network.pop_mut(0).unwrap();
        let cable = pop
            .as_any_mut()
            .downcast_mut::<super::discretize::CablePopulation>()
            .unwrap();
        cable.ca_soma(0)
    };
    assert!(ca1 > 1e-4, "[Ca] должен вырасти под Ca-током, got {ca1}");
    assert!(ca1.is_finite(), "[Ca] конечен: {ca1}");

    // Публичный доступ к [Ca] через Network::cable_ca (зеркало, neuron-major).
    let ca_view = ln.network.cable_ca(0).unwrap();
    assert_eq!(ca_view.n_neurons, 1);
    assert_eq!(ca_view.ca[root], ca1, "cable_ca должен совпасть с downcast");

    // Nernst-обращение + Ca-ток меняют V (деполяризация), без NaN.
    let v = ln.network.state(0, 0).to_vec();
    assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
    assert!(
        (v[0] - (-65.0)).abs() > 0.1,
        "V отклонился от покоя: {}",
        v[0]
    );
}

/// `channelPopulationNernst` (дискретная популяция Ca-каналов с Nernst-
/// обращением) парсится без warning и запускается: [Ca] растёт, V конечен.
#[test]
fn morpho_cell_channel_population_nernst_runs() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ca_pop_cell">
  <ionChannelHH id="caChan" conductance="10pS" species="ca">
    <gateHHtauInf id="m" instances="1">
      <steadyState type="HHSigmoidVariable" rate="1" midpoint="-100mV" scale="10mV"/>
      <timeCourse type="fixedTimeCourse" tau="0.1ms"/>
    </gateHHtauInf>
  </ionChannelHH>
  <decayingPoolConcentrationModel id="CaPool" restingConc="1e-4 mM" decayConstant="100 ms" shellThickness="0.1 um"/>
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="0.3 mS_per_cm2" erev="-65mV" ion="non_specific" segmentGroup="soma_group"/>
      <channelPopulationNernst id="caPop" ionChannel="caChan" number="1257" ion="ca"/>
      <initMembPotential value="-65mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
    </membraneProperties>
    <intracellularProperties><resistivity value="100 ohm_cm"/></intracellularProperties>
    <species id="ca" ion="ca" concentrationModel="CaPool" initialConcentration="1e-4 mM" initialExtConcentration="2 mM"/>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.05, 50.0).unwrap();
    assert!(
        ln.warnings.is_empty(),
        "channelPopulationNernst не должен давать warning: {:?}",
        ln.warnings
    );

    let res = ln.run();
    assert!(res.steps > 0);

    // [Ca] растёт, V конечен (стохастический Ca-канал не ломает численку).
    let root = ln.cable_info(0).unwrap().root;
    let ca = ln.network.cable_ca(0).unwrap();
    assert!(ca.ca[root] > 1e-4, "[Ca] вырос: {}", ca.ca[root]);
    assert!(ca.ca[root].is_finite(), "[Ca] конечен: {}", ca.ca[root]);
    let v = ln.network.state(0, 0).to_vec();
    assert!(v.iter().all(|x| x.is_finite()), "V конечен: {v:?}");
}

/// Неизвестная segmentGroup в channelDensity — честная ошибка.
#[test]
fn morpho_cell_unknown_segment_group_errors() {
    let xml = MORPHO_HH
        .replace("segmentGroup=\"soma_group\"/>", "segmentGroup=\"nope\"/>")
        .replace("segmentGroup=\"soma_group\">", "segmentGroup=\"nope\">");
    // Ошибку должен дать именно первый не-пассивный channelDensity.
    assert!(load_with(&xml, 0.025, 100.0).is_err());
}

/// `expTwoSynapse` с tauRise/tauDecay должен уходить в кинетический движок
/// (syn_type > 0 в графе), а `expOneSynapse` без tauDecay остаётся
/// статическим delta-синапсом (совместимость с exc↔inh-примером).
#[test]
fn exp_two_synapse_routes_to_kinetic_engine() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="kin">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <expTwoSynapse id="exc" gbase="0.5" tauRise="1ms" tauDecay="5ms" erev="0mV"/>
  <expOneSynapse id="delta" gbase="1"/>
  <network id="net">
    <population id="pre" component="rs" size="1"/>
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="exc">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="1"/>
    </projection>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="delta">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="0.1"/>
    </projection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    let types = ln.network.graph().unwrap().syn_types().to_vec();
    assert!(
        types.contains(&1),
        "expTwo должен получить kinetic id: {types:?}"
    );
    assert!(
        types.contains(&0),
        "expOne без tauDecay остаётся static: {types:?}"
    );
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // Кинетический путь не портит состояние: прогон без NaN.
    ln.run();
    for pid in 0..ln.network.num_populations() {
        assert!(ln.network.state(pid, 0).iter().all(|v| v.is_finite()));
    }
}

/// `alphaSynapse` и `expThreeSynapse` — полные кинетические профили:
/// регистрируются в кинетическом движке (syn_type > 0) и прогоняются без
/// NaN (прежний skip-list их молча отбрасывал бы с warning).
#[test]
fn alpha_and_exp_three_synapses_route_to_kinetic_engine() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="kin2">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <alphaSynapse id="alpha" gbase="0.3" tau="2ms" erev="0mV"/>
  <expThreeSynapse id="e3" tauRise="1ms" tauDecay1="4ms" tauDecay2="12ms" gbase1="0.5" gbase2="0.2" erev="-65mV"/>
  <network id="net">
    <population id="pre" component="rs" size="1"/>
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="alpha">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="1"/>
    </projection>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="e3">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="1"/>
    </projection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    let types = ln.network.graph().unwrap().syn_types().to_vec();
    assert!(
        types.contains(&1),
        "alpha должен получить kinetic id: {types:?}"
    );
    assert!(
        types.contains(&2),
        "expThree должен получить kinetic id: {types:?}"
    );
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // Кинетический путь не портит состояние: прогон без NaN.
    ln.run();
    for pid in 0..ln.network.num_populations() {
        assert!(ln.network.state(pid, 0).iter().all(|v| v.is_finite()));
    }
}

/// `blockingPlasticSynapse` с `tsodyksMarkramDepMechanism` — коротко-
/// временная пластика: парсер больше не пропускает её с warning, а
/// компилятор регистрирует Tsodyks–Markram-профиль (plast_types > 0)
/// рядом с кинетикой (syn_types > 0). Прогон не портит состояние.
#[test]
fn blocking_plastic_synapse_registers_tm_profile() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="plast">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <blockingPlasticSynapse id="pl" gbase="0.5" tauRise="1ms" tauDecay="5ms" erev="0mV">
    <tsodyksMarkramDepMechanism initReleaseProb="0.5" tauRec="500ms"/>
  </blockingPlasticSynapse>
  <network id="net">
    <population id="pre" component="rs" size="1"/>
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="pl">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="1"/>
    </projection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let plast = ln.network.plast_types().to_vec();
    assert!(
        plast.iter().any(|&p| p > 0),
        "blockingPlasticSynapse должен получить TM-профиль: {plast:?}"
    );
    let syn = ln.network.graph().unwrap().syn_types().to_vec();
    assert!(
        syn.iter().any(|&s| s > 0),
        "expTwo-основа должна получить kinetic id: {syn:?}"
    );
    ln.run();
    for pid in 0..ln.network.num_populations() {
        assert!(ln.network.state(pid, 0).iter().all(|v| v.is_finite()));
    }
}

/// `doubleSynapse` — одна связь, два независимых механизма (AMPA + NMDA):
/// компилятор опускает её в **два ребра** на connection — static-ампа и
/// кинетический-нмда, каждый со своим gbase/профилем. Прогон не портит
/// состояние.
#[test]
fn double_synapse_lowers_to_two_edges() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ds">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <expOneSynapse id="ampa" gbase="0.5"/>
  <expTwoSynapse id="nmda" gbase="0.3" tauRise="1ms" tauDecay="5ms" erev="0mV"/>
  <doubleSynapse id="ds" synapse1="ampa" synapse2="nmda"/>
  <network id="net">
    <population id="pre" component="rs" size="1"/>
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pre" postsynapticPopulation="post" synapse="ds">
      <connection preCellId="../pre[0]" postCellId="../post[0]" weight="1"/>
    </projection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let graph = ln.network.graph().unwrap();
    let syn = graph.syn_types().to_vec();
    assert_eq!(syn.len(), 2, "одна связь → два ребра: {syn:?}");
    assert!(syn.contains(&0), "AMPA (без tauDecay) → static: {syn:?}");
    assert!(syn.contains(&1), "NMDA (с тау) → kinetic id 1: {syn:?}");
    // Оба ребра — от pre к post.
    let row = graph.row(0);
    assert!(
        row.col.contains(&1),
        "pre(0) должен иметь оба ребра к post(1)"
    );
    ln.run();
    for pid in 0..ln.network.num_populations() {
        assert!(ln.network.state(pid, 0).iter().all(|v| v.is_finite()));
    }
}

/// `doubleSynapse` как синапс poisson-проекции — бессмысленно (два
/// механизма на одном источнике) → честная ошибка.
#[test]
fn double_synapse_as_poisson_synapse_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ds_pois">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <expOneSynapse id="ampa" gbase="0.5"/>
  <expTwoSynapse id="nmda" gbase="0.3" tauRise="1ms" tauDecay="5ms" erev="0mV"/>
  <doubleSynapse id="ds" synapse1="ampa" synapse2="nmda"/>
  <poissonFiringSynapse id="pois" averageRate="100Hz"/>
  <network id="net">
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pois" postsynapticPopulation="post" synapse="ds">
      <connection preCellId="../pois[0]" postCellId="../post[0]" weight="1"/>
    </projection>
  </network>
</neuroml>
"#;
    let err = match load_with(xml, 0.5, 50.0) {
        Err(e) => e,
        Ok(_) => panic!("doubleSynapse + poisson-источник должен быть ошибкой"),
    };
    assert!(
        err.to_string().contains("doubleSynapse"),
        "ошибка должна называть doubleSynapse: {err}"
    );
}

/// `to_hz` разбирает NeuroML-единицы частоты (`Hz`/`per_s`/`kHz`/`per_ms`)
/// и отклоняет несовместимые единицы.
#[test]
fn to_hz_parses_rate_units() {
    assert_eq!(parse::to_hz("100Hz").unwrap(), 100.0);
    assert_eq!(parse::to_hz("100 per_s").unwrap(), 100.0);
    assert_eq!(parse::to_hz("0.1kHz").unwrap(), 100.0);
    assert_eq!(parse::to_hz("0.1per_ms").unwrap(), 100.0);
    assert!(parse::to_hz("100mA").is_err());
}

/// `intracellularProperties/resistivity` — NeuroML2 пишет `kohm_cm`
/// (Poirazi `0.05kohm_cm` = 50 Ω·cm). Раньше `raw` молча отбрасывал суффикс
/// → Ra = 0.05 Ω·cm, что замыкало аксиальный импеданс в 1000× и давало
/// неправильный покой полной клетки.
#[test]
fn to_ohm_cm_parses_resistivity_units() {
    assert_eq!(parse::to_ohm_cm("0.05kohm_cm").unwrap(), 50.0);
    assert_eq!(parse::to_ohm_cm("100 ohm_cm").unwrap(), 100.0);
    assert_eq!(parse::to_ohm_cm("1Mohm_cm").unwrap(), 1e6);
    assert!(parse::to_ohm_cm("0.05kohm").is_err());
}

/// `spikeArray` (и `spikeArrayRef`) через `explicitInput` дают детерминированные
/// одиночные `dt`-импульсы тока в перечисленные моменты, квантованные к сетке.
#[test]
fn spike_array_input_schedules_deterministic_pulses() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="spikes">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeArray id="sa">
    <spike id="0" time="5ms"/>
    <spike id="1" time="15ms"/>
  </spikeArray>
  <spikeArrayRef id="sar" ref="sa"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="sar"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 20.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // До первого спайка и между спайками — тишина.
    assert!(ln.injections_at(0.0).is_empty());
    assert!(ln.injections_at(7.0).is_empty());
    // Ровно в перечисленные моменты — один dt-импульс.
    let at5 = ln.injections_at(5.0);
    assert_eq!(at5.len(), 1, "{at5:?}");
    assert_eq!(at5[0].0, 0);
    assert_eq!(at5[0].1, vec![0u32]);
    assert!((at5[0].2[0] - 1.0).abs() < 1e-6);
    assert_eq!(ln.injections_at(15.0).len(), 1);
    // После последнего спайка — снова тишина.
    assert!(ln.injections_at(15.5).is_empty());
}

/// `poissonFiringSynapse` через `explicitInput` — детерминированный (seeded,
/// stateless) пуассоновский трейн: `rate·dt` ограничен, не каждый шаг, и
/// воспроизводим между вызовами.
#[test]
fn poisson_input_deterministic_and_bounded() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="pois">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <poissonFiringSynapse id="p" averageRate="1000Hz"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="p"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 100.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // p = rate·dt = 1000 Гц · 0.5 мс = 0.5 на шаг.
    let mut hits = 0usize;
    let mut steps = 0usize;
    let mut t = 0.0f32;
    while t < 100.0 {
        steps += 1;
        let n = ln.injections_at(t).len();
        hits += n;
        // Детерминированность: тот же t → тот же бросок.
        assert_eq!(ln.injections_at(t).len(), n);
        t += 0.5;
    }
    assert!(steps > 0);
    assert!(hits > 0, "p=0.5 должен стрелять на 200 шагах");
    assert!(
        hits < steps,
        "p=0.5 не должен стрелять каждый шаг: {hits}/{steps}"
    );
}

/// `poissonFiringSynapse` при 0 Гц никогда не инжектирует.
#[test]
fn poisson_zero_rate_never_spikes() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="pois0">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <poissonFiringSynapse id="p" averageRate="0Hz"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="p"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 50.0).unwrap();
    for s in 0..100 {
        assert!(ln.injections_at(s as f32 * 0.5).is_empty());
    }
}

/// `transientPoissonFiringSynapse` через `explicitInput` — Poisson только
/// внутри `[delay, delay+duration]`, тишина до/после окна.
#[test]
fn transient_poisson_input_respects_window() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="tpois">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <transientPoissonFiringSynapse id="tp" averageRate="2000Hz" delay="10ms" duration="20ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="tp"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // До delay (10 мс) — тишина; внутри [10, 30) — стреляет (p = rate·dt = 1);
    // после delay+duration (30 мс) — снова тишина.
    assert!(ln.injections_at(5.0).is_empty());
    assert!(!ln.injections_at(10.0).is_empty());
    assert!(!ln.injections_at(29.5).is_empty());
    assert!(ln.injections_at(30.0).is_empty());
}

/// `spikeGenerator` (регулярный период) и `spikeGeneratorPoisson` через
/// `explicitInput`: регулярный — точный период; Poisson — ограниченный и
/// детерминированный.
#[test]
fn spike_generator_inputs() {
    // Регулярный: первый спайк на t=period, затем каждый period.
    let regular = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sgreg">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeGenerator id="sg" period="5ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="sg"/>
  </network>
</neuroml>
"#;
    let ln = load_with(regular, 0.5, 100.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(!ln.injections_at(5.0).is_empty());
    assert!(!ln.injections_at(10.0).is_empty());
    assert!(ln.injections_at(2.0).is_empty());
    assert!(ln.injections_at(5.5).is_empty()); // нет спайка в [5.5, 6.0)

    // Poisson: p = 0.5 на шаге — не каждый шаг, но детерминирован.
    let poisson = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sgpois">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeGeneratorPoisson id="sgp" averageRate="1000Hz"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="sgp"/>
  </network>
</neuroml>
"#;
    let ln = load_with(poisson, 0.5, 100.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let mut hits = 0usize;
    let mut steps = 0usize;
    let mut t = 0.0f32;
    while t < 100.0 {
        steps += 1;
        let n = ln.injections_at(t).len();
        hits += n;
        assert_eq!(ln.injections_at(t).len(), n); // детерминирован
        t += 0.5;
    }
    assert!(hits > 0, "p=0.5 должен стрелять");
    assert!(hits < steps, "p=0.5 не каждый шаг: {hits}/{steps}");
}

/// Total current injected into the (single-neuron) target at `t_ms` across
/// all scheduled stimuli — compounds and ramp baselines may stack several.
fn sum_injected(ln: &LoadedNetwork, t_ms: f32) -> f32 {
    ln.injections_at(t_ms)
        .iter()
        .map(|(_, _, amps)| amps.iter().sum::<f32>())
        .sum()
}

/// `sineGenerator` — LEMS: `i = A·sin(phase + 2π·(t−delay)/period)` внутри
/// окна, 0 вне (phase в радианах). t=0 → 0, t=1 → +5, t=3 → −5, t=4 → 0.
#[test]
fn sine_generator_injects_sinusoid() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sine">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <sineGenerator id="s" amplitude="5" delay="0ms" duration="10ms" period="4ms" phase="0"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="s"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 20.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(sum_injected(&ln, 0.0).abs() < 1e-4, "sin(0) = 0");
    assert!((sum_injected(&ln, 1.0) - 5.0).abs() < 1e-4, "t=1 → +5");
    assert!(sum_injected(&ln, 2.0).abs() < 1e-4, "t=2 → 0");
    assert!((sum_injected(&ln, 3.0) + 5.0).abs() < 1e-4, "t=3 → −5");
    assert!(sum_injected(&ln, 4.0).abs() < 1e-3, "t=4 → ~0");
    // Вне окна [0, 10) — тишина.
    assert!(ln.injections_at(10.0).is_empty());
    assert!(ln.injections_at(15.0).is_empty());
}

/// `rampGenerator` — LEMS: линейно `startAmplitude → finishAmplitude`
/// внутри `[delay, delay+duration)`; с `baselineAmplitude = 0` вне окна
/// тока нет.
#[test]
fn ramp_generator_injects_linear_ramp() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ramp">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <rampGenerator id="r" startAmplitude="0" finishAmplitude="10" baselineAmplitude="0" delay="2ms" duration="4ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="r"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 20.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(ln.injections_at(1.0).is_empty(), "до delay — тишина");
    assert!(sum_injected(&ln, 2.0).abs() < 1e-4, "t=delay → start");
    assert!((sum_injected(&ln, 3.0) - 2.5).abs() < 1e-4, "t=3 → 25%");
    assert!((sum_injected(&ln, 5.0) - 7.5).abs() < 1e-4, "t=5 → 75%");
    assert!(
        (sum_injected(&ln, 5.5) - 8.75).abs() < 1e-4,
        "t=5.5 → 87.5%"
    );
    assert!(
        ln.injections_at(6.0).is_empty(),
        "после окна (baseline 0) — тишина"
    );
}

/// `rampGenerator` с `baselineAmplitude ≠ 0`: baseline держится до delay и
/// после окна (LEMS), рампа — только внутри.
#[test]
fn ramp_generator_holds_baseline_outside_window() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="rampb">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <rampGenerator id="r" startAmplitude="0" finishAmplitude="10" baselineAmplitude="3" delay="2ms" duration="4ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="r"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 20.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(
        (sum_injected(&ln, 0.0) - 3.0).abs() < 1e-4,
        "baseline до delay"
    );
    assert!((sum_injected(&ln, 1.0) - 3.0).abs() < 1e-4);
    assert!(
        (sum_injected(&ln, 3.0) - 2.5).abs() < 1e-4,
        "рампа без baseline"
    );
    assert!(
        (sum_injected(&ln, 6.0) - 3.0).abs() < 1e-4,
        "baseline после окна"
    );
    assert!(
        (sum_injected(&ln, 12.0) - 3.0).abs() < 1e-4,
        "baseline держится"
    );
}

/// `compoundInput` — сумма детей: каждый ребёнок становится отдельным
/// стимулом на ту же цель, итоговый ток = Σ.
#[test]
fn compound_input_sums_child_generators() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="comp">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <pulseGenerator id="p" delay="0ms" duration="5ms" amplitude="2"/>
  <sineGenerator id="s" amplitude="3" delay="0ms" duration="20ms" period="10ms" phase="0"/>
  <compoundInput id="c" weight="1">
    <input id="0" component="p"/>
    <input id="1" component="s"/>
  </compoundInput>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="c"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 25.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // t=1: pulse 2 + sine 3·sin(2π/10) ≈ 2 + 1.7634 = 3.7634
    assert!(
        (sum_injected(&ln, 1.0) - 3.7634).abs() < 1e-3,
        "t=1: got {}",
        sum_injected(&ln, 1.0)
    );
    // t=6: pulse кончился; sine 3·sin(2π·6/10) ≈ −1.7634
    assert!(
        (sum_injected(&ln, 6.0) + 1.7634).abs() < 1e-3,
        "t=6: got {}",
        sum_injected(&ln, 6.0)
    );
}

/// `timedSynapticInput` — спайки по расписанию через синапс: импульсы
/// `weight·gbase` в перечисленные моменты (delta-понижение, как spikeArray).
#[test]
fn timed_synaptic_input_pulses_with_gbase() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="tsi">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <expOneSynapse id="exc" gbase="0.5"/>
  <timedSynapticInput id="tsi" synapse="exc" weight="2">
    <spike id="0" time="5ms"/>
    <spike id="1" time="15ms"/>
  </timedSynapticInput>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="tsi"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 25.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(
        (sum_injected(&ln, 5.0) - 1.0).abs() < 1e-4,
        "weight·gbase = 2·0.5"
    );
    assert!((sum_injected(&ln, 15.0) - 1.0).abs() < 1e-4);
    assert!(ln.injections_at(10.0).is_empty());
}

/// `spikeGeneratorRandom` — равномерный ISI в `[minISI, maxISI]`:
/// расписание детерминировано (один seed → одинаковый трейн), все ISIs в
/// границах, окно [0, duration) уважается.
#[test]
fn spike_generator_random_isi_respects_bounds_and_seed() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sgrnd">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeGeneratorRandom id="sr" minISI="5ms" maxISI="20ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="sr"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 1000.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let times = match &ln.inputs[0].kind {
        StimulusKind::SpikeTimes { times_ms } => times_ms.clone(),
        k => panic!("expected SpikeTimes, got {k:?}"),
    };
    assert!(
        times.len() > 20,
        "1000ms / ~12.5ms ISI → десятки спайков: {}",
        times.len()
    );
    assert!(times.last().unwrap() < &1000.0);
    let mut prev = 0.0f32;
    for &t in &times {
        let isi = t - prev;
        assert!((4.99..=20.01).contains(&isi), "ISI {isi} вне [5, 20]");
        prev = t;
    }
    // Детерминизм: повторная загрузка даёт побитово тот же трейн.
    let again = load_with(xml, 0.5, 1000.0).unwrap();
    let times2 = match &again.inputs[0].kind {
        StimulusKind::SpikeTimes { times_ms } => times_ms.clone(),
        _ => unreachable!(),
    };
    assert_eq!(times, times2, "один seed → одинаковое расписание");
}

/// `spikeGeneratorRefPoisson` — максимально-энтропийный ISI на
/// `[minimumISI, ∞)` со средним 1/rate: все ISIs ≥ minimumISI, среднее ≈
/// 1/rate, детерминировано.
#[test]
fn spike_generator_ref_poisson_isi_respects_minimum_and_mean() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sgrp">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeGeneratorRefPoisson id="srp" averageRate="100Hz" minimumISI="2ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="srp"/>
  </network>
</neuroml>
"#;
    let ln = load_with(xml, 0.5, 5000.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let times = match &ln.inputs[0].kind {
        StimulusKind::SpikeTimes { times_ms } => times_ms.clone(),
        k => panic!("expected SpikeTimes, got {k:?}"),
    };
    assert!(
        times.len() > 200,
        "~500 спайков за 5s при 100Hz: {}",
        times.len()
    );
    let mut sum = 0.0f32;
    let mut prev = 0.0f32;
    for &t in &times {
        let isi = t - prev;
        assert!(isi >= 1.99, "ISI {isi} < minimumISI 2ms");
        sum += isi;
        prev = t;
    }
    let mean = sum / times.len() as f32;
    assert!(
        (mean - 10.0).abs() < 1.0,
        "mean ISI {mean:.2} ≈ 1/rate = 10ms"
    );
    // Детерминизм.
    let again = load_with(xml, 0.5, 5000.0).unwrap();
    let times2 = match &again.inputs[0].kind {
        StimulusKind::SpikeTimes { times_ms } => times_ms.clone(),
        _ => unreachable!(),
    };
    assert_eq!(times, times2);
}

/// `spikeGeneratorRefPoisson` со средним ISI ≤ minimumISI физически
/// невозможен (среднее распределения на [min, ∞) ≥ min) → честная ошибка.
#[test]
fn spike_generator_ref_poisson_mean_below_minimum_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sgrp_bad">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeGeneratorRefPoisson id="srp" averageRate="1000Hz" minimumISI="5ms"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="srp"/>
  </network>
</neuroml>
"#;
    let err = match load_with(xml, 0.5, 100.0) {
        Err(e) => e,
        Ok(_) => panic!("avg ISI 1ms <= minimumISI 5ms должен быть ошибкой"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("average ISI") && msg.contains("minimumISI"),
        "ошибка должна объяснять ограничение: {msg}"
    );
}

/// `voltageClamp` — патч-кламп: пока активно окно [delay, delay+duration],
/// каждый шаг инжектит `weight·(target − V)/R` (пропорциональный
/// контроллер) против живого V — мембрана тянется к target; до окна —
/// покой, после — релаксация к leakReversal.
#[test]
fn voltage_clamp_pulls_membrane_to_target() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="clamp">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <voltageClamp id="vc" targetVoltage="-60mV" simpleSeriesResistance="200Mohm" delay="5ms" duration="20ms" weight="1"/>
  <network id="net">
    <population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="vc"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert_eq!(ln.clamps.len(), 1, "clamp зарегистрирован");
    assert!(matches!(
        ln.clamps[0].schedule,
        ClampSchedule::Window { target_mv: -60.0 }
    ));
    assert_eq!(ln.clamps[0].series_resistance_mohm, 200.0);

    let mut v4 = 0.0f32;
    let mut v15 = 0.0f32;
    let mut v24 = 0.0f32;
    let mut v45 = 0.0f32;
    for s in 0..100 {
        let t = s as f32 * 0.5;
        for (pop, indices, amps) in ln.injections_at(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        for (pop, indices, amps) in ln.clamp_currents(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        ln.network.step();
        let v = ln.network.state(0, 0)[0];
        match t as i32 {
            4 => v4 = v,
            15 => v15 = v,
            24 => v24 = v,
            45 => v45 = v,
            _ => {}
        }
    }
    assert!((v4 - (-70.0)).abs() < 0.5, "до окна — покой: {v4}");
    // Сильный контроллер (R = 200 MΩ ≪ R_in) держит V в ~0.4 мВ от target.
    assert!((v15 + 60.0).abs() < 1.5, "в окне V ≈ target −60: {v15}");
    assert!((v24 + 60.0).abs() < 1.5, "конец окна: {v24}");
    assert!(v45 < -64.0, "после окна релаксация к −70: {v45}");
}

/// `voltageClampTriple` — три ступени: conditioning до delay, testing в
/// окне, return держится **навсегда** после окна. `active=0` — зажим
/// выключен (покой всё время).
#[test]
fn voltage_clamp_triple_steps_and_holds_return() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="vc3">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <voltageClampTriple id="vc3" active="1" delay="5ms" duration="20ms" conditioningVoltage="-70mV" testingVoltage="-60mV" returnVoltage="-70mV" simpleSeriesResistance="200Mohm"/>
  <network id="net">
    <population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="vc3"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    assert!(matches!(
        ln.clamps[0].schedule,
        ClampSchedule::Triple {
            conditioning_mv: -70.0,
            testing_mv: -60.0,
            return_mv: -70.0,
            active: true
        }
    ));

    let mut v2 = 0.0f32;
    let mut v15 = 0.0f32;
    let mut v45 = 0.0f32;
    for s in 0..100 {
        let t = s as f32 * 0.5;
        for (pop, indices, amps) in ln.injections_at(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        for (pop, indices, amps) in ln.clamp_currents(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        ln.network.step();
        let v = ln.network.state(0, 0)[0];
        match t as i32 {
            2 => v2 = v,
            15 => v15 = v,
            45 => v45 = v,
            _ => {}
        }
    }
    assert!((v2 + 70.0).abs() < 0.5, "conditioning −70: {v2}");
    assert!((v15 + 60.0).abs() < 1.5, "testing −60: {v15}");
    assert!((v45 + 70.0).abs() < 1.5, "return держится −70: {v45}");

    // active=0 → зажим выключен: V остаётся на покое весь прогон.
    let off = xml.replace("active=\"1\"", "active=\"0\"");
    let mut ln_off = load_with(&off, 0.5, 50.0).unwrap();
    for s in 0..100 {
        let t = s as f32 * 0.5;
        for (pop, indices, amps) in ln_off.injections_at(t) {
            ln_off.network.inject_current(pop, &indices, &amps);
        }
        for (pop, indices, amps) in ln_off.clamp_currents(t) {
            ln_off.network.inject_current(pop, &indices, &amps);
        }
        ln_off.network.step();
    }
    let v_end = ln_off.network.state(0, 0)[0];
    assert!(
        (v_end + 70.0).abs() < 0.5,
        "active=0 → покой всю дорогу: {v_end}"
    );
}

/// `voltageClamp` (Window) против NEURON `SEClamp`: тот же proportional
/// controller `i = (V_cmd − V)/rs` и та же оконная семантика. LEMS
/// `voltageClamp` даёт `i=0` до `delay` и после `delay+duration`, ток только
/// в окне; NEURON `SEClamp` (dur1=delay, amp1=покой, dur2=duration,
/// amp2=target) выключается после `dur1+dur2` — 1:1. Эталон `REF_V_CLAMP`
/// генерится `gen_clamp_ref.py` (pas-компартмент C=3.2 пФ / 0.2 нС +
/// SEClamp rs=200 МΩ → равновесие −60.38 мВ).
///
/// NB: `voltageClampTriple` (LEMS держит `returnVoltage` навсегда) **не**
/// мапится на SEClamp — тот выключается после 2-й ступени (amp3 игнорируется).
#[test]
fn voltage_clamp_matches_neuron_seclamp() {
    const REF_V: [f32; 12] = [
        -70.000000, // t=0.0
        -70.000000, // t=2.5
        -60.876175, // t=7.5
        -60.409745, // t=10.0
        -60.384681, // t=15.0
        -60.384616, // t=20.0
        -60.384615, // t=22.5
        -61.755832, // t=27.5
        -62.931504, // t=30.0
        -64.803781, // t=35.0
        -66.724874, // t=42.5
        -67.592378, // t=47.5
    ];
    // Индексы шага (dt=0.5 мс): t = i·0.5 → v[i] = V(i·0.5).
    const GRID: [usize; 12] = [0, 5, 15, 20, 30, 40, 45, 55, 60, 70, 85, 95];

    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="vc">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <voltageClamp id="vc" targetVoltage="-60mV" simpleSeriesResistance="200Mohm" delay="5ms" duration="20ms" weight="1"/>
  <network id="net">
    <population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="vc"/>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 50.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);

    // Сэмплирование как в NEURON `Vector.record`: v[k] = V(k·0.5).
    let mut v = Vec::with_capacity(101);
    v.push(ln.network.state(0, 0)[0]); // V(0)
    for s in 0..100 {
        let t = s as f32 * 0.5;
        for (pop, indices, amps) in ln.clamp_currents(t) {
            ln.network.inject_current(pop, &indices, &amps);
        }
        ln.network.step();
        v.push(ln.network.state(0, 0)[0]); // V((s+1)·0.5)
    }

    let mut d = 0.0f32;
    for (&idx, &ref_v) in GRID.iter().zip(REF_V.iter()) {
        d = d.max((v[idx] - ref_v).abs());
    }
    println!("voltageClamp vs NEURON SEClamp: max |ΔV| = {d:.6} mV");
    // Полу-неявный контроллер (читает V на старте шага) vs неявный SEClamp —
    // расхождение только на быстром фронте (τ_eff≈0.6 мс vs dt=0.5 мс);
    // равновесие −60.38 совпадает до <0.001 мВ.
    assert!(d < 0.6, "max |ΔV| = {d:.4} mV");
}

/// `spikeArrayRef` на несуществующий массив — честная ошибка, а не тишина.
#[test]
fn spike_array_ref_to_unknown_array_errors() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="badref">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <spikeArrayRef id="sar" ref="missing"/>
  <network id="net">
    <population id="pop" component="rs" size="1"/>
    <explicitInput target="pop[0]" input="sar"/>
  </network>
</neuroml>
"#;
    assert!(load(xml).is_err());
}

/// `poissonFiringSynapse` как **пресинаптическая** популяция в `<projection>`:
/// её пуассоновские спайки идут через синапсы (weight × gbase) на пост-нейрон.
/// Высокая частота (p=1) заставляет пост-нейрон стрелять; 0 Гц — тишина.
#[test]
fn poisson_presynaptic_projection_drives_post_neuron() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="poisproj">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <poissonFiringSynapse id="pois" averageRate="2000Hz"/>
  <network id="net">
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pois" postsynapticPopulation="post">
      <connection preCellId="../pois[0]" postCellId="../post[0]" weight="10"/>
    </projection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 200.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    let res = ln.run();
    assert!(
        res.spike_counts[0] > 0,
        "p=1 Poisson drive (weight 10) должен стрелять RS-нейрон: {:?}",
        res.spike_counts
    );

    // 0 Гц — пост-нейрон молчит.
    let xml0 = xml.replace("averageRate=\"2000Hz\"", "averageRate=\"0Hz\"");
    let mut ln0 = load_with(&xml0, 0.5, 200.0).unwrap();
    let res0 = ln0.run();
    assert_eq!(res0.spike_counts[0], 0, "0 Гц не должен стрелять");
}

/// У `poissonFiringSynapse` один источник: `preCellId` с индексом != 0 —
/// честная ошибка, а не тихое игнорирование.
#[test]
fn poisson_projection_rejects_nonzero_source_index() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="poisbad">
  <izhikevichCell id="rs" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65.0mV" d="8"/>
  <poissonFiringSynapse id="pois" averageRate="100Hz"/>
  <network id="net">
    <population id="post" component="rs" size="1"/>
    <projection presynapticPopulation="pois" postsynapticPopulation="post">
      <connection preCellId="../pois[1]" postCellId="../post[0]" weight="10"/>
    </projection>
  </network>
</neuroml>
"#;
    assert!(load(xml).is_err());
}

/// `gapJunction` + `electricalProjection` → симметричная омическая связь:
/// деполяризованный нейрон тянет второй вверх, а сам падает (обмен `g·ΔV`).
#[test]
fn electrical_projection_couples_neurons() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="electrical">
  <iafTauCell id="lif" tau="10ms" leakReversal="-70mV" thresh="-54mV" reset="-70mV" refract="0ms"/>
  <gapJunction id="gj" conductance="0.5nS"/>
  <network id="net">
    <population id="pop" component="lif" size="2"/>
    <electricalProjection id="ep" presynapticPopulation="pop" postsynapticPopulation="pop" gapJunction="gj">
      <electricalConnection id="0" preCellId="../pop[0]" postCellId="../pop[1]"/>
    </electricalProjection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 10.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // Нейрон 0 деполяризован; щелевой контакт тянет нейрон 1 вверх.
    ln.network
        .pop_typed_mut::<crate::zoo::neurons::LIFModel>(0)
        .unwrap()
        .state_slice_mut(0)[0] = -55.0;
    ln.network.step();
    let v0 = ln.network.state(0, 0)[0];
    let v1 = ln.network.state(0, 0)[1];
    assert!(v1 > -70.0, "связанный нейрон должен подняться: {v1}");
    assert!(v0 < -55.0, "ведущий нейрон должен упасть: {v0}");
}

/// `linearGradedSynapse` + `continuousProjection` → односторонняя омическая
/// связь («односторонний gap junction»): пост-нейрон получает `g·(V_pre − V_post)`.
#[test]
fn continuous_projection_is_one_way() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="graded">
  <iafTauCell id="lif" tau="10ms" leakReversal="-70mV" thresh="-54mV" reset="-70mV" refract="0ms"/>
  <linearGradedSynapse id="gs" conductance="0.5nS"/>
  <network id="net">
    <population id="pop" component="lif" size="2"/>
    <continuousProjection id="cp" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="gs">
      <continuousConnection id="0" preCellId="../pop[0]" postCellId="../pop[1]"/>
    </continuousProjection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 10.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    ln.network
        .pop_typed_mut::<crate::zoo::neurons::LIFModel>(0)
        .unwrap()
        .state_slice_mut(0)[0] = -55.0;
    ln.network.step();
    let v0 = ln.network.state(0, 0)[0];
    let v1 = ln.network.state(0, 0)[1];
    assert!(v1 > -70.0, "пост-нейрон должен подняться: {v1}");
    // Пре-нейрон не тянется обратно (односторонняя связь).
    assert!(
        (v0 - (-55.75)).abs() < 1e-3,
        "пре-нейрон не должен получать ток от graded-связи: {v0}"
    );
}

/// Сигмоидный `gradedSynapse` + `continuousProjection`: активация `s` следует
/// за V_pre, ток `w·g·s·(erev − V_post)` деполяризует пост-нейрон.
#[test]
fn sigmoid_graded_synapse_drives_post() {
    let xml = r#"
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sigmoid">
  <iafTauCell id="lif" tau="10ms" leakReversal="-70mV" thresh="-54mV" reset="-70mV" refract="0ms"/>
  <gradedSynapse id="gs" conductance="1nS" delta="5mV" k="0.5per_ms" Vth="-50mV" erev="0mV"/>
  <network id="net">
    <population id="pop" component="lif" size="2"/>
    <continuousProjection id="cp" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="gs">
      <continuousConnection id="0" preCellId="../pop[0]" postCellId="../pop[1]"/>
    </continuousProjection>
  </network>
</neuroml>
"#;
    let mut ln = load_with(xml, 0.5, 10.0).unwrap();
    assert!(ln.warnings.is_empty(), "warnings: {:?}", ln.warnings);
    // Пре-нейрон выше Vth=-50, пост в покое.
    ln.network
        .pop_typed_mut::<crate::zoo::neurons::LIFModel>(0)
        .unwrap()
        .state_slice_mut(0)[0] = -40.0;
    ln.network.step();
    let v1 = ln.network.state(0, 0)[1];
    assert!(v1 > -70.0, "пост должен деполяризоваться: {v1}");
}
