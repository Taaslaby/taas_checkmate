//! 🔬 Cable (Hines) solver benchmark — end-to-end via `load()` → `run()`.
//!
//! Measures the morpho-cell pipeline at NEURON scale:
//!
//! | Bench | Морфология | Каналы |
//! |-------|-----------|--------|
//! | `cable_passive_1k` | сома + дендрит L=100000 мкм → 1001 компартмент | pas |
//! | `cable_passive_10k` | сома + дендрит L=1e6 мкм → 10001 компартмент | pas |
//! | `cable_active_1k` | 1001 компартмент | pas + NaV/KV (m³h/n⁴) на соме |
//!
//! Протокол: dt=0.025 мс, 5 мс симуляции (200 шагов), импульс 0.1 nA в сому.
//! Через `Throughput` отдаются компартмент-шаги/сек.
//!
//! Запуск:
//! ```bash
//! cargo bench --package pillars --bench bench_cable
//! ```

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use pillars::api::neuroml::{load_with, LoadedNetwork};

const DT_MS: f32 = 0.025;
/// 5 мс симуляции = 200 шагов на итерацию.
const DURATION_MS: f32 = 5.0;

/// NeuroML для морфо-клетки: сома L=20/d=20 + дендрит L=`dend_len`/d=2.
/// При d_λ=0.1 дендрит даёт `dend_len/100` компартментов (λ=1000 мкм).
fn morpho_xml(dend_len_um: f32, active: bool) -> String {
    let channels = if active {
        r#"
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
  <channelDensity id="na" ionChannel="naChans" ion="na" condDensity="120 mS_per_cm2" erev="50mV" segmentGroup="soma_group"/>
  <channelDensity id="k" ionChannel="kChans" ion="k" condDensity="36 mS_per_cm2" erev="-77mV" segmentGroup="soma_group"/>
"#
    } else {
        ""
    };
    format!(
        r#"<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="cable_bench">
  <morphology id="m1">
    <segment id="0" name="soma">
      <proximal x="0" y="0" z="0" diameter="20"/>
      <distal x="0" y="20" z="0" diameter="20"/>
    </segment>
    <segment id="1" name="dend">
      <segmentParent segment="0"/>
      <proximal x="0" y="20" z="0" diameter="2"/>
      <distal x="0" y="{dend_y}" z="0" diameter="2"/>
    </segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
    <segmentGroup id="dendrite_group"><member segment="1"/></segmentGroup>
  </morphology>
  <biophysicalProperties id="bp1">
    <membraneProperties>
      <channelDensity id="pasChans" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific" segmentGroup="all"/>
      <initMembPotential value="-70mV"/>
      <specificCapacitance value="1.0 uF_per_cm2"/>
      {channels}
    </membraneProperties>
    <intracellularProperties>
      <resistivity value="100 ohm_cm"/>
    </intracellularProperties>
  </biophysicalProperties>
  <cell id="c1" morphology="m1" biophysicalProperties="bp1"/>
  <pulseGenerator id="pg" delay="0ms" duration="5ms" amplitude="0.1"/>
  <network id="net">
    <population id="pop" component="c1" size="1"/>
    <explicitInput target="pop[0]" input="pg"/>
  </network>
</neuroml>
"#,
        dend_y = 20.0 + dend_len_um,
        channels = channels
    )
}

fn load_cable(dend_len_um: f32, active: bool) -> LoadedNetwork {
    let xml = morpho_xml(dend_len_um, active);
    load_with(&xml, DT_MS, DURATION_MS).expect("cable bench XML must load")
}

fn bench_group(c: &mut Criterion, name: &str, dend_len_um: f32, active: bool, n_comps: usize) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(
        (n_comps * (DURATION_MS / DT_MS) as usize) as u64,
    ));
    group.bench_function("run", |b| {
        b.iter_batched(
            || load_cable(dend_len_um, active),
            |mut ln| {
                let res = ln.run();
                black_box(res.steps);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn cable_benchmarks(c: &mut Criterion) {
    // d_λ=0.1, λ(дендрит)=1000 мкм → сегменты 100 мкм.
    bench_group(c, "cable_passive_1k", 100_000.0, false, 1001);
    bench_group(c, "cable_passive_10k", 1_000_000.0, false, 10_001);
    bench_group(c, "cable_active_1k", 100_000.0, true, 1001);
}

criterion_group!(benches, cable_benchmarks);
criterion_main!(benches);
