//! Валидационные бенчмарки против эталонов NEURON / Brian2 / OSB.
//!
//! Эталоны:
//! - **HH** — классический кальмаровый аксон: под импульсом 50 нА обязан
//!   спайкать (NEURON `bench_neuron_hh.py`);
//! - **AdEx Figure 4** (Naud et al. 2008, Brian2): 3 спайка (RS) и 6
//!   (bursting) за 200 мс — регрессия на нормализацию единиц;
//! - **Poirazi CA1** (OSB): реальная морфология + реальные каналы грузятся
//!   без warnings и дают конечную траекторию; тяжёлый full-cell тест на
//!   NEURON-покой — в [`poirazi_ca1_full_rest`] (минуты в debug).

use std::path::Path;

use crate::api::neuroml::{load_file, load_with};

use super::{metrics, ValidationResult};

// ─────────────────────────────────────────────────────────────────────────────
//  HH: классический кальмаровый аксон
// ─────────────────────────────────────────────────────────────────────────────

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

/// Классический HH: спайкует под импульсом, V конечен, каналы распознаны.
pub fn hh_classic() -> ValidationResult {
    let mut ln = match load_with(HH_SQUID, 0.01, 100.0) {
        Ok(ln) => ln,
        Err(err) => return ValidationResult::failed("hh_classic", err.to_string()),
    };
    if !ln.warnings.is_empty() {
        return ValidationResult::failed(
            "hh_classic",
            format!("unexpected warnings: {:?}", ln.warnings),
        );
    }
    let res = ln.run();
    let v = ln.network.state(0, 0).to_vec();
    if v.iter().any(|x| !x.is_finite()) {
        return ValidationResult::failed("hh_classic", "non-finite voltage in trace");
    }
    match res.spike_counts.first() {
        Some(&n) if n > 0 => ValidationResult::passed("hh_classic", 0.0, 0.0),
        other => ValidationResult::failed(
            "hh_classic",
            format!("classic HH must spike under 50 nA pulse, got {other:?}"),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  AdEx: Naud et al. 2008, Figure 4 (Brian2-эталон)
// ─────────────────────────────────────────────────────────────────────────────

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

/// Эталон Figure 4: RS-клетка → 3 спайка, bursting → 6 (Brian2, dt=0.5).
const ADEX_FIG4_REF_SPIKES: [usize; 2] = [3, 6];

/// AdEx Figure 4: счётчики спайков против Brian2-эталона.
pub fn adex_regular_spiking() -> ValidationResult {
    let mut ln = match load_with(ADEX_FIG4, 0.5, 200.0) {
        Ok(ln) => ln,
        Err(err) => return ValidationResult::failed("adex_regular_spiking", err.to_string()),
    };
    let res = ln.run();
    let got: Vec<usize> = res.spike_counts.clone();
    if got == ADEX_FIG4_REF_SPIKES {
        ValidationResult::passed("adex_regular_spiking", 0.0, 0.0)
    } else {
        ValidationResult::failed(
            "adex_regular_spiking",
            format!("spike counts {got:?} != reference {ADEX_FIG4_REF_SPIKES:?}"),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Poirazi CA1 (OSB)
// ─────────────────────────────────────────────────────────────────────────────

fn poirazi_soma_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_ca1_soma.nml")
}

/// Быстрая проверка сомы Poirazi: грузится без warnings (кастомные LEMS
/// `ComponentType` для Ih), один компартмент, траектория конечна.
pub fn poirazi_ca1_soma() -> ValidationResult {
    let path = poirazi_soma_path();
    let mut ln = match load_file(&path) {
        Ok(ln) => ln,
        Err(err) => {
            return ValidationResult::failed(
                "poirazi_ca1_soma",
                format!("failed to load {}: {err}", path.display()),
            )
        }
    };
    if !ln.warnings.is_empty() {
        return ValidationResult::failed(
            "poirazi_ca1_soma",
            format!("unexpected warnings: {:?}", ln.warnings),
        );
    }
    match ln.cable_info(0) {
        Some(info) if info.compartments.len() == 1 => {}
        other => {
            return ValidationResult::failed(
                "poirazi_ca1_soma",
                format!("expected single-compartment soma, got {other:?}"),
            )
        }
    }
    ln.run();
    let v = ln.network.state(0, 0).to_vec();
    if v.iter().all(|x| x.is_finite()) {
        ValidationResult::passed("poirazi_ca1_soma", 0.0, 0.0)
    } else {
        ValidationResult::failed("poirazi_ca1_soma", "non-finite voltage in trace")
    }
}

/// NEURON-эталон покоя полной Poirazi 2003 CA1 пирамиды
/// (`gen_poirazi_full_ref.py`, secondorder=0, dt=0.025, 35 °C):
/// t = 0, 5, 10, 15, 20 мс → −70.00, −75.07, −75.77, −75.98, −76.02 мВ.
pub const POIRAZI_FULL_REST_T_MS: [f32; 5] = [0.0, 5.0, 10.0, 15.0, 20.0];
/// Допуск 1.0 мВ: остаток ~0.5 мВ к NEURON численный (жёсткий транзиент).
pub const POIRAZI_FULL_REST_TOLERANCE_MV: f32 = 1.0;

/// Тяжёлая проверка (~3.7 мин в debug): полная Poirazi (5249 компартментов)
/// садится на NEURON-покой. В быстрый набор не входит — вызывать явно.
pub fn poirazi_ca1_full_rest() -> ValidationResult {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../resources/validate/osb/poirazi_full/poirazi.net.nml");
    let mut ln = match load_file(&path) {
        Ok(ln) => ln,
        Err(err) => {
            return ValidationResult::failed(
                "poirazi_ca1_full_rest",
                format!("failed to load {}: {err}", path.display()),
            )
        }
    };
    if !ln.warnings.is_empty() {
        return ValidationResult::failed(
            "poirazi_ca1_full_rest",
            format!("unexpected warnings: {:?}", ln.warnings),
        );
    }
    let Some(info) = ln.cable_info(0) else {
        return ValidationResult::failed("poirazi_ca1_full_rest", "no cable info");
    };
    if info.compartments.len() != 5249 {
        return ValidationResult::failed(
            "poirazi_ca1_full_rest",
            format!(
                "expected 5249 compartments, got {}",
                info.compartments.len()
            ),
        );
    }

    // 20 мс @ 0.025 мс, сома на сетке 5 мс.
    let mut traj = vec![ln.network.state(0, 0)[0]];
    for s in 0..800usize {
        ln.network.step();
        if (s + 1) % 200 == 0 {
            traj.push(ln.network.state(0, 0)[0]);
        }
    }
    let ref_v: Vec<f32> = [-70.0, -75.07, -75.77, -75.98, -76.02].to_vec();
    let m = metrics::compare(&traj, &ref_v);
    if m.max_abs_error <= POIRAZI_FULL_REST_TOLERANCE_MV {
        ValidationResult::passed("poirazi_ca1_full_rest", m.max_abs_error, m.rmse)
    } else {
        ValidationResult::failed(
            "poirazi_ca1_full_rest",
            format!(
                "rest trajectory {traj:?} vs NEURON {ref_v:?}: max error {:.3} mV",
                m.max_abs_error
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn suite_passes() {
        let results = super::super::run_validation_suite();
        for r in &results {
            assert!(
                r.passed,
                "{} failed: {:?}",
                r.name,
                r.failure_reason
            );
        }
    }
}
