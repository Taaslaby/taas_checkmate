#!/usr/bin/env python3
"""Smoke test: Python-мост NeuroML2 — Ca²⁺-подсистема (ca / run_ca_trace /
run_trace / channelPopulationNernst).

Запуск (как и остальные smoke-тесты — `python tests/test_*.py` после
`maturin build && pip install` или `PYTHONPATH=target/debug`):

    cd taas_checkmate
    python tests/test_neuroml_bridge.py

Ожидаемый результат: OK — все проверки проходят.
"""

import sys
import os

# Windows-consoles default to cp1251 — force UTF-8 so the test runs on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np

passed = 0
failed = 0


def check(cond: bool, msg: str):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


def section(name: str):
    print(f"\n── {name} ─{'─' * max(0, 60 - len(name))}")


# ─── import ──────────────────────────────────────────────

section("import")

try:
    from taas_checkmate.neuroml import load, load_file, NeuromlNetwork

    check(True, "from taas_checkmate.neuroml import load, load_file, NeuromlNetwork")
except Exception as e:  # noqa: BLE001
    check(False, f"import taas_checkmate.neuroml → {e}")
    sys.exit(1)

# ─── фикстуры ────────────────────────────────────────────

# Единственная сферическая сома + ohmic-подобный Ca-канал (gate почти открыт) +
# decayingPool — модель из `morpho_cell_ca_nernst_and_decaying_pool` (Rust).
CA_CELL_XML = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ca_cell">
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
      <channelDensityNernst id="caDens" ionChannel="caChan" condDensity="1.0 mS_per_cm2" ion="ca" segmentGroup="soma_group"/>
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
</neuroml>"""

# Тот же Ca-клеточный скелет, но канал — дискретная популяция
# (`channelPopulationNernst`): number=1257 × 10 pS ≈ 1 mS/cm² плотность.
CA_POPULATION_XML = CA_CELL_XML.replace(
    '<channelDensityNernst id="caDens" ionChannel="caChan" condDensity="1.0 mS_per_cm2" ion="ca" segmentGroup="soma_group"/>',
    '<channelPopulationNernst id="caPop" ionChannel="caChan" number="1257" ion="ca"/>',
).replace('id="ca_cell"', 'id="ca_pop_cell"')


def _finite(x) -> bool:
    return bool(np.isfinite(x).all())


# ─── ca() — начальная концентрация ───────────────────────

section("ca() — начальная [Ca²⁺]")

net = load(CA_CELL_XML, dt_ms=0.05, duration_ms=50.0)
check(net.warnings == [], f"Ca-модель без warning: {net.warnings}")
check(isinstance(net, NeuromlNetwork), "load возвращает NeuromlNetwork")
check(abs(net.dt - 0.05) < 1e-6 and abs(net.duration - 50.0) < 1e-6, "dt/duration геттеры")

ca0 = net.ca(0)
check(isinstance(ca0, np.ndarray) and ca0.shape == (1,), f"ca(0) shape (1,), got {ca0.shape}")
check(abs(float(ca0[0]) - 1e-4) < 1e-6, f"ca(0)[0] == 1e-4 мМ, got {float(ca0[0])}")

# ─── run_ca_trace ────────────────────────────────────────

section("run_ca_trace — трасса [Ca²⁺](t)")

ca_trace = net.run_ca_trace(0)
n_steps = int(round(50.0 / 0.05)) + 1
check(ca_trace.shape == (n_steps, 1), f"run_ca_trace shape ({n_steps}, 1), got {ca_trace.shape}")
check(abs(float(ca_trace[0, 0]) - 1e-4) < 1e-6, f"ca_trace[0] == 1e-4, got {float(ca_trace[0, 0])}")
check(
    float(ca_trace[-1, 0]) > 1e-4 and _finite(ca_trace),
    f"[Ca] растёт и конечен: {float(ca_trace[-1, 0])}",
)

# ─── run_trace ───────────────────────────────────────────

section("run_trace — трасса V(t)")

v_trace = net.run_trace(0)
check(v_trace.shape == (n_steps, 1), f"run_trace shape ({n_steps}, 1), got {v_trace.shape}")
check(_finite(v_trace), f"V конечен: {float(v_trace[-1, 0])}")

# ─── ca() на не-Ca популяции → ValueError ────────────────

section("ca() — контракт ошибок")

# iafCell — точечный нейрон без Ca²⁺ species → ca() должен дать ValueError.
point = load(
    """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="pt">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV" thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <network id="net"><population id="pop" component="iaf" size="1"/></network>
</neuroml>""",
    dt_ms=0.5,
    duration_ms=10.0,
)
try:
    point.ca(0)
    check(False, "ca() на точечной популяции отклонён")
except ValueError:
    check(True, "ca() на точечной популяции → ValueError")

# Выход за диапазон популяций.
try:
    net.ca(99)
    check(False, "ca(99) отклонён")
except ValueError:
    check(True, "ca(99) → ValueError (out of range)")

# ─── channelPopulationNernst ─────────────────────────────

section("channelPopulationNernst — дискретная Ca-популяция")

pop_net = load(CA_POPULATION_XML, dt_ms=0.05, duration_ms=50.0)
check(pop_net.warnings == [], f"channelPopulationNernst без warning: {pop_net.warnings}")
pop_ca0 = pop_net.ca(0)
check(abs(float(pop_ca0[0]) - 1e-4) < 1e-6, f"ca(0)[0] == 1e-4, got {float(pop_ca0[0])}")

pop_ca_trace = pop_net.run_ca_trace(0)
check(
    float(pop_ca_trace[-1, 0]) > 1e-4 and _finite(pop_ca_trace),
    f"[Ca] стохастической популяции растёт: {float(pop_ca_trace[-1, 0])}",
)
pop_v_trace = pop_net.run_trace(0)
check(_finite(pop_v_trace), f"V стохастической популяции конечен: {float(pop_v_trace[-1, 0])}")

# ─── load_file ───────────────────────────────────────────

section("load_file — файл .nml")

example = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "examples", "neuroml", "exc_inh_loop.nml",
)
if os.path.exists(example):
    file_net = load_file(example)
    spikes = file_net.run()
    check(isinstance(spikes, list) and len(spikes) == file_net.num_populations(), "load_file → run() → счётчики")
    check(sum(spikes) >= 0, f"load_file прогон: спайки {spikes}")
else:
    check(True, f"фикстура {example} отсутствует — load_file пропущен")

# ─── итог ────────────────────────────────────────────────

print()
if failed == 0:
    print(f"OK — все {passed} проверок прошли 🧠")
    sys.exit(0)
else:
    print(f"FAIL — {failed} из {passed + failed} проверок не прошли")
    sys.exit(1)
