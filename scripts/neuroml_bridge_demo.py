#!/usr/bin/env python3
"""Python-мост NeuroML2 → Rust-движок + круговая сверка с NEURON.

Демонстрирует полный интероп-цикл из Python:

  1. модель описана как NeuroML2 (XML-строка или файл, либо сгенерирована
     pyNeuroML-объектами, если пакет `pyneuroml` установлен);
  2. `taas_checkmate.neuroml.load()` компилирует её в Rust-движок и гоняет
     за миллисекунды (SIMD, без интерпретатора);
  3. тот же протокол прогоняется в NEURON (SEClamp) и следы сравниваются —
     доказательство, что мост не врёт.

Зависимости: `numpy` (обязательно), `neuron` (для сверки, опционально),
`pyneuroml` (для генерации `.nml` из объектов, опционально).

Пример:
    python scripts/neuroml_bridge_demo.py
    python scripts/neuroml_bridge_demo.py --cross-check   # + сверка с NEURON
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Windows-консоль (cp1251) не умеет печатать Δ/кириллицу — пишем UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

# ── 1. Каноническая модель: iafCell + voltageClamp (ровно та, что сверена
#     с NEURON SEClamp в crates/pillars/src/api/neuroml/mod.rs). ──────────────
MODEL_XML = """<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bridge">
  <iafCell id="iaf" leakConductance="0.2nS" leakReversal="-70mV"
           thresh="-55mV" reset="-70mV" C="3.2pF"/>
  <voltageClamp id="vc" delay="5ms" duration="20ms" targetVoltage="-60mV"
                simpleSeriesResistance="200Mohm"/>
  <network id="net">
    <population id="pop" component="iaf" size="1"/>
    <explicitInput target="pop[0]" input="vc"/>
  </network>
</neuroml>"""

DT_MS = 0.5
DURATION_MS = 60.0


def load_bridge():
    """Импорт Rust-расширения (ищет .pyd/.dll рядом с проектом)."""
    root = Path(__file__).resolve().parents[1]
    for cand in (root / "target" / "debug", root / "target" / "release", root):
        if (cand / "taas_checkmate.pyd").exists() or (cand / "taas_checkmate.dll").exists():
            sys.path.insert(0, str(cand))
            break
    from taas_checkmate.neuroml import load  # noqa: F401

    return load


def build_model_xml():
    """Модель как XML-строка.

    Если установлен `pyneuroml`/`libNeuroML`, модель строится из типизированных
    объектов (`NeuroMLDocument` → `NeuroMLWriter.write`) — это и есть
    «pyNeuroML-интероп» (генерация .nml из Python). Иначе — literal XML.
    """
    try:
        from neuroml import (  # type: ignore
            ExplicitInput,
            IafCell,
            Network,
            NeuroMLDocument,
            Population,
            VoltageClamp,
        )
        from neuroml.writers import NeuroMLWriter  # type: ignore

        doc = NeuroMLDocument(id="bridge")
        doc.iaf_cells.append(IafCell(
            id="iaf", leak_reversal="-70mV", thresh="-55mV", reset="-70mV",
            C="3.2pF", leak_conductance="0.2nS",
        ))
        doc.voltage_clamps.append(VoltageClamp(
            id="vc", delay="5ms", duration="20ms", target_voltage="-60mV",
            simple_series_resistance="200Mohm",
        ))
        pop = Population(id="pop", component="iaf", size=1)
        net = Network(id="net", populations=[pop])
        net.explicit_inputs.append(ExplicitInput(target="pop[0]", input="vc"))
        doc.networks.append(net)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nml", delete=False, encoding="utf-8"
        ) as fh:
            path = fh.name
        NeuroMLWriter.write(doc, path)
        with open(path, encoding="utf-8") as fh:
            xml = fh.read()
        os.unlink(path)
        return xml, "pyNeuroML (NeuroMLDocument → writer)"
    except ImportError:
        return MODEL_XML, "literal XML (pyNeuroML not installed)"


def run_rust(xml: str):
    """Прогон модели на Rust-движке через мост. Возвращает (t, V, net)."""
    load = load_bridge()
    net = load(xml, dt_ms=DT_MS, duration_ms=DURATION_MS)
    trace = np.asarray(net.run_trace(0), dtype=np.float64)  # [steps+1, 1]
    t = np.arange(trace.shape[0]) * net.dt
    return t, trace[:, 0], net


def run_neuron():
    """Тот же протокол в NEURON: pas-компартмент + SEClamp.

    Геометрия/единицы — ровно `gen_clamp_ref.py` (area 3.2e-6 cm^2 →
    C_m = 3.2 pF, g_pas = 0.2 nS total), т.е. тот же эталон, что в Rust-тесте.
    """
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())  # NEURON ломается на кириллическом пути
    try:
        from neuron import h  # type: ignore

        h.load_file("stdrun.hoc")
        diam = 10.0
        L = 3.2e-6 / (np.pi * diam * 1e-8)  # µm → π·diam·L·1e-8 = 3.2e-6 cm²
        g_pas = 0.2e-9 / 3.2e-6            # S/cm² → 0.2 nS total

        soma = h.Section(name="soma")
        soma.L = L
        soma.diam = diam
        soma.nseg = 1
        soma.cm = 1.0                       # µF/cm² → 3.2 pF total
        soma.insert("pas")
        soma(0.5).g_pas = g_pas
        soma(0.5).e_pas = -70.0

        c = h.SEClamp(soma(0.5))
        c.rs = 200.0
        c.dur1 = 5.0
        c.amp1 = -70.0   # pre-window: command=rest → i=0
        c.dur2 = 20.0
        c.amp2 = -60.0   # window: clamp to target
        c.amp3 = -70.0   # ignored — SEClamp turns off after dur1+dur2

        t = h.Vector()
        v = h.Vector()
        t.record(h._ref_t)
        v.record(soma(0.5)._ref_v)
        h.dt = DT_MS
        h.tstop = DURATION_MS
        h.finitialize(-70.0)
        h.continuerun(DURATION_MS)
        return np.asarray(t), np.asarray(v)
    finally:
        os.chdir(old_cwd)


def cross_check(t_rust, v_rust):
    """Сверка Rust-моста с NEURON SEClamp. Возвращает (полный, settled) max |ΔV|.

    Методология — ровно Rust-тест `voltage_clamp_matches_neuron_seclamp`:
    сравниваем на сетке, избегающей краёв окна (t=5 и t=25 мс), где
    полу-неявный контроллер pillars и неявный SEClamp расходятся на
    недорозрешённом фронте (τ_eff≈0.6 мс vs dt=0.5 мс). На полке равновесие
    −60.38 мВ совпадает до <0.001 мВ.
    """
    try:
        t_neuron, v_neuron = run_neuron()
    except ImportError as e:
        print(f"[skip] NEURON not installed ({e})")
        return None, None

    v_neuron_at = np.interp(t_rust, t_neuron, v_neuron)
    full_err = float(np.abs(v_rust - v_neuron_at).max())

    # Сетка «settled»: кратные 0.5 мс, без 5.0 и 25.0 (края окна).
    step = int(round(DT_MS / DT_MS))  # 1 шаг = 0.5 мс
    settled_idx = [
        i for i in range(0, len(t_rust), 5)  # каждые 2.5 мс
        if abs(t_rust[i] - 5.0) > 1e-9 and abs(t_rust[i] - 25.0) > 1e-9
    ]
    settled_err = float(np.abs(v_rust[settled_idx] - v_neuron_at[settled_idx]).max())

    print(f"  NEURON:  {len(t_neuron)} points, t_max={t_neuron[-1]:.1f} ms")
    print(f"  max |dV| full trace      = {full_err:.4f} mV (incl. under-resolved edges)")
    print(f"  max |dV| settled (no 5/25) = {settled_err:.4f} mV")
    return full_err, settled_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-check", action="store_true",
                    help="compare the trace against NEURON SEClamp")
    args = ap.parse_args()

    xml, how = build_model_xml()
    print(f"Model built via: {how}")
    t, v, net = run_rust(xml)
    print(f"Rust engine: {net.num_populations()} population, "
          f"dt={net.dt} ms, {len(t)} steps")
    print(f"  V(t=0)={v[0]:.3f}, V(t=5)={v[10]:.3f}, "
          f"V(t=25)={v[50]:.3f}, V(t={t[-1]:.0f})={v[-1]:.3f} mV")
    print(f"  warnings: {net.warnings or 'none'}")

    if args.cross_check:
        full_err, settled_err = cross_check(t, v)
        if settled_err is not None:
            ok = settled_err < 0.6  # порог Rust-теста `voltage_clamp_matches_neuron_seclamp`
            print(f"  {'PASS: cross-check ok' if ok else 'FAIL: mismatch'} (settled <0.6 mV)")
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
