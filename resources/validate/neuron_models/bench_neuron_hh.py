#!/usr/bin/env python3
"""
NEURON HH speed benchmark - side-by-side vs taas-checkmate bench_hh_200k.

Same workload as examples/bench_hh_200k.rs:
  - N HH neurons (single compartment, L=10um, diam=10um), mechanism `hh`
  - dt = 0.01 ms, fixed step (CVode off), 5000 steps
  - half of the cells under 50 uA/cm2 (IClamp ~0.1571 nA on a 314 um2 soma),
    half silent
  - single thread (h.nthread() = 1)

Usage (Windows NEURON install at C:\\nrn):
    python bench_neuron_hh.py [N] [steps]
"""

import os
import time

# Windows NEURON (C:\nrn install): set the env vars the 9.x bootstrap needs
# (NRN_PYTHONEXE / NRN_PYLIB / mingw bash on PATH), then import.
NRN_HOME = r'C:\nrn'
PY_BASE = os.environ.get(
    'NRN_PY_BASE', r'C:\Users\anton\AppData\Local\Programs\Python\Python313'
)
os.environ['NRNHOME'] = NRN_HOME
os.environ['NRN_PYTHONEXE'] = PY_BASE + r'\python.exe'
os.environ['NRN_PYLIB'] = PY_BASE + r'\python313.dll'
for d in (os.path.join(NRN_HOME, 'bin'), os.path.join(NRN_HOME, 'mingw', 'usr', 'bin'), PY_BASE):
    if d not in os.environ.get('PATH', ''):
        os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    for d in (os.path.join(NRN_HOME, 'bin'), PY_BASE):
        os.add_dll_directory(d)

from neuron import h  # noqa: E402

# nrniv -python does not forward a clean sys.argv (its own args leak in),
# so the workload size comes from env vars instead.
N = int(os.environ.get('NRN_BENCH_N', 200_000))
STEPS = int(os.environ.get('NRN_BENCH_STEPS', 5_000))
DT = 0.01  # ms
TSTOP = DT * STEPS

# Single-threaded run (default), make it explicit. `h.nthread` is gone in
# NEURON 9 (multi-threaded runs use nrn_.../CoreNEURON), so guard it.
try:
    h.nthread(1)
except AttributeError:
    pass
h.CVode().active(0)
h.dt = DT
# Та же температура, что в bench_hh_200k (q10 = 1.0 при 6.3 °C), чтобы
# кинетика ворот совпадала — иначе частоты спайков разъедутся.
h.celsius = 6.3

# ---- build N sections ----
cells = [h.Section() for _ in range(N)]
for s in cells:
    s.L = 10.0
    s.diam = 10.0
    s.insert('hh')
    s(0.5).v = -65.0

# Current clamp on the first half: 50 uA/cm2 * 3.1416e-6 cm2 = 0.1571 nA.
clamps = []
for i, s in enumerate(cells):
    if i < N // 2:
        stim = h.IClamp(s(0.5))
        stim.delay = 0.0
        stim.dur = 1e9
        stim.amp = 0.1571  # nA ~= 50 uA/cm2 on 314.16 um2
        clamps.append(stim)

h.finitialize(-65.0)

# Warm-up, then time the fixed-step loop.
for _ in range(100):
    h.fadvance()

t0 = time.perf_counter()
for _ in range(STEPS):
    h.fadvance()
wall = time.perf_counter() - t0

total_steps = N * STEPS
mps = total_steps / wall / 1e6
sim_ms = DT * STEPS
rt = sim_ms / (wall * 1000.0)

print(f"[NEURON] HH x {N} neurons x {STEPS} steps, dt={DT} ms, 1 thread")
print(f"  wall          = {wall:.3f} s")
print(f"  throughput    = {mps:.1f} M neuron-steps/s")
print(f"  realtime      = {rt:.4f}x")
print(f"  final v       = {cells[0](0.5).v:.2f} mV (cell 0, stimulated)")

# ---- NEURON spike rate (separate SMALL run, NetCon counters) ----
# Outside the timed loop so the counters don't skew the throughput number.
# 2000 neurons (1000 driven) x 5000 steps — expect ~7000 spikes if the
# firing rate matches the CPU/GPU bench (700k spikes / 100k driven / 5000
# steps = 7 spikes per neuron per run).
spike_n = 2000
spike_cells = [h.Section() for _ in range(spike_n)]
for s in spike_cells:
    s.L = 10.0
    s.diam = 10.0
    s.insert('hh')
    s(0.5).v = -65.0
spike_clamps = []
for s in spike_cells[:spike_n // 2]:
    stim = h.IClamp(s(0.5))
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.1571
    spike_clamps.append(stim)  # держим ссылки — иначе GC убьёт ток
# record() должен быть зарегистрирован ДО finitialize, иначе вектор пуст.
# Частота NEURON на 50 µA/cm2 ~120 Hz против ~140 Hz у pillars (cnexp vs
# Euler + double vs f32) — на валидированных токах (10.6-42.4 µA/cm2)
# тест crates/pillars/tests/hh_vs_neuron.rs даёт точное совпадение спайков.
recs = []
for s in spike_cells:
    vv = h.Vector()
    vv.record(s(0.5)._ref_v)
    recs.append(vv)
h.finitialize(-65.0)
for _ in range(STEPS):
    h.fadvance()
# Rising-edge crossings of v = 0 — same threshold semantics as the CPU/GPU
# bench (v_thresh = 0, edge-triggered), independent of NetCon.
spikes = 0
for vv in recs:
    arr = list(vv)
    spikes += sum(1 for i in range(1, len(arr)) if arr[i - 1] < 0.0 <= arr[i])
print(f"  NEURON spikes   = {spikes} (2000 neurons x {STEPS} steps, separate run)")
print(f"  pillars ref ~   = {spike_n // 2 * 7} (7 spikes/neuron, fires ~14% faster)")
