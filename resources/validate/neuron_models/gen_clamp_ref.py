#!/usr/bin/env python3
"""Generate NEURON reference traces for `voltageClamp`/`voltageClampTriple`.

pillars models a patch-clamp as a **proportional controller**:
`i = weight·(target(t) − V)/series_resistance` (LEMS `voltageClamp` semantics),
read against the live membrane potential at the start of each step. NEURON's
built-in `SEClamp` is the same finite-gain controller — `i = (V_cmd − V)/rs`
with `rs` in MΩ. Its command is a **two-step** protocol: `amp1` for t<dur1,
`amp2` for dur1 ≤ t < dur1+dur2, then the clamp **turns off** (the `amp3`
parameter is ignored — verified empirically: amp3=−70 and amp3=−80 give
identical traces after dur1+dur2).

LEMS `voltageClamp` (Window) is exactly this: `i=0` before `delay`, then
`i=weight·(target−v)/R` on `[delay, delay+duration]`, then `i=0`. Mapping:

    voltageClamp → SEClamp:  dur1=delay, amp1=rest (no current), dur2=duration,
                             amp2=target, off afterwards.

LEMS `voltageClampTriple` holds `returnVoltage` **forever** after the window,
so it does **not** map to SEClamp — that phase is validated analytically.

The model here mirrors the Rust regression test's `iafCell` (a voltage-
normalised LIF): leak conductance 0.2 nS, reversal −70 mV, capacitance 3.2 pF
(τ_m = 16 ms), clamped by a 200 MΩ electrode to −60 mV over [5, 25] ms. The
leak is a `pas` density over a single compartment of area 3.2e-6 cm²
(π·10 µm·10.1859 µm), so C_m = 3.2 pF and g_pas = 0.2 nS / area exactly.
"""
import os
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DT = 0.5
TSTOP = 50.0
V_REST = -70.0
TARGET = -60.0
RS = 200.0  # MΩ
DELAY = 5.0
DURATION = 20.0

# Single compartment with area 3.2e-6 cm² → C_m = 3.2 pF at cm=1 µF/cm².
DIAM = 10.0
L = 3.2e-6 / (np.pi * DIAM * 1e-8)  # µm, so π·diam·L·1e-8 = 3.2e-6 cm²
G_PAS = 0.2e-9 / 3.2e-6  # S/cm² → 0.2 nS total

# Grid (ms) — all multiples of DT, none at the 5/25 ms window edges.
TIMES = [0.0, 2.5, 7.5, 10.0, 15.0, 20.0, 22.5, 27.5, 30.0, 35.0, 42.5, 47.5]


def main():
    old_cwd = os.getcwd()
    # NEURON imports break on non-ASCII paths (Cyrillic project dir).
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h

        h.load_file("stdrun.hoc")
        soma = h.Section(name="soma")
        soma.L = L
        soma.diam = DIAM
        soma.nseg = 1
        soma.cm = 1.0
        soma.insert("pas")
        soma(0.5).g_pas = G_PAS
        soma(0.5).e_pas = V_REST

        c = h.SEClamp(soma(0.5))
        c.rs = RS
        c.dur1 = DELAY
        c.amp1 = V_REST  # pre-window: command=rest → i=0 (cell sits at rest)
        c.dur2 = DURATION
        c.amp2 = TARGET  # window: clamp to target
        c.amp3 = V_REST  # ignored — SEClamp turns off after dur1+dur2

        v = h.Vector()
        v.record(soma(0.5)._ref_v)
        h.dt = DT
        h.tstop = TSTOP
        h.finitialize(V_REST)
        h.continuerun(TSTOP)

        vv = np.array(v)
        print(f"# C_m = {soma.cm * np.pi*soma.L*soma.diam*1e-8 * 1e6:.6f} pF")
        print(f"# g_pas = {G_PAS:.6e} S/cm2 (0.2 nS total), rs = {RS} MOhm")
        print(f"# protocol: {V_REST} -> {TARGET} @[{DELAY},{DELAY+DURATION}] -> {V_REST}")
        print("REF_V_CLAMP = [")
        for tt in TIMES:
            i = int(round(tt / DT))
            print(f"    {vv[i]:.6f},  # t={tt:.1f}")
        print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
