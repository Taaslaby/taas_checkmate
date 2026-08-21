#!/usr/bin/env python3
"""Generate NEURON reference traces for the passive ball-and-stick cable.

Runs a subthreshold current-clamp protocol on a soma + dendrite (passive
`pas`), records V at the soma and at the dendrite tip, and prints both on a
coarse grid so the values can be hardcoded into the Rust
`cable_vs_neuron.rs` regression tests (NEURON env is not available in CI).

Geometry and passive constants match `discretize::CableParams` defaults:
    soma  L=20 um, diam=20 um (1 compartment);
    dend  L=1000 um, diam=2 um (10 compartments via d_lambda=0.1, lam=1000 um);
    Ra=100 ohm*cm, cm=1 uF/cm2, pas g=5e-5 S/cm2 (Rm=20000 ohm*cm2), E=-65 mV;
    V0=-65; I = +0.01 nA into the soma from t=10 to t=60 ms;
    dt=0.025, secondorder=0 (backward Euler, matching the Hines solver), tstop=100.

NEURON `nseg` is set to match pillars' compartment counts exactly (soma 1,
dend 10), so agreement should be near machine precision.
"""
import os
import tempfile

import numpy as np

DT = 0.025
TSTOP = 100.0
V0 = -65.0
I_AMP = 0.01  # nA, into the soma
PULSE_ON = 10.0
PULSE_OFF = 60.0
RA = 100.0
CM = 1.0
G_PAS = 5e-5  # S/cm2 = 1 / 20000 ohm*cm2
E_PAS = -65.0


def main():
    old_cwd = os.getcwd()
    # NEURON imports break on non-ASCII paths (Cyrillic project dir) — chdir to
    # the ASCII temp dir first, exactly as gen_ks_ref.py does.
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h

        h.load_file("stdrun.hoc")
        h.secondorder = 0  # backward Euler == pillars Hines solver
        h.dt = DT
        h.tstop = TSTOP

        soma = h.Section(name="soma")
        soma.L = 20
        soma.diam = 20
        soma.nseg = 1
        soma.Ra = RA
        soma.cm = CM
        soma.insert("pas")
        soma(0.5).g_pas = G_PAS
        soma(0.5).e_pas = E_PAS

        dend = h.Section(name="dend")
        dend.L = 1000
        dend.diam = 2
        dend.nseg = 10
        dend.Ra = RA
        dend.cm = CM
        dend.insert("pas")
        for seg in dend:
            seg.g_pas = G_PAS
            seg.e_pas = E_PAS
        dend.connect(soma(1.0))

        stim = h.IClamp(soma(0.5))
        stim.delay = PULSE_ON
        stim.dur = PULSE_OFF - PULSE_ON
        stim.amp = I_AMP

        v_soma = h.Vector()
        v_soma.record(soma(0.5)._ref_v)
        v_tip = h.Vector()
        v_tip.record(dend(1.0)._ref_v)

        h.finitialize(V0)
        h.continuerun(TSTOP)

        for name, vec in (("SOMA", np.array(v_soma)), ("TIP", np.array(v_tip))):
            n = len(vec)
            print(f"# {name}: n_steps={n}  V range: [{vec.min():.4f}, {vec.max():.4f}] mV")
            step = int(5.0 / DT)
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            print(f"REF_V_{name} = [")
            for i in idxs:
                t = i * DT
                print(f"    {vec[i]:.6f},  # t={t:.1f}")
            print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
