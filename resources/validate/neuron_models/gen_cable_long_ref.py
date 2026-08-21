#!/usr/bin/env python3
"""Generate a NEURON reference for a **1000-compartment** passive cable.

Same ball-and-stick as `gen_cable_ref.py`, but the dendrite is a **1000-
compartment** long cable (L=100000 um, nseg=1000 at d_lambda=0.1 = lambda/10)
to validate the pillars Hines solver at NEURON scale (f32 elimination over a
long chain, no degradation).

Note: a *fine* mesh (d_lambda=0.001 -> 1 um segments) is numerically
pathological: the axial conductance dwarfs the membrane conductance (~550x)
and the f32 tridiagonal elimination drifts. The lambda/10 mesh keeps the
matrix well-conditioned.

    soma  L=20 um, diam=20 um, nseg=1 (d_lambda=0.1);
    dend  L=100000 um, diam=2 um, nseg=1000 (lambda=1000 um -> 100 um segs);
    Ra=100 ohm*cm, cm=1 uF/cm2, pas g=5e-5 S/cm2 (Rm=20000), E=-65 mV;
    V0=-65; I = +0.01 nA into the soma from t=10 to 60 ms;
    dt=0.025, secondorder=0 (backward Euler), tstop=100.

Records V at the soma center, at dend(0.5) (compartment 500) and at the tip
dend(1.0); prints on a coarse 5 ms grid for hardcoding into the Rust test.
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
DEND_NSEG = 1000


def main():
    old_cwd = os.getcwd()
    # NEURON imports break on non-ASCII paths (Cyrillic project dir) — chdir to
    # the ASCII temp dir first, exactly as gen_cable_ref.py does.
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
        soma.nseg = 1  # d_lambda=0.1: ceil(20 / (0.1 * 3162.3)) = 1
        soma.Ra = RA
        soma.cm = CM
        soma.insert("pas")
        soma(0.5).g_pas = G_PAS
        soma(0.5).e_pas = E_PAS

        dend = h.Section(name="dend")
        dend.L = 100000
        dend.diam = 2
        dend.nseg = DEND_NSEG
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
        v_mid = h.Vector()
        v_mid.record(dend(0.5)._ref_v)
        v_tip = h.Vector()
        v_tip.record(dend(1.0)._ref_v)

        h.finitialize(V0)
        h.continuerun(TSTOP)

        for name, vec in (
            ("SOMA", np.array(v_soma)),
            ("MID", np.array(v_mid)),
            ("TIP", np.array(v_tip)),
        ):
            n = len(vec)
            print(f"# {name}: n_steps={n}  V range: [{vec.min():.4f}, {vec.max():.4f}] mV")
            step = int(5.0 / DT)
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            print(f"REF_V_LONG_{name} = [")
            for i in idxs:
                t = i * DT
                print(f"    {vec[i]:.6f},  # t={t:.1f}")
            print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
