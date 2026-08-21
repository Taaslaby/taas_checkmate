#!/usr/bin/env python3
"""Generate NEURON reference traces for a passive branching (Y-shaped) cable.

Extends `gen_cable_ref.py` from a linear ball-and-stick to a **branching**
tree, so the Rust Hines solver is validated on non-trivial topology (one
parent segment feeding two asymmetric daughters) — the topology case that a
single chain cannot exercise.

Geometry (matches `discretize`'s d_lambda=0.1 with Ra=100, Rm=20000):
    soma  L=20,   diam=20  (1 compartment);
    trunk L=1000, diam=2   (10 compartments, lam=1000 um);
    da    L=500,  diam=1   (8 compartments, lam~707 um);   thin/long → attenuates
    db    L=200,  diam=2   (2 compartments, lam=1000 um);  thick/short → passes
    da and db both connect at trunk(1.0) (the branch point).

Passive `pas` everywhere (g=5e-5 S/cm2, E=-65 mV), Ra=100 ohm*cm, cm=1 uF/cm2.
Protocol: I = +0.01 nA into the soma from t=10 to t=60 ms, dt=0.025,
secondorder=0 (backward Euler == pillars Hines solver), tstop=100.

NEURON `nseg` is set to match pillars' d_lambda compartment counts exactly, so
agreement should be near machine precision. Records V at soma, the branch
point (trunk tip), and both daughter tips.
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


def make_section(h, name, length, diam, nseg):
    sec = h.Section(name=name)
    sec.L = length
    sec.diam = diam
    sec.nseg = nseg
    sec.Ra = RA
    sec.cm = CM
    sec.insert("pas")
    for seg in sec:
        seg.g_pas = G_PAS
        seg.e_pas = E_PAS
    return sec


def main():
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h

        h.load_file("stdrun.hoc")
        h.secondorder = 0
        h.dt = DT
        h.tstop = TSTOP

        soma = make_section(h, "soma", 20, 20, 1)
        trunk = make_section(h, "trunk", 1000, 2, 10)
        da = make_section(h, "da", 500, 1, 8)
        db = make_section(h, "db", 200, 2, 2)

        trunk.connect(soma(1.0))
        da.connect(trunk(1.0))
        db.connect(trunk(1.0))

        stim = h.IClamp(soma(0.5))
        stim.delay = PULSE_ON
        stim.dur = PULSE_OFF - PULSE_ON
        stim.amp = I_AMP

        vecs = {
            "SOMA": h.Vector().record(soma(0.5)._ref_v),
            "BRANCH": h.Vector().record(trunk(1.0)._ref_v),
            "TIP_A": h.Vector().record(da(1.0)._ref_v),
            "TIP_B": h.Vector().record(db(1.0)._ref_v),
        }

        h.finitialize(V0)
        h.continuerun(TSTOP)

        for name, vec in vecs.items():
            arr = np.array(vec)
            n = len(arr)
            print(f"# {name}: n_steps={n}  V range: [{arr.min():.4f}, {arr.max():.4f}] mV")
            step = int(5.0 / DT)
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            print(f"REF_V_{name} = [")
            for i in idxs:
                t = i * DT
                print(f"    {arr[i]:.6f},  # t={t:.1f}")
            print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
