#!/usr/bin/env python3
"""Generate NEURON reference traces for the Markov (kschan) channel in a cable.

Runs a subthreshold current-clamp protocol on a ball-and-stick cable with the
3-state kinetic scheme `ks3.mod` (closed <-> open <-> inactivated, Na-m/Na-h
rates) inserted on the **soma** only, and prints V at the soma and at the
dendrite tip on a coarse grid so the values can be hardcoded into the Rust
`discretize.rs` regression tests (NEURON env is not available in CI).

The geometry/mesh/units match the pillars cable exactly:
    soma  L=20 um, diam=20 um (1 compartment);
    dend  L=1000 um, diam=2 um (10 compartments via d_lambda=0.1);
    Ra=100 ohm*cm, cm=1 uF/cm2;
    pas g=0.3 mS/cm2 / E=-54.3 everywhere;
    ks3 on soma: gbar=36 mS/cm2, ek=-77 mV (K-like, stays subthreshold);
    V0=-65; I = +10 uA/cm2 (0.12566 nA) into the soma from t=10 to 60 ms;
    dt=0.025, secondorder=0 (backward Euler, matching the Hines solver).

NEURON's KINETIC is solved with METHOD sparse (backward Euler on the chain),
pillars uses the exact matrix exponential — agreement is integrator accuracy
(like ks_vs_neuron.rs, ~1e-3 mV), not machine precision.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEURON_HOME = r"C:\nrn"
MKNrndll = os.path.join(NEURON_HOME, "bin", "mknrndll.bat")

DT = 0.025
TSTOP = 100.0
V0 = -65.0
PULSE_ON = 10.0
PULSE_OFF = 60.0
RA = 100.0
CM = 1.0
G_PAS = 0.3 / 1000.0  # S/cm2
E_PAS = -54.3
G_KS3 = 36.0 / 1000.0  # S/cm2
E_K = -77.0
I_DENSITY = 10.0  # uA/cm2, subthreshold depolarizing
AREA_CM2 = np.pi * 20.0 * 20.0 * 1e-8  # soma 20um x 20um cylinder
I_AMP = I_DENSITY * 1000.0 * AREA_CM2  # nA into the soma


def main():
    tmp = tempfile.mkdtemp(prefix="ks3_mods_")
    shutil.copy(os.path.join(HERE, "ks3.mod"), os.path.join(tmp, "ks3.mod"))
    subprocess.run(["cmd.exe", "/c", MKNrndll], cwd=tmp, check=True, capture_output=True)
    dll = os.path.join(tmp, "nrnmech.dll")

    old_cwd = os.getcwd()
    # NEURON imports break on non-ASCII paths (Cyrillic project dir) — chdir to
    # the ASCII temp dir first, exactly as gen_ks_ref.py does.
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h

        h.nrn_load_dll(dll)
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
        soma.insert("ks3")
        soma(0.5).gbar_ks3 = G_KS3
        soma(0.5).ek = E_K

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

        print(f"# I_AMP = {I_AMP:.6f} nA (10 uA/cm2 over soma area {AREA_CM2:.8f} cm2)")
        for name, vec in (("SOMA", np.array(v_soma)), ("TIP", np.array(v_tip))):
            n = len(vec)
            print(f"# {name}: n_steps={n}  V range: [{vec.min():.4f}, {vec.max():.4f}] mV")
            step = int(5.0 / DT)
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            print(f"REF_V_KS3_{name} = [")
            for i in idxs:
                t = i * DT
                print(f"    {vec[i]:.6f},  # t={t:.1f}")
            print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
