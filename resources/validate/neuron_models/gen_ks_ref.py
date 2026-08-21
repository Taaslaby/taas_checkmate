#!/usr/bin/env python3
"""Generate NEURON reference traces for the kinetic-scheme channels.

Runs the same subthreshold current-clamp protocol on `ks2.mod` (2-state
closed <-> open) and `ks3.mod` (3-state closed <-> open <-> inactivated), then
prints V(t) on a coarse grid so the values can be hardcoded into the Rust
`ks_vs_neuron.rs` regression tests (NEURON env is not available in CI).

Both channels share the membrane setup:
    soma L=10 um, diam=10 um, cm=1; pas g=0.3/E=-54.3;
    channel g=36 (S/cm2 = 36/1000) / E_rev=-77 (K-like);
    V0=-65; I = +10 uA/cm2 from t=10 to t=60 ms; dt=0.025; tstop=100 ms.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEURON_HOME = r"C:\nrn"
MKNrndll = os.path.join(NEURON_HOME, "bin", "mknrndll.bat")

DT = 0.025
AREA_CM2 = np.pi * 10.0 * 10.0 * 1e-8  # 10um x 10um cylinder
CM = 1.0
G_LEAK = 0.3
E_LEAK = -54.3
G_CH = 36.0
E_K = -77.0
V0 = -65.0
I_AMP = 10.0  # uA/cm2, subthreshold depolarizing
PULSE_ON = 10.0
PULSE_OFF = 60.0
TSTOP = 100.0


def run_one(mod_name, suffix):
    tmp = tempfile.mkdtemp(prefix=f"{mod_name}_mods_")
    shutil.copy(os.path.join(HERE, mod_name), os.path.join(tmp, mod_name))
    subprocess.run(["cmd.exe", "/c", MKNrndll], cwd=tmp, check=True, capture_output=True)
    dll = os.path.join(tmp, "nrnmech.dll")

    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h
        h.nrn_load_dll(dll)
        h.load_file("stdrun.hoc")

        soma = h.Section(name="soma")
        soma.L = 10
        soma.diam = 10
        soma.cm = CM
        soma.insert("pas")
        soma(0.5).g_pas = G_LEAK / 1000.0
        soma(0.5).e_pas = E_LEAK
        soma.insert(suffix)
        mech = getattr(soma(0.5), suffix)
        mech.gbar = G_CH / 1000.0
        soma(0.5).ek = E_K

        stim = h.IClamp(soma(0.5))
        stim.delay = PULSE_ON
        stim.dur = PULSE_OFF - PULSE_ON
        stim.amp = I_AMP * 1000.0 * AREA_CM2 * CM

        v_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        h.dt = DT
        h.tstop = TSTOP
        h.finitialize(V0)
        h.continuerun(TSTOP)

        v = np.array(v_vec)
        n = len(v)
        print(f"# {suffix} ({mod_name}): n_steps={n}  V range: [{v.min():.4f}, {v.max():.4f}] mV")
        step = int(5.0 / DT)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        print(f"REF_V_{suffix.upper()} = [")
        for i in idxs:
            t = i * DT
            print(f"    {v[i]:.6f},  # t={t:.1f}")
        print("]")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    run_one("ks2.mod", "ks2")
    run_one("ks3.mod", "ks3")


if __name__ == "__main__":
    main()
