#!/usr/bin/env python3
"""Generate NEURON reference traces for the Ca²⁺ subsystem parity test.

Model: a single spherical soma (diam = L = 20 µm, area = 1256.6 µm²) with a
passive leak (pas: 0.3 mS/cm² → −65 mV), an always-open ohmic Ca²⁺ channel
(`cachan_simple.mod`: `ica = gbar·(v − eca)`, gbar = 1.0 mS/cm², eca = Nernst)
and calcium accumulation (`cacum.mod`: thin shell depth = 0.1 µm = 100 nm,
tau = 100 ms, cai0 = 1e-4 mM). This is the NEURON analogue of the Rust test
`morpho_cell_ca_nernst_and_decaying_pool`'s model, and of the NeuroML
`channelDensityNernst` + `species` + `decayingPoolConcentrationModel`.

The NEURON `cacum.mod` is the **thin-shell** approximation (volume = area·depth),
while LEMS `decayingPoolConcentrationModel` uses the exact spherical shell
(4π/3)(R³ − (R−d)³). For depth/R = 0.1/10 = 1e-2 the two agree to ~1%, so the
Rust parity test asserts a tolerance that absorbs this known geometric
difference.

`secondorder=0` (backward Euler) matches the pillars Hines solver.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEURON_HOME = r"C:\nrn"
MKNrndll = os.path.join(NEURON_HOME, "bin", "mknrndll.bat")

DT = 0.05
TSTOP = 50.0
V_INIT = -65.0
DIAM = 20.0
L = 20.0
G_PAS = 0.3e-3  # S/cm² (0.3 mS/cm²)
E_PAS = -65.0
GBAR_CA = 1.0e-3  # S/cm² (1.0 mS/cm²)
DEPTH_NM = 100.0  # 0.1 µm
TAU = 100.0
CAI0 = 1e-4  # mM
CAO = 2.0  # mM

# Sample grid (ms), multiples of DT, no edge transients.
TIMES = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 50.0]


def main():
    tmp = tempfile.mkdtemp(prefix="ca_mods_")
    for f in ("cachan_simple.mod", "cacum.mod"):
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    subprocess.run(["cmd.exe", "/c", MKNrndll], cwd=tmp, check=True, capture_output=True)
    dll = os.path.join(tmp, "nrnmech.dll")

    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())  # NEURON breaks on non-ASCII (Cyrillic) paths
    try:
        from neuron import h

        h.nrn_load_dll(dll)
        h.load_file("stdrun.hoc")
        h.secondorder = 0  # backward Euler == pillars Hines solver
        h.dt = DT
        h.tstop = TSTOP
        h.celsius = 6.3

        soma = h.Section(name="soma")
        soma.L = L
        soma.diam = DIAM
        soma.nseg = 1
        soma.cm = 1.0
        soma.insert("pas")
        soma(0.5).g_pas = G_PAS
        soma(0.5).e_pas = E_PAS
        soma.insert("cachan_s")
        soma(0.5).gbar_cachan_s = GBAR_CA
        soma.insert("cacum")
        soma(0.5).depth_cacum = DEPTH_NM
        soma(0.5).tau_cacum = TAU
        soma(0.5).cai0_cacum = CAI0

        # Ion concentrations (mM). eca = Nernst(cao/cai) at celsius=6.3.
        h.cao0_ca_ion = CAO
        h.cai0_ca_ion = CAI0

        vv = h.Vector()
        vv.record(soma(0.5)._ref_v)
        cai = h.Vector()
        cai.record(soma(0.5)._ref_cai)

        h.finitialize(V_INIT)
        h.continuerun(TSTOP)

        v_arr = np.array(vv)
        cai_arr = np.array(cai)
        area = np.pi * DIAM * L
        print(f"# area = {area:.4f} um2, depth = {DEPTH_NM} nm, tau = {TAU} ms, celsius = 6.3")
        print(f"# g_pas = {G_PAS} S/cm2, gbar_ca = {GBAR_CA} S/cm2, cai0 = {CAI0} mM, cao = {CAO} mM")
        print("REF_V_CA = [")
        for tt in TIMES:
            i = int(round(tt / DT))
            print(f"    {v_arr[i]:.6f},  # t={tt:.1f}")
        print("]")
        print("REF_CAI_CA = [")
        for tt in TIMES:
            i = int(round(tt / DT))
            print(f"    {cai_arr[i]:.6f},  # t={tt:.1f}")
        print("]")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
