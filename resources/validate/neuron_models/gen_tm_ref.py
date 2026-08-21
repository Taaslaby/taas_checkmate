#!/usr/bin/env python3
"""Generate NEURON reference traces for Tsodyks-Markram short-term plasticity.

pillars scales every delivered spike weight by `R·U` (the pre-spike
plasticityFactor), then depletes `R ← R·(1−U)` and (optionally) facilitates
`U ← U + U0·(1−U)`, with exact-exponential recovery between spikes. The
canonical `tm.mod` implements the same Tsodyks–Markram (1997/1998) model with
`METHOD cnexp` (exact exponential), so the two must agree to floating point.

A NetStim drives a fixed spike train into a passive compartment through a
`TsodyksMarkram` synapse; `delivered` (RANGE) holds the weight of the most
recent spike, so sampling it one `dt` after each spike yields the delivered-
weight sequence. Two profiles are printed:

  DEPRESSION  U0=0.5, tau_rec=100 ms, 12 spikes @20 ms  → geometric decay
  FACIL       U0=0.1, tau_rec=1e6, tau_facil=50 ms       → growing release prob
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NEURON_HOME = r"C:\nrn"
MKNrndll = os.path.join(NEURON_HOME, "bin", "mknrndll.bat")

DT = 0.5
TSTOP = 250.0
START = 10.0
INTERVAL = 20.0
N_SPIKES = 12


def _load(tmp):
    dll = os.path.join(tmp, "nrnmech.dll")
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())  # ASCII cwd (Cyrillic project dir breaks NEURON)
    try:
        from neuron import h

        h.nrn_load_dll(dll)
        h.load_file("stdrun.hoc")
        return h
    finally:
        os.chdir(old_cwd)


def run_profile(h, label, u0, tau_rec, tau_facil):
    post = h.Section(name="post_" + label)
    post.L = 10
    post.diam = 10
    post.cm = 1.0
    post.insert("pas")
    post(0.5).g_pas = 1e-4
    post(0.5).e_pas = -70.0

    syn = h.TsodyksMarkram(post(0.5))
    syn.tau = 3.0
    syn.e = 0.0
    syn.U0 = u0
    syn.tau_rec = tau_rec
    syn.tau_facil = tau_facil

    stim = h.NetStim()
    stim.number = N_SPIKES
    stim.start = START
    stim.interval = INTERVAL
    stim.noise = 0

    nc = h.NetCon(stim, syn)
    nc.delay = 0.0
    nc.weight[0] = 1.0

    delivered = h.Vector()
    delivered.record(syn._ref_delivered)

    h.dt = DT
    h.tstop = TSTOP
    h.finitialize(-70.0)
    h.continuerun(TSTOP)

    dd = np.array(delivered)
    seq = []
    for k in range(N_SPIKES):
        # Sample one dt after spike k (spike at START + k·INTERVAL), so the
        # value is unambiguously that spike's delivered weight.
        t = START + k * INTERVAL + DT
        seq.append(dd[int(round(t / DT))])
    return seq


def main():
    tmp = tempfile.mkdtemp(prefix="tm_mods_")
    shutil.copy(os.path.join(HERE, "tm.mod"), os.path.join(tmp, "tm.mod"))
    subprocess.run(["cmd.exe", "/c", MKNrndll], cwd=tmp, check=True, capture_output=True)

    h = _load(tmp)
    dep = run_profile(h, "dep", 0.5, 100.0, 0.0)
    fac = run_profile(h, "fac", 0.1, 1e6, 50.0)

    print("REF_TM_DEPRESSION = [")
    for i, w in enumerate(dep):
        print(f"    {w:.6f},  # spike {i+1} delivered weight")
    print("]")
    print("REF_TM_FACILITATION = [")
    for i, w in enumerate(fac):
        print(f"    {w:.6f},  # spike {i+1} delivered weight")
    print("]")


if __name__ == "__main__":
    main()
