#!/usr/bin/env python3
"""Generate NEURON reference traces for the normalized AdEx mechanism.

Runs the *subthreshold* current-clamp protocol on `adex_taas.mod` and prints
V(t) on a coarse grid so the values can be hardcoded into the Rust
`adex_vs_neuron.rs` regression tests (NEURON env is not available in CI).

The mechanism integrates the pillars-normalized AdEx directly (no membrane
capacitor):
    v' = (v_rest - v + delta_t*exp((v - v_thresh)/delta_t) - w + i_inj) * inv_tau_m
    w' = (a*(v - v_rest) - w) * inv_tau_w
`i_inj` is the injected current already divided by gL (units: mV), matching
pillars `i_syn` exactly. The spike reset (v >= v_peak) is intentionally NOT
tested here — the protocol stays below threshold, so both sides only exercise
the continuous ODE (as ks_vs_neuron.rs does for the Markov scheme).

pillars integrates with forward Euler; NEURON with `derivimplicit` (backward
Euler). Both are first-order, so agreement is to integrator accuracy (≲0.05 mV
at dt=0.025 ms for this smooth trajectory), not bit-exact.

Protocol: RS parameters from Brette & Gerstner 2005;
    v_rest=-70.6, v_thresh=-50.4, delta_t=2.0, inv_tau_m=0.107 (/ms),
    inv_tau_w=0.00694 (/ms), a=0.133, b=2.68, v_peak=0.0;
    i_inj = 0 for t in [0, 50), +11.0 mV for t in [50, 500);
    dt = 0.025 ms, tstop = 500 ms.
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
TSTOP = 500.0
V0 = -70.6
PULSE_ON = 50.0
PULSE_OFF = TSTOP
I_INJ = 11.0  # mV (subthreshold: settles near -61 mV, no spike)

# RS parameter set (pillars-normalized units).
P = dict(
    inv_tau_m=0.107,
    inv_tau_w=0.00694,
    v_rest=-70.6,
    v_thresh=-50.4,
    v_reset=-70.6,
    delta_t=2.0,
    a=0.133,
    b=2.68,
    tau_ref=2.0,
    v_peak=0.0,
)


def main():
    tmp = tempfile.mkdtemp(prefix="adex_mods_")
    shutil.copy(os.path.join(HERE, "adex_taas.mod"), os.path.join(tmp, "adex_taas.mod"))
    subprocess.run(["cmd.exe", "/c", MKNrndll], cwd=tmp, check=True, capture_output=True)
    dll = os.path.join(tmp, "nrnmech.dll")

    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h
        h.nrn_load_dll(dll)
        h.load_file("stdrun.hoc")
        h.dt = DT

        sec = h.Section(name="soma")
        sec.L = 10
        sec.diam = 10
        sec.nseg = 1
        sec.insert("adex_taas")
        mech = sec(0.5).adex_taas
        for k, v in P.items():
            setattr(mech, k, v)

        h.finitialize()
        n_steps = int(TSTOP / DT)
        v_trace = []
        for i in range(n_steps + 1):
            t = i * DT
            mech.i_inj = I_INJ if PULSE_ON <= t < PULSE_OFF else 0.0
            if i > 0:
                h.fadvance()
            v_trace.append(mech.v_mem)

        v = np.array(v_trace)
        n = len(v)
        print(f"# adex_taas: n_steps={n}  V range: [{v.min():.4f}, {v.max():.4f}] mV")
        step = int(5.0 / DT)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        print("REF_V_ADEX = [")
        for i in idxs:
            t = i * DT
            print(f"    {v[i]:.6f},  # t={t:.1f}")
        print("]")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
