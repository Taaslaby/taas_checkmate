#!/usr/bin/env python3
"""Generate NEURON reference for active HH channels on a cable soma.

A single soma compartment (L=20 um, diam=20 um) with the built-in `hh`
mechanism (NaV/KV + leak), current-clamped, is the exact active-channel
counterpart of the passive `gen_cable_ref.py`. Records V(soma) and counts
spikes (rising edge through 0 mV) so the values can be hardcoded into the Rust
`cable_hh_vs_neuron` regression (NEURON env is not available in CI).

NEURON `hh` defaults are the original squid-axon parameters, matching pillars'
`standard_channels()` NaV/KV + leak:
    gnabar=0.12 S/cm2, gkbar=0.036 S/cm2, gl=0.0003 S/cm2, el=-54.3 mV,
    celsius=6.3 (Q10=1). V0=-65. I = +0.3 nA into the soma from t=10 to 90 ms.
    dt=0.025, secondorder=0, tstop=100 ms.
"""
import os
import tempfile

import numpy as np

DT = 0.025
TSTOP = 100.0
V0 = -65.0
I_AMP = 0.3  # nA
PULSE_ON = 10.0
PULSE_OFF = 90.0
CM = 1.0
RA = 100.0
THRESH = 0.0  # spike detection threshold (mV)


def main():
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        from neuron import h

        h.load_file("stdrun.hoc")
        h.celsius = 6.3  # Q10 = 1
        h.secondorder = 0  # backward Euler == pillars Hines solver
        h.dt = DT
        h.tstop = TSTOP

        soma = h.Section(name="soma")
        soma.L = 20
        soma.diam = 20
        soma.nseg = 1
        soma.Ra = RA
        soma.cm = CM
        soma.insert("hh")
        seg = soma(0.5)
        seg.gnabar_hh = 0.12
        seg.gkbar_hh = 0.036
        seg.gl_hh = 0.0003
        seg.el_hh = -54.3

        stim = h.IClamp(soma(0.5))
        stim.delay = PULSE_ON
        stim.dur = PULSE_OFF - PULSE_ON
        stim.amp = I_AMP

        v = h.Vector()
        v.record(soma(0.5)._ref_v)

        h.finitialize(V0)
        h.continuerun(TSTOP)

        arr = np.array(v)
        n = len(arr)
        # Rising edge through THRESH.
        spikes = int(np.sum((arr[:-1] < THRESH) & (arr[1:] >= THRESH)))
        print(f"# SOMA: n_steps={n}  spikes={spikes}  V range: [{arr.min():.4f}, {arr.max():.4f}] mV")
        step = int(5.0 / DT)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        print("REF_V_SOMA = [")
        for i in idxs:
            t = i * DT
            print(f"    {arr[i]:.6f},  # t={t:.1f}")
        print("]")
        print(f"REF_SPIKES = {spikes}")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
