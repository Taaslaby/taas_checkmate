#!/usr/bin/env python3
"""Generate NEURON reference for the reduced Poirazi 2003 CA1 soma.

The real published CA1 pyramidal cell (Poirazi–Brannon–Mel 2003, OSB
Poirazi2003-CA1PyramidalCell, NeuroML2 export) reduced to its soma: real soma
geometry (d=4.58 um, L=0.01 um) + real soma channels (na m^2h, kdr n^2,
leak 5e-3 mS/cm2, hs Ih 1.872e-2 mS/cm2, km M-current 60 mS/cm2), Cm=1,
Ra=0.05 kohm-cm, V0=-70, spikeThresh=0.

The four channel mechanisms (poirazi_na/kdr/km/h.mod) are faithful NEURON
ports of the published NeuroML2 channels. Validation runs at celsius=35
(secondorder=0, dt=0.025) — physiological CA1 temperature. The published
per-channel q10Settings are applied by BOTH sides: km carries q10 =
2.3^((celsius-23)/10) in the mod (jNeuroML semantics), na carries q10Fixed=1
(no temperature dependence), and the pillars engine applies the same factors
per gate. At 35C km accelerates ~2.7x vs the old 6.3C Q10=1 runs.

Prints V(soma) on a 5 ms grid + spike counts per protocol; values are
hardcoded into the Rust regression (NEURON is not available in CI).
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
V0 = -70.0
CM = 1.0
RA = 50.0  # 0.05 kohm_cm
L = 0.01
DIAM = 4.58
THRESH = 0.0

# Published soma densities (mS/cm2) and reversals (mV).
CHANNELS = [
    ("pas", dict(g_pas=5e-3 / 1000.0, e_pas=-70.0)),
    ("poirazi_na", dict(gbar=7.0 / 1000.0, ena=50.0)),
    ("poirazi_kdr", dict(gbar=1.4 / 1000.0, ek=-77.0)),
    ("poirazi_km", dict(gbar=60.0 / 1000.0, ek=-80.0)),
    ("poirazi_h", dict(gbar=1.872e-2 / 1000.0, eh=-10.0)),
]

# (name, I nA, pulse on, pulse off, tstop ms)
PROTOCOLS = [
    ("depolarizing_spikes", +2.0e-7, 50.0, 100.0, 150.0),
    ("hyperpolarizing_ih", -1.0e-7, 50.0, 100.0, 150.0),
]

MODS = ["poirazi_na.mod", "poirazi_kdr.mod", "poirazi_km.mod", "poirazi_h.mod"]


def compile_mods():
    """Compile the four poirazi mods in an ASCII temp dir (NEURON's c_str()
    cannot handle the Cyrillic project path). Returns (dir, dll_path)."""
    tmp = tempfile.mkdtemp(prefix="poirazi_mods_")
    for f in MODS:
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    subprocess.run(
        ["cmd.exe", "/c", MKNrndll],
        cwd=tmp, check=True, capture_output=True,
    )
    return tmp, os.path.join(tmp, "nrnmech.dll")


def run_neuron(dll):
    """Build the real soma with the published channels. Returns a callable
    run(i_amp, on_ms, off_ms, tstop_ms) -> V(t) (n_steps+1 points)."""
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())

    from neuron import h
    h.nrn_load_dll(dll)
    h.load_file("stdrun.hoc")
    h.celsius = 35.0
    h.secondorder = 0
    h.dt = DT

    soma = h.Section(name="soma")
    soma.L = L
    soma.diam = DIAM
    soma.nseg = 1
    soma.Ra = RA
    soma.cm = CM
    for mech, params in CHANNELS:
        soma.insert(mech)
        seg = soma(0.5)
        if mech == "pas":
            # Built-in leak — range vars carry no suffix (g_pas/e_pas).
            for k, val in params.items():
                setattr(seg, k, val)
        else:
            mech_obj = getattr(seg, mech)
            for k, val in params.items():
                setattr(mech_obj, k, val)

    stim = h.IClamp(soma(0.5))

    def run(i_amp, on_ms, off_ms, tstop_ms):
        v_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        stim.delay = on_ms
        stim.dur = off_ms - on_ms
        stim.amp = i_amp
        h.tstop = tstop_ms
        h.finitialize(V0)
        h.continuerun(tstop_ms)
        return np.array(v_vec)

    return run, old_cwd


def main():
    tmp, dll = compile_mods()
    print(f"compiled {len(MODS)} mods -> {dll}")
    run, old_cwd = run_neuron(dll)
    try:
        for name, i_amp, on_ms, off_ms, tstop in PROTOCOLS:
            v = run(i_amp, on_ms, off_ms, tstop)
            n = len(v)
            spikes = int(np.sum((v[:-1] < THRESH) & (v[1:] >= THRESH)))
            print(f"# {name}: I={i_amp:+.1e} nA  n_steps={n}  spikes={spikes}  "
                  f"V range: [{v.min():.4f}, {v.max():.4f}] mV")
            const = f"REF_V_{name.upper()}"
            step = int(5.0 / DT)
            idxs = list(range(0, n, step))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)
            print(f"{const} = [")
            for i in idxs:
                print(f"    {v[i]:.6f},  # t={i * DT:.1f}")
            print("]")
            print(f"REF_SPIKES_{name.upper()} = {spikes}")
    finally:
        os.chdir(old_cwd)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
