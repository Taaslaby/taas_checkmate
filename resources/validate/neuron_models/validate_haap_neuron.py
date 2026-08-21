#!/usr/bin/env python3
"""HAAP vs NEURON validation (biologically-correct signs, subthreshold).

Validates the conductance/gate physics of the pillars HAAP model against
NEURON `pas` + `km` + `hcn` mechanisms.

IMPORTANT — why subthreshold only
----------------------------------
The pillars HAAP is a **spiking** model: `V >= V_thresh` resets `V` to
`V_reset` and bumps `u += delta_u`. NEURON's `pas`/`km`/`hcn` are **passive
conductances** with no spike mechanism, so the two can only be compared while
`V` stays below `V_thresh` (the reset never fires). In that regime the
equations are identical:

    C_m·dV/dt = I − g_L·(V−E_L) − g_M·u·(V−E_K) − g_h·h·(V−E_h)
    u' = (u_∞(V) − u)/tau_u      u_∞ = 1/(1+exp(−(V−V_half_u)/k_u))
    h' = (h_∞(V) − h)/tau_h      h_∞ = 1/(1+exp(+(V−V_half_h)/k_h))

Two protocols exercise the full subthreshold dynamics:
1. **Depolarizing** pulse (I=+1.5 µA/cm²): leak + M-current opening + HCN
   closing as V depolarizes.
2. **Hyperpolarizing** pulse (I=−3.0 µA/cm²): HCN *opening* during inhibition,
   then the rebound depolarization after release (the model's raison d'être).

A pass means RMSE < RTOL (default 0.5 mV) — in practice it is ~0.03 mV,
i.e. machine precision, confirming the unit conversions (`g/1000` → S/cm²,
`I·area·1000` → nA) and the gate signs are correct.

Units mapping:
    pillars `g` (mS/cm²) · `V` (mV) = current (µA/cm²)  →  NEURON `gbar` = g/1000 S/cm²
    pillars `I` (µA/cm²)                                →  NEURON `IClamp.amp` = I·area_cm²·1000 nA

Usage:
    python validate/neuron_models/validate_haap_neuron.py [--plot]
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Project root is three levels up: resources/validate/neuron_models -> root.
# (Pointing at root matters: it is where the freshly built `taas_checkmate.pyd`
# lives. Pointing at `resources/` would silently import the stale copy in
# site-packages instead.)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

NEURON_HOME = r"C:\nrn"
MKNrndll = os.path.join(NEURON_HOME, "bin", "mknrndll.bat")

DT = 0.05
# 10 µm diameter × 10 µm length cylinder — converts injected current density
# (µA/cm²) into NEURON's absolute nA for the same membrane area.
AREA_CM2 = np.pi * 10.0 * 10.0 * 1e-8
RTOL_MV = 0.5  # pass threshold (mV); measured is ~0.03 mV

# VNC profile (matches zoo::neurons::haap `make_vnc_params` / HaapProfile::vnc).
PARAMS = dict(
    g_L=0.1, E_L=-65.0, cm=1.0,
    g_M=0.5, E_K=-80.0, V_half_u=-35.0, k_u=5.0, tau_u=100.0,
    g_h=0.3, E_h=-30.0, V_half_h=-75.0, k_h=6.0, tau_h=200.0,
    V_thresh=-40.0, V_reset=-55.0, delta_u=0.02,
)

# (name, current µA/cm², pulse duration ms, extra trace time after pulse end).
PROTOCOLS = [
    ("depolarizing_subthreshold", +1.5, 100.0, 100.0),
    ("hyperpolarizing_rebound", -3.0, 100.0, 150.0),
]
PULSE_START_MS = 50.0


def compile_mods():
    """Compile km.mod/hcn.mod in an ASCII temp dir (NEURON's c_str() cannot
    handle the Cyrillic project path), return (dir, dll_path)."""
    tmp = tempfile.mkdtemp(prefix="haap_mods_")
    for f in ("km.mod", "hcn.mod"):
        shutil.copy(os.path.join(HERE, f), os.path.join(tmp, f))
    subprocess.run(
        ["cmd.exe", "/c", MKNrndll],
        cwd=tmp, check=True, capture_output=True,
    )
    return tmp, os.path.join(tmp, "nrnmech.dll")


def setup_neuron(dll):
    """Load the compiled mechanisms and build the pas+km+hcn soma. Returns a
    callable `run(i_amp, pulse_ms, duration_ms) -> V(t)` for one session."""
    old_cwd = os.getcwd()
    os.chdir(tempfile.gettempdir())  # ASCII cwd for NEURON's c_str()

    from neuron import h
    h.nrn_load_dll(dll)
    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = 10
    soma.diam = 10
    soma.cm = PARAMS["cm"]

    soma.insert("pas")
    soma(0.5).g_pas = PARAMS["g_L"] / 1000.0
    soma(0.5).e_pas = PARAMS["E_L"]

    soma.insert("km")
    soma(0.5).km.gbar = PARAMS["g_M"] / 1000.0
    soma(0.5).km.vhalf = PARAMS["V_half_u"]
    soma(0.5).km.kslope = PARAMS["k_u"]
    soma(0.5).km.mtau = PARAMS["tau_u"]

    soma.insert("hcn")
    soma(0.5).hcn.gbar = PARAMS["g_h"] / 1000.0
    soma(0.5).hcn.vhalf = PARAMS["V_half_h"]
    soma(0.5).hcn.kslope = PARAMS["k_h"]
    soma(0.5).hcn.htau = PARAMS["tau_h"]

    stim = h.IClamp(soma(0.5))

    def run(i_amp, pulse_ms, duration_ms):
        v_vec = h.Vector()
        v_vec.record(soma(0.5)._ref_v)
        stim.delay = PULSE_START_MS
        stim.dur = pulse_ms
        stim.amp = i_amp * 1000.0 * AREA_CM2 * PARAMS["cm"]
        h.dt = DT
        h.tstop = duration_ms
        h.finitialize(PARAMS["E_L"])
        h.continuerun(duration_ms)
        # h.Vector records V at t = 0, DT, 2·DT, … duration_ms (n_steps+1 pts).
        return np.array(v_vec)

    return run, old_cwd


def run_taas(i_amp, pulse_ms, duration_ms):
    """Run pillars HAAP via `step_haap` (subthreshold), return V(t) array
    with the same t-grid as NEURON (t = 0, DT, … duration_ms)."""
    import taas_checkmate as tcm

    n_steps = int(duration_ms / DT)
    n = 8
    # Gates at their Boltzmann steady state at V = E_L, matching NEURON's
    # INITIAL block (u_∞ ≈ 0.0025, h_∞ ≈ 0.159).
    u_rest = 1.0 / (1.0 + np.exp(-(PARAMS["E_L"] - PARAMS["V_half_u"]) / PARAMS["k_u"]))
    h_rest = 1.0 / (1.0 + np.exp((PARAMS["E_L"] - PARAMS["V_half_h"]) / PARAMS["k_h"]))

    v = np.full(n, PARAMS["E_L"], dtype=np.float32)
    u = np.full(n, u_rest, dtype=np.float32)
    h = np.full(n, h_rest, dtype=np.float32)
    # `spikes` is a per-step flag (overwritten each step), not a counter.
    spikes = np.zeros(n, dtype=np.int32)

    p = np.zeros(n * 17, dtype=np.float32)
    cols = [DT, PARAMS["cm"], PARAMS["g_L"], PARAMS["E_L"], PARAMS["g_M"],
            PARAMS["E_K"], PARAMS["g_h"], PARAMS["E_h"], PARAMS["tau_u"],
            PARAMS["tau_h"], PARAMS["V_half_u"], PARAMS["k_u"],
            PARAMS["V_half_h"], PARAMS["k_h"], PARAMS["V_thresh"],
            PARAMS["V_reset"], PARAMS["delta_u"]]
    for c, val in enumerate(cols):
        p[c * n:(c + 1) * n] = val

    i_inj = np.zeros(n_steps)
    i_inj[int(PULSE_START_MS / DT):int((PULSE_START_MS + pulse_ms) / DT)] = i_amp

    v_trace = [float(v[0])]  # t = 0 (before the first step)
    for i in range(n_steps):
        tcm.step_haap(v, u, h, np.full(n, i_inj[i], dtype=np.float32), spikes, p)
        v_trace.append(float(v[0]))
    return np.array(v_trace)


def main():
    plot = "--plot" in sys.argv
    tmp, dll = compile_mods()
    print(f"Compiled km.mod + hcn.mod -> {dll}")

    run_neuron, old_cwd = setup_neuron(dll)
    all_pass = True
    traces = {}
    try:
        for name, i_amp, pulse_ms, extra_ms in PROTOCOLS:
            duration_ms = PULSE_START_MS + pulse_ms + extra_ms
            v_neuron = run_neuron(i_amp, pulse_ms, duration_ms)
            v_taas = run_taas(i_amp, pulse_ms, duration_ms)
            traces[name] = (v_neuron, v_taas)

            m = min(len(v_neuron), len(v_taas))
            rmse = float(np.sqrt(np.mean((v_neuron[:m] - v_taas[:m]) ** 2)))
            v_range = float(np.ptp(v_taas[:m]))
            ok = rmse < RTOL_MV
            all_pass &= ok
            print(f"  {name:28s} I={i_amp:+5.1f} uA/cm2  "
                  f"RMSE={rmse:7.4f} mV  (range {v_range:5.2f} mV)  "
                  f"{'PASS' if ok else 'FAIL'}")
    finally:
        os.chdir(old_cwd)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(PROTOCOLS), 1, figsize=(10, 3 * len(PROTOCOLS)), sharex=False)
        if len(PROTOCOLS) == 1:
            axes = [axes]
        for ax, (name, i_amp, _, _) in zip(axes, PROTOCOLS):
            vn, vt = traces[name]
            m = min(len(vn), len(vt))
            t = np.arange(m) * DT
            rmse = float(np.sqrt(np.mean((vn[:m] - vt[:m]) ** 2)))
            ax.plot(t, vn[:m], "r-", label="NEURON", alpha=0.85)
            ax.plot(t, vt[:m], "b--", label="TAAS", alpha=0.85)
            ax.axvline(PULSE_START_MS, color="gray", ls=":", alpha=0.5)
            ax.set_ylabel("V (mV)")
            ax.set_title(f"{name} (I={i_amp:+} uA/cm2): RMSE={rmse:.4f} mV")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("t (ms)")
        plt.tight_layout()
        out = os.path.join(HERE, "haap_comparison.png")
        plt.savefig(out, dpi=150)
        print(f"  plot: {out}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("RESULT:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
