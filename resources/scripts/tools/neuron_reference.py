"""Generate ground-truth HH reference traces using NEURON simulator.

Produces data/ground_truth/hh_neuron.ref (binary) and hh_neuron_if.csv.

NEURON model matches taas-checkmate's HH engine:
  - Hodgkin-Huxley 1952 squid axon @ 6.3°C
  - NaV (m³h): gnabar=120 mS/cm², ena=50 mV
  - KV (n⁴):   gkbar=36  mS/cm², ek=-77 mV
  - Leak:      gl=0.3    mS/cm², el=-54.3 mV (default NEURON)
  - Cm = 1 µF/cm²
  - dt = 0.01 ms, duration = 500 ms
"""

import os
import struct
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ground_truth"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── HH parameters (match tasa-checkmate) ──────────────────────────────
DT = 0.01        # ms
DURATION = 500   # ms
N_STEPS = int(DURATION / DT)

I_EXTS = (0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0)  # µA/cm²

# Soma geometry (tiny cylindrical compartment → area ≈ 1e-5 cm²)
SOMA_DIAM = 10.0   # µm
SOMA_L = 10.0      # µm


def soma_area_cm2(d, l):
    """Lateral surface area in cm² (NEURON uses lateral only)."""
    return (np.pi * d * l) * 1e-8  # µm² → cm²


AREA_CM2 = soma_area_cm2(SOMA_DIAM, SOMA_L)


def run_neuron_hh(i_ext_density):
    """Run NEURON HH for one I_ext value, return (t, v, m, h, n)."""
    from neuron import h

    h.dt = DT
    # Set tstop via HOC
    h("tstop = {}".format(DURATION))

    soma = h.Section("soma")
    soma.L = SOMA_L
    soma.diam = SOMA_DIAM
    soma.cm = 1.0
    soma.insert("hh")

    seg = soma(0.5)
    seg.hh.gnabar = 0.12    # S/cm² = 120 mS/cm²
    seg.hh.gkbar = 0.036    # S/cm² = 36  mS/cm²
    seg.hh.gl = 0.0003      # S/cm² = 0.3 mS/cm²
    seg.hh.el = -54.3       # mV (NEURON default)
    seg.ena = 50.0          # mV
    seg.ek = -77.0          # mV

    # Current injection: µA/cm² density → nA total
    i_clamp_nA = i_ext_density * AREA_CM2 * 1000.0
    stim = h.IClamp(seg)
    stim.delay = 0.0
    stim.dur = DURATION
    stim.amp = i_clamp_nA

    # Recording vectors
    t_vec = h.Vector()
    v_vec = h.Vector()
    m_vec = h.Vector()
    h_vec = h.Vector()
    n_vec = h.Vector()

    t_vec.record(h._ref_t)
    v_vec.record(seg._ref_v)
    m_vec.record(seg.hh._ref_m)
    h_vec.record(seg.hh._ref_h)
    n_vec.record(seg.hh._ref_n)

    # Manual run — finitialize + fadvance loop
    h.finitialize(-65)
    tstop_val = DURATION
    target_steps = int(tstop_val / DT)
    for _ in range(target_steps):
        h.fadvance()

    return (
        np.array(t_vec, dtype=np.float32),
        np.array(v_vec, dtype=np.float32),
        np.array(m_vec, dtype=np.float32),
        np.array(h_vec, dtype=np.float32),
        np.array(n_vec, dtype=np.float32),
    )


def write_reference():
    """Generate and save all reference data."""
    print(f"NEURON HH reference — area = {AREA_CM2:.4e} cm²")
    print(f"I_ext sweep: {I_EXTS}")
    print()

    n_currents = len(I_EXTS)
    v_all = np.empty((n_currents, N_STEPS), dtype=np.float32)
    m_all = np.empty((n_currents, N_STEPS), dtype=np.float32)
    h_all = np.empty((n_currents, N_STEPS), dtype=np.float32)
    n_all = np.empty((n_currents, N_STEPS), dtype=np.float32)

    for idx, i_ext in enumerate(I_EXTS):
        print(f"  Running I_ext = {i_ext:6.1f} µA/cm²  ", end="", flush=True)
        t, v, m, h_gate, n = run_neuron_hh(i_ext)

        # NEURON may produce more steps than N_STEPS; truncate/pad
        actual_steps = min(len(v), N_STEPS)
        v_all[idx, :actual_steps] = v[:actual_steps]
        m_all[idx, :actual_steps] = m[:actual_steps]
        h_all[idx, :actual_steps] = h_gate[:actual_steps]
        n_all[idx, :actual_steps] = n[:actual_steps]
        if actual_steps < N_STEPS:
            v_all[idx, actual_steps:] = v_all[idx, actual_steps - 1]
            m_all[idx, actual_steps:] = m_all[idx, actual_steps - 1]
            h_all[idx, actual_steps:] = h_all[idx, actual_steps - 1]
            n_all[idx, actual_steps:] = n_all[idx, actual_steps - 1]

        spikes = np.sum(np.diff((v > 0).astype(int)) > 0)
        print(f"  spikes={spikes}")

    # ── Write binary .ref file ────────────────────────────────────────
    ref_path = DATA_DIR / "hh_neuron.ref"
    print(f"\n  Writing {ref_path} ...", end=" ", flush=True)

    with open(ref_path, "wb") as f:
        # [4B magic "HHRF", 1B version=1]
        f.write(b"HHRF")
        f.write(struct.pack("B", 1))
        # [4B u32 num_currents, 4B u32 num_steps, 4B f32 dt]
        f.write(struct.pack("IIf", n_currents, N_STEPS, DT))
        # [ num_currents × 4B ] I_ext values
        f.write(np.array(I_EXTS, dtype=np.float32).tobytes())
        # [ num_currents × num_steps × 4B ] V traces, row-major
        f.write(v_all.tobytes())
        # [ num_currents × num_steps × 3 × 4B ] m,h,n gates, row-major
        gate_stack = np.stack([m_all, h_all, n_all], axis=-1)  # [nc, ns, 3]
        f.write(gate_stack.tobytes())

    print("OK")

    # ── Write F-I curve CSV ──────────────────────────────────────────
    csv_path = DATA_DIR / "hh_neuron_if.csv"
    print(f"  Writing {csv_path} ...", end=" ", flush=True)

    with open(csv_path, "w") as f:
        f.write("I_ext_uA_per_cm2,firing_rate_Hz\n")
        for idx, i_ext in enumerate(I_EXTS):
            v_trace = v_all[idx]
            crossings = np.diff((v_trace > 0).astype(int))
            spike_times = np.where(crossings > 0)[0] * DT
            interval = DURATION / 1000.0  # seconds
            n_spikes = len(spike_times)

            if n_spikes >= 2:
                isi_mean = np.mean(np.diff(spike_times[:20])) * DT / 1000.0
                freq = 1.0 / isi_mean if isi_mean > 0 else 0
            elif n_spikes == 1:
                freq = n_spikes / interval
            else:
                freq = 0.0

            f.write(f"{i_ext},{freq:.2f}\n")

    print("OK")
    print("\nDone!")


if __name__ == "__main__":
    write_reference()
