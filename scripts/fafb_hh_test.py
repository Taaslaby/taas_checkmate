#!/usr/bin/env python3
"""
Load baked .taas connectome → HH simulation → spike stats.

Usage:
  python scripts/fafb_hh_test.py                          # princeton.taas, 100 steps
  python scripts/fafb_hh_test.py --taas fafb.taas          # 614MB full FAFB
  python scripts/fafb_hh_test.py --taas fafb_15m.taas      # 117MB subsample
  python scripts/fafb_hh_test.py --steps 1000 --dt 0.1     # long run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Project root
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "examples" / "data"

sys.path.insert(0, str(PROJECT))
# Assuming taas_checkmate is installed or in PYTHONPATH
import taas_checkmate as tcm


def load_taas(path: Path) -> tcm.ConnectomeData:
    """Load .taas file and return ConnectomeData."""
    print(f"  Loading {path} ({path.stat().st_size / 1e6:.0f} MB) ...")
    t0 = time.time()

    # .taas file is a binary format with header + CSR arrays.
    # ConnectomeData::load_baked reads row_ptr, col_indices, values, nt_types
    # We'll reconstruct from numpy arrays saved alongside .taas.
    # For now, use the Python-side loader via taas_format if available,
    # or reconstruct from known structure.
    import struct

    with open(path, "rb") as f:
        # Magic: 0x54414153 = "TAAS" in LE = b"SAAT"
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"

        version = struct.unpack("<I", f.read(4))[0]
        n_neurons = struct.unpack("<Q", f.read(8))[0]
        n_synapses = struct.unpack("<Q", f.read(8))[0]

        print(f"    Format v{version}: {n_neurons} neurons, {n_synapses} synapses")

        # Read row_ptr (int32, n_neurons+1)
        row_ptr = np.frombuffer(f.read((n_neurons + 1) * 4), dtype=np.int32).copy()
        # Read col_indices (int32, n_synapses)
        col_indices = np.frombuffer(f.read(n_synapses * 4), dtype=np.int32).copy()
        # Read values (float32, n_synapses)
        values = np.frombuffer(f.read(n_synapses * 4), dtype=np.float32).copy()
        # Read nt_types (float32, n_neurons)
        nt_types = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32).copy()
        # Read has_receptors flag + optional receptor_types array
        # Older .taas files don't have this byte — graceful EOF fallback
        hr_byte = f.read(1)
        has_receptors = struct.unpack("<B", hr_byte)[0] if hr_byte else 0
        receptor_types = None
        if has_receptors:
            receptor_types = np.frombuffer(f.read(n_synapses), dtype=np.uint8).copy()
            print(f"    receptor_types: {len(receptor_types)} bytes (per-synapse)")
        else:
            print(f"    No receptor_types")

    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Basic E/I stats from nt_types (+1 = exc, -1 = inh)
    e_count = int((nt_types > 0).sum())
    i_count = int((nt_types < 0).sum())
    print(f"    E/I: {e_count}E / {i_count}I")

    conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
    return conn


def run_hh_simulation(conn: tcm.ConnectomeData, n_steps: int, dt_ms: float):
    """Run HH on connectome and report spike statistics."""
    n = conn.n_neurons

    print(f"\n── Creating BrainSimulator(model='hh') for {n} neurons ──")
    sim = tcm.BrainSimulator(n, "hh")

    # Set HH parameters for realistic cortical neurons
    # Standard: NaV 120, KV 36, g_leak=0.3
    sim.set_hh_params(
        np.array([120.0, 36.0], dtype=np.float32),  # g_max: NaV, KV
        np.array([50.0, -77.0], dtype=np.float32),   # E_rev: Na, K
        np.array([0.3] * n, dtype=np.float32),        # g_leak per neuron
        np.array([-65.0] * n, dtype=np.float32),      # E_leak per neuron
        dt_ms=dt_ms,
    )

    # Drive current for HH spiking
    # HH standard: I > 7 µA/cm² triggers spiking at NaV=120, KV=36
    external_I = np.zeros(n, dtype=np.float32)
    n_drive = max(1, n // 5)
    # Drive first 20% of excitatory neurons with 10 µA/cm²
    nt = conn.nt_types()[:n_drive]
    exc_mask = nt > 0
    external_I[:n_drive] = np.where(exc_mask, 10.0, 0.0)
    # Add a brief 50-step pulse to the first 5% to kickstart network
    n_kick = max(1, n // 20)
    kick = np.zeros(n, dtype=np.float32)
    kick[:n_kick] = 15.0  # strong kick to E neurons

    print(f"\n── Running {n_steps} steps at dt={dt_ms}ms ──")
    t0 = time.time()

    # Determine appropriate scale factor for baked weights
    # Princeton baked: weight = syn_count * 1e-12, so scale=1e10 gives ~1e-2
    scale = 1e10 if "princeton" in str(conn._taas_path if hasattr(conn, '_taas_path') else '') else 1e10

    # Step 1: kick with strong input for first 50 steps
    spike_buf = np.zeros(n_steps * n, dtype=np.int32)
    for step in range(n_steps):
        ext = kick if step < 50 else external_I
        step_buf = sim.step_batch(conn, 1, dt_ms=dt_ms, external_I=ext, scale=scale)
        spike_buf[step * n:(step + 1) * n] = step_buf

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.2f}s ({n_steps/elapsed:.0f} steps/s)")

    # Analyze spikes
    spike_buf_2d = spike_buf.reshape(n_steps, n).astype(np.int32)
    total_spikes = spike_buf_2d.sum()
    spike_counts = spike_buf_2d.sum(axis=0)  # per neuron

    # Compute firing rate per neuron (Hz)
    duration_s = n_steps * dt_ms / 1000.0
    fr_per_neuron = spike_counts / duration_s

    # E/I groups
    nt = conn.nt_types()
    exc_neurons = nt > 0
    inh_neurons = nt < 0

    print(f"\n═══ Spike Statistics ═══")
    print(f"  Total spikes:       {total_spikes}")
    print(f"  Mean rate (all):    {fr_per_neuron.mean():.2f} Hz")
    if exc_neurons.sum() > 0:
        print(f"  Mean rate (E):      {fr_per_neuron[exc_neurons.nonzero()[0]].mean():.2f} Hz")
        e_spiking = (spike_counts[exc_neurons.nonzero()[0]] > 0).sum()
        print(f"  Spiking E neurons:  {e_spiking}/{exc_neurons.sum()}")
    if inh_neurons.sum() > 0:
        print(f"  Mean rate (I):      {fr_per_neuron[inh_neurons.nonzero()[0]].mean():.2f} Hz")
        i_spiking = (spike_counts[inh_neurons.nonzero()[0]] > 0).sum()
        print(f"  Spiking I neurons:  {i_spiking}/{inh_neurons.sum()}")

    # Top 5 firing neurons
    top_idx = np.argsort(fr_per_neuron)[-5:][::-1]
    print(f"\n  Top 5 firing neurons:")
    for idx in top_idx:
        nt_label = "E" if nt[idx] > 0 else ("I" if nt[idx] < 0 else "?")
        print(f"    Neuron {idx:>6d}  {nt_label:>3s}  {fr_per_neuron[idx]:>6.1f} Hz  ({spike_counts[idx]} spikes)")

    # Timecourse: spikes per step
    spikes_per_step = spike_buf_2d.sum(axis=1)
    print(f"\n  Spikes per step:  min={spikes_per_step.min()}, "
          f"mean={spikes_per_step.mean():.1f}, max={spikes_per_step.max()}")
    
    # Show first few and last few step counts
    print(f"    First 5 steps: {spikes_per_step[:5].tolist()}")
    
    # Check if any neurons reached sustained firing (>10 spikes)
    sustained = (spike_counts > 10).sum()
    print(f"  Sustained firing (>10 spikes): {sustained}/{n} neurons")

    return fr_per_neuron, spike_buf_2d


def main():
    parser = argparse.ArgumentParser(description="FAFB HH simulation test")
    parser.add_argument("--taas", default="princeton.taas",
                        help=".taas file in examples/data/ (default: princeton.taas)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Simulation steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Timestep in ms (default: 0.5)")
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found")
        print(f"Available .taas files:")
        for f in sorted(DATA.glob("*.taas")):
            size = f.stat().st_size / 1e6
            print(f"  {f.name:30s}  {size:>6.0f} MB")
        sys.exit(1)

    print(f"╔════════════════════════════════════════════╗")
    print(f"║  FAFB HH Simulation Test                  ║")
    print(f"╚════════════════════════════════════════════╝")
    print()

    conn = load_taas(taas_path)
    fr, spike_buf = run_hh_simulation(conn, args.steps, args.dt)

    print(f"\n✅ Test complete.")


if __name__ == "__main__":
    main()
