#!/usr/bin/env python3
"""
Compare instant-current HH vs conductance-based HH on FAFB connectome.

Instant-current: synaptic input → directly added to Ii (step_batch)
Conductance-based: synaptic input → g_syn * (E_rev - V) with τ decay (step_batch_synaptic)

Usage:
  python scripts/fafb_hh_compare.py                          # princeton.taas
  python scripts/fafb_hh_compare.py --taas fafb.taas         # full FAFB
  python scripts/fafb_hh_compare.py --steps 200 --dt 0.5     # longer run
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "examples" / "data"
sys.path.insert(0, str(PROJECT))
import taas_checkmate as tcm


def load_taas(path: Path) -> tcm.ConnectomeData:
    """Load .taas file → ConnectomeData."""
    print(f"  Loading {path} ({path.stat().st_size / 1e6:.0f} MB) ...", end=" ", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"
        version = struct.unpack("<I", f.read(4))[0]
        n_neurons = struct.unpack("<Q", f.read(8))[0]
        n_synapses = struct.unpack("<Q", f.read(8))[0]
        row_ptr = np.frombuffer(f.read((n_neurons + 1) * 4), dtype=np.int32).copy()
        col_indices = np.frombuffer(f.read(n_synapses * 4), dtype=np.int32).copy()
        values = np.frombuffer(f.read(n_synapses * 4), dtype=np.float32).copy()
        nt_types = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32).copy()
        hr_byte = f.read(1)
        _ = struct.unpack("<B", hr_byte)[0] if hr_byte else 0

    elapsed = time.time() - t0
    e_count = int((nt_types > 0).sum())
    i_count = int((nt_types < 0).sum())
    print(f"{elapsed:.1f}s — {n_neurons}N, {n_synapses}S ({e_count}E/{i_count}I)")

    conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
    # Build receptor_types for conductance-based simulation
    print("  Building receptor_types (AMPA/GABA_A mapping) ...", end=" ", flush=True)
    t0 = time.time()
    conn.build_receptor_types()
    print(f"{time.time()-t0:.2f}s")
    return conn


def simulate(sim, conn, n_steps, dt_ms, external_I, scale, method):
    """Run simulation and return spike stats."""
    t0 = time.time()

    if method == "instant":
        spike_buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms,
                                    external_I=external_I, scale=scale)
    elif method == "synaptic":
        spike_buf = sim.step_batch_synaptic(conn, n_steps, dt_ms=dt_ms,
                                             external_I=external_I, scale=scale)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    steps_s = n_steps / elapsed if elapsed > 0 else 0

    sb = spike_buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = sb.sum()
    per_step = sb.sum(axis=1)
    duration = n_steps * dt_ms / 1000.0
    fr = sb.sum(axis=0) / duration

    nt = conn.nt_types()
    exc = nt > 0
    inh = nt < 0
    exc_idx = exc.nonzero()[0] if exc.sum() > 0 else np.array([], dtype=int)
    inh_idx = inh.nonzero()[0] if inh.sum() > 0 else np.array([], dtype=int)

    # Compute stats
    fr_mean = fr.mean()
    fr_e = fr[exc_idx].mean() if len(exc_idx) > 0 else 0.0
    fr_i = fr[inh_idx].mean() if len(inh_idx) > 0 else 0.0
    e_active = (sb[:, exc_idx].sum(axis=0) > 0).sum() if len(exc_idx) > 0 else 0
    i_active = (sb[:, inh_idx].sum(axis=0) > 0).sum() if len(inh_idx) > 0 else 0

    # Peak rate step
    peak_step = per_step.max()
    peak_step_idx = per_step.argmax()

    print(f"    Elapsed:  {elapsed:.2f}s  ({steps_s:.0f} steps/s)")
    print(f"    Total:    {total:,} spikes")
    print(f"    Mean fr:  {fr_mean:.1f} Hz  (E={fr_e:.1f}, I={fr_i:.1f})")
    print(f"    Active:   {e_active}/{exc.sum()} E, {i_active}/{inh.sum()} I")
    print(f"    Per step: min={per_step.min()}, mean={per_step.mean():.0f}, "
          f"max={peak_step} (step {peak_step_idx})")

    return {
        "total": total,
        "fr_mean": fr_mean,
        "fr_e": fr_e,
        "fr_i": fr_i,
        "e_active": e_active,
        "i_active": i_active,
        "per_step": per_step,
        "elapsed": elapsed,
        "steps_s": steps_s,
    }


def main():
    parser = argparse.ArgumentParser(description="FAFB HH: instant vs conductance comparison")
    parser.add_argument("--taas", default="princeton.taas",
                        help=".taas file in examples/data/")
    parser.add_argument("--steps", type=int, default=100,
                        help="Simulation steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Timestep in ms (default: 0.5)")
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found")
        sys.exit(1)

    n_steps = args.steps
    dt_ms = args.dt

    print("╔═══════════════════════════════════════════════════╗")
    print("║  HH: Instant-Current vs Conductance-Based       ║")
    print(f"║  {taas_path.name}  —  {n_steps} steps @ dt={dt_ms}ms         ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    # ── Load connectome ──
    conn = load_taas(taas_path)
    n = conn.n_neurons
    scale = 1e10

    # ── External drive: first 20% E neurons get 10 µA/cm² ──
    print("  Preparing drive current ...", end=" ", flush=True)
    nt = conn.nt_types()
    external_I = np.zeros(n, dtype=np.float32)
    n_drive = max(1, n // 5)
    external_I[:n_drive] = np.where(nt[:n_drive] > 0, 10.0, 0.0)
    print(f"{n_drive} neurons driven ({int((nt[:n_drive] > 0).sum())}E)")

    # ── Create simulators (fresh state for each method) ──
    def make_sim():
        sim = tcm.BrainSimulator(n, "hh")
        sim.set_hh_params(
            np.array([120.0, 36.0], dtype=np.float32),
            np.array([50.0, -77.0], dtype=np.float32),
            np.array([0.3] * n, dtype=np.float32),
            np.array([-65.0] * n, dtype=np.float32),
            dt_ms=dt_ms,
        )
        return sim

    # ════════════════════════════════════════════════════
    #  Test 1: Instant-current HH (step_batch)
    # ════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [1/2] Instant-current HH  (step_batch)")
    print(f"{'─'*55}")
    sim1 = make_sim()
    r1 = simulate(sim1, conn, n_steps, dt_ms, external_I, scale, "instant")
    del sim1

    # ════════════════════════════════════════════════════
    #  Test 2: Conductance-based HH (step_batch_synaptic)
    # ════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [2/2] Conductance-based HH  (step_batch_synaptic)")
    print(f"        AMPA τ=2ms, GABA_A τ=10ms")
    print(f"{'─'*55}")
    sim2 = make_sim()
    r2 = simulate(sim2, conn, n_steps, dt_ms, external_I, scale, "synaptic")
    del sim2

    # ════════════════════════════════════════════════════
    #  Comparison Table
    # ════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Comparison: Instant vs Conductance-based")
    print(f"{'='*55}")
    print(f"  {'Metric':<30s} {'Instant':>10s} {'Synaptic':>10s} {'Δ%':>8s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'Total spikes':<30s} {r1['total']:>10,} {r2['total']:>10,} "
          f"{((r2['total']-r1['total'])/max(r1['total'],1)*100):>+7.1f}%")
    print(f"  {'Mean rate (all)':<30s} {r1['fr_mean']:>10.1f} {r2['fr_mean']:>10.1f} "
          f"{((r2['fr_mean']-r1['fr_mean'])/max(r1['fr_mean'],0.1)*100):>+7.1f}%")
    print(f"  {'Mean rate E':<30s} {r1['fr_e']:>10.1f} {r2['fr_e']:>10.1f} "
          f"{((r2['fr_e']-r1['fr_e'])/max(r1['fr_e'],0.1)*100):>+7.1f}%")
    print(f"  {'Mean rate I':<30s} {r1['fr_i']:>10.1f} {r2['fr_i']:>10.1f} "
          f"{((r2['fr_i']-r1['fr_i'])/max(r1['fr_i'],0.1)*100):>+7.1f}%")
    print(f"  {'E/I ratio':<30s} {r1['fr_e']/max(r1['fr_i'],0.1):>10.2f} "
          f"{r2['fr_e']/max(r2['fr_i'],0.1):>10.2f}")
    print(f"  {'Active E':<30s} {r1['e_active']:>10,} {r2['e_active']:>10,} "
          f"{((r2['e_active']-r1['e_active'])/max(r1['e_active'],1)*100):>+7.1f}%")
    print(f"  {'Active I':<30s} {r1['i_active']:>10,} {r2['i_active']:>10,} "
          f"{((r2['i_active']-r1['i_active'])/max(r1['i_active'],1)*100):>+7.1f}%")
    print(f"  {'Time':<30s} {r1['elapsed']:>10.2f}s {r2['elapsed']:>10.2f}s "
          f"{((r2['elapsed']-r1['elapsed'])/r1['elapsed']*100):>+7.1f}%")
    print(f"  {'Throughput':<30s} {r1['steps_s']:>10.0f} {r2['steps_s']:>10.0f} steps/s")
    print(f"{'='*55}")
    print()

    # Summary
    print("  Interpretation:")
    if r1['fr_mean'] > 0 or r2['fr_mean'] > 0:
        if r2['fr_mean'] < r1['fr_mean'] * 0.8:
            print("    • Conductance-based (τ decay) reduces firing vs instant — expected,")
            print("      because synaptic current decays over time instead of instant delivery.")
        elif r2['fr_mean'] > r1['fr_mean'] * 1.2:
            print("    • Conductance-based increases firing — suggests temporal summation")
            print("      from τ decay creates sustained depolarization.")
        else:
            print("    • Comparable firing rates — FAFB dynamics are similar in both modes.")
    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
