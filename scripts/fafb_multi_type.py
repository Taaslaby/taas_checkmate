#!/usr/bin/env python3
"""
Multi-type HH: per-neuron g_max на коннектоме.

Сравнивает uniform HH (все нейроны одинаковые) vs multi-type HH
(возбуждающие/тормозные нейроны с разной плотностью ионных каналов).

Типы нейронов:
  - Excitatory (nt > 0): NaV=120, KV=36  — standard pyramidal-like
  - Inhibitory (nt < 0): NaV=80,  KV=45  — fast-spiking PV-like (меньше Na = выше порог, 
    больше KV = быстрее реполяризация → выше частота)

Usage:
  python scripts/fafb_multi_type.py                          # princeton.taas
  python scripts/fafb_multi_type.py --taas fafb.taas         # full FAFB
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
    print(f"  Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...", end=" ", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"
        _ = struct.unpack("<I", f.read(4))[0]  # version
        n_neurons = struct.unpack("<Q", f.read(8))[0]
        n_synapses = struct.unpack("<Q", f.read(8))[0]
        row_ptr = np.frombuffer(f.read((n_neurons + 1) * 4), dtype=np.int32).copy()
        col_indices = np.frombuffer(f.read(n_synapses * 4), dtype=np.int32).copy()
        values = np.frombuffer(f.read(n_synapses * 4), dtype=np.float32).copy()
        nt_types = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32).copy()

    e_count = int((nt_types > 0).sum())
    i_count = int((nt_types < 0).sum())
    print(f"{time.time()-t0:.1f}s — {n_neurons}N, {n_synapses}S ({e_count}E/{i_count}I)")
    return tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)


def make_uniform_sim(n: int, dt_ms: float) -> tcm.BrainSimulator:
    """Uniform HH: все нейроны с стандартными NaV=120, KV=36."""
    sim = tcm.BrainSimulator(n, "hh")
    g_max = np.array([120.0, 36.0], dtype=np.float32)  # broadcast: [n_channels]
    sim.set_hh_params(
        g_max,
        np.array([50.0, -77.0], dtype=np.float32),
        np.array([0.3] * n, dtype=np.float32),
        np.array([-65.0] * n, dtype=np.float32),
        dt_ms=dt_ms,
    )
    return sim


def make_multi_type_sim(conn: tcm.ConnectomeData, dt_ms: float) -> tcm.BrainSimulator:
    """Multi-type HH: E/I нейроны с разной плотностью каналов."""
    n = conn.n_neurons
    nt = conn.nt_types()
    n_channels = 2  # NaV, KV

    # Per-neuron g_max: column-major [ch0_n0..ch0_nN, ch1_n0..ch1_nN]
    # NaV (ch0): E=120, I=80 (меньше Na = выше порог для I)
    # KV  (ch1): E=36,  I=45 (больше KV = быстрее реполяризация)
    g_max_per_neuron = np.zeros(n_channels * n, dtype=np.float32)

    for i in range(n):
        if nt[i] > 0:  # Excitatory
            g_max_per_neuron[0 * n + i] = 120.0  # NaV: pyramidal-like
            g_max_per_neuron[1 * n + i] = 36.0   # KV
        elif nt[i] < 0:  # Inhibitory
            g_max_per_neuron[0 * n + i] = 80.0   # NaV: lower → higher threshold
            g_max_per_neuron[1 * n + i] = 45.0   # KV: higher → faster repolarization
        else:
            g_max_per_neuron[0 * n + i] = 120.0
            g_max_per_neuron[1 * n + i] = 36.0

    sim = tcm.BrainSimulator(n, "hh")
    sim.set_hh_params(
        g_max_per_neuron,  # [n_channels * n] → per-neuron!
        np.array([50.0, -77.0], dtype=np.float32),
        np.array([0.3] * n, dtype=np.float32),
        np.array([-65.0] * n, dtype=np.float32),
        dt_ms=dt_ms,
    )
    return sim


def simulate(sim, conn, n_steps, dt_ms, external_I, scale, label):
    """Run BB simulation + return stats."""
    t0 = time.time()
    spike_buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms,
                                external_I=external_I, scale=scale)
    elapsed = time.time() - t0

    sb = spike_buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = sb.sum()
    duration = n_steps * dt_ms / 1000.0
    fr = sb.sum(axis=0) / duration

    nt = conn.nt_types()
    exc = nt > 0
    inh = nt < 0
    fr_e = fr[exc].mean() if exc.sum() > 0 else 0.0
    fr_i = fr[inh].mean() if inh.sum() > 0 else 0.0
    e_active = (sb[:, exc].sum(axis=0) > 0).sum() if exc.sum() > 0 else 0
    i_active = (sb[:, inh].sum(axis=0) > 0).sum() if inh.sum() > 0 else 0

    return {
        "label": label,
        "total": total,
        "fr_mean": fr.mean(),
        "fr_e": fr_e,
        "fr_i": fr_i,
        "ei_ratio": fr_e / max(fr_i, 0.1),
        "e_active": e_active,
        "i_active": i_active,
        "elapsed": elapsed,
        "steps_s": n_steps / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-type HH on FAFB")
    parser.add_argument("--taas", default="princeton.taas")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found"); sys.exit(1)

    n_steps, dt_ms = args.steps, args.dt

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Multi-Type HH: Uniform vs E/I-differentiated      ║")
    print(f"║  {taas_path.name}  —  {n_steps} steps @ dt={dt_ms}ms               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Load connectome ──
    conn = load_taas(taas_path)
    n = conn.n_neurons
    scale = 1e10

    # ── Drive current ──
    print("  Preparing drive ...", end=" ", flush=True)
    external_I = np.zeros(n, dtype=np.float32)
    n_drive = max(1, n // 5)
    nt = conn.nt_types()[:n_drive]
    external_I[:n_drive] = np.where(nt > 0, 10.0, 0.0)
    print(f"{n_drive} neurons driven")

    # ═══════════════════════════════════════
    #  1. Uniform HH (baseline)
    # ═══════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [1/2] Uniform HH  (NaV: all=120, KV: all=36)")
    print(f"{'─'*55}")
    sim1 = make_uniform_sim(n, dt_ms)
    r1 = simulate(sim1, conn, n_steps, dt_ms, external_I, scale, "Uniform")
    del sim1

    # ═══════════════════════════════════════
    #  2. Multi-type HH
    # ═══════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [2/2] Multi-type HH  (E: NaV=120,KV=36 | I: NaV=80,KV=45)")
    print(f"{'─'*55}")
    sim2 = make_multi_type_sim(conn, dt_ms)
    r2 = simulate(sim2, conn, n_steps, dt_ms, external_I, scale, "Multi-type")
    del sim2

    # ═══════════════════════════════════════
    #  Comparison
    # ═══════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Comparison: Uniform vs Multi-type HH")
    print(f"{'='*55}")
    print(f"  {'Metric':<30s} {'Uniform':>10s} {'Multi':>10s} {'Δ%':>8s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    for key, fmt, unit in [("total", ",d", ""), ("fr_mean", ".1f", " Hz"),
                            ("fr_e", ".1f", " Hz"), ("fr_i", ".1f", " Hz"),
                            ("e_active", ",d", ""), ("i_active", ",d", "")]:
        v1 = r1[key]
        v2 = r2[key]
        keyname = {"total":"Total spikes", "fr_mean":"Mean rate", "fr_e":"E rate",
                    "fr_i":"I rate", "e_active":"Active E", "i_active":"Active I"}.get(key, key)
        delta = ((v2 - v1) / max(abs(v1), 1)) * 100
        if fmt == ",d":
            print(f"  {keyname:<30s} {v1:>10,} {v2:>10,} {delta:>+7.1f}%")
        else:
            print(f"  {keyname:<30s} {v1:>10{fmt}}{unit} {v2:>10{fmt}}{unit} {delta:>+7.1f}%")

    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'E/I ratio':<30s} {r1['ei_ratio']:>10.2f} {r2['ei_ratio']:>10.2f}")
    print(f"  {'Time':<30s} {r1['elapsed']:>10.2f}s {r2['elapsed']:>10.2f}s")
    print(f"  {'Throughput':<30s} {r1['steps_s']:>10.0f} {r2['steps_s']:>10.0f} steps/s")
    print(f"{'='*55}")

    # Summary
    print()
    print("  Выводы:")
    if r2['ei_ratio'] < r1['ei_ratio']:
        print(f"    • Multi-type улучшил E/I баланс: {r1['ei_ratio']:.2f} → {r2['ei_ratio']:.2f}")
        print("      (I-нейроны стали активнее за счёт меньшего NaV)")
    elif r2['ei_ratio'] > r1['ei_ratio']:
        print(f"    • Multi-type ухудшил E/I баланс: {r1['ei_ratio']:.2f} → {r2['ei_ratio']:.2f}")
    else:
        print("    • E/I баланс не изменился")

    if r2['fr_i'] > r1['fr_i'] * 1.1:
        print(f"    • I rate выросла: {r1['fr_i']:.1f} → {r2['fr_i']:.1f} Hz "
              f"(больше KV даёт быструю реполяризацию)")
    elif r2['fr_i'] < r1['fr_i'] * 0.9:
        print(f"    • I rate упала: {r1['fr_i']:.1f} → {r2['fr_i']:.1f} Hz "
              f"(меньше NaV поднял порог — I недополучают возбуждения)")

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
