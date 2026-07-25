#!/usr/bin/env python3
"""
Bio + Synaptic HH: per-neuron C_m/g_max/g_leak + conductance-based synapses.

Сравнивает 3 режима на fafb.taas:
  1. Uniform instant-current HH (baseline)
  2. Bio-realistic instant-current HH (per-neuron C_m, g_max, g_leak)
  3. Bio-realistic conductance-based HH (per-neuron params + AMPA τ=2ms, GABA_A τ=10ms)

Usage:
  python scripts/fafb_bio_synaptic.py --steps 100 --dt 0.5
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
    """Load .taas + build receptor_types for synaptic mode."""
    print(f"  Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...", end=" ", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"
        _ = struct.unpack("<I", f.read(4))[0]
        n_neurons = struct.unpack("<Q", f.read(8))[0]
        n_synapses = struct.unpack("<Q", f.read(8))[0]
        row_ptr = np.frombuffer(f.read((n_neurons + 1) * 4), dtype=np.int32).copy()
        col_indices = np.frombuffer(f.read(n_synapses * 4), dtype=np.int32).copy()
        values = np.frombuffer(f.read(n_synapses * 4), dtype=np.float32).copy()
        nt_types = np.frombuffer(f.read(n_neurons * 4), dtype=np.float32).copy()

    print(f"{time.time()-t0:.1f}s — {n_neurons}N, {n_synapses}S")
    conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)

    # Build receptor_types for synaptic mode
    print("  Building receptor_types (AMPA/GABA_A) ...", end=" ", flush=True)
    t0 = time.time()
    conn.build_receptor_types()
    print(f"{time.time()-t0:.2f}s")
    return conn


def build_bio_params(n, nt):
    """Построить per-neuron параметры из nt_types (E/I)."""
    g_max = np.zeros(2 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    for i in range(n):
        if nt[i] > 0:  # Excitatory (ACh)
            g_max[0 * n + i] = 120.0  # NaV
            g_max[1 * n + i] = 36.0   # KV
            C_m[i] = 1.0
            g_leak[i] = 0.3
        else:           # Inhibitory (GABA)
            g_max[0 * n + i] = 120.0  # NaV — сохраняем возбудимость
            g_max[1 * n + i] = 50.0   # KV ↑ быстрее реполяризация
            C_m[i] = 0.6              # ↓ быстрее динамика
            g_leak[i] = 0.5           # ↑ стабильнее покой

    return {"g_max": g_max, "C_m": C_m, "g_leak": g_leak, "E_leak": E_leak}


def make_sim(n, dt_ms, bio=None):
    """Создать BrainSimulator с uniform или bio-параметрами."""
    sim = tcm.BrainSimulator(n, "hh")
    if bio is not None:
        sim.set_hh_params(
            bio["g_max"],
            np.array([50.0, -77.0], dtype=np.float32),
            bio["g_leak"],
            bio["E_leak"],
            dt_ms=dt_ms,
        )
        sim.set_C_m(bio["C_m"])
    else:
        sim.set_hh_params(
            np.array([120.0, 36.0], dtype=np.float32),
            np.array([50.0, -77.0], dtype=np.float32),
            np.array([0.3] * n, dtype=np.float32),
            np.array([-65.0] * n, dtype=np.float32),
            dt_ms=dt_ms,
        )
    return sim


def simulate(sim, conn, n_steps, dt_ms, external_I, scale, label, method):
    """Запустить симуляцию и вернуть статистику."""
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
    parser = argparse.ArgumentParser(description="Bio + Synaptic HH on FAFB")
    parser.add_argument("--taas", default="fafb.taas")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found"); sys.exit(1)

    n_steps, dt_ms = args.steps, args.dt

    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Bio + Synaptic HH                                   ║")
    print("║  per-neuron C_m/g_max/g_leak + AMPA τ=2ms/GABA τ=10ms║")
    print(f"║  {taas_path.name}  —  {n_steps} steps @ dt={dt_ms}ms                ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    # ── Load connectome ──
    conn = load_taas(taas_path)
    n = conn.n_neurons
    scale = 1e10
    nt = conn.nt_types()

    # ── Bio params ──
    bio = build_bio_params(n, nt)
    print(f"\n  Bio-params:")
    print(f"    Mean C_m:    {bio['C_m'].mean():.3f} μF/cm² "
          f"(E={bio['C_m'][nt>0].mean():.3f}, I={bio['C_m'][nt<0].mean():.3f})")
    print(f"    Mean g_leak: {bio['g_leak'].mean():.3f} mS/cm²")
    print(f"    Mean NaV:    {bio['g_max'][:n].mean():.1f} mS/cm²")
    print(f"    Mean KV:     {bio['g_max'][n:].mean():.1f} mS/cm²")

    # ── Drive ──
    print("\n  Preparing drive ...", end=" ", flush=True)
    external_I = np.zeros(n, dtype=np.float32)
    n_drive = max(1, n // 5)
    nt_drive = nt[:n_drive]
    external_I[:n_drive] = np.where(np.array([v > 0 for v in nt_drive]), 10.0, 0.0)
    print(f"{n_drive} neurons driven")

    def t(desc):
        print(f"\n{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")

    results = []

    # ════════════════════════════════════════════
    #  1. Uniform instant HH
    # ════════════════════════════════════════════
    t("[1/3] Uniform instant HH (NaV=120, KV=36, Cm=1.0, gL=0.3)")
    sim = make_sim(n, dt_ms)
    r = simulate(sim, conn, n_steps, dt_ms, external_I, scale, "Uniform", "instant")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  E={r['fr_e']:.1f} Hz  I={r['fr_i']:.1f} Hz  "
          f"E/I={r['ei_ratio']:.2f}  {r['elapsed']:.2f}s")
    del sim

    # ════════════════════════════════════════════
    #  2. Bio instant HH
    # ════════════════════════════════════════════
    t("[2/3] Bio instant HH (E: 120/36/1.0/0.3  I: 120/50/0.6/0.5)")
    sim = make_sim(n, dt_ms, bio)
    r = simulate(sim, conn, n_steps, dt_ms, external_I, scale, "Bio", "instant")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  E={r['fr_e']:.1f} Hz  I={r['fr_i']:.1f} Hz  "
          f"E/I={r['ei_ratio']:.2f}  {r['elapsed']:.2f}s")
    del sim

    # ════════════════════════════════════════════
    #  3. Bio synaptic HH
    # ════════════════════════════════════════════
    t("[3/3] Bio synaptic HH (same params + AMPA τ=2ms, GABA_A τ=10ms)")
    sim = make_sim(n, dt_ms, bio)
    r = simulate(sim, conn, n_steps, dt_ms, external_I, scale, "Bio+Syn", "synaptic")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  E={r['fr_e']:.1f} Hz  I={r['fr_i']:.1f} Hz  "
          f"E/I={r['ei_ratio']:.2f}  {r['elapsed']:.2f}s")
    del sim

    # ════════════════════════════════════════════
    #  Comparison table
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Comparison: 3 режима на {taas_path.name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<28s} {'Uniform':>10s} {'Bio':>10s} {'Bio+Syn':>10s}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")

    metrics = [
        ("total", ",d", "Total spikes"),
        ("fr_e", ".1f", "E rate (Hz)"),
        ("fr_i", ".1f", "I rate (Hz)"),
        ("fr_mean", ".1f", "Mean rate (Hz)"),
        ("e_active", ",d", "Active E"),
        ("i_active", ",d", "Active I"),
    ]
    for key, fmt, name in metrics:
        vals = [r[key] for r in results]
        if fmt == ",d":
            print(f"  {name:<28s} {vals[0]:>10,} {vals[1]:>10,} {vals[2]:>10,}")
        else:
            print(f"  {name:<28s} {vals[0]:>10{fmt}} {vals[1]:>10{fmt}} {vals[2]:>10{fmt}}")

    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
    ei_vals = [r["ei_ratio"] for r in results]
    print(f"  {'E/I ratio':<28s} {ei_vals[0]:>10.2f} {ei_vals[1]:>10.2f} {ei_vals[2]:>10.2f}")
    time_vals = [r["elapsed"] for r in results]
    print(f"  {'Time':<28s} {time_vals[0]:>10.2f}s {time_vals[1]:>10.2f}s {time_vals[2]:>10.2f}s")
    step_vals = [r["steps_s"] for r in results]
    print(f"  {'Throughput':<28s} {step_vals[0]:>10.0f} {step_vals[1]:>10.0f} {step_vals[2]:>10.0f} steps/s")
    print(f"{'='*60}")

    # Summary
    print("\n  Выводы:")
    e0, e1, e2 = ei_vals
    if e2 < e0 and e2 < e1:
        print(f"    ✅ Bio+Synaptic даёт лучший E/I баланс: {e0:.2f} → {e2:.2f}")
    elif e1 < e0:
        print(f"    ✅ Bio параметры улучшили E/I: {e0:.2f} → {e1:.2f}")
    if e2 < e1:
        print(f"    ✅ Синаптическая физика дополнительно улучшила баланс: {e1:.2f} → {e2:.2f}")
    elif e2 > e1:
        print(f"    ❌ Синаптическая физика ухудшила баланс: {e1:.2f} → {e2:.2f}")

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
