#!/usr/bin/env python3
"""
CaV-T pacemaking: спонтанная активность без внешнего тока.

Добавляет T-type calcium каналы (CaV, g_max=0.5) для JO/LC/pacemaking типов.
Эти нейроны должны спайкать спонтанно, даже без external_I.

Сравнивает:
  1. Uniform HH (2ch: NaV+KV) — стандарт
  2. Bio HH (2ch: NaV+KV, per-neuron C_m/g_max/g_leak)
  3. Bio+CaV HH (3ch: NaV+KV+CaT) — CaV=0.5 для JO/LC, 0 для остальных

Все тесты БЕЗ external_I — смотрим на спонтанную активность.
"""

import argparse
import gzip
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "examples" / "data"
PRINCETON_CSV = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets/connections_princeton.csv.gz")
CELL_TYPES = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets/consolidated_cell_types.csv.gz")

sys.path.insert(0, str(PROJECT))
import taas_checkmate as tcm


# ═══════════════════════════════════════════════════════════
#  Биокарта с CaV
# ═══════════════════════════════════════════════════════════
# Формат: (prefix, NaV, KV, CaV, C_m, g_leak, E_leak)
# CaV = 0 для обычных нейронов, 0.5 для pacemaking

BIO_MAP = [
    # Excitatory (E)
    ("R1-6",   100.0, 40.0, 0.0, 0.5, 0.8, -65.0),
    ("R7",     100.0, 40.0, 0.0, 0.5, 0.7, -65.0),
    ("R8",     100.0, 40.0, 0.0, 0.5, 0.7, -65.0),
    ("TmY",    130.0, 34.0, 0.0, 0.8, 0.3, -65.0),
    ("TmZ",    130.0, 34.0, 0.0, 0.8, 0.3, -65.0),
    ("Tm",     120.0, 36.0, 0.0, 1.0, 0.3, -65.0),
    ("T4",     130.0, 34.0, 0.0, 0.8, 0.3, -65.0),
    ("T5",     130.0, 34.0, 0.0, 0.8, 0.3, -65.0),
    ("T3",     120.0, 36.0, 0.0, 1.0, 0.3, -65.0),
    ("T2",     120.0, 36.0, 0.0, 1.0, 0.3, -65.0),
    ("T1",     110.0, 38.0, 0.0, 0.9, 0.3, -65.0),
    ("Mi",     120.0, 36.0, 0.0, 1.0, 0.3, -65.0),
    ("L5",     110.0, 38.0, 0.0, 0.9, 0.3, -65.0),
    ("L4",     110.0, 38.0, 0.0, 0.9, 0.3, -65.0),
    ("L3",     120.0, 36.0, 0.0, 0.8, 0.3, -65.0),
    ("L2",     120.0, 36.0, 0.0, 0.8, 0.3, -65.0),
    ("L1",     120.0, 36.0, 0.0, 0.8, 0.3, -65.0),
    ("KC",     100.0, 40.0, 0.0, 0.7, 0.4, -65.0),
    ("LC",     120.0, 36.0, 0.0, 1.0, 0.3, -65.0),
    # Inhibitory (I)
    ("C3",     120.0, 50.0, 0.0, 0.6, 0.5, -65.0),
    ("C2",     120.0, 50.0, 0.0, 0.6, 0.5, -65.0),
    ("Lawf",   120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    ("Lai",    120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    ("Dm",     120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    ("Sm",     120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    ("Y3",     120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    ("BM",     120.0, 48.0, 0.0, 0.6, 0.5, -65.0),
    # Pacemaking (with CaV!)
    ("JO",      90.0, 40.0, 0.5, 0.8, 0.4, -65.0),
]
BIO_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def load_princeton_mapping():
    """neuron_index → root_id из Princeton CSV."""
    t0 = time.time()
    root_to_idx = {}
    next_idx = 0
    with gzip.open(PRINCETON_CSV, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                pre = int(parts[0])
                post = int(parts[1])
            except ValueError:
                continue
            if pre not in root_to_idx:
                root_to_idx[pre] = next_idx
                next_idx += 1
            if post not in root_to_idx:
                root_to_idx[post] = next_idx
                next_idx += 1

    print(f"  {len(root_to_idx)} neurons in {time.time()-t0:.1f}s")
    return root_to_idx


def load_cell_types(root_to_idx):
    """root_id → cell_type_name."""
    n = len(root_to_idx)
    cell_types = [""] * n
    with gzip.open(CELL_TYPES, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",", 2)
            if len(parts) < 2:
                continue
            try:
                root_id = int(parts[0])
            except ValueError:
                continue
            if root_id in root_to_idx:
                cell_types[root_to_idx[root_id]] = parts[1]
    n_found = sum(1 for ct in cell_types if ct)
    print(f"  {n_found}/{n} neurons mapped")
    return cell_types


def load_taas(path: Path):
    """Load .taas → ConnectomeData."""
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        s = struct.unpack("<Q", f.read(8))[0]
        rp = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32).copy()
        ci = np.frombuffer(f.read(s * 4), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(s * 4), dtype=np.float32).copy()
        nt = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    print(f"  {path.name} loaded in {time.time()-t0:.1f}s — {n}N, {s}S")
    return tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32), vals, nt), nt


def build_params(cell_types, nt):
    """Построить per-neuron params с CaV=0.5 для pacemaking типов."""
    n = len(cell_types)
    n_channels = 3  # NaV, KV, CaT

    g_max = np.zeros(n_channels * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    pacemaking_count = 0
    matched = 0

    for i in range(n):
        ct = cell_types[i]
        params = None
        if ct:
            ct_u = ct.upper()
            for prefix, nav, kv, cav, cm, gl, el in BIO_MAP:
                if ct_u.startswith(prefix.upper()):
                    params = (nav, kv, cav, cm, gl, el)
                    if cav > 0:
                        pacemaking_count += 1
                    break

        if params is None:
            # Fallback по nt
            if nt[i] > 0:
                params = (120.0, 36.0, 0.0, 1.0, 0.3, -65.0)
            else:
                params = (120.0, 50.0, 0.0, 0.6, 0.5, -65.0)
        else:
            matched += 1

        nav, kv, cav, cm, gl, el = params
        g_max[0 * n + i] = nav
        g_max[1 * n + i] = kv
        g_max[2 * n + i] = cav  # CaV g_max
        C_m[i] = cm
        g_leak[i] = gl
        E_leak[i] = el

    print(f"  Matched: {matched}/{n}  Pacemaking (CaV>0): {pacemaking_count}")
    print(f"  Mean C_m: {C_m.mean():.3f}  g_leak: {g_leak.mean():.3f}")
    print(f"  Mean NaV: {g_max[:n].mean():.1f}  KV: {g_max[n:2*n].mean():.1f}  "
          f"CaV: {g_max[2*n:].mean():.4f} (nonzero: {(g_max[2*n:] > 0).sum()})")
    return {"g_max": g_max, "C_m": C_m, "g_leak": g_leak, "E_leak": E_leak}, pacemaking_count


def simulate(sim, conn, n_steps, dt_ms, label):
    """Запустить БЕЗ external_I — проверить спонтанную активность."""
    t0 = time.time()
    scale = 1e10
    spike_buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms, scale=scale)
    elapsed = time.time() - t0

    sb = spike_buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = sb.sum()
    duration = n_steps * dt_ms / 1000.0
    fr = sb.sum(axis=0) / duration
    active = (sb.sum(axis=0) > 0).sum()

    nt = conn.nt_types()
    fr_e = fr[nt > 0].mean() if (nt > 0).sum() > 0 else 0.0
    fr_i = fr[nt < 0].mean() if (nt < 0).sum() > 0 else 0.0
    e_active = (sb[:, nt > 0].sum(axis=0) > 0).sum() if (nt > 0).sum() > 0 else 0
    i_active = (sb[:, nt < 0].sum(axis=0) > 0).sum() if (nt < 0).sum() > 0 else 0

    return {"label": label, "total": total, "active": active,
            "fr_mean": fr.mean(), "fr_e": fr_e, "fr_i": fr_i,
            "e_active": e_active, "i_active": i_active, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="CaV-T pacemaking on Princeton")
    parser.add_argument("--taas", default="princeton.taas")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found")
        sys.exit(1)

    n_steps, dt_ms = args.steps, args.dt

    print("╔══════════════════════════════════════════════════════╗")
    print("║  CaV-T Pacemaking HH                                ║")
    print("║  3-channel (NaV+KV+CaT) — спонтанная активность     ║")
    print(f"║  {taas_path.name}  —  {n_steps} steps @ dt={dt_ms}ms               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    conn, nt = load_taas(taas_path)
    n = conn.n_neurons

    # ── Определяем: Princeton (cell type mapping) или FAFB (nt-only) ──
    is_princeton = "princeton" in args.taas.lower()

    if is_princeton:
        print("\n  Rebuilding cell type mapping from Princeton + FlyWire ...")
        root_to_idx = load_princeton_mapping()
        cell_types = load_cell_types(root_to_idx)

        print("\n  ── Cell types with CaV pacemaking ──")
        for ct, cnt in Counter(cell_types).most_common(10):
            if ct:
                has_cav = any(ct.upper().startswith(p.upper()) and cav > 0
                              for p, _, _, cav, _, _, _ in BIO_MAP)
                marker = " 🔥CaV" if has_cav else ""
                print(f"    {ct:30s} {cnt:>6,}{marker}")

        print("\n  Building per-neuron 3-channel params from cell types ...")
        bio, pc_count = build_params(cell_types, nt)
    else:
        print("\n  FAFB detected — using nt_type-based defaults + CaV on 5% neurons")
        pc_count = int(n * 0.05)
        print(f"  Assigning CaV=0.5 to {pc_count:,} random neurons (5%)")
        bio = {
            "g_max": np.zeros(3 * n, dtype=np.float32),
            "C_m": np.ones(n, dtype=np.float32),
            "g_leak": np.ones(n, dtype=np.float32) * 0.3,
            "E_leak": np.ones(n, dtype=np.float32) * -65.0,
        }
        for i in range(n):
            if nt[i] > 0:
                bio["g_max"][0 * n + i] = 120.0
                bio["g_max"][1 * n + i] = 36.0
                bio["C_m"][i] = 1.0
                bio["g_leak"][i] = 0.3
            else:
                bio["g_max"][0 * n + i] = 120.0
                bio["g_max"][1 * n + i] = 50.0
                bio["C_m"][i] = 0.6
                bio["g_leak"][i] = 0.5
        # Mark 5% of neurons as pacemaking
        rng = np.random.default_rng(42)
        cav_indices = rng.choice(n, size=pc_count, replace=False)
        bio["g_max"][2 * n + cav_indices] = 0.5

        print(f"  Mean C_m: {bio['C_m'].mean():.3f}  g_leak: {bio['g_leak'].mean():.3f}")
        print(f"  CaV nonzero: {pc_count:,}  ({pc_count/n*100:.1f}%)")

    print()

    def t(desc):
        print(f"{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")

    results = []

    # ═══════════════════════════════════════════
    #  1. Uniform HH (2ch)
    # ═══════════════════════════════════════════
    t("[1/3] Uniform 2ch HH (NaV=120,KV=36)")
    sim = tcm.BrainSimulator(n, "hh")
    sim.set_hh_params(
        np.array([120.0, 36.0], dtype=np.float32),
        np.array([50.0, -77.0], dtype=np.float32),
        np.array([0.3] * n, dtype=np.float32),
        np.array([-65.0] * n, dtype=np.float32),
        dt_ms=dt_ms,
    )
    r = simulate(sim, conn, n_steps, dt_ms, "Uniform")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  Active: {r['active']:,}/{n}  "
          f"E={r['fr_e']:.2f}Hz I={r['fr_i']:.2f}Hz")
    del sim

    # ═══════════════════════════════════════════
    #  2. Bio HH (2ch, per-neuron)
    # ═══════════════════════════════════════════
    t("[2/3] Bio 2ch HH (per-neuron C_m/g_max/g_leak, NO CaV)")
    sim = tcm.BrainSimulator(n, "hh")
    # Use only first 2 channels of bio params for 2ch sim
    bio2 = {
        "g_max": bio["g_max"][:2*n].copy(),
        "C_m": bio["C_m"],
        "g_leak": bio["g_leak"],
        "E_leak": bio["E_leak"],
    }
    sim.set_hh_params(
        bio2["g_max"],
        np.array([50.0, -77.0], dtype=np.float32),
        bio2["g_leak"],
        bio2["E_leak"],
        dt_ms=dt_ms,
    )
    sim.set_C_m(bio2["C_m"])
    r = simulate(sim, conn, n_steps, dt_ms, "Bio 2ch")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  Active: {r['active']:,}/{n}  "
          f"E={r['fr_e']:.2f}Hz I={r['fr_i']:.2f}Hz")
    del sim

    # ═══════════════════════════════════════════
    #  3. Bio+CaV HH (3ch, with CaT channels)
    # ═══════════════════════════════════════════
    t(f"[3/3] Bio+CaV 3ch HH (CaV=0.5 for {pc_count} pacemaking neurons)")
    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    sim.set_hh_params(
        bio["g_max"],         # [3*n] — NaV, KV, CaT
        np.array([50.0, -77.0, 120.0], dtype=np.float32),  # Na, K, Ca reversal
        bio["g_leak"],
        bio["E_leak"],
        dt_ms=dt_ms,
    )
    sim.set_C_m(bio["C_m"])
    r = simulate(sim, conn, n_steps, dt_ms, "Bio+CaV 3ch")
    results.append(r)
    print(f"    Spikes: {r['total']:,}  Active: {r['active']:,}/{n}  "
          f"E={r['fr_e']:.2f}Hz I={r['fr_i']:.2f}Hz")
    del sim

    # ═══════════════════════════════════════════
    #  Comparison
    # ═══════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Comparison: Спонтанная активность БЕЗ external_I")
    print(f"{'='*55}")
    print(f"  {'Metric':<25s} {'Uniform':>10s} {'Bio 2ch':>10s} {'Bio+CaV':>10s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")

    for key, name in [("total", "Spikes"), ("active", "Active N"),
                       ("fr_mean", "Mean Hz"), ("fr_e", "E Hz"),
                       ("fr_i", "I Hz")]:
        vals = [r[key] for r in results]
        if key in ("total", "active"):
            print(f"  {name:<25s} {vals[0]:>10,} {vals[1]:>10,} {vals[2]:>10,}")
        else:
            print(f"  {name:<25s} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}")

    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Time':<25s} {results[0]['elapsed']:>10.2f}s "
          f"{results[1]['elapsed']:>10.2f}s {results[2]['elapsed']:>10.2f}s")
    print(f"{'='*55}")

    # Summary
    print("\n  Выводы:")
    if results[2]["total"] > results[0]["total"]:
        print(f"  ✅ CaV-T добавил спонтанной активности: "
              f"{results[2]['total'] - results[0]['total']:,} доп. спайков")
    if results[2]["active"] > results[0]["active"]:
        print(f"  ✅ Больше активных нейронов: "
              f"{results[2]['active'] - results[0]['active']:,} доп.")
    if results[2]["total"] == 0 and results[0]["total"] == 0:
        print(f"  ❌ Никто не спайкает — CaV-T не хватает или "
              f"JO/LC нейроны не нашлись в Princeton")

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
