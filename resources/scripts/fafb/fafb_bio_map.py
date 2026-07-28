#!/usr/bin/env python3
"""
Bio-realistic multi-type HH: per-neuron g_max, C_m, g_leak из FlyWire cell types.

Пайплайн:
  1. Загружает connections_princeton.csv.gz → neuron_index → root_id
  2. Загружает consolidated_cell_types.csv.gz → root_id → cell_type
  3. Маппит cell_type → биофизические параметры (g_max, C_m, g_leak, E_leak)
  4. Запускает multi-type HH с per-neuron параметрами, сравнивает с uniform

Биоклассы (Drosophila melanogaster):
  - Cholinergic (ACh): NaV=120, KV=36, Cm=1.0, gL=0.3 — стандарт
  - GABAergic (PV-like): NaV=120, KV=50, Cm=0.6, gL=0.5 — быстрая реполяризация
  - Glutamatergic: NaV=140, KV=30, Cm=1.0, gL=0.2 — высокая возбудимость
  - Sensory/фоторецепторы: NaV=100, KV=40, Cm=0.5, gL=0.8 — быстрые, шумные
  - Kenyon cells: NaV=100, KV=40, Cm=0.7, gL=0.4 — MB нейроны
  - Neuromodulatory (JO, etc): CaV 0.5 + NaV=90, KV=40, Cm=0.8, gL=0.4

Usage:
  python scripts/fafb_bio_map.py --taas princeton.taas --steps 100 --dt 0.5
  python scripts/fafb_bio_map.py --taas fafb.taas --steps 100 --dt 0.5
"""

import argparse
import gzip
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ── Paths ───────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "examples" / "data"
PRINCETON_CSV = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets/connections_princeton.csv.gz")
CELL_TYPES = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets/consolidated_cell_types.csv.gz")

sys.path.insert(0, str(PROJECT))
import taas_checkmate as tcm


# ═══════════════════════════════════════════════════════════
#  Биокарта
# ═══════════════════════════════════════════════════════════
#
# Формат: (prefix, NaV, KV, C_m, g_leak, E_leak, has_CaT)
# Матчинг: longest-prefix-first (сортировка по длине префикса убыванию)
#
# Ключевое изменение vs первой версии:
# - GABAergic (I) получают NaV=120 (не 80!) — порог не поднимаем,
#   но KV=50 для быстрой реполяризации
# - Добавлены TmY, T1, разделены Mi/Tm/TmY/TmZ
# - C_m реально применяется через set_C_m()
BIO_MAP = [
    # Cholinergic / Excitatory (E)
    ("R1-6",   100.0, 40.0, 0.5, 0.8, -65.0, False),   # фоторецепторы: быстрые, шумные
    ("R7",     100.0, 40.0, 0.5, 0.7, -65.0, False),
    ("R8",     100.0, 40.0, 0.5, 0.7, -65.0, False),
    ("TmY",    130.0, 34.0, 0.8, 0.3, -65.0, False),    # TmY — motion-selective (до Tm!)
    ("TmZ",    130.0, 34.0, 0.8, 0.3, -65.0, False),
    ("Tm",     120.0, 36.0, 1.0, 0.3, -65.0, False),    # transmedullary
    ("T4",     130.0, 34.0, 0.8, 0.3, -65.0, False),    # motion detection
    ("T5",     130.0, 34.0, 0.8, 0.3, -65.0, False),
    ("T3",     120.0, 36.0, 1.0, 0.3, -65.0, False),    # T-cells
    ("T2",     120.0, 36.0, 1.0, 0.3, -65.0, False),
    ("T1",     110.0, 38.0, 0.9, 0.3, -65.0, False),    # T1 (из L5)
    ("Mi",     120.0, 36.0, 1.0, 0.3, -65.0, False),    # medulla intrinsic
    ("L5",     110.0, 38.0, 0.9, 0.3, -65.0, False),    # lamina
    ("L4",     110.0, 38.0, 0.9, 0.3, -65.0, False),
    ("L3",     120.0, 36.0, 0.8, 0.3, -65.0, False),
    ("L2",     120.0, 36.0, 0.8, 0.3, -65.0, False),
    ("L1",     120.0, 36.0, 0.8, 0.3, -65.0, False),
    ("KC",     100.0, 40.0, 0.7, 0.4, -65.0, False),    # Kenyon cells
    ("LC",     120.0, 36.0, 1.0, 0.3, -65.0, False),    # lateral complex
    # GABAergic / Inhibitory (I)
    ("C3",     120.0, 50.0, 0.6, 0.5, -65.0, False),    # centrifugal — NaV=120, KV=50
    ("C2",     120.0, 50.0, 0.6, 0.5, -65.0, False),
    ("Lawf",   120.0, 48.0, 0.6, 0.5, -65.0, False),    # local — fast-spiking PV-like
    ("Lai",    120.0, 48.0, 0.6, 0.5, -65.0, False),    # lamina amacrine
    ("Dm",     120.0, 48.0, 0.6, 0.5, -65.0, False),    # distal medulla (GABAergic)
    ("Sm",     120.0, 48.0, 0.6, 0.5, -65.0, False),    # serotonergic
    ("Y3",     120.0, 48.0, 0.6, 0.5, -65.0, False),    # Yahoo (GABAergic)
    ("BM",     120.0, 48.0, 0.6, 0.5, -65.0, False),    # bilateral medial (GABA)
    # Neuromodulatory / pacemaking
    ("JO",      90.0, 40.0, 0.8, 0.4, -65.0, True),     # Johnston's organ (CaV!)
]

# Sort by prefix length descending (longest first) for correct matching
BIO_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def load_princeton_mapping():
    """
    Восстанавливает neuron_index → root_id из Princeton CSV.
    Тот же алгоритм что в bake_princeton(): порядок первого появления.
    """
    print("  Rebuilding neuron index → root_id from Princeton CSV ...", end=" ", flush=True)
    t0 = time.time()
    root_to_idx = {}
    next_idx = 0
    with gzip.open(PRINCETON_CSV, "rt") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
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

    n_neurons = len(root_to_idx)
    print(f"{n_neurons} neurons in {time.time()-t0:.1f}s")
    return root_to_idx


def load_cell_types(root_to_idx):
    """
    Загружает consolidated_cell_types.csv.gz, маппит root_id → cell_type_name.
    Возвращает: [n_neurons] cell_type_name (или пустая строка если нет)
    """
    n = len(root_to_idx)
    cell_types = [""] * n

    print("  Loading consolidated_cell_types ...", end=" ", flush=True)
    t0 = time.time()
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
                idx = root_to_idx[root_id]
                cell_types[idx] = parts[1]

    n_found = sum(1 for ct in cell_types if ct)
    print(f"{n_found}/{n} neurons mapped in {time.time()-t0:.1f}s")
    return cell_types


def map_biophysics(cell_types, nt_types):
    """
    Маппит cell_type_name → биофизические параметры per neuron.

    Longest-prefix matching: BIO_MAP отсортирован по убыванию длины префикса,
    поэтому "TmY" будет проверен перед "Tm".

    nt_types: исходные +1/-1 из .taas — fallback для E/I дискриминации
    """
    n = len(cell_types)
    g_max = np.zeros(2 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    # Default per neurotransmitter type (fallback если cell type не найден)
    # nt_types: +1 = ACh (E), -1 = GABA (I)
    def default_params(nt_val):
        if nt_val > 0:   # Excitatory
            return (120.0, 36.0, 1.0, 0.3, -65.0)
        else:             # Inhibitory
            return (120.0, 50.0, 0.6, 0.5, -65.0)

    matched = 0
    for i in range(n):
        ct = cell_types[i]

        # Try longest-prefix match
        params = None
        if ct:
            ct_upper = ct.upper()
            for prefix, nav, kv, cm, gl, el, _ in BIO_MAP:
                if ct_upper.startswith(prefix.upper()):
                    params = (nav, kv, cm, gl, el)
                    break

        # Fallback to nt_type-based defaults
        if params is None:
            params = default_params(nt_types[i] if i < len(nt_types) else 1.0)
        else:
            matched += 1

        nav, kv, cm, gl, el = params
        g_max[0 * n + i] = nav
        g_max[1 * n + i] = kv
        C_m[i] = cm
        g_leak[i] = gl
        E_leak[i] = el

    print(f"    Longest-prefix matched: {matched}/{n} neurons")
    if matched < n:
        print(f"    Fallback to nt_type defaults: {n - matched} neurons")

    return {"g_max": g_max, "C_m": C_m, "g_leak": g_leak, "E_leak": E_leak}


# ═══════════════════════════════════════════════════════════
#  Sim helpers
# ═══════════════════════════════════════════════════════════

def load_taas(path: Path) -> tcm.ConnectomeData:
    """Load .taas file."""
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
    return tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)


def make_sim(n, dt_ms, bio=None):
    """
    Создаёт BrainSimulator(model='hh') с per-neuron параметрами.

    Если bio не None: применяет g_max per-neuron + C_m per-neuron + g_leak/E_leak.
    Если bio None: uniform NaV=120, KV=36, Cm=1.0, gL=0.3.
    """
    sim = tcm.BrainSimulator(n, "hh")

    if bio is not None:
        # Per-neuron: g_max [2*n], g_leak [n], E_leak [n]
        sim.set_hh_params(
            bio["g_max"],
            np.array([50.0, -77.0], dtype=np.float32),
            bio["g_leak"],
            bio["E_leak"],
            dt_ms=dt_ms,
        )
        # Per-neuron C_m — новый API!
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


def simulate(sim, conn, n_steps, dt_ms, external_I, scale, label):
    """Run simulation + return stats."""
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
    }


def main():
    parser = argparse.ArgumentParser(description="Bio-realistic multi-type HH")
    parser.add_argument("--taas", default="princeton.taas",
                        help=".taas file (default: princeton.taas)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--skip-mapping", action="store_true",
                        help="Skip cell type mapping (use nt_type defaults)")
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found"); sys.exit(1)

    n_steps, dt_ms = args.steps, args.dt

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Bio-realistic Multi-Type HH                       ║")
    print("║  FlyWire cell types → per-neuron parameters        ║")
    print(f"║  {taas_path.name}  —  {n_steps} steps @ dt={dt_ms}ms               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Load connectome ──
    conn = load_taas(taas_path)
    n = conn.n_neurons
    scale = 1e10
    nt = conn.nt_types()

    # ── Reconstruct bio-map (only for Princeton — has root_id data) ──
    if args.skip_mapping or "princeton" not in taas_path.name:
        # fafb.taas doesn't have root_id mapping → use nt_type-based defaults
        print("\n  No cell type mapping available — using nt_type-based defaults")
        print("  E: NaV=120, KV=36, Cm=1.0, gL=0.3")
        print("  I: NaV=120, KV=50, Cm=0.6, gL=0.5")
        bio = {
            "g_max": np.zeros(2 * n, dtype=np.float32),
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
    else:
        root_to_idx = load_princeton_mapping()
        cell_types = load_cell_types(root_to_idx)
        bio = map_biophysics(cell_types, nt)

        # Show cell type distribution
        print("\n  ── Cell type distribution (top 20) ──")
        ct_counter = Counter(cell_types)
        for ct, cnt in ct_counter.most_common(20):
            if ct:
                prefix = ""
                for p, _, _, _, _, _, _ in BIO_MAP:
                    if ct.upper().startswith(p.upper()):
                        prefix = f" → {p}"
                        break
                print(f"    {ct:30s} {cnt:>6,}{prefix}")
        other = sum(cnt for ct, cnt in ct_counter.items() if not ct)
        if other:
            print(f"    {'(unmapped → nt_type fallback)':30s} {other:>6,}")

    # Stats
    print(f"\n  Bio-params stats:")
    print(f"    Mean C_m:    {bio['C_m'].mean():.3f} μF/cm²  "
          f"(E={bio['C_m'][nt.cpu().numpy() > 0 if hasattr(nt, 'cpu') else (nt > 0)].mean():.3f}, "
          f"I={bio['C_m'][nt.cpu().numpy() < 0 if hasattr(nt, 'cpu') else (nt < 0)].mean():.3f})")
    print(f"    Mean g_leak: {bio['g_leak'].mean():.3f} mS/cm²")
    print(f"    Mean NaV:    {bio['g_max'][0*n:1*n].mean():.1f} mS/cm²")
    print(f"    Mean KV:     {bio['g_max'][1*n:2*n].mean():.1f} mS/cm²")

    # ── Drive ──
    print("\n  Preparing drive ...", end=" ", flush=True)
    external_I = np.zeros(n, dtype=np.float32)
    n_drive = max(1, n // 5)
    nt_drive = nt[:n_drive]
    exc_mask = nt_drive > 0 if hasattr(nt_drive, 'gt') else np.array([v > 0 for v in nt_drive])
    external_I[:n_drive] = np.where(exc_mask, 10.0, 0.0)
    print(f"{n_drive} neurons driven")

    # ════════════════════════════════════════════════════
    #  Test 1: Uniform HH (baseline)
    # ════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [1/2] Uniform HH (NaV=120, KV=36, Cm=1.0, gL=0.3)")
    print(f"{'─'*55}")
    sim1 = make_sim(n, dt_ms)
    r1 = simulate(sim1, conn, n_steps, dt_ms, external_I, scale, "Uniform")
    del sim1

    # ════════════════════════════════════════════════════
    #  Test 2: Bio-realistic HH
    # ════════════════════════════════════════════════════
    print(f"\n{'─'*55}")
    print("  [2/2] Bio-realistic HH (per-neuron g_max + C_m + g_leak)")
    print(f"{'─'*55}")
    sim2 = make_sim(n, dt_ms, bio)
    r2 = simulate(sim2, conn, n_steps, dt_ms, external_I, scale, "Bio")
    del sim2

    # ════════════════════════════════════════════════════
    #  Comparison
    # ════════════════════════════════════════════════════
    print(f"\n{'='*55}")
    print(f"  Comparison: Uniform vs Bio-realistic HH")
    print(f"{'='*55}")
    print(f"  {'Metric':<30s} {'Uniform':>10s} {'Bio':>10s} {'Δ%':>8s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")

    for key, fmt in [("total", ",d"), ("fr_mean", ".1f"), ("fr_e", ".1f"),
                      ("fr_i", ".1f"), ("e_active", ",d"), ("i_active", ",d")]:
        v1 = r1[key]
        v2 = r2[key]
        keyname = {"total":"Total spikes", "fr_mean":"Mean rate", "fr_e":"E rate",
                    "fr_i":"I rate", "e_active":"Active E", "i_active":"Active I"}.get(key, key)
        delta = ((v2 - v1) / max(abs(v1), 1)) * 100
        if fmt == ",d":
            print(f"  {keyname:<30s} {v1:>10,} {v2:>10,} {delta:>+7.1f}%")
        else:
            print(f"  {keyname:<30s} {v1:>10{fmt}} Hz {v2:>10{fmt}} Hz {delta:>+7.1f}%")

    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'E/I ratio':<30s} {r1['ei_ratio']:>10.2f} {r2['ei_ratio']:>10.2f}")
    print(f"  {'Time':<30s} {r1['elapsed']:>10.2f}s {r2['elapsed']:>10.2f}s")
    print(f"{'='*55}")

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
