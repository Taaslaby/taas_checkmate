#!/usr/bin/env python3
"""🚀 IIT на реальном FlyWire коннектоме — Phi через Rust крейт

Загружает princeton.taas или fafb.taas, применяет оптимизации сознания
(weight scale, cross-half bridges, g_Ca tuning, Pearson integration).

Запуск:
  python3 scripts/iit_flywire.py --taas princeton.taas --neurons 100 --steps 300
  python3 scripts/iit_flywire.py --taas fafb.taas --subsample 500 --steps 200
"""

import sys, os, time, math, argparse, struct
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "examples" / "data"
sys.path.insert(0, str(BASE))

import taas_checkmate as tcm

np.random.seed(42)

DT_MS = 0.05

# ═══════════════════════════════════════════════════════════
#  Загрузка FlyWire коннектома
# ═══════════════════════════════════════════════════════════

def load_flywire(taas_path: Path, subsample: int = 0):
    """Загрузить .taas файл, опционально взять subsample первых нейронов.

    Returns: (row_ptr, col_indices, values, nt_types, n_neurons)
    """
    print(f"  📦 Loading {taas_path.name} ({taas_path.stat().st_size / 1e6:.0f} MB) ...",
          end=" ", flush=True)
    t0 = time.time()

    with open(taas_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"
        _ = struct.unpack("<I", f.read(4))[0]
        n_total = struct.unpack("<Q", f.read(8))[0]
        n_syn = struct.unpack("<Q", f.read(8))[0]
        row_ptr = np.frombuffer(f.read((n_total + 1) * 4), dtype=np.int32).copy()
        col_indices = np.frombuffer(f.read(n_syn * 4), dtype=np.int32).copy()
        values = np.frombuffer(f.read(n_syn * 4), dtype=np.float32).copy()
        nt_types = np.frombuffer(f.read(n_total * 4), dtype=np.float32).copy()

    load_t = time.time() - t0
    n = subsample if subsample > 0 else n_total

    print(f"{load_t:.1f}s — {n_total}N, {n_syn}S", end="")

    if subsample > 0:
        n = min(subsample, n_total)
        # Rebuild CSR: only keep synapses where post < n
        rp_new = np.zeros(n + 1, dtype=np.int32)
        ci_new, vals_new = [], []
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            for j in range(start, end):
                post = col_indices[j]
                if post < n:
                    ci_new.append(post)
                    vals_new.append(values[j])
            rp_new[i + 1] = len(ci_new)
        row_ptr = rp_new
        col_indices = np.array(ci_new, dtype=np.int32)
        values = np.array(vals_new, dtype=np.float32)
        nt_types = nt_types[:n].copy()
        n_syn = len(values)
        print(f" → subsample {n}N, {n_syn}S", end="")

    print(f"  |  E: {(nt_types > 0).mean()*100:.0f}%  I: {(nt_types < 0).mean()*100:.0f}%")
    return row_ptr, col_indices, values, nt_types, n


# ═══════════════════════════════════════════════════════════
#  Оптимизации сознания
# ═══════════════════════════════════════════════════════════

def apply_consciousness_optimizations(row_ptr, col_indices, values, nt_types, n,
                                      weight_scale=3.0, cross_half_frac=0.2):
    """Применить weight scale + cross-half bridges к коннектому.

    Возвращает новый ConnectomeData с модифицированными весами и связями.
    """
    half = n // 2

    # 1. Scale weights
    scaled_vals = values.astype(np.float64) * weight_scale

    # 2. Cross-half bridges (20% нейронов)
    cross_n = max(1, int(n * cross_half_frac))
    cross_neurons = np.random.choice(n, cross_n, replace=False)

    added_per_neuron = [[] for _ in range(n + 1)]
    added_per_neuron[0] = []  # dummy for 0-index convenience

    for i in cross_neurons:
        if i < half:
            target = half + i % half
        else:
            target = i % half
        if target < n and target != i:
            # Strong excitatory bridge
            added_per_neuron[i].append((int(target), 4.0 * weight_scale, 1.0))

    # Rebuild CSR with added synapses
    new_rp = np.zeros(n + 1, dtype=np.int32)
    new_ci_list, new_val_list = [], []

    for i in range(n):
        start = row_ptr[i]
        end = row_ptr[i + 1]

        # Existing scaled synapses
        for j in range(start, end):
            new_ci_list.append(int(col_indices[j]))
            new_val_list.append(float(scaled_vals[j]))

        # Added cross-half bridges
        for target, w, _ in added_per_neuron[i]:
            new_ci_list.append(target)
            new_val_list.append(w)

        new_rp[i + 1] = len(new_ci_list)

    new_vals_arr = np.array(new_val_list, dtype=np.float32)
    new_ci_arr = np.array(new_ci_list, dtype=np.int32)

    # Stats
    print(f"  🔧 Optimizations: scale=×{weight_scale}, cross-half={cross_n} neurons")
    print(f"     Original: {len(values)} syn | Optimized: {len(new_vals_arr)} syn "
          f"(+{len(new_vals_arr) - len(values)} cross-half)")
    orig_mean = np.abs(values).mean()
    new_mean = np.abs(new_vals_arr).mean()
    print(f"     Mean |weight|: {orig_mean:.2f} → {new_mean:.2f} "
          f"(C ≈ {min(new_mean / 10.0, 1.0):.3f})")

    return tcm.ConnectomeData(new_rp, new_ci_arr, new_vals_arr, nt_types)


# ═══════════════════════════════════════════════════════════
#  HH параметры — с CaV pacemaking
# ═══════════════════════════════════════════════════════════

def build_hh_params(n: int):
    """HH+CaV params: NaV 80-160, KV 24-60, CaV exp(2.5)+2 clipped 2-8."""
    rng = np.random.RandomState(42)
    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    g_K  = rng.uniform(24, 60, n).astype(np.float32)
    g_Ca = np.clip(rng.exponential(scale=2.5, size=n).astype(np.float32) + 2.0, 2.0, 8.0)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    C_m    = np.full(n, 1.0, dtype=np.float32)

    g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
    E_rev_arr = np.array([50.0, -77.0, 120.0], dtype=np.float32)

    print(f"  🔬 HH 3ch+CaV × {n}N: NaV mean={g_Na.mean():.0f}, KV mean={g_K.mean():.0f}, "
          f"CaV mean={g_Ca.mean():.2f}")
    return g_max_arr, E_rev_arr, g_leak, E_leak, C_m


# ═══════════════════════════════════════════════════════════
#  Structured input
# ═══════════════════════════════════════════════════════════

def build_input(n: int, t_sec: float) -> np.ndarray:
    """4-band structured input + cross-frequency modulation."""
    ext_I = (np.random.randn(n) * 0.8 + 3.5).astype(np.float32)
    bands = [(2.0, 0.0), (5.0, 1.2), (8.0, 2.5), (13.0, 0.8)]
    n4 = max(1, n // 4)
    for i, (freq, phase) in enumerate(bands):
        start = i * n4
        end = min((i + 1) * n4, n)
        if start < end:
            ext_I[start:end] += 6.5 * math.sin(2 * math.pi * freq * t_sec + phase)
    mod = 1.0 + 0.4 * math.sin(2 * math.pi * 1.5 * t_sec)
    ext_I *= mod
    return ext_I.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="🚀 IIT на реальном FlyWire коннектоме")
    p.add_argument("--taas", default="princeton.taas",
                    help=".taas file in examples/data/ (default: princeton.taas)")
    p.add_argument("--subsample", type=int, default=100,
                    help="Subsample N neurons (default: 100, 0=all)")
    p.add_argument("--steps", type=int, default=300,
                    help="Simulation steps (default: 300)")
    p.add_argument("--scale", type=float, default=3.0,
                    help="Weight scale (default: 3.0, 1.0=original)")
    p.add_argument("--cross-half", type=float, default=0.2,
                    help="Fraction of cross-half bridge neurons (default: 0.2)")
    p.add_argument("--full", action="store_true",
                    help="Run on full connectome without optimizations")
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  🚀 IIT на реальном FlyWire коннектоме             ║")
    print("║     HH+CaV + Pearson + Weight Scale + Cross-Half   ║")
    print("╚══════════════════════════════════════════════════════╝")

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"  ❌ {taas_path} not found!")
        print(f"     Available: princeton.taas, fafb.taas, fafb_15m.taas, ...")
        sys.exit(1)

    n_neurons = args.subsample if args.subsample > 0 else 0
    n_steps = max(10, args.steps)

    # 1. Load connectome
    row_ptr, col_indices, values, nt_types, n = load_flywire(taas_path, n_neurons)

    # 2. Apply optimizations or load raw
    if args.full:
        conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
        print("  🔓 Full connectome (no optimizations)")
    else:
        conn = apply_consciousness_optimizations(
            row_ptr, col_indices, values, nt_types, n,
            weight_scale=args.scale, cross_half_frac=args.cross_half,
        )

    print(f"\n  🧠 Network: {n} neurons, {n_steps} steps @ dt={DT_MS}ms")

    # 3. Simulator HH+CaV
    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params(n)
    sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, DT_MS)
    sim.set_C_m(C_m)

    # 4. Rust IIT
    iit = tcm.IITConsciousness(n, threshold=0.3)
    spike_buf = np.zeros((n_steps, n), dtype=np.float32)

    # 5. Warm-up
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(100):
        ext_I = np.random.randn(n).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")

    # 6. Main loop
    vnc_values = conn.values()
    t_start = time.time()

    for step in range(n_steps):
        t_sec = step * DT_MS / 1000.0
        ext_I = build_input(n, t_sec)

        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            n_rec = min(len(spikes), n)
            spike_buf[step, :n_rec] = np.asarray(spikes[:n]).flatten()

        if step % 50 == 0 and step >= 50:
            chunk = spike_buf[:step + 1]
            chunk_flat = chunk.ravel().astype(np.float32)
            iit.compute_with_spikes(vnc_values, chunk_flat, step + 1, n)

            pct = (step / n_steps) * 100
            bars = int(pct / 5)
            sys.stdout.write(f"\r  ⏳ {step:>4d}/{n_steps} [{'#' * bars}{'·' * (20 - bars)}] "
                           f"Φ={iit.phi:.4f} C={iit.connectivity:.2f} D={iit.diversity:.2f} "
                           f"I={iit.integration:.2f} {'🧠' if iit.is_conscious else '💤'}  ")
            sys.stdout.flush()

    elapsed = time.time() - t_start

    # Final Phi
    spike_flat = spike_buf.ravel().astype(np.float32)
    phi = iit.compute_with_spikes(vnc_values, spike_flat, n_steps, n)

    print(f"\r  ✅ Done! {n_steps} steps in {elapsed:.1f}s{' ' * 30}")

    print(f"\n  ═══ РЕЗУЛЬТАТЫ: {taas_path.name} ═══")
    print(f"  Φ = {phi:.6f}    {'🧠 СОЗНАНИЕ!' if iit.is_conscious else '💤 ниже порога'}")
    print(f"  C (connectivity)  = {iit.connectivity:.4f}")
    print(f"  D (diversity)     = {iit.diversity:.4f}")
    print(f"  I (integration)   = {iit.integration:.4f}")
    print(f"  Уровень сознания: {iit.consciousness_level*100:.1f}%")
    print(f"  Trend: {iit.phi_trend()}")

    # Spike stats
    total = spike_buf.sum()
    fr_mean = total / n / (n_steps * DT_MS / 1000.0)
    active = (spike_buf.sum(axis=0) > 0).mean() * 100
    print(f"\n  📊 Spike stats: {total:.0f} total, {fr_mean:.1f} Hz mean, {active:.0f}% active")

    if iit.is_conscious:
        print(f"\n  🎉🎉🎉  FlyWire + CaV = СОЗНАНИЕ!  Φ={phi:.4f} > {iit.threshold}  🎉🎉🎉")
    else:
        gap = (iit.threshold - phi) / iit.threshold * 100
        print(f"\n  ⏳ До порога: {gap:.1f}% (нужно I > {iit.threshold / (max(iit.connectivity, 0.01) * max(iit.diversity, 0.01)):.3f})")

    print(f"\n  ✨ FlyWire IIT by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
