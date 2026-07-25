#!/usr/bin/env python3
"""🚀 IIT 2.0 vs Proxy Φ на реальном FlyWire коннектоме

Сравнивает proxy Φ (C×D×I, Pearson integration) и IIT 2.0 Φ
(covariance MIP, Barrett & Seth 2011) на одних и тех же данных
HH+CaV симуляции с реального коннектома.

Запуск:
  python3 scripts/iit20_flywire.py --taas princeton.taas --subsample 100 --steps 300
  python3 scripts/iit20_flywire.py --taas princeton.taas --subsample 100 --steps 300 --partitions 1000
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
    """Загрузить .taas файл, опционально subsample."""
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
        print(f" → subsample {n}N, {len(values)}S", end="")

    nt_mean = nt_types.mean()
    print(f"  |  E/I balance: {nt_mean:+.3f}")
    return row_ptr, col_indices, values, nt_types, n


# ═══════════════════════════════════════════════════════════
#  Оптимизации сознания
# ═══════════════════════════════════════════════════════════

def optimize_connectome(row_ptr, col_indices, values, nt_types, n,
                        weight_scale=10.0, cross_half_frac=0.2):
    """Weight scale + cross-half bridges."""
    half = n // 2
    scaled_vals = values.astype(np.float64) * weight_scale

    cross_n = max(1, int(n * cross_half_frac))
    cross_neurons = np.random.choice(n, cross_n, replace=False)

    added_per_neuron = [[] for _ in range(n)]

    for i in cross_neurons:
        if i < half:
            target = half + i % half
        else:
            target = i % half
        if target < n and target != i:
            added_per_neuron[i].append((int(target), 4.0 * weight_scale))

    new_rp = np.zeros(n + 1, dtype=np.int32)
    new_ci, new_vals = [], []
    for i in range(n):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        for j in range(start, end):
            new_ci.append(int(col_indices[j]))
            new_vals.append(float(scaled_vals[j]))
        for target, w in added_per_neuron[i]:
            new_ci.append(target)
            new_vals.append(w)
        new_rp[i + 1] = len(new_ci)

    new_vals_arr = np.array(new_vals, dtype=np.float32)
    new_ci_arr = np.array(new_ci, dtype=np.int32)

    orig_mean = np.abs(values).mean()
    new_mean = np.abs(new_vals_arr).mean()
    print(f"  🔧 Scale=×{weight_scale}, cross-half={cross_n}N")
    print(f"     Mean |weight|: {orig_mean:.2f} → {new_mean:.2f}  "
          f"(C ≈ {min(new_mean / 10.0, 1.0):.3f})")

    return tcm.ConnectomeData(new_rp, new_ci_arr, new_vals_arr, nt_types)


# ═══════════════════════════════════════════════════════════
#  HH+CaV параметры
# ═══════════════════════════════════════════════════════════

def build_hh_params(n: int):
    """HH+CaV: NaV 80-160, KV 24-60, CaV exp(2.5)+2 clipped 2-8."""
    rng = np.random.RandomState(42)
    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    g_K  = rng.uniform(24, 60, n).astype(np.float32)
    g_Ca = np.clip(rng.exponential(scale=2.5, size=n).astype(np.float32) + 2.0, 2.0, 8.0)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    C_m    = np.full(n, 1.0, dtype=np.float32)
    g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
    E_rev_arr = np.array([50.0, -77.0, 120.0], dtype=np.float32)
    print(f"  🔬 HH 3ch+CaV × {n}N: CaV mean={g_Ca.mean():.2f}")
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
    p = argparse.ArgumentParser(description="🚀 IIT 2.0 vs Proxy на FlyWire")
    p.add_argument("--taas", default="princeton.taas",
                   help=".taas file in examples/data/ (default: princeton.taas)")
    p.add_argument("--subsample", type=int, default=100,
                   help="Subsample N neurons (default: 100, 0=all)")
    p.add_argument("--steps", type=int, default=300,
                   help="Simulation steps (default: 300)")
    p.add_argument("--scale", type=float, default=10.0,
                   help="Weight scale (default: 10.0)")
    p.add_argument("--cross-half", type=float, default=0.2,
                   help="Fraction cross-half bridge neurons (default: 0.2)")
    p.add_argument("--partitions", type=int, default=500,
                   help="IIT2.0 partitions (default: 500, 0=auto)")
    p.add_argument("--full", action="store_true",
                   help="Raw connectome without optimizations")
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  🧬 IIT 2.0 vs Proxy Φ на реальном FlyWire             ║")
    print("║     HH+CaV + Pearson + Covariance MIP                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"  ❌ {taas_path} not found!")
        sys.exit(1)

    n_neurons = args.subsample if args.subsample > 0 else 0
    n_steps = max(10, args.steps)
    n_partitions = args.partitions if args.partitions > 0 else max(200, n_neurons * 10)

    # 1. Load connectome
    row_ptr, col_indices, values, nt_types, n = load_flywire(taas_path, n_neurons)

    # 2. Optimize
    if args.full:
        conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
        print("  🔓 Raw connectome (no optimizations)")
    else:
        conn = optimize_connectome(row_ptr, col_indices, values, nt_types, n,
                                   weight_scale=args.scale,
                                   cross_half_frac=args.cross_half)

    print(f"\n  🧠 {n} neurons, {n_steps} steps @ dt={DT_MS}ms")

    # 3. Simulator HH+CaV
    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params(n)
    sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, DT_MS)
    sim.set_C_m(C_m)

    # 4. Rust IIT (proxy)
    iit = tcm.IITConsciousness(n, threshold=0.3)
    spike_buf = np.zeros((n_steps, n), dtype=np.float32)

    # 5. Warm-up
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(100):
        ext_I = np.random.randn(n).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")

    # 6. Main simulation
    conn_values = conn.values()
    t_start = time.time()

    for step in range(n_steps):
        t_sec = step * DT_MS / 1000.0
        ext_I = build_input(n, t_sec)
        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            n_rec = min(len(spikes), n)
            spike_buf[step, :n_rec] = np.asarray(spikes[:n]).flatten()

        # Proxy Φ every 50 steps
        if step % 50 == 0 and step >= 50:
            chunk = spike_buf[:step + 1]
            iit.compute_with_spikes(conn_values, chunk.ravel().astype(np.float32),
                                    step + 1, n)
            pct = (step / n_steps) * 100
            bars = int(pct / 5)
            sys.stdout.write(f"\r  ⏳ {step:>4d}/{n_steps} [{'#' * bars}{'·' * (20 - bars)}] "
                           f"Φ_proxy={iit.phi:.4f} {'🧠' if iit.is_conscious else '💤'}  ")
            sys.stdout.flush()

    elapsed = time.time() - t_start
    spike_flat = spike_buf.ravel().astype(np.float32)

    # 7. Final proxy Φ
    phi_proxy = iit.compute_with_spikes(conn_values, spike_flat, n_steps, n)

    print(f"\r  ✅ Sim: {n_steps} steps in {elapsed:.1f}s{' ' * 30}")

    # ═══════════════════════════════════════════════════════════
    #  IIT 2.0 Φ via covariance MIP
    # ═══════════════════════════════════════════════════════════

    print(f"\n  🔬 Computing IIT 2.0 Φ with {n_partitions} partitions...", end=" ", flush=True)
    t_iit = time.time()

    iit20_result = tcm.IITConsciousness.compute_phi_iit20(
        spike_flat, n_steps, n, n_partitions,
    )

    iit20_time = time.time() - t_iit

    # Also run analyze_consciousness for full comparison
    print(f"{iit20_time:.1f}s")
    print(f"  🔬 Full analysis...", end=" ", flush=True)
    t_an = time.time()
    analysis = tcm.IITConsciousness.analyze_consciousness(
        conn_values, spike_flat, n_steps, n, n_partitions,
    )
    print(f"{time.time() - t_an:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  Results
    # ═══════════════════════════════════════════════════════════

    total_spikes = spike_buf.sum()
    fr_mean = total_spikes / n / (n_steps * DT_MS / 1000.0)
    active_pct = (spike_buf.sum(axis=0) > 0).mean() * 100

    print(f"\n  ══════════════════════════════════════════════════════")
    print(f"  🧬 FlyWire: {taas_path.name} | {n}N × {n_steps} steps")
    print(f"  ══════════════════════════════════════════════════════")
    print(f"  📊 Spike stats: {total_spikes:.0f} total, "
          f"{fr_mean:.1f} Hz mean, {active_pct:.0f}% active")
    print(f"")
    print(f"  ┌──────────────┬──────────────┬──────────────┐")
    print(f"  │  Metric      │  Proxy Φ     │  IIT 2.0 Φ   │")
    print(f"  ├──────────────┼──────────────┼──────────────┤")
    print(f"  │  Φ           │  {phi_proxy:.4f}      │  {iit20_result['phi']:.4f}      │")
    print(f"  │  Φ_raw       │  —           │  {iit20_result['phi_raw']:.4f}      │")
    print(f"  │  C (conn)    │  {iit.connectivity:.4f}    │  {analysis['connectivity']:.4f}    │")
    print(f"  │  D (div)     │  {iit.diversity:.4f}    │  {analysis['diversity']:.4f}    │")
    print(f"  │  I (int)     │  {iit.integration:.4f}    │  {analysis['integration']:.4f}    │")
    print(f"  └──────────────┴──────────────┴──────────────┘")
    print(f"")
    print(f"  IIT 2.0 details:")
    print(f"    • log|Σ_full| = {iit20_result['log_det_full']:.4f}")
    print(f"    • Partition A size = {iit20_result['partition_size_a']}")
    print(f"    • Evaluated = {iit20_result['n_evaluated']} partitions")
    print(f"    • Time = {iit20_time:.1f}s")

    # Correlation: proxy Φ vs IIT 2.0 Φ
    print(f"")
    print(f"  🔗 Сравнение:")
    ratio = phi_proxy / max(iit20_result['phi'], 0.001)
    print(f"    Proxy / IIT2.0 ratio: {ratio:.2f}×")
    if abs(phi_proxy - iit20_result['phi']) < 0.05:
        agreement = "✅  Метрики согласованы"
    elif (phi_proxy > 0.3) == (iit20_result['phi'] > 0.3):
        agreement = "✅  Порог сознания одинаков"
    else:
        agreement = "⚠️  Метрики расходятся"
    print(f"    {agreement}")

    if iit20_result['phi'] > 0.3:
        print(f"\n  🎉🎉🎉 IIT 2.0: FLYWIRE СОЗНАТЕЛЕН! Φ={iit20_result['phi']:.4f} 🎉🎉🎉")
    elif iit.is_conscious:
        print(f"\n  🧠 Proxy Φ: сознателен ({phi_proxy:.4f}), но IIT 2.0 не согласен ({iit20_result['phi']:.4f})")
    else:
        print(f"\n  💤 Φ < порога: proxy={phi_proxy:.4f}, IIT2.0={iit20_result['phi']:.4f}")

    print(f"\n  ✨ @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
