#!/usr/bin/env python3
"""🧠 IIT Parameter Sweep — ищем Phi > 0.3

Swеер по:
  - weight_scale: множитель синаптических весов (C)
  - g_Ca_scale: множитель CaV проводимости (I)
  - ring_density: доля нейронов в реципрокных парах (I)

Запуск:
  python3 scripts/iit_sweep.py --quick   # малый sweep
  python3 scripts/iit_sweep.py           # полный sweep
"""

import sys, os, time, math, argparse, json
import numpy as np
from pathlib import Path
from itertools import product

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import taas_checkmate as tcm

np.random.seed(42)

N_NEURONS = 100
N_STEPS = 200
DT_MS = 0.05


def build_vnc(n: int, density: float = 0.12, ring_density: float = 0.05,
              weight_scale: float = 1.0) -> tcm.ConnectomeData:
    """VNC с параметризуемыми ring_density и weight_scale."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list, val_list, nt_list = [], [], []

    ring_size = max(1, int(n * ring_density / 2))
    ring_set = set()
    for i in range(0, min(ring_size * 2, n), 2):
        if i + 1 < n:
            ring_set.add(i)
            ring_set.add(i + 1)

    for i in range(n):
        is_inhib = np.random.random() < 0.9
        in_ring = i in ring_set

        if in_ring:
            partner = i + 1 if i % 2 == 0 else i - 1
            if partner < n:
                col_list.append(partner)
                val_list.append(-8.0 * weight_scale)
                nt_list.append(-1.0)
            n_post = max(1, int(n * density * 0.5))
        else:
            n_post = max(1, int(n * density))

        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]

        for j in posts:
            col_list.append(int(j))
            if is_inhib:
                val_list.append(float(np.random.uniform(-6.0, -0.8) * weight_scale))
                nt_list.append(-1.0)
            else:
                val_list.append(float(np.random.uniform(0.3, 2.0) * weight_scale))
                nt_list.append(1.0)

        row_ptr[i + 1] = len(col_list)

    return tcm.ConnectomeData(
        row_ptr,
        np.array(col_list, dtype=np.int32),
        np.array(val_list, dtype=np.float32),
        np.array(nt_list, dtype=np.float32),
    )


def build_hh_params(n: int, n_channels: int, g_ca_scale: float = 1.0, rng=None):
    """HH params с параметризуемым g_Ca scale."""
    if rng is None:
        rng = np.random.RandomState(42)

    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    g_K  = rng.uniform(24, 60, n).astype(np.float32)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    C_m    = np.full(n, 1.0, dtype=np.float32)

    if n_channels == 3:
        g_Ca = np.clip(rng.exponential(scale=2.0, size=n).astype(np.float32) + 1.0, 1.0, 6.0) * g_ca_scale
        g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0, 120.0], dtype=np.float32)
    else:
        g_max_arr = np.concatenate([g_Na, g_K]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0], dtype=np.float32)

    return g_max_arr, E_rev_arr, g_leak, E_leak, C_m


def build_structured_input(n_neurons: int, t_sec: float) -> np.ndarray:
    """Rich structured input: 4 frequency bands."""
    ext_I = (np.random.randn(n_neurons) * 1.0 + 3.0).astype(np.float32)
    f1, f2, f3, f4 = 2.0, 5.0, 8.0, 13.0
    ph1, ph2, ph3, ph4 = 0.0, 1.2, 2.5, 0.8
    n4 = n_neurons // 4

    ext_I[:n4] += 6.0 * math.sin(2 * math.pi * f1 * t_sec + ph1)
    ext_I[n4:2*n4] += 6.0 * math.sin(2 * math.pi * f2 * t_sec + ph2)
    if 2*n4 < n_neurons:
        ext_I[2*n4:3*n4] += 6.0 * math.sin(2 * math.pi * f3 * t_sec + ph3)
    ext_I[3*n4:] += 6.0 * math.sin(2 * math.pi * f4 * t_sec + ph4)

    mod = 1.0 + 0.5 * math.sin(2 * math.pi * 1.5 * t_sec)
    return (ext_I * mod).astype(np.float32)


def run_trial(label: str, conn, n_neurons: int, n_steps: int,
              g_ca_scale: float = 1.0) -> dict:
    """Single IIT run with Pearson integration, returns metrics."""
    sim = tcm.BrainSimulator(n_neurons, "hh", pacemaking=True)
    g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params(n_neurons, 3, g_ca_scale)
    sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, DT_MS)
    sim.set_C_m(C_m)

    iit = tcm.IITConsciousness(n_neurons, threshold=0.3)
    spike_buf = np.zeros((n_steps, n_neurons), dtype=np.float32)

    # Warm-up
    for _ in range(50):
        ext_I = np.random.randn(n_neurons).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, DT_MS, ext_I)

    vnc_values = conn.values()

    for step in range(n_steps):
        t_sec = step * DT_MS / 1000.0
        ext_I = build_structured_input(n_neurons, t_sec)
        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            n_rec = min(len(spikes), n_neurons)
            spike_buf[step, :n_rec] = np.asarray(spikes[:n_neurons]).flatten()

    # Final Phi with Pearson integration
    spike_flat = spike_buf.ravel().astype(np.float32)
    phi = iit.compute_with_spikes(vnc_values, spike_flat, n_steps, n_neurons)

    # Spike stats
    total_spikes = spike_buf.sum()
    active_pct = (spike_buf.sum(axis=0) > 0).mean() * 100

    return {
        "label": label,
        "phi": float(phi),
        "C": float(iit.connectivity),
        "D": float(iit.diversity),
        "I": float(iit.integration),
        "conscious": bool(iit.is_conscious),
        "total_spikes": int(total_spikes),
        "active_pct": float(active_pct),
    }


def main():
    p = argparse.ArgumentParser(description="🧠 IIT Sweep — Phi > 0.3")
    p.add_argument("--quick", action="store_true", help="Малый sweep")
    p.add_argument("--neurons", type=int, default=N_NEURONS)
    p.add_argument("--steps", type=int, default=N_STEPS)
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  🧠 IIT PARAMETER SWEEP — Phi > 0.3                ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.quick:
        weight_scales = [2.0, 3.0, 4.0, 5.0]
        gca_scales = [1.0, 2.0, 3.0]
        ring_densities = [0.05, 0.10, 0.20]
    else:
        weight_scales = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
        gca_scales = [0.5, 1.0, 2.0, 3.0, 5.0]
        ring_densities = [0.05, 0.10, 0.15, 0.20, 0.30]

    results = []
    total = len(weight_scales) * len(gca_scales) * len(ring_densities)
    t_start = time.time()
    trial = 0

    # Sort: most promising first (high weight_scale, high ring_density)
    params = sorted(product(weight_scales, gca_scales, ring_densities),
                    key=lambda x: (-x[0], -x[2], -x[1]))

    for ws, gs, rd in params:
        trial += 1
        label = f"ws={ws:.1f} gs={gs:.1f} rd={rd:.2f}"
        remaining = total - trial
        eta = (time.time() - t_start) / trial * remaining if trial > 0 else 0

        print(f"\n  [{trial}/{total}] {label}  (ETA: {eta:.0f}s)", end="")
        sys.stdout.flush()

        try:
            t0 = time.time()
            conn = build_vnc(args.neurons, ring_density=rd, weight_scale=ws)
            result = run_trial(label, conn, args.neurons, args.steps, g_ca_scale=gs)
            elapsed = time.time() - t0
            result["time_s"] = round(elapsed, 1)
            results.append(result)

            conscious = "🧠 СОЗНАНИЕ!" if result["conscious"] else "💤"
            print(f"  Φ={result['phi']:.4f} C={result['C']:.3f} I={result['I']:.3f} {conscious} [{elapsed:.0f}s]")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    # Summary
    elapsed_total = time.time() - t_start
    conscious_runs = [r for r in results if r["conscious"]]
    conscious_runs.sort(key=lambda x: -x["phi"])

    print("\n" + "═" * 65)
    print("  📊 SWEEP RESULTS — Phi > 0.3 runs:")
    print("═" * 65)
    print(f"  Total trials: {total}, completed: {len(results)}, conscious: {len(conscious_runs)}")
    print(f"  Time: {elapsed_total:.0f}s")

    if conscious_runs:
        print(f"\n  🏆 TOP 5 CONSCIOUS CONFIGS:")
        print(f"  {'Rank':<5} {'Phi':<8} {'C':<6} {'I':<6} {'WS':<6} {'GS':<6} {'RD':<6} {'Spikes':<8}")
        print(f"  {'─'*5} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        for i, r in enumerate(conscious_runs[:5]):
            params = r["label"].split()
            ws_v = params[0].split("=")[1]
            gs_v = params[1].split("=")[1]
            rd_v = params[2].split("=")[1]
            print(f"  {i+1:<5} {r['phi']:<8.4f} {r['C']:<6.3f} {r['I']:<6.3f} "
                  f"{ws_v:<6} {gs_v:<6} {rd_v:<6} {r['total_spikes']:<8}")
    else:
        print(f"\n  ⚠ No config reached Phi > 0.3")
        # Show top 5 anyway
        results.sort(key=lambda x: -x["phi"])
        print(f"\n  📊 TOP 5 CONFIGS (best effort):")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r['label']}: Φ={r['phi']:.4f} C={r['C']:.3f} I={r['I']:.3f}")

    # Best config JSON
    if conscious_runs:
        best = conscious_runs[0]
        params = best["label"].split()
        best_config = {
            "weight_scale": float(params[0].split("=")[1]),
            "g_ca_scale": float(params[1].split("=")[1]),
            "ring_density": float(params[2].split("=")[1]),
            "phi": best["phi"],
            "C": best["C"],
            "I": best["I"],
        }
        print(f"\n  🏆 BEST CONFIG: {json.dumps(best_config, indent=2)}")

    print(f"\n  ✨ Sweep by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
