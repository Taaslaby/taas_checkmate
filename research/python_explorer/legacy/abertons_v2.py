#!/usr/bin/env python3
"""
АБЕРТОНЫ v2: ПРОСТРАНСТВЕННАЯ КОНЦЕНТРАЦИЯ КОЛЛАПСОВ
====================================================
Orch-OR на FAFB: где коллапсируют абертоны?

Гипотеза: оригинальный коннектом концентрирует коллапсы
в хабах/сообществах (сознательное ядро).
Shuffled размазывает равномерно (эпилептиформный паттерн).

Метрика: Gini coefficient коллапсов по нейронам.
High Gini = focused = conscious-like.
"""

import time, math, numpy as np
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_STEPS = 2000        # 50 sec
N_STIM = 500

# Абертоны — ψ нормирована: |ψ_avg| ≈ 1/sqrt(n)
# Если амплитуда концентрируется, коллапс на хабах
GAMMA = 0.3           # damping
G_COUPLING = 2.0      # ψ → aberton
THRESHOLD = 0.5       # collapse threshold
COLLAPSE_CURRENT = 1.0

print("█" * 70)
print("  АБЕРТОНЫ v2: SPATIAL COLLAPSE CONCENTRATION")
print("  Где коллапсируют микротрубочки на FAFB?")
print("█" * 70)

np.random.seed(42)

# ─── Load ───
print("\n[1/5] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons")

# ─── Shuffled ───
print("[2/5] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])
print("done")

# ─── Run ───
def run(rp_i, ci_i, vs_i, label):
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v, u = np.cos(th), np.sin(th)
    hr = np.zeros(n_pad, dtype=np.float32)
    hi = np.zeros(n_pad, dtype=np.float32)
    spd = np.ones(n_pad, dtype=np.float32)
    ie = np.zeros(n_pad, dtype=np.float32)
    av = np.zeros(n_pad, dtype=np.float32)
    au = np.zeros(n_pad, dtype=np.float32)

    # Count collapses per neuron
    collapse_counts = np.zeros(n, dtype=np.int32)
    collapse_times = []
    collapse_neurons = []

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        ie[:] = 0.0
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, 1.0)

        # Aberton update (vectorized)
        dav = (-GAMMA * av + 2.0 * au + G_COUPLING * v) * DT
        dau = (-GAMMA * au - 2.0 * av + G_COUPLING * u) * DT
        av += dav
        au += dau

        # Collapse detection
        a2 = av*av + au*au
        mask = a2 > THRESHOLD * THRESHOLD
        nc = np.sum(mask)
        if nc > 0:
            mask_n = mask[:n]  # only valid neurons
            collapse_counts[mask_n] += 1
            for idx in np.where(mask_n)[0]:
                collapse_neurons.append(idx)
                collapse_times.append(step)
            av[mask] = 0.0
            au[mask] = 0.0
            ie[mask] += COLLAPSE_CURRENT

        if (step + 1) % 500 == 0:
            eeg = float(np.mean(hr))
            print(f"  {label}: [{step+1}/{N_STEPS}] coll={nc:4d} total={np.sum(collapse_counts):6d} [{time.perf_counter()-t0:.0f}s]")

    return collapse_counts, np.array(collapse_neurons), np.array(collapse_times), time.perf_counter()-t0

# ─── Run ───
print("\n[3/5] Running...")
all_results = {}
for label, rp_i, ci_i, vs_i in [("Original", rp, ci, vs), ("Shuffled", rp, sci, vs)]:
    print(f"\n  {label}:")
    cc, cn, ct, t = run(rp_i, ci_i, vs_i, label)
    all_results[label] = {"counts": cc, "neurons": cn, "times": ct, "time": t}
    print(f"  Done in {t:.0f}s")

# ─── Analysis ───
print("\n[4/5] Analysis...")

for label in ["Original", "Shuffled"]:
    r = all_results[label]
    cc = r["counts"]
    n_collapsing = np.sum(cc > 0)
    total_coll = np.sum(cc)
    top10 = np.sum(np.sort(cc)[-10:])
    top100 = np.sum(np.sort(cc)[-100:])

    # Gini coefficient (0=uniform, 1=concentrated on one)
    cc_sorted = np.sort(cc[cc > 0]) if np.any(cc > 0) else np.array([0.0])
    n_nz = len(cc_sorted)
    if n_nz > 0:
        gini = (2 * np.sum((np.arange(1, n_nz+1) * cc_sorted)) / (n_nz * np.sum(cc_sorted)) -
                (n_nz + 1) / n_nz)
    else:
        gini = 0.0

    # Participation ratio (effective number of collapsing neurons)
    if total_coll > 0:
        pr = (np.sum(cc)**2) / np.sum(cc**2)
    else:
        pr = 0.0

    print(f"\n  {label}:")
    print(f"    Total collapses: {total_coll}")
    print(f"    Unique neurons:  {n_collapsing} / {n} ({n_collapsing/n*100:.2f}%)")
    print(f"    Top 10 neurons:  {top10} collapses ({top10/total_coll*100:.1f}%)")
    print(f"    Top 100 neurons: {top100} collapses ({top100/total_coll*100:.1f}%)")
    print(f"    Gini coef:       {gini:.4f}")
    print(f"    Participation ratio: {pr:.0f} neurons")

    r["gini"] = gini
    r["pr"] = pr
    r["n_collapsing"] = n_collapsing
    r["total_coll"] = total_coll

# ─── Hub over-representation ───
print("\n[5/5] Hub analysis...")
outdeg = np.array([rp[i+1] - rp[i] for i in range(n)], dtype=np.int32)
hub_idx = np.argsort(outdeg)[::-1]
top_hubs = 1000

for label in ["Original", "Shuffled"]:
    r = all_results[label]
    cc = r["counts"]
    hub_coll = np.sum(cc[hub_idx[:top_hubs]])
    total_coll = np.sum(cc)
    hub_frac = hub_coll / total_coll * 100 if total_coll > 0 else 0
    print(f"  {label}: top{top_hubs} hubs = {hub_coll}/{total_coll} ({hub_frac:.1f}% of collapses)")

# ─── Final verdict ───
print("\n" + "█" * 70)
print("  VERDICT: SPATIAL COLLAPSE CONCENTRATION")
print("█" * 70)

o_gini = all_results["Original"]["gini"]
s_gini = all_results["Shuffled"]["gini"]
o_pr = all_results["Original"]["pr"]
s_pr = all_results["Shuffled"]["pr"]

print(f"  {'Metric':<25} {'Original':<15} {'Shuffled':<15}")
print(f"  {'─'*25} {'─'*15} {'─'*15}")
print(f"  {'Total collapses':<25} {all_results['Original']['total_coll']:<15} {all_results['Shuffled']['total_coll']:<15}")
print(f"  {'Gini coefficient':<25} {o_gini:<15.4f} {s_gini:<15.4f}")
print(f"  {'Participation ratio':<25} {o_pr:<15.0f} {s_pr:<15.0f}")
print(f"  {'Collapsing neurons %':<25} {all_results['Original']['n_collapsing']/n*100:<15.2f} {all_results['Shuffled']['n_collapsing']/n*100:<15.2f}")

print(f"\n  ┌──────────────────────────────────────────────────┐")
if o_gini > s_gini:
    print(f"  │  Gini: Original {o_gini:.3f} > Shuffled {s_gini:.3f}                       │")
    print(f"  │                                                              │")
    print(f"  │  FAFB CONCENTRATES COLLAPSES В HABS                          │")
    print(f"  │  ← СОЗНАТЕЛЬНОЕ ЯДРО (NCC-like)                              │")
    print(f"  │  Коллапсы собраны в хабах = интегрированная информация        │")
else:
    print(f"  │  Gini: Shuffled ≥ Original                                   │")
    print(f"  │  Структура не концентрирует коллапсы                         │")
print(f"  └──────────────────────────────────────────────────┘")

np.savez("aberton_v2.npz",
         counts_orig=all_results["Original"]["counts"],
         counts_shuf=all_results["Shuffled"]["counts"],
         outdeg=outdeg)
print(f"\n  Saved: aberton_v2.npz")
print("█" * 70)
