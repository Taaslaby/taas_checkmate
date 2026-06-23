#!/usr/bin/env python3
"""
АБЕРТОНЫ v4: ДРАЙВ ЧЕРЕЗ СИНАПТИЧЕСКОЕ ПОЛЕ h = W*ψ
======================================================
Правильная физика:
  compute_hamiltonian → h_real, h_imag (W*ψ) → ABERTON → step_psi_wave → normalize

Aberton equation:
  da/dt = (-γ + iω)*a + (h_real + i*h_imag)   [synaptic field drives collapse]

Коллапс там, где сильное синаптическое поле = hubs.
Gini coefficient: Original vs Shuffled.
"""

import time, numpy as np, gzip
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_STEPS = 1000
GAMMA = 0.3          # aberton damping
G_H = 1.0            # coupling from h_field to aberton
THRESHOLD = 0.8      # collapse threshold
OMEGA = 0.5          # aberton natural frequency (Hz-like, *2π for rad)
SPEED = 1.0
HBAR = 1.0

print("█" * 70)
print("  АБЕРТОНЫ v4: SYNAPTIC FIELD DRIVE")
print("  h = W*ψ → aberton → collapse → сознательное ядро?")
print("█" * 70)
np.random.seed(42)

# ─── Load ───
print("\n[1/5] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
npad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vs):,} synapses")

# ─── Shuffle ───
print("[2/5] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])
print("done")

outdeg = np.array([rp[i+1] - rp[i] for i in range(n)], dtype=np.int32)

# ─── Run ───
def run(label, rp_i, ci_i, vs_i):
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2*np.pi, npad).astype(np.float32)
    v = np.cos(th); u = np.sin(th)
    hr = np.zeros(npad, dtype=np.float32)
    hi = np.zeros(npad, dtype=np.float32)
    ie = np.zeros(npad, dtype=np.float32)
    spd = np.full(npad, SPEED, dtype=np.float32)

    av = np.zeros(npad, dtype=np.float32)  # Re(aberton)
    au = np.zeros(npad, dtype=np.float32)  # Im(aberton)
    w = 2 * np.pi * OMEGA
    
    coll_counts = np.zeros(n, dtype=np.int32)
    coll_log = np.zeros(N_STEPS, dtype=np.int32)
    hfield_log = np.zeros(N_STEPS, dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        # 1. Compute synaptic field: h = W*ψ
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)

        # Local field magnitude for each neuron
        h_mag = np.sqrt(hr*hr + hi*hi)
        h_mean = float(np.mean(h_mag[:n]))
        hfield_log[step] = h_mean

        # 2. Aberton update: da/dt = (-γ + iω)*a + G_H * (hr - i*hi)  [wait, check sign]
        # From Rust: hr = ΣW*u, hi = -ΣW*v. So h = hr + i*hi = ΣW*u - i*ΣW*v
        # This is -i*W*ψ (see earlier derivation).
        # For the aberton, the driving force is |h| (field magnitude)
        # Or the complex field h = hr + i*hi
        # da/dt = (-γ + iω)*a + G_H * (hr + i*hi)
        # Real: dav_dt = -γ*av - ω*au + G_H*hr
        # Imag: dau_dt = -γ*au + ω*av + G_H*hi
        dav = (-GAMMA * av - w * au + G_H * hr) * DT
        dau = (-GAMMA * au + w * av + G_H * hi) * DT
        av += dav
        au += dau

        # 3. Check collapse: |a|² > θ²
        a2 = av*av + au*au
        mask = a2 > THRESHOLD * THRESHOLD
        nc = np.sum(mask)
        if nc > 0:
            mask_n = mask[:n]
            coll_counts[mask_n] += 1
            # Collapse: reset aberton, add feedback to ψ
            av[mask] = 0.0
            au[mask] = 0.0
            ie[mask] += 2.0  # feedback current

        # 4. ψ-wave step
        step_psi_wave(v, u, hr, hi, ie, spd, DT, HBAR)
        ie[:] = 0.0

        coll_log[step] = nc
        if (step + 1) % 200 == 0:
            print(f"  {label}: [{step+1}/{N_STEPS}] coll={nc:4d} tot={np.sum(coll_counts):5d} "
                  f"h_field={h_mean:.2f} [{time.perf_counter()-t0:.0f}s]")

    return {"coll_counts": coll_counts, "coll_log": coll_log, "hfield": hfield_log,
            "time": time.perf_counter()-t0}

# ─── Run ───
print("\n[3/5] Running...")
R = {}
for label, rp_i, ci_i, vs_i in [("Original", rp, ci, vs), ("Shuffled", rp, sci, vs)]:
    print(f"\n  {label}:")
    R[label] = run(label, rp_i, ci_i, vs_i)
    print(f"  Done in {R[label]['time']:.0f}s")

# ─── Analysis ───
print("\n[4/5] Analysis...")

hub_order = np.argsort(outdeg)[::-1]

for label in ["Original", "Shuffled"]:
    cc = R[label]["coll_counts"]
    total = np.sum(cc)
    n_coll = np.sum(cc > 0)

    # Gini (collapse concentration)
    if total > 0:
        cc_s = np.sort(cc[cc > 0].astype(np.float64))
        n_nz = len(cc_s)
        gini = (2*np.sum((np.arange(1,n_nz+1)*cc_s))/(n_nz*np.sum(cc_s)) - (n_nz+1)/n_nz)
        pr = total**2 / np.sum(cc.astype(np.float64)**2) if np.sum(cc**2) > 0 else 0
    else:
        gini, pr = 0.0, 0.0

    hub_coll = np.sum(cc[hub_order[:1000]])
    corr = np.corrcoef(outdeg, cc.astype(np.float32))[0,1] if np.std(cc) > 0 else 0

    R[label].update(gini=gini, pr=pr, hub_pct=hub_coll/total*100 if total > 0 else 0, corr=corr)
    print(f"  {label}: tot={total} n_coll={n_coll} Gini={gini:.4f} PR={pr:.0f} hubs={hub_coll/total*100:.1f}% corr(outdeg,coll)={corr:.4f}")

# ─── Final ───
print("\n[5/5] VERDICT")
print("█" * 70)
for label in ["Original", "Shuffled"]:
    r = R[label]
    print(f"  {label:<12}: Gini={r['gini']:.4f}  PR={r['pr']:<6.0f}  "
          f"hubs_top1000={r['hub_pct']:.1f}%  corr(degree,coll)={r['corr']:+.4f}")

o, s = R["Original"], R["Shuffled"]
print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
if o["gini"] > s["gini"] + 0.005:
    print(f"  │  Gini: Original {o['gini']:.3f} > Shuffled {s['gini']:.3f}                        │")
    print(f"  │                                                              │")
    print(f"  │  FAFB CONCENTRATES COLLAPSES В ХАБАХ                          │")
    print(f"  │  ← СОЗНАТЕЛЬНОЕ ЯДРО (Orch-OR conscious nucleus)              │")
    print(f"  │  Коллапсы сфокусированы = интегрированная информация           │")
elif o["gini"] > s["gini"]:
    print(f"  │  Gini: Original {o['gini']:.3f} ≈ Shuffled {s['gini']:.3f} (weak)                │")
    print(f"  │  Topology has small effect on collapse focusing               │")
else:
    print(f"  │  Gini: Shuffled {s['gini']:.3f} ≥ Original {o['gini']:.3f}                        │")
    print(f"  │  Collapses are spread uniformly regardless of topology        │")
print(f"  └──────────────────────────────────────────────────────────────┘")

np.savez("aberton_v4.npz", **{f"{k}_{l}": R[l][k] for l in ["Original","Shuffled"] 
         for k in ["coll_counts","coll_log","hfield"]})
print(f"\n  Saved: aberton_v4.npz")
print("█" * 70)
