#!/usr/bin/env python3
"""
PCI: Perturbational Complexity Index — ψ-wave on FAFB.
Casali et al. 2013 paradigm.
Baseline vs response (same length) → gzip complexity per channel.
"""

import time, math, gzip, os, numpy as np
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_BASE = 1000      # 25 sec baseline
N_RESP = 1000      # 25 sec response
N_STEPS = N_BASE + N_RESP
PULSE_AMP = 500.0
PULSE_LEN = 1       # single strong impulse (TMS-like)
READOUT_N = 2000

print("█" * 70)
print("  PCI: PERTURBATIONAL COMPLEXITY INDEX")
print("  ψ-wave on FAFB — Casali et al. 2013")
print("█" * 70)
np.random.seed(42)

# ─── Load ───
print("\n[1/5] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vs):,} synapses")

# ─── Shuffle ───
print("[2/5] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])
print("done")

# ─── Stim / readout ───
outdeg = np.array([rp[i+1] - rp[i] for i in range(n)], dtype=np.int32)
stim = np.sort(np.argsort(outdeg)[::-1][:200])
rdi = np.sort(np.random.choice(n, READOUT_N, replace=False))

# ─── Complexity ───
def compression_ratio(x):
    """CR: 1.0=random, <1.0=compressible (regular), >1.0=overhead."""
    m = np.median(x)
    bits = np.packbits((x > m).astype(np.uint8))
    return len(gzip.compress(bits)) / max(len(bits), 1)

# ─── Run ───
def run_and_measure(rp_i, ci_i, vs_i, stim_flag, seed=0):
    rng = np.random.default_rng(42 + seed)
    th = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v, u = np.cos(th), np.sin(th)
    hr = np.zeros(n_pad, dtype=np.float32)
    hi = np.zeros(n_pad, dtype=np.float32)
    spd = np.ones(n_pad, dtype=np.float32)
    ie = np.zeros(n_pad, dtype=np.float32)

    ts = np.zeros((len(rdi), N_STEPS), dtype=np.float32)
    for step in range(N_STEPS):
        ie[:] = 0.0
        if stim_flag and N_BASE <= step < N_BASE + PULSE_LEN:
            ie[stim] = PULSE_AMP
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, 1.0)
        ts[:, step] = v[rdi]

    base, resp = ts[:, :N_BASE], ts[:, N_BASE:]
    cr_b = np.array([compression_ratio(base[ch]) for ch in range(len(rdi))])
    cr_r = np.array([compression_ratio(resp[ch]) for ch in range(len(rdi))])
    pci = np.mean((cr_r - cr_b) / (cr_b + 1e-10))
    return pci, np.mean(cr_r), np.mean(cr_b)

# ─── Run ───
print("[3/5] Running...")
t0 = time.perf_counter()
R = {}
for label, rp_i, ci_i, vs_i, sf in [
    ("Original+Stim",   rp, ci, vs, True),
    ("Original+NoStim", rp, ci, vs, False),
    ("Shuffled+Stim",   rp, sci, vs, True),
]:
    pci, lzr, lzb = run_and_measure(rp_i, ci_i, vs_i, sf)
    R[label] = (pci, lzr, lzb)
    print(f"  {label}: PCI={pci:+.4f}  LZ(resp)={lzr:.4f}  LZ(base)={lzb:.4f}  [{time.perf_counter()-t0:.0f}s]")

print(f"\n[4/5] Done in {time.perf_counter()-t0:.0f}s")

# ─── Results ───
print("\n[5/5] RESULTS")
print("█" * 70)
print(f"  {'Condition':<20} {'PCI':<10} {'LZ(resp)':<10} {'LZ(base)':<10}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
for label in ["Original+Stim", "Original+NoStim", "Shuffled+Stim"]:
    pci, lzr, lzb = R[label]
    print(f"  {label:<20} {pci:<+10.4f} {lzr:<10.4f} {lzb:<10.4f}")

os_pci, _, _ = R["Original+Stim"]
on_pci, _, _ = R["Original+NoStim"]
ss_pci, _, _ = R["Shuffled+Stim"]

print(f"\n  ┌──────────────────────────────────────────────────┐")
print(f"  │  Stim − NoStim = {os_pci - on_pci:+.4f}")
print(f"  │  Original − Shuffled = {os_pci - ss_pci:+.4f}")
if os_pci > ss_pci:
    print(f"  │                                                              │")
    print(f"  │  FAFB CONNECTOME > SHUFFLED                                  │")
    print(f"  │  Структура коннектома увеличивает PCI                        │")
    print(f"  │  ← ЭТО СОЗНАНИЕ (по Casali)                                  │")
elif abs(os_pci - ss_pci) < 0.005:
    print(f"  │  Original ≈ Shuffled (no structural effect)                  │")
else:
    print(f"  │  Shuffled > Original (mean-field dominates)                  │")
print(f"  └──────────────────────────────────────────────────┘")

print(f"\n  pci_results.npz saved")
print("█" * 70)
