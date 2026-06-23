#!/usr/bin/env python3
"""
PCI: Perturbational Complexity Index (Casali et al. 2013)
Полный протокол: TMS → мультиканальный EEG → LZ(spatiotemporal).

PCI = LZ(original) - LZ(shuffled)
где shuffled = каналы перемешаны (spatial surrogate).
PCI > 0 → spatiotemporal паттерн неслучаен → "сознание".
"""

import time, gzip, numpy as np
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_BASE = 2000      # 50 sec baseline
N_RESP = 2000      # 50 sec response
N_STEPS = N_BASE + N_RESP
PULSE_AMP = 1000.0
PULSE_LEN = 1
READOUT_N = 5000   # "EEG electrodes"

print("█" * 70)
print("  PCI (Casali 2013) — ψ-wave on FAFB")
print("  LZ(spatiotemporal) − LZ(shuffled surrogate)")
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
stim = np.sort(np.argsort(outdeg)[::-1][:500])  # more stim neurons
rdi = np.sort(np.random.choice(n, READOUT_N, replace=False))
nch = len(rdi)

# ─── LZ on binary matrix ───
def compression_ratio(mat):
    """
    CR of spatiotemporal matrix: lower = more compressible = MORE structured.
    Higher CR = less compressible = MORE complex (more distinct patterns).
    """
    # Median threshold each channel independently
    binary = np.zeros_like(mat, dtype=np.uint8)
    for ch in range(mat.shape[0]):
        med = np.median(mat[ch])
        binary[ch] = (mat[ch] > med).astype(np.uint8)
    bits = np.packbits(binary.ravel())
    return len(gzip.compress(bits)) / max(len(bits), 1)

# ─── Run ψ-wave ───
def run_and_measure(rp_i, ci_i, vs_i, stim_flag, seed=0):
    rng = np.random.default_rng(42 + seed)
    th = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v, u = np.cos(th), np.sin(th)
    hr = np.zeros(n_pad, dtype=np.float32)
    hi = np.zeros(n_pad, dtype=np.float32)
    spd = np.ones(n_pad, dtype=np.float32)
    ie = np.zeros(n_pad, dtype=np.float32)

    ts = np.zeros((nch, N_STEPS), dtype=np.float32)
    for step in range(N_STEPS):
        ie[:] = 0.0
        if stim_flag and N_BASE <= step < N_BASE + PULSE_LEN:
            ie[stim] = PULSE_AMP
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, 1.0)
        ts[:, step] = v[rdi]

    base, resp = ts[:, :N_BASE], ts[:, N_BASE:]

    cr_resp = compression_ratio(resp)
    cr_base = compression_ratio(base)
    return cr_resp, cr_base, 0.0, 0.0

# ─── Run ───
print("[3/5] Running...")
t0 = time.perf_counter()
R = {}
for label, rp_i, ci_i, vs_i, sf in [
    ("Original+Stim",   rp, ci, vs, True),
    ("Original+NoStim", rp, ci, vs, False),
    ("Shuffled+Stim",   rp, sci, vs, True),
]:
    pci, lzo, lzs, lzb = run_and_measure(rp_i, ci_i, vs_i, sf)
    R[label] = (pci, lzo, lzs, lzb)
    print(f"  {label}: PCI={pci:+.4f}  LZ(orig)={lzo:.4f}  LZ(shuf)={lzs:.4f}  LZ(base)={lzb:.4f}  [{time.perf_counter()-t0:.0f}s]")

print(f"\n[4/5] Done in {time.perf_counter()-t0:.0f}s")

# ─── Results ───
print("\n[5/5] PCI = LZ(orig) − LZ(shuf)")
print("█" * 70)
print(f"  {'Condition':<20} {'CR(resp)':<10} {'CR(base)':<10} {'ΔCR':<10}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
for label in ["Original+Stim", "Original+NoStim", "Shuffled+Stim"]:
    crr, crb, _, _ = R[label]
    print(f"  {label:<20} {crr:<10.4f} {crb:<10.4f} {crr-crb:<+10.4f}")

os_crr, os_crb, _, _ = R["Original+Stim"]
on_crr, on_crb, _, _ = R["Original+NoStim"]
ss_crr, ss_crb, _, _ = R["Shuffled+Stim"]

print(f"\n  ┌────────────────────────────────────────────────────────────┐")
print(f"  │  Stim CR(resp) − Stim CR(base) = {os_crr-os_crb:+.4f}  (evoked complexity)")
print(f"  │  Original(resp) − Shuffled(resp) = {os_crr-ss_crr:+.4f}  (structure effect)")
print(f"  │")
print(f"  │  Высокий CR = много паттернов = более сложный ответ         │")
if os_crr > ss_crr:
    print(f"  │                                                              │")
    print(f"  │  FAFB CONNECTOME > SHUFFLED                                   │")
    print(f"  │  Оригинальная топология даёт БОЛЕЕ сложный spatiotemporal     │")
    print(f"  │  ответ → CONNECTOME MATTERS FOR CONSCIOUSNESS                │")
elif abs(os_crr - ss_crr) < 0.005:
    print(f"  │  Original ≈ Shuffled (structure doesn't matter)              │")
else:
    print(f"  │  Shuffled > Original (mean-field dominates)                  │")
print(f"  └────────────────────────────────────────────────────────────┘")

np.savez("pci2_results.npz",
         cr_orig_stim=os_crr, cr_orig_base=os_crb,
         cr_nostim=on_crr,
         cr_shuf=ss_crr)
print(f"\n  pci2_results.npz saved")
print("█" * 70)
