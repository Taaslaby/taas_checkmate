#!/usr/bin/env python3
"""
PCI: Perturbational Complexity Index — финальный
Casali et al. 2013: TMS → spatiotemporal response compression.

Гипотеза: ORIGINAL FAFB даёт более сложный (менее сжатый) response
чем SHUFFLED. Топология коннектома = сознание.
"""

import time, numpy as np, gzip
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_BASE = 1000       # 25 sec baseline  
N_RESP = 1000       # 25 sec response
N_STEPS = N_BASE + N_RESP
PULSE_AMP = 10000.0
PULSE_LEN = 5
READOUT_N = 2000
SPEED = 1.0
HBAR = 1.0

print("█" * 70)
print("  PCI — ФИНАЛ: SPATIOTEMPORAL CR AFTER TMS")
print("  FAFB Original vs Shuffled")
print("█" * 70)
np.random.seed(42)

# ─── Load & Shuffle ───
print("\n[1/4] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
npad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vs):,} synapses")

print("[2/4] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])

# Stim (hubs) & readout (random)
outdeg = np.array([rp[i+1]-rp[i] for i in range(n)], dtype=np.int32)
stim = np.sort(np.argsort(outdeg)[::-1][:500])
rdi = np.sort(np.random.choice(n, READOUT_N, replace=False))
nch = len(rdi)
print("done")

# ─── Complexity ───
def cr_matrix(mat):
    """Spatiotemporal compression ratio: lower = more structured."""
    binary = np.zeros_like(mat, dtype=np.uint8)
    for ch in range(mat.shape[0]):
        m = np.median(mat[ch])
        binary[ch] = (mat[ch] > m).astype(np.uint8)
    bits = np.packbits(binary.ravel())
    return len(gzip.compress(bits)) / max(len(bits), 1)

# ─── Run ───
def run(label, rp_i, ci_i, vs_i):
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2*np.pi, npad).astype(np.float32)
    v = np.cos(th); u = np.sin(th)
    ie = np.zeros(npad, dtype=np.float32)
    spd = np.full(npad, SPEED, dtype=np.float32)
    hr = np.zeros(npad, dtype=np.float32)
    hi = np.zeros(npad, dtype=np.float32)
    ts = np.zeros((nch, N_STEPS), dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        ie[:] = 0.0
        if N_BASE <= step < N_BASE + PULSE_LEN:
            ie[stim] = PULSE_AMP
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, HBAR)
        ts[:, step] = v[rdi]
        if (step+1) % 500 == 0:
            print(f"  {label}: [{step+1}/{N_STEPS}] [{time.perf_counter()-t0:.0f}s]")

    base, resp = ts[:, :N_BASE], ts[:, N_BASE:]
    cr_base = cr_matrix(base)
    cr_resp = cr_matrix(resp)
    return {"cr_base": cr_base, "cr_resp": cr_resp, "ts": ts, "time": time.perf_counter()-t0}

# ─── Run ───
print("\n[3/4] Running...")
R = {}
for label, rp_i, ci_i, vs_i in [("Original", rp, ci, vs), ("Shuffled", rp, sci, vs)]:
    print(f"\n  {label}:")
    R[label] = run(label, rp_i, ci_i, vs_i)
    print(f"  Done in {R[label]['time']:.0f}s")
    print(f"  CR(base)={R[label]['cr_base']:.4f}  CR(resp)={R[label]['cr_resp']:.4f}  Δ={R[label]['cr_resp']-R[label]['cr_base']:+.4f}")

# ─── Results ───
print("\n[4/4] PCI RESULTS")
print("█" * 70)
print(f"  {'Condition':<15} {'CR(base)':<12} {'CR(resp)':<12} {'ΔCR':<12} {'PCI(ΔCR/base)':<14}")
print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")

for label in ["Original", "Shuffled"]:
    r = R[label]
    dcr = r["cr_resp"] - r["cr_base"]
    pci = dcr / (r["cr_base"] + 1e-10)
    print(f"  {label:<15} {r['cr_base']:<12.4f} {r['cr_resp']:<12.4f} {dcr:<+12.4f} {pci:<+14.4f}")

o_resp = R["Original"]["cr_resp"]
s_resp = R["Shuffled"]["cr_resp"]
print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
print(f"  │  Высокий CR = много spatiotemporal паттернов = сложный ответ     │")
print(f"  │  CR(Original) = {o_resp:.4f}  CR(Shuffled) = {s_resp:.4f}                      │")
if o_resp > s_resp + 0.01:
    print(f"  │                                                              │")
    print(f"  │  ORIGINAL > SHUFFLED: FAFB даёт БОЛЕЕ сложный ответ          │")
    print(f"  │  ← CONNECTOME MATTERS FOR CONSCIOUSNESS (Casali)             │")
    print(f"  │  Biologically plausible topology → richer dynamics           │")
elif o_resp > s_resp:
    print(f"  │  ORIGINAL ≈ SHUFFLED (weak difference)                        │")
    print(f"  │  Topology has modest effect on spatiotemporal complexity      │")
else:
    print(f"  │  SHUFFLED > ORIGINAL (mean-field dominates)                   │")
    print(f"  │  Random graph produces more complex responses                 │")
print(f"  └──────────────────────────────────────────────────────────────────┘")

np.savez("pci_final.npz", cr_orig=R["Original"]["cr_resp"], cr_shuf=R["Shuffled"]["cr_resp"],
         ts_orig=R["Original"]["ts"], ts_shuf=R["Shuffled"]["ts"])
print(f"\n  Saved: pci_final.npz")
print("█" * 70)
