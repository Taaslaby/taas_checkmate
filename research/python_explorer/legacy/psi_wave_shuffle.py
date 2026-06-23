#!/usr/bin/env python3
"""
ψ-WAVE: ORIGINAL vs SHUFFLED CONNECTOME
Сравнение когерентности на реальной топологии vs случайной.
"""

import time, sys
import numpy as np
from taas_checkmate import step_psi_wave_full, compute_hamiltonian

DT = 0.1
HBAR = 1.0
N_STEPS = 50

print("█" * 60)
print("ψ-WAVE: FAFB vs SHUFFLED CONNECTOME")
print("█" * 60)

# ─── Load ───
print("\nLoading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr = data["row_ptr"]
col_idx = data["col_indices"]
vals = data["values"]
n = len(row_ptr) - 1
print(f"{n:,} neurons, {len(vals):,} synapses")

# ─── Init identical ψ state ───
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n).astype(np.float32)
v0 = np.cos(theta, dtype=np.float32)
u0 = np.sin(theta, dtype=np.float32)

# ─── Shuffle: preserve degree, randomise targets ───
print("\nShuffling connectome...", end=" ", flush=True)
t0 = time.perf_counter()
col_shuff = col_idx.copy()
for i in range(n):
    start = row_ptr[i]
    end = row_ptr[i + 1]
    deg = end - start
    if deg > 1:
        col_shuff[start:end] = rng.integers(0, n, deg).astype(np.int32)
print(f"{time.perf_counter()-t0:.1f}s")


def run(label, col):
    print(f"\n─── {label} ───")
    v = v0.copy()
    u = u0.copy()
    h_real = np.zeros(n, dtype=np.float32)
    h_imag = np.zeros(n, dtype=np.float32)
    print(f"  {'Step':>4}  {'Energy':>10}  {'Coherence':>10}")
    for step in range(N_STEPS):
        compute_hamiltonian(row_ptr, col, vals, v, u, h_real, h_imag)
        step_psi_wave_full(v, u, row_ptr, col, vals, DT, HBAR)
        if step % 5 == 0 or step == N_STEPS - 1:
            energy = float(np.mean(h_real**2 + h_imag**2))
            mv, mu = float(np.mean(v)), float(np.mean(u))
            coh = float(np.sqrt(mv**2 + mu**2))
            print(f"  {step:4d}  {energy:10.1f}  {coh:10.6f}")
    mv, mu = float(np.mean(v)), float(np.mean(u))
    return float(np.sqrt(mv**2 + mu**2))


coh_real = run("FAFB ORIGINAL", col_idx)
coh_shuf = run("FAFB SHUFFLED", col_shuff)

print("\n" + "█" * 60)
print(f"  FAFB original:  coherence = {coh_real:.6f}")
print(f"  FAFB shuffled:  coherence = {coh_shuf:.6f}")
print(f"  Difference:     {abs(coh_real - coh_shuf):.6f}")
print(f"  Topology matters? {'YES!' if coh_real > coh_shuf * 1.1 else 'No, mean-field'} ")
print("█" * 60)
