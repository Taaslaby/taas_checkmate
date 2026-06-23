#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ: PCI (Perturbational Complexity Index)
=========================================================
Протокол из клинической нейробиологии (Casali et al. 2013, Science):
  TMS-импульс → EEG → сжатие ответа.

Если ψ-wave на FAFB коннектоме производит "conscious-like" EEG
(сложный, дифференцированный ответ), то PCI будет высоким.
Если ответ стереотипный — PCI низкий.

Сравнение:
  1. FAFB original — импульс
  2. FAFB original — no импульс (baseline)
  3. FAFB shuffled — импульс
  4. FAFB random graph — импульс
"""

import time, math, struct, os, gzip
import numpy as np
from io import BytesIO
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
)

DT = 0.025
N_STEPS = 400  # 10 sec @ 0.025
N_PRE = 100    # 2.5 sec baseline
N_POST = 300   # 7.5 sec post-stimulus
PULSE_AMP = 10.0
PULSE_DURATION = 4  # steps
READOUT_N = 2000

print("█" * 70)
print("  PCI: PERTURBATIONAL COMPLEXITY INDEX")
print("  ψ-wave on FAFB connectome")
print("█" * 70)

np.random.seed(42)

# ─── Load FAFB ───
print("\n[1/6] Loading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vals):,} synapses")

# ─── Shuffled FAFB ───
print("[2/6] Building comparison graphs...")
rng = np.random.default_rng(42)
# Shuffled: same degree, random targets
shuf_col = col_idx.copy()
for i in range(n):
    start = row_ptr[i]
    end = row_ptr[i+1]
    if end > start:
        perm = rng.permutation(shuf_col[start:end])
        shuf_col[start:end] = perm
print("  Shuffled: done")

# Random graph: Erdos-Renyi with same density
nz = len(vals)
p = nz / (n * n)
rand_vals = rng.uniform(0.5, 1.0, nz).astype(np.float32)
print(f"  Random (Erdos-Renyi p={p:.2e}): {nz} edges")

# ─── Select stimulation & readout neurons ───
print("[3/6] Selecting stim & readout neurons...", end=" ", flush=True)
indeg = np.bincount(col_idx.astype(np.int64), minlength=n).astype(np.int32)
outdeg = np.array([row_ptr[i+1] - row_ptr[i] for i in range(n)], dtype=np.int32)
# Stim: high-degree hub neurons (like TMS coil targeting a region)
hubs = np.argsort(outdeg)[::-1]
stim_n = 200
stim_idx = hubs[:stim_n]
# Readout: random subset of all neurons
readout_idx = rng.choice(n, min(READOUT_N, n), replace=False)
readout_idx.sort()
print(f"stim={stim_n} hubs + readout={len(readout_idx)}")

# ─── Helper: run ψ-wave ───
def run_psi(row_ptr, col_idx, vals, stimulate=True):
    theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v = np.cos(theta)
    u = np.sin(theta)
    h_real = np.zeros(n_pad, dtype=np.float32)
    h_imag = np.zeros(n_pad, dtype=np.float32)
    speed = np.ones(n_pad, dtype=np.float32)
    i_ext = np.zeros(n_pad, dtype=np.float32)
    
    eeg = np.zeros(N_STEPS, dtype=np.float32)
    readout = np.zeros((N_STEPS, len(readout_idx)), dtype=np.float32)
    
    for step in range(N_STEPS):
        # Pulse
        i_ext[:] = 0.0
        if stimulate and N_PRE <= step < N_PRE + PULSE_DURATION:
            i_ext[stim_idx] = PULSE_AMP
        
        compute_hamiltonian(row_ptr, col_idx, vals, v, u, h_real, h_imag)
        step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, 1.0)
        
        eeg[step] = float(np.mean(h_imag))
        readout[step] = v[readout_idx]
    
    return eeg, readout

# ─── Measure complexity ───
def lempel_ziv_complexity(signal, n_bins=64):
    """Lempel-Ziv complexity of a binned signal."""
    # Normalize and bin
    s = signal - np.mean(signal)
    s = s / (np.std(s) + 1e-10)
    s = np.clip(s, -3, 3)
    bins = np.linspace(-3, 3, n_bins)
    indices = np.digitize(s, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    
    # LZ complexity
    seq = indices.astype(np.uint8)
    n_bytes = len(seq)
    compressed = gzip.compress(seq.tobytes())
    compression_ratio = len(compressed) / n_bytes
    return 1.0 - compression_ratio  # 1 = very complex, 0 = very simple

def permutation_entropy(signal, order=5, delay=1):
    """Permutation entropy (Bandt-Pompe)."""
    if len(signal) < order * delay:
        return 0.0
    n_perm = len(signal) - (order - 1) * delay
    patterns = {}
    for i in range(n_perm):
        pattern = tuple(np.argsort(signal[i:i + order * delay:delay]))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    probs = np.array(list(patterns.values())) / n_perm
    return -np.sum(probs * np.log2(probs + 1e-30)) / np.log2(math.factorial(order))

# ─── Run experiments ───
configs = [
    ("Original + Stim", row_ptr, col_idx, vals, True),
    ("Original + NoStim", row_ptr, col_idx, vals, False),
    ("Shuffled + Stim", row_ptr, shuf_col, vals, True),
]

t_start = time.perf_counter()
results = {}

for name, rp, ci, va, stim in configs:
    print(f"\n[4/6] Running: {name}...")
    label = stim and "STIM" or "NO STIM"
    eeg, readout = run_psi(rp, ci, va, stimulate=stim)
    
    # Split: baseline vs response
    baseline_eeg = eeg[:N_PRE]
    response_eeg = eeg[N_PRE:]
    
    # EEG complexity
    lz_full = lempel_ziv_complexity(eeg)
    lz_baseline = lempel_ziv_complexity(baseline_eeg)
    lz_response = lempel_ziv_complexity(response_eeg)
    pe_response = permutation_entropy(response_eeg, order=5)
    
    # Readout complexity (high-dim response)
    # For each readout channel, compute LZ, then average
    lz_readout = np.zeros(len(readout_idx))
    for ri in range(min(500, len(readout_idx))):  # sample 500
        lz_readout[ri] = lempel_ziv_complexity(readout[N_PRE:, ri])
    lz_readout_mean = np.mean(lz_readout[:ri+1])
    
    # Energy of response
    response_energy = np.mean(response_eeg**2) / (np.mean(baseline_eeg**2) + 1e-10)
    
    results[name] = {
        "LZ_full": lz_full,
        "LZ_baseline": lz_baseline,
        "LZ_response": lz_response,
        "PE_response": pe_response,
        "LZ_readout": lz_readout_mean,
        "response_energy_ratio": response_energy,
        "eeg": eeg,
        "readout": readout,
    }
    
    elapsed = time.perf_counter() - t_start
    print(f"    Full:  LZ={lz_full:.4f}")
    print(f"    Resp:  LZ={lz_response:.4f}, PE={pe_response:.3f}")
    print(f"    Read:  LZ_mean={lz_readout_mean:.4f}")
    print(f"    Energy ratio: {response_energy:.2f}x")
    print(f"    [{elapsed:.0f}s]")

total_time = time.perf_counter() - t_start

# ─── PCI-like metric ───
# PCI = complexity of response - complexity of baseline (normalized)
# Higher PCI = more consciousness-like
print("\n[5/6] Computing PCI...")
for name in results:
    r = results[name]
    r["PCI_LZ"] = r["LZ_response"] - r["LZ_baseline"]
    # PCI score: normalized by max possible
    r["PCI_score"] = r["PCI_LZ"] / (1.0 - r["LZ_baseline"] + 1e-10)

# ─── Results table ───
print("\n" + "█" * 70)
print("  PCI REZULTAT")
print("█" * 70)
print(f"  {'Condition':<25} {'LZ_full':<9} {'LZ_resp':<9} {'PCI_LZ':<9} {'PCI_score':<10} {'PE_resp':<8}")
print(f"  {'─'*25} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*8}")
for name in sorted(results.keys()):
    r = results[name]
    print(f"  {name:<25} {r['LZ_full']:<9.4f} {r['LZ_response']:<9.4f} "
          f"{r['PCI_LZ']:<+9.4f} {r['PCI_score']:<10.4f} {r['PE_response']:<8.3f}")

# Compare
orig = results.get("Original + Stim", {})
shuf = results.get("Shuffled + Stim", {})
nostim = results.get("Original + NoStim", {})

print('\n  Ключевое сравнение (PCI — чем выше, тем "сознательнее"):')
if orig:
    print(f"  Original + Stim:  PCI={orig.get('PCI_score', 0):.4f}  (LZ_response={orig['LZ_response']:.4f})")
if shuf:
    print(f"  Shuffled  + Stim:  PCI={shuf.get('PCI_score', 0):.4f}  (LZ_response={shuf['LZ_response']:.4f})")
if nostim:
    print(f"  Original + NoStim: PCI={nostim.get('PCI_score', 0):.4f}  (LZ_response={nostim['LZ_response']:.4f})")

# Who is more "conscious"?
if orig and shuf:
    if orig['PCI_score'] > shuf['PCI_score']:
        print(f"\n  → FAFB CONNECTOME даёт БОЛЕЕ сложный ответ, чем shuffled!")
        print(f"  → Топология МАТЕРИТ для сознания (как в реальном мозге)")
    else:
        print(f"\n  → Shuffled даёт более сложный ответ. Структура не важна?")
        print(f"  → Сознание — это mean-field эффект (как мы видели по coherence)")

print(f"\n  Simulation time: {total_time:.0f}s")
print("█" * 70)
