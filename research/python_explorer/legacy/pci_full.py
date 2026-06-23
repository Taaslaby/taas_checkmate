#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ: PCI (Perturbational Complexity Index)
=========================================================
Casali et al. 2013, Science Translational Medicine.
Полный протокол: TMS-импульс → мультиканальный EEG → Lempel-Ziv-76.

Гипотеза: ψ-wave на FAFB должна давать PCI выше для оригинального
коннектома, чем для shuffled — как в реальном мозге.
"""

import time, math, numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
)

DT = 0.025
N_STEPS = 2000        # 50 sec total
N_PRE = 400            # 10 sec baseline
N_POST = 1600          # 40 sec post-stimulus
PULSE_AMP = 3.0
PULSE_DURATION = int(0.05 / DT)  # ~2 steps
READOUT_N = 4000       # "EEG channels"
N_REPEAT = 5           # repeat each condition for statistics

print("█" * 70)
print("  PCI: PERTURBATIONAL COMPLEXITY INDEX — FULL PROTOCOL")
print("  ψ-wave on FAFB connectome")
print("  Casali et al. 2013 paradigm")
print("█" * 70)

np.random.seed(42)

# ─── Load FAFB ───
print("\n[1/6] Loading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vals):,} synapses")

# ─── Shuffled ───
print("[2/6] Building shuffled graph...", end=" ", flush=True)
shuf_col = col_idx.copy()
for i in range(n):
    start, end = row_ptr[i], row_ptr[i+1]
    if end > start:
        shuf_col[start:end] = np.random.permutation(shuf_col[start:end])
del data
print("done")

# ─── Stim & readout ───
print("[3/6] Selecting stim + readout neurons...", end=" ", flush=True)
outdeg = np.array([row_ptr[i+1] - row_ptr[i] for i in range(n)], dtype=np.int32)
hubs = np.argsort(outdeg)[::-1]
stim_idx = hubs[:200]           # TMS coil → top 200 hubs
readout_idx = np.sort(np.random.choice(n, min(READOUT_N, n), replace=False))
print(f"stim=200 hubs, readout={len(readout_idx)} channels")

# ─── LZ76 ───
def lempel_ziv_76(signal):
    """LZ76 complexity: binarize around median, count distinct substrings."""
    med = np.median(signal)
    binary = ''.join('1' if s > med else '0' for s in signal)
    n = len(binary)
    c = 1
    i, k, kk = 0, 1, 1
    while True:
        if i + k - 1 >= n:
            c -= 1
            break
        sub = binary[i:i + k]
        found = False
        for j in range(i + 1, i + kk + 1):
            if j + k - 1 >= n:
                found = True
                break
            if binary[j:j + k] == sub:
                found = True
                break
        if found:
            k += 1
        else:
            c += 1
            i = i + kk
            kk = 1
            k = 1
        if i + 1 >= n:
            break
    return c * np.log2(n) / n

def compute_pci(responses, baselines):
    """
    PCI: compare LZ complexity of response vs baseline for each channel.
    responses: (n_channels, n_timepoints) post-stimulus
    baselines: (n_channels, n_timepoints) pre-stimulus
    """
    n_ch = responses.shape[0]
    lz_resp = np.array([lempel_ziv_76(responses[ch]) for ch in range(n_ch)])
    lz_base = np.array([lempel_ziv_76(baselines[ch]) for ch in range(n_ch)])
    # PCI = (LZ_resp - LZ_base) / LZ_base (normalized difference)
    pci = (lz_resp - lz_base) / (lz_base + 1e-10)
    return np.mean(pci), np.mean(lz_resp), np.mean(lz_base)

# ─── Run ψ-wave ───
def run_and_measure(row_ptr, col_idx, vals_val, stimulate=True, seed_offset=0):
    rng = np.random.default_rng(42 + seed_offset)
    theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v = np.cos(theta)
    u_state = np.sin(theta)
    h_real = np.zeros(n_pad, dtype=np.float32)
    h_imag = np.zeros(n_pad, dtype=np.float32)
    speed = np.ones(n_pad, dtype=np.float32)
    i_ext = np.zeros(n_pad, dtype=np.float32)

    # Record per-readout time series
    n_ch = len(readout_idx)
    readout_ts = np.zeros((n_ch, N_STEPS), dtype=np.float32)
    global_eeg = np.zeros(N_STEPS, dtype=np.float32)

    for step in range(N_STEPS):
        i_ext[:] = 0.0
        if stimulate and N_PRE <= step < N_PRE + PULSE_DURATION:
            i_ext[stim_idx] = PULSE_AMP

        compute_hamiltonian(row_ptr, col_idx, vals_val, v, u_state, h_real, h_imag)
        step_psi_wave(v, u_state, h_real, h_imag, i_ext, speed, DT, 1.0)
        global_eeg[step] = float(np.mean(h_imag))
        readout_ts[:, step] = v[readout_idx]

    baseline = readout_ts[:, :N_PRE]
    response = readout_ts[:, N_PRE:]
    return baseline, response, global_eeg

# ─── Run ───
print("\n[4/6] Running ψ-wave experiments...")
print(f"  {N_STEPS} steps, {N_REPEAT} repeats per condition")
t_start = time.perf_counter()

all_results = {f"Original+Stim": [], "Original+NoStim": [], "Shuffled+Stim": []}

for cond_name, rp, ci, va, stim_flag in [
    ("Original+Stim",    row_ptr, col_idx, vals, True),
    ("Original+NoStim",  row_ptr, col_idx, vals, False),
    ("Shuffled+Stim",    row_ptr, shuf_col, vals, True),
]:
    print(f"\n  {cond_name}:")
    for rep in range(N_REPEAT):
        baseline, response, eeg = run_and_measure(rp, ci, va, stim_flag, rep)
        pci, lz_r, lz_b = compute_pci(response, baseline)
        # Also compute single-channel EEG complexity
        lz_eeg_resp = lempel_ziv_76(eeg[N_PRE:])
        lz_eeg_base = lempel_ziv_76(eeg[:N_PRE])
        all_results[cond_name].append({
            "pci": pci, "lz_resp": lz_r, "lz_base": lz_b,
            "lz_eeg_resp": lz_eeg_resp, "lz_eeg_base": lz_eeg_base,
            "eeg": eeg,
        })
        elapsed = time.perf_counter() - t_start
        pci_str = f"{pci:+.4f}" if not math.isnan(pci) else " nan "
        print(f"    rep{rep}: PCI={pci_str}  LZ(r/b)={lz_r:.3f}/{lz_b:.3f}  [{elapsed:.0f}s]")

total_time = time.perf_counter() - t_start

# ─── Results ───
print("\n" + "█" * 70)
print("  PCI RESULTS")
print("█" * 70)

for cond_name in ["Original+Stim", "Original+NoStim", "Shuffled+Stim"]:
    reps = all_results[cond_name]
    pcis = np.array([r["pci"] for r in reps])
    lz_rs = np.array([r["lz_resp"] for r in reps])
    lz_ee = np.array([r["lz_eeg_resp"] for r in reps])
    pci_valid = pcis[~np.isnan(pcis)]
    if len(pci_valid) > 0:
        print(f"\n  {cond_name}:")
        print(f"    PCI μ={np.mean(pci_valid):+.4f}  σ={np.std(pci_valid):.4f}")
        print(f"    LZ(response) μ={np.mean(lz_rs):.4f}  σ={np.std(lz_rs):.4f}")
        print(f"    LZ(global EEG) μ={np.mean(lz_ee):.4f}")

# Compare: Stim vs NoStim for original
orig_s = [r["pci"] for r in all_results["Original+Stim"] if not math.isnan(r["pci"])]
orig_n = [r["pci"] for r in all_results["Original+NoStim"] if not math.isnan(r["pci"])]
shuf_s = [r["pci"] for r in all_results["Shuffled+Stim"] if not math.isnan(r["pci"])]

print(f"\n  ┌──────────────────────────────────────────────┐")
if orig_s and orig_n:
    d = np.mean(orig_s) - np.mean(orig_n)
    print(f"  │ O.Stim − O.NoStim = {d:+.4f}  {'(evoked response ↑)' if d > 0 else '(no evoked response)'}")
if orig_s and shuf_s:
    d = np.mean(orig_s) - np.mean(shuf_s)
    win = "ORIGINAL > shuffled" if d > 0 else "shuffled > original"
    print(f"  │ Original − Shuffled = {d:+.4f}  →  {win}")
    if d > 0:
        print(f"  │                                                              │")
        print(f"  │  ТОПОЛОГИЯ FAFB ДАЁТ БОЛЕЕ СЛОЖНЫЙ ОТВЕТ                   │")
        print(f"  │  CONNECTOME MATTERS — consciousness в ψ-волне                │")
    else:
        print(f"  │  Топология не увеличивает сложность ответа                   │")
        print(f"  │  (mean-field эффект)                                         │")
print(f"  └──────────────────────────────────────────────┘")

# Show one evoked EEG response
import os
eeg_stim = all_results["Original+Stim"][0]["eeg"]
eeg_nostim = all_results["Original+NoStim"][0]["eeg"]
np.savez("pci_results.npz",
         eeg_stim=eeg_stim, eeg_nostim=eeg_nostim,
         dt=DT, n_pre=N_PRE,
         pci_orig=np.array(orig_s) if orig_s else np.array([]),
         pci_shuf=np.array(shuf_s) if shuf_s else np.array([]),
         )
print(f"\n  Saved: pci_results.npz")
print(f"  Total time: {total_time:.0f}s")
print("█" * 70)
