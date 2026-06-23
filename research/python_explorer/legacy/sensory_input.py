#!/usr/bin/env python3
"""
СЕНСОРНЫЙ ВХОД В МОЗГ МУХИ ЧЕРЕЗ ψ-WAVE
========================================
Находим сенсорные нейроны (нулевой indegree), подаём
осциллирующий стимул, смотрим как ψ-поле откликается.

Эксперимент:
  1. Тишина (500 steps) — baseline
  2. Стимул (500 steps) — визуальный oscillation
  3. Последействие (500 steps) — что осталось
"""

import time, numpy as np
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.05
HBAR = 1.0
N_BASELINE = 500
N_STIM = 500
N_AFTER = 500
N_TOTAL = N_BASELINE + N_STIM + N_AFTER

print("█" * 60)
print("СЕНСОРНЫЙ ВХОД В FAFB CONNECTOME")
print("█" * 60)

# ─── Load ───
print("\nLoading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
print(f"{n:,} neurons, {len(vals):,} syn")

# ─── Find sensory neurons (indegree = 0 or near-0) ───
print("Finding sensory neurons (indegree ≤ 1)...", end=" ", flush=True)
t0 = time.perf_counter()
# Count incoming synapses per neuron (vectorized)
indeg = np.bincount(col_idx.astype(np.int64), minlength=n).astype(np.int32)
# Also find neurons with no outdegree or extremely low
outdeg = np.array([row_ptr[i+1] - row_ptr[i] for i in range(n)], dtype=np.int32)
# Sensory: indegree < 2 (no/few inputs = sensor), with at least some output
sensory = np.where((indeg <= 1) & (outdeg >= 5))[0]
print(f"{len(sensory):,} candidates ({time.perf_counter()-t0:.1f}s)")
print(f"  First 10 sensory neuron IDs: {sensory[:10].tolist()}")

# Pick N_SENSORY neurons to drive
N_SENSORY = min(50, len(sensory))
if N_SENSORY == 0:
    # Fallback: pick random neurons
    sensory = np.arange(1000, 1000 + N_SENSORY)
    print("  (no indeg=0 found, using random)")
else:
    sensory = sensory[:N_SENSORY]
    print(f"  Driving sensor IDs: {sensory[:5].tolist()}...")

# ─── Init ψ-state (pad to ×8 for SIMD) ───
print("\nInit ψ...", end=" ", flush=True)
n_pad = (n + 7) // 8 * 8
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
v = np.cos(theta, dtype=np.float32)
u = np.sin(theta, dtype=np.float32)
h_real = np.zeros(n_pad, dtype=np.float32)
h_imag = np.zeros(n_pad, dtype=np.float32)
speed = np.ones(n_pad, dtype=np.float32)
i_ext = np.zeros(n_pad, dtype=np.float32)
print(f"done (padded {n} → {n_pad})")

# Stimulus parameters
STIM_FREQ = 0.2   # oscillation frequency (dimensionless)
STIM_AMP = 5.0

# ─── Run ───
print(f"\nRunning ψ-wave: baseline {N_BASELINE} + stim {N_STIM} + after {N_AFTER}")
print()

record_every = 5
n_record = N_TOTAL // record_every
eeg = np.full(n_record, np.nan)
coh = np.full(n_record, np.nan)
time_ax = np.arange(n_record) * DT * record_every

# Mark phases
base_end = N_BASELINE // record_every
stim_end = (N_BASELINE + N_STIM) // record_every

t_start = time.perf_counter()
record_i = 0

for step in range(N_TOTAL):
    # Compute Hamiltonian
    compute_hamiltonian(row_ptr, col_idx, vals, v, u, h_real, h_imag)
    
    # Inject stimulus
    if step >= N_BASELINE and step < N_BASELINE + N_STIM:
        t = step * DT
        i_ext[:] = 0.0
        stim = STIM_AMP * np.sin(2 * np.pi * STIM_FREQ * t)
        i_ext[sensory] = stim
    else:
        i_ext[:] = 0.0
    
    # ψ-wave step
    step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, HBAR)
    
    if step % record_every == 0:
        # EEG = mean of H_imag (weighted synaptic input)
        mv, mu = float(np.mean(v)), float(np.mean(u))
        coh[record_i] = float(np.sqrt(mv**2 + mu**2))
        # Local EEG = response near sensory area
        local = float(np.mean(h_imag[sensory]) if len(sensory) > 0 else 0)
        eeg[record_i] = local
        record_i += 1

    if step % 500 == 0 and step > 0:
        print(f"  step {step:5d}/{N_TOTAL}  ({time.perf_counter()-t_start:.0f}s)")

total = time.perf_counter() - t_start
print(f"  done in {total:.1f}s")

# ─── Analysis ───
print("\n" + "█" * 60)
print("EVOKED RESPONSE ANALYSIS")
print("█" * 60)

# Evoked response potential (ERP)
erp_baseline = np.nanmean(eeg[:base_end])
erp_stim = np.nanmean(eeg[base_end:stim_end])
erp_after = np.nanmean(eeg[stim_end:])
print(f"\n  Mean local EEG:")
print(f"    Baseline (silence): {erp_baseline:.4f}")
print(f"    Stimulus:           {erp_stim:.4f}")
print(f"    After:              {erp_after:.4f}")
print(f"    Response Δ:         {erp_stim - erp_baseline:+.4f}")

# Coherence change
coh_base = np.nanmean(coh[:base_end])
coh_stim = np.nanmean(coh[base_end:stim_end])
coh_aft = np.nanmean(coh[stim_end:])
print(f"\n  Global coherence:")
print(f"    Baseline:           {coh_base:.6f}")
print(f"    Stimulus:           {coh_stim:.6f}")
print(f"    After:              {coh_aft:.6f}")
print(f"    Coherence Δ:        {coh_stim - coh_base:+.6f}")

# FFT of stimulus period vs baseline
from numpy.fft import rfft, rfftfreq
fs = 1.0 / (DT * record_every)

stim_signal = eeg[base_end:stim_end] - np.nanmean(eeg[base_end:stim_end])
base_signal = eeg[:base_end] - np.nanmean(eeg[:base_end])

# Remove NaNs
stim_signal = stim_signal[~np.isnan(stim_signal)]
base_signal = base_signal[~np.isnan(base_signal)]

if len(stim_signal) >= 16:
    stim_fft = np.abs(rfft(stim_signal * np.hanning(len(stim_signal))))**2
    base_fft = np.abs(rfft(base_signal * np.hanning(len(base_signal))))**2
    freqs = rfftfreq(len(stim_signal), 1/fs)
    
    # Find stimulus frequency in spectrum
    idx = np.argmin(np.abs(freqs - STIM_FREQ))
    print(f"\n  Power at stimulus freq ({STIM_FREQ:.2f}):")
    print(f"    Baseline: {base_fft[idx]:.2e}")
    print(f"    Stimulus: {stim_fft[idx]:.2e}")
    print(f"    Ratio:    {stim_fft[idx]/(base_fft[idx]+1e-30):.1f}x")
    
    # Check for harmonics
    print(f"\n  Frequency response:")
    for h in [1, 2, 3]:
        f = STIM_FREQ * h
        if f < fs/2:
            idx_h = np.argmin(np.abs(freqs - f))
            ratio = stim_fft[idx_h] / (base_fft[idx_h] + 1e-30)
            if ratio > 2:
                print(f"    f={f:.2f} (×{h}): {ratio:.1f}x power — HARMONIC!")
            else:
                print(f"    f={f:.2f} (×{h}): {ratio:.1f}x — no response")

print(f"\n  Entrainment check:")
print(f"    Stim freq: {STIM_FREQ:.3f}")
print(f"    Natural freq (from Peaks): — see run above")

print("\n" + "█" * 60)
print("DONE — sensory stimulus injected into FAFB ψ-wave")
print("█" * 60)
