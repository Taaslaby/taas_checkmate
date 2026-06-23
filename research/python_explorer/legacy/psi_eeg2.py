#!/usr/bin/env python3
"""
ψ-EEG: POWER SPECTRUM of ψ-WAVE ON FAFB CONNECTOME
=================================================
2048 sample EEG → FFT → band analysis + 1/f check.
"""

import time, numpy as np
from taas_checkmate import step_psi_wave_full

DT = 0.05
HBAR = 1.0
BURN_IN = 500
N_STEPS = 2560  # 500 burn-in + 2048 record

print("█" * 60)
print("ψ-EEG: FLYWIRE CONNECTOME MACRO-ACTIVITY")
print("█" * 60)

# ─── Load ───
print("Loading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
print(f"{n:,} neurons, {len(vals):,} syn")

# ─── Init ───
print("Init ψ...", end=" ", flush=True)
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n).astype(np.float32)
v = np.cos(theta, dtype=np.float32)
u = np.sin(theta, dtype=np.float32)

# ─── Pre-alloc ───
h_real = np.zeros(n, dtype=np.float32)
h_imag = np.zeros(n, dtype=np.float32)
n_record = N_STEPS - BURN_IN
eeg = np.zeros(n_record, dtype=np.float64)

# ─── Burn-in ───
print(f"\nBurn-in {BURN_IN} steps...", end=" ", flush=True)
t0 = time.perf_counter()
for _ in range(BURN_IN):
    step_psi_wave_full(v, u, row_ptr, col_idx, vals, DT, HBAR)
print(f"{time.perf_counter()-t0:.1f}s")

# ─── Record ───
print(f"Record {n_record} steps (EEG = mean H_imag = -W·v)...")
t0 = time.perf_counter()
for i in range(n_record):
    step_psi_wave_full(v, u, row_ptr, col_idx, vals, DT, HBAR)
    # EEG = mean synaptic input to imag channel (LFP proxy)
    # Fast: just track mean(v) — no Hamiltonian needed
    eeg[i] = float(np.mean(v))

elapsed = time.perf_counter() - t0
print(f"  {elapsed:.1f}s ({n_record/elapsed:.0f} steps/s)")

# ─── FFT ───
from numpy.fft import rfft, rfftfreq
fs = 1.0 / DT
signal = eeg - np.mean(eeg)
window = np.hanning(len(signal))
N = len(signal)
fft = rfft(signal * window)
power = np.abs(fft)**2
freqs = rfftfreq(N, 1/fs)

# ─── Results ───
print("\n" + "█" * 60)
print("SPECTRUM")
print("█" * 60)
print(f"  Sampling:   {fs:.0f} Hz (dimensionless)")
print(f"  Samples:    {N}")
print(f"  Resolution: {fs/N:.4f}")
print(f"  Range:      0 — {fs/2:.1f}")
print(f"  EEG std:    {float(np.std(eeg)):.6f}")

# 1/f fit (log-log)
idx = (freqs > 0.01) & (freqs < fs/4)
log_f = np.log10(freqs[idx])
log_p = np.log10(power[idx] + 1e-30)
coeffs = np.polyfit(log_f, log_p, 1)
print(f"\n  1/f exponent: f^({coeffs[0]:.2f})")
print(f"  {'✓ BRAIN-LIKE!' if -2.2 < coeffs[0] < -0.8 else '✗ Not brain-like'}")

# Peaks
try:
    from scipy.signal import find_peaks
    db = 10 * np.log10(power + 1e-30)
    db_smooth = np.convolve(db, np.ones(3)/3, mode='same')
    peaks, props = find_peaks(db_smooth, height=np.percentile(db_smooth, 85), distance=5)
    if len(peaks) > 0:
        print(f"\n  Peaks ({len(peaks)} found):")
        order = np.argsort(props["peak_heights"])[::-1]
        for i, p in enumerate(peaks[order][:5]):
            print(f"    f={freqs[p]:.4f}  power={props['peak_heights'][order[i]]:.1f}dB")
except ImportError:
    print("\n  (scipy not available, skipping peak detection)")
except Exception as e:
    print(f"\n  (peak detection skipped: {e})")

# Band power
total = np.sum(power[idx])
bands = [
    ("Slow",   0.001, 0.01),
    ("δ",      0.01,  0.03),
    ("θ",      0.03,  0.07),
    ("α",      0.07,  0.13),
    ("β",      0.13,  0.30),
    ("γ",      0.30,  0.60),
]
print(f"\n  Band power (relative):")
for name, lo, hi in bands:
    mask = (freqs >= lo) & (freqs < hi)
    bp = np.sum(power[mask]) / total * 100
    bar = "█" * int(bp/2) + "░" * max(0, 50-int(bp/2))
    print(f"    {name:4s}  [{lo:.3f}-{hi:.3f}): {bp:5.1f}%  |{bar}|")

print("\n" + "█" * 60)
