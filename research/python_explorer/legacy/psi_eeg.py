#!/usr/bin/env python3
"""
ψ-EEG: МАКРОАКТИВНОСТЬ МОЗГА МУХИ ЧЕРЕЗ ψ-WAVE
=============================================
Записываем LFP-подобный сигнал с 139K ψ-нейронов,
ищем brain-like oscillations.
"""

import time, sys
import numpy as np
from taas_checkmate import step_psi_wave_full

# ─── Параметры ────────────────────────────────────────────
DT = 0.05
HBAR = 1.0
N_STEPS = 2048        # быстрый прогон, потом можно 8192
RECORD_CHUNK = 16      # записываем каждые 16 шагов (экономия места)

print("█" * 60)
print("██  ψ-EEG: MACRO-ACTIVITY OF FLYWIRE CONNECTOME")
print("█" * 60)

# ─── Загрузка ───
print("\nLoading FAFB connectome...", end=" ", flush=True)
t0 = time.perf_counter()
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
print(f"{n:,} neurons, {len(vals):,} synapses ({time.perf_counter()-t0:.1f}s)")

# ─── Инициализация ───
print("Initializing ψ-state...", end=" ", flush=True)
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n).astype(np.float32)
v = np.cos(theta, dtype=np.float32)
u = np.sin(theta, dtype=np.float32)
print("done")

# ─── Симуляция с записью ───
n_record = N_STEPS // RECORD_CHUNK
print(f"\nSimulating ψ-wave ({N_STEPS} steps, recording {n_record} frames)...")

eeg = np.zeros(n_record, dtype=np.float64)
energy = np.zeros(n_record, dtype=np.float64)
coherence = np.zeros(n_record, dtype=np.float64)
times = np.arange(n_record) * (DT * RECORD_CHUNK)

t_start = time.perf_counter()
record_i = 0

for step in range(N_STEPS):
    step_psi_wave_full(v, u, row_ptr, col_idx, vals, DT, HBAR)

    if step % RECORD_CHUNK == 0:
        # EEG proxy = mean(v) — средний мембранный потенциал (LFP)
        eeg[record_i] = float(np.mean(v))
        # Energy trace
        energy[record_i] = float(np.mean(v**2 + u**2))
        # Coherence
        mv, mu = float(np.mean(v)), float(np.mean(u))
        coherence[record_i] = float(np.sqrt(mv**2 + mu**2))
        record_i += 1

    if step % 1000 == 0 and step > 0:
        elapsed = time.perf_counter() - t_start
        print(f"  step {step:5d}/{N_STEPS}  |  {elapsed:.0f}s")

total = time.perf_counter() - t_start
print(f"  done in {total:.1f}s  ({N_STEPS/total:.0f} steps/s)")

# ─── Анализ ───
print("\n" + "█" * 60)
print("SPECTRAL ANALYSIS")
print("█" * 60)

# Remove DC offset + detrend
eeg_detrend = eeg - np.mean(eeg)

# FFT
from numpy.fft import rfft, rfftfreq
N = len(eeg_detrend)
fs = 1.0 / (DT * RECORD_CHUNK)  # sampling rate in dimensionless units
freqs = rfftfreq(N, 1.0/fs)
fft = rfft(eeg_detrend * np.hanning(N))
power = np.abs(fft)**2
db = 10 * np.log10(power + 1e-30)

print(f"\n  Sampling rate:  {fs:.1f} (dimensionless)")
print(f"  Frequency range: 0 — {fs/2:.1f}")
print(f"  FFT bins:    {len(freqs)}")
print(f"  Signal std:  {float(np.std(eeg)):.6f}")

# Dominant frequencies (find peaks)
from scipy.signal import find_peaks
peaks, props = find_peaks(db, height=np.percentile(db, 90), distance=5)
if len(peaks) > 0:
    # Sort by power
    order = np.argsort(props["peak_heights"])[::-1]
    peaks = peaks[order]
    print(f"\n  Dominant frequencies:")
    for i, p in enumerate(peaks[:10]):
        print(f"    {i+1:2d}.  {freqs[p]:8.3f}  (power={props['peak_heights'][order[i]]:.1f} dB)")
else:
    print("  No dominant peaks found")

# 1/f fit
low_idx = np.searchsorted(freqs, 0.1)
high_idx = np.searchsorted(freqs, fs/2)
log_f = np.log10(freqs[low_idx:high_idx])
log_p = np.log10(power[low_idx:high_idx] + 1e-30)
if len(log_f) > 10:
    coeffs = np.polyfit(log_f, log_p, 1)
    print(f"\n  1/f exponent:  f^({coeffs[0]:.2f})  (ideal EEG: f^(-1) to f^(-2))")
    print(f"  {'→ Brain-like 1/f noise!' if -2.5 < coeffs[0] < -0.5 else '→ Not brain-like'}")

# Band power ratios
print(f"\n  Band analysis:")
bands = [
    ("Slow", 0.0, 0.05),
    ("Theta", 0.05, 0.15),
    ("Alpha", 0.15, 0.3),
    ("Beta",  0.3, 0.6),
    ("Gamma", 0.6, 1.5),
    ("Fast",  1.5, fs/2),
]
total_power = np.sum(power)
for name, lo, hi in bands:
    mask = (freqs >= lo) & (freqs < hi)
    bp = np.sum(power[mask])
    print(f"    {name:6s}  [{lo:.2f}-{hi:.2f}):  {bp/total_power*100:5.1f}%")

# Time-domain stats
print(f"\n  Time-domain:")
print(f"    Mean EEG:   {float(np.mean(eeg)):.6f}")
print(f"    Std EEG:    {float(np.std(eeg)):.6f}")
print(f"    Final coherence: {coherence[-1]:.6f}")
print(f"    Coherence range: [{coherence.min():.6f}, {coherence.max():.6f}]")

# Check for oscillations (zero crossings = frequency proxy)
zc = np.sum(np.diff(np.signbit(eeg_detrend)))
print(f"    Zero crossings: {zc}  (mean freq: {zc/(2*N*DT*RECORD_CHUNK):.3f})")

print("\n" + "█" * 60)
print("██  ψ-EEG ANALYSIS COMPLETE")
print("█" * 60)
