#!/usr/bin/env python3
"""
T3 → FAFB CONNECTOME → ψ-WAVE
==============================
T3 осцилляторы (Kuramoto + Stuart-Landau) гонят сигнал
в сенсорные нейроны мухи, ψ-wave разносит по всему мозгу.
"""

import time, numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
    PyTriOscillator,
)

# ─── Параметры ────────────────────────────────────────────
DT = 0.05
HBAR = 1.0
N_T3_OSC = 32
T3_FREQ_MIN = 0.5
T3_FREQ_MAX = 2.0
N_BASELINE = 300
N_T3_DRIVE = 500
N_AFTER = 300
N_TOTAL = N_BASELINE + N_T3_DRIVE + N_AFTER
STIM_AMP = 5.0

print("█" * 60)
print("T3 → FAFB CONNECTOME → ψ-WAVE RESPONSE")
print("█" * 60)

# ─── Load FAFB ───
print("\nLoading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons")

# ─── Find sensory neurons ───
print("Finding sensory neurons...", end=" ", flush=True)
indeg = np.bincount(col_idx.astype(np.int64), minlength=n).astype(np.int32)
outdeg = np.array([row_ptr[i+1] - row_ptr[i] for i in range(n)], dtype=np.int32)
sensory = np.where((indeg <= 1) & (outdeg >= 5))[0]
N_SENS = min(96, len(sensory))
if N_SENS < 96:
    extra = np.random.choice(np.where(outdeg >= 5)[0], 96 - N_SENS, replace=False)
    sensory = np.concatenate([sensory, extra])
    N_SENS = 96
sensory = sensory[:96].copy()
print(f"{N_SENS} sensors found")

# ─── Init T3 ───
print("Init T3 oscillator...", end=" ", flush=True)
tri = PyTriOscillator(N_T3_OSC, T3_FREQ_MIN, T3_FREQ_MAX)
tri.set_coupling(0.3, 0.15, 0.08)
print(f"done (n={N_T3_OSC}, pst_dim={tri.pst_dim()})")

# ─── Init ψ-state ───
print("Init ψ-wave...", end=" ", flush=True)
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
v = np.cos(theta, dtype=np.float32)
u = np.sin(theta, dtype=np.float32)
h_real = np.zeros(n_pad, dtype=np.float32)
h_imag = np.zeros(n_pad, dtype=np.float32)
speed = np.ones(n_pad, dtype=np.float32)
i_ext = np.zeros(n_pad, dtype=np.float32)
print("done")

# ─── Pre-find sensor groups ───
# Each T3 oscillator drives N_SENS // N_T3_OSC sensors
sensors_per_osc = N_SENS // N_T3_OSC
sensor_groups = sensory[:sensors_per_osc * N_T3_OSC].reshape(N_T3_OSC, sensors_per_osc)

# ─── Run ───
print(f"\nRunning: baseline {N_BASELINE} + T3 {N_T3_DRIVE} + after {N_AFTER}")

record_every = 3
n_record = N_TOTAL // record_every
eeg = np.zeros(n_record, dtype=np.float64)
coh = np.zeros(n_record, dtype=np.float64)
t3_ph1_log = np.zeros(n_record, dtype=np.float64)  # T3 phase tracking
base_end = N_BASELINE // record_every
drive_end = (N_BASELINE + N_T3_DRIVE) // record_every

t_start = time.perf_counter()
record_i = 0

for step in range(N_TOTAL):
    # T3 step (one step per connectome step)
    # Run T3 with random drive (no external, just internal dynamics)
    ph1 = np.array(tri.phases(1))
    ph2 = np.array(tri.phases(2))
    ph3 = np.array(tri.phases(3))
    r1 = np.array(tri.amplitudes(1))
    r2 = np.array(tri.amplitudes(2))
    r3 = np.array(tri.amplitudes(3))
    tri.step(DT)

    # ── Build ψ drive from T3 ──
    i_ext[:] = 0.0
    if step >= N_BASELINE and step < N_BASELINE + N_T3_DRIVE:
        for t3_i in range(N_T3_OSC):
            # Use combined phase-amplitude signal: r·sin(θ)
            drive_val = STIM_AMP * (
                r1[t3_i] * np.sin(ph1[t3_i]) +
                r2[t3_i] * np.sin(ph2[t3_i]) +
                r3[t3_i] * np.cos(ph3[t3_i])
            )
            idx = sensor_groups[t3_i]
            i_ext[idx] = drive_val / 3.0  # average across 3 pops
    # ── Compute Hamiltonian ──
    compute_hamiltonian(row_ptr, col_idx, vals, v, u, h_real, h_imag)

    # ── ψ-wave step ──
    step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, HBAR)

    if step % record_every == 0 and record_i < n_record:
        mv, mu = float(np.mean(v)), float(np.mean(u))
        coh[record_i] = float(np.sqrt(mv**2 + mu**2))
        eeg[record_i] = float(np.mean(h_imag[:n]))
        t3_ph1_log[record_i] = float(ph1[0])  # track phase of first oscillator
        record_i += 1

    if step > 0 and step % 300 == 0:
        print(f"  step {step:5d}/{N_TOTAL}  ({time.perf_counter()-t_start:.0f}s)")

total = time.perf_counter() - t_start
print(f"  done in {total:.1f}s")

# ─── Analysis ───
print("\n" + "█" * 60)
print("T3-EVOKED RESPONSE")
print("█" * 60)

base_eeg = np.mean(eeg[:base_end])
drive_eeg = np.mean(eeg[base_end:drive_end])
after_eeg = np.mean(eeg[drive_end:])

base_coh = np.mean(coh[:base_end])
drive_coh = np.mean(coh[base_end:drive_end])
after_coh = np.mean(coh[drive_end:])

print(f"\n  Phase       Mean EEG    Coherence")
print(f"  ─────       ────────    ─────────")
print(f"  Baseline:   {base_eeg:+.4f}     {base_coh:.6f}")
print(f"  T3 drive:   {drive_eeg:+.4f}     {drive_coh:.6f}")
print(f"  After:      {after_eeg:+.4f}     {after_coh:.6f}")
print(f"  Δ EEG:      {drive_eeg-base_eeg:+.4f}")
print(f"  Δ Coh:      {drive_coh-base_coh:+.6f}")

# FFT
from numpy.fft import rfft, rfftfreq
fs = 1.0 / (DT * record_every)

for name, lo, hi in [("Baseline", 0, base_end), ("T3-drive", base_end, drive_end), ("After", drive_end, n_record)]:
    segment = eeg[lo:hi]
    if len(segment) >= 32:
        sig = segment - np.mean(segment)
        f = rfftfreq(len(sig), 1/fs)
        p = np.abs(rfft(sig * np.hanning(len(sig))))**2
        # 1/f fit
        idx = (f > 0.01) & (f < fs/3)
        if np.sum(idx) > 5:
            coeffs = np.polyfit(np.log10(f[idx]), np.log10(p[idx] + 1e-30), 1)
            print(f"\n  {name}:  1/f^{coeffs[0]:.2f}")

# T3 phase vs EEG correlation
print(f"\n  T3-ψ coupling:")
phase_seg = t3_ph1_log[base_end:drive_end]
eeg_seg = eeg[base_end:drive_end]
if len(phase_seg) > 10:
    corr = np.corrcoef(phase_seg, eeg_seg)[0, 1]
    print(f"    Phase-EEG correlation: r = {corr:.4f}")

# Final stats
print(f"\n  Entrainment depth:")
print(f"    Coherence increase: {drive_coh-base_coh:+.6f}")
print(f"    After-effect:       {after_coh-base_coh:+.6f} (0 = no memory)")

print("\n" + "█" * 60)
print("T3 → FAFB → ψ-WAVE: DONE")
print("█" * 60)
