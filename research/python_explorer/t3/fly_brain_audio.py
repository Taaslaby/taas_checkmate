#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ: АУДИО МОЗГА МУХИ
T3 → FAFB connectome → ψ-wave → WAV.

Генерим EEG на ψ-wave rate (DT=0.05, 20 Hz),
интерполируем до 44100 Hz для аудио.
5 сек baseline + 10 сек T3 drive.
"""

import time, struct, math, os
import numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
    PyTriOscillator,
)

DT = 0.05                   # ψ-wave time step
DURATION = 15               # total seconds
N_PSI_STEPS = int(DURATION / DT)  # 300
STIM_AMP = 3.0
N_T3_OSC = 32
BASELINE_STEPS = int(5 / DT)  # 100

SAMPLE_RATE = 44100

print("█" * 70)
print("  САМЫЙ ЖЁСТКИЙ ТЕСТ")
print("  T3 → FAFB CONNECTOME → ψ-WAVE → 44.1 kHz AUDIO")
print("█" * 70)

# ─── Load FAFB ───
print("\n[1/6] Loading FAFB connectome...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons / {len(vals):,} synapses")

# ─── Sensors ───
print("[2/6] Finding sensors...", end=" ", flush=True)
indeg = np.bincount(col_idx.astype(np.int64), minlength=n).astype(np.int32)
outdeg = np.array([row_ptr[i+1] - row_ptr[i] for i in range(n)], dtype=np.int32)
sensory = np.where((indeg <= 1) & (outdeg >= 5))[0]
if len(sensory) < 96:
    extra = np.random.choice(np.where(outdeg >= 5)[0], 96 - len(sensory), replace=False)
    sensory = np.concatenate([sensory, extra])
sensory = sensory[:96]
sensor_groups = sensory.reshape(N_T3_OSC, 96 // N_T3_OSC)
print(f"{96} sensors")

# ─── Init T3 ───
print("[3/6] Init T3 oscillator...", end=" ", flush=True)
tri = PyTriOscillator(N_T3_OSC, 0.5, 2.0)
tri.set_coupling(0.3, 0.15, 0.08)
for _ in range(200):
    tri.step(DT)
print("ok")

# ─── Init ψ ───
print("[4/6] Init ψ-wave...", end=" ", flush=True)
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
v = np.cos(theta)
u = np.sin(theta)
h_real = np.zeros(n_pad, dtype=np.float32)
h_imag = np.zeros(n_pad, dtype=np.float32)
speed = np.ones(n_pad, dtype=np.float32)
i_ext = np.zeros(n_pad, dtype=np.float32)
print("ok")

# ─── Record ───
print("[5/6] Recording ψ-EEG...")
print(f"  {N_PSI_STEPS} steps @ DT={DT} ({1/DT:.0f} Hz)")
print(f"  Baseline: {BASELINE_STEPS} steps ({BASELINE_STEPS*DT:.0f}s)")
print(f"  T3 drive: {N_PSI_STEPS - BASELINE_STEPS} steps ({(N_PSI_STEPS - BASELINE_STEPS)*DT:.0f}s)")
print()

eeg_raw = np.zeros(N_PSI_STEPS, dtype=np.float32)
coh_log = np.zeros(N_PSI_STEPS, dtype=np.float32)
energy_log = np.zeros(N_PSI_STEPS, dtype=np.float32)
t_start = time.perf_counter()

for step in range(N_PSI_STEPS):
    tri.step(DT)

    # Build T3 drive (after baseline)
    i_ext[:] = 0.0
    if step >= BASELINE_STEPS:
        ph1 = np.array(tri.phases(1))
        ph2 = np.array(tri.phases(2))
        ph3 = np.array(tri.phases(3))
        r1 = np.array(tri.amplitudes(1))
        r2 = np.array(tri.amplitudes(2))
        r3 = np.array(tri.amplitudes(3))
        for t3_i in range(N_T3_OSC):
            drive_val = STIM_AMP * (
                r1[t3_i] * math.sin(ph1[t3_i]) +
                r2[t3_i] * math.sin(ph2[t3_i]) +
                r3[t3_i] * math.cos(ph3[t3_i])
            )
            for j in sensor_groups[t3_i]:
                i_ext[j] = drive_val / 3.0

    compute_hamiltonian(row_ptr, col_idx, vals, v, u, h_real, h_imag)
    step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, 1.0)

    eeg_raw[step] = float(np.mean(h_imag[:n]))
    coh_log[step] = float(math.sqrt(np.mean(v[:n])**2 + np.mean(u[:n])**2))
    energy_log[step] = float(np.mean(v[:n]**2 + u[:n]**2))

    if (step + 1) % 50 == 0:
        elapsed = time.perf_counter() - t_start
        label = "baseline" if step < BASELINE_STEPS else "T3 DRIVE"
        print(f"  [{step+1:3d}/{N_PSI_STEPS}] EEG={eeg_raw[step]:+.4f}  coh={coh_log[step]:.4f}  E={energy_log[step]:.2f}  {label}  [{elapsed:.0f}s]")

total_time = time.perf_counter() - t_start
print(f"\n  Done in {total_time:.0f}s ({total_time/N_PSI_STEPS:.2f}s/step)")

# ─── Interpolate to audio ───
print("\n[6/6] Interpolating to 44.1 kHz...", end=" ", flush=True)
t_psi = np.arange(N_PSI_STEPS) * DT
t_audio = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
audio = np.interp(t_audio, t_psi, eeg_raw).astype(np.float32)
# High-pass at ~40 Hz to remove DC drift for listenability
audio = audio - np.mean(audio)
from scipy import signal
b, a = signal.butter(4, 40.0 / (SAMPLE_RATE / 2), btype='high')
audio = signal.filtfilt(b, a, audio)
# Normalize
peak = np.max(np.abs(audio)) + 1e-10
audio = np.clip(audio / peak * 0.95, -1.0, 1.0)
print("ok")

# ─── Save WAV ───
wav_path = "fly_brain_t3.wav"
n_samples = len(audio)
with open(wav_path, "wb") as f:
    channels, bits = 1, 16
    byte_rate = SAMPLE_RATE * channels * bits // 8
    block = channels * bits // 8
    data_size = n_samples * block
    f.write(b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE")
    f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, SAMPLE_RATE, byte_rate, block, bits))
    f.write(b"data" + struct.pack("<I", data_size))
    for vv in audio:
        f.write(struct.pack("<h", int(vv * 32767)))

# ─── Stats ───
print("\n" + "█" * 70)
print("  РЕЗУЛЬТАТЫ")
print("█" * 70)
print(f"  File:      {wav_path}  ({os.path.getsize(wav_path)/1024:.0f} KB)")
print(f"  Duration:  {DURATION}s  ({BASELINE_STEPS*DT:.0f}s baseline + {(N_PSI_STEPS-BASELINE_STEPS)*DT:.0f}s T3)")
print(f"  ψ-steps:   {N_PSI_STEPS}  ({total_time:.0f}s simulation)")
print(f"  Audio:     {SAMPLE_RATE} Hz / 16-bit mono")

m = BASELINE_STEPS
print(f"\n  ┌──────┬──────────┬──────────┬──────────┐")
print(f"  │      │  EEG μ   │  σ(EEG)  │ Coherence│")
print(f"  ├──────┼──────────┼──────────┼──────────┤")
print(f"  │ Base │ {np.mean(eeg_raw[:m]):+.4f}  │ {np.std(eeg_raw[:m]):.4f}  │ {np.mean(coh_log[:m]):.4f}  │")
print(f"  │ T3   │ {np.mean(eeg_raw[m:]):+.4f}  │ {np.std(eeg_raw[m:]):.4f}  │ {np.mean(coh_log[m:]):.4f}  │")
print(f"  │ Δ    │ {np.mean(eeg_raw[m:]) - np.mean(eeg_raw[:m]):+.4f}  │ {np.std(eeg_raw[m:]) - np.std(eeg_raw[:m]):+.4f}  │ {np.mean(coh_log[m:]) - np.mean(coh_log[:m]):+.4f}  │")
print(f"  └──────┴──────────┴──────────┴──────────┘")

# FFT on T3 segment
sig = eeg_raw[m:] - np.mean(eeg_raw[m:])
freqs = np.fft.rfftfreq(len(sig), DT)
fft = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))**2
idx = (freqs > 0.5) & (freqs < 8)
if np.sum(idx) > 3:
    coeffs = np.polyfit(np.log10(freqs[idx][freqs[idx] > 0]), np.log10(fft[idx][freqs[idx] > 0] + 1e-30), 1)
    print(f"\n  1/f exponent (T3 ψ-EEG, 0.5-8 Hz): {coeffs[0]:.2f}")

print(f"\n  СЛУШАЙ: {os.path.abspath(wav_path)}")
print("  5 sec тишина → 10 sec мозг мухи под T3")
print("█" * 70)
