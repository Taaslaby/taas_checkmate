#!/usr/bin/env python3
"""
Feed audio through T3 oscillator reservoir.
Shows PST heatmap per frequency.
"""

import sys, time
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from taas_checkmate import (PyTriOscillator, pst_linear_dim, pst_linear_features)

N_OSC = 32
N_PST = pst_linear_dim(N_OSC)
DT = 0.05
FREQ_MIN, FREQ_MAX = 0.5, 3.0
N_T3_STEPS = 10       # T3 steps per audio window
WINDOW = 256          # samples per window


def process_audio(samples, sr):
    """Feed audio through T3, return PST features over time."""
    tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
    tri.set_uniform_coupling(0.3, 0.15, 0.08)

    # Random projection: audio samples → N_OSC drives
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 0.5, (N_OSC, WINDOW)).astype(np.float32)

    fv = np.zeros(N_PST, dtype=np.float32)
    features = []

    n_windows = len(samples) // WINDOW
    for wi in range(n_windows):
        chunk = samples[wi*WINDOW : (wi+1)*WINDOW]
        chunk -= chunk.mean()
        chunk /= chunk.std() + 1e-8
        chunk *= 2.0

        drives = (proj @ chunk).astype(np.float32)
        for _ in range(N_T3_STEPS):
            tri.set_drive(drives, 1)
            tri.step(DT)

        pst_linear_features(tri, fv)
        features.append(fv.copy())

    return np.array(features)


def main():
    for freq, name in [(220, 'A3 (220Hz)'), (440, 'A4 (440Hz)'), (880, 'A5 (880Hz)')]:
        sr, samples = wavfile.read(f'/tmp/{name.split()[0]}.wav')
        samples = samples.astype(np.float32)

        print(f"\n{name}: {len(samples)} samples @ {sr}Hz")
        F = process_audio(samples, sr)
        print(f"  T3 windows: {F.shape[0]}, PST dim: {F.shape[1]}")

        # Plot PST feature heatmap
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for pop in range(3):
            idx = pop * N_OSC * 6
            im = axes[pop].imshow(F[:, idx:idx+N_OSC*6].T, aspect='auto',
                                   cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[pop].set_ylabel(f'Pop {pop+1}')
        axes[2].set_xlabel('Time window')
        fig.suptitle(f'T3 PST response — {name}')
        plt.tight_layout()
        plt.savefig(f'/tmp/t3_audio_{name.split()[0]}.png', dpi=150)
        print(f"  Saved /tmp/t3_audio_{name.split()[0]}.png")
        plt.close()

    # Compare all 3 frequencies on one plot
    print("\n── Cross-frequency comparison ──")
    fig, ax = plt.subplots(figsize=(10, 4))
    for freq, name, color in [(220, 'A3', 'b'), (440, 'A4', 'g'), (880, 'A5', 'r')]:
        _, samples = wavfile.read(f'/tmp/{name}.wav')
        F = process_audio(samples.astype(np.float32), 22050)
        # Take mean PST feature per window as summary
        ax.plot(F.mean(axis=1), color=color, alpha=0.8, label=name)
    ax.set_xlabel('Time window')
    ax.set_ylabel('Mean |PST|')
    ax.legend()
    ax.set_title('T3 response to different frequencies')
    plt.tight_layout()
    plt.savefig('/tmp/t3_audio_comparison.png', dpi=150)
    print("Saved /tmp/t3_audio_comparison.png")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
