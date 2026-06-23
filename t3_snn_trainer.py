#!/usr/bin/env python3
"""
T3-as-trainer for SNN proof of concept.

Problem: SNN (spiking) is non-differentiable → hard to train.
Solution: Feed spike patterns into T3 → get continuous PST features
→ train decoder with gradients → backprop through T3 into spike drives.

The key insight: T3 oscillator dynamics are differentiable w.r.t.
drive input. PST features = smooth encoding of discrete spike trains.
"""

import numpy as np
import matplotlib.pyplot as plt
from taas_checkmate import (
    PyTriOscillator, PyNlmsDecoder,
    pst_linear_dim, pst_linear_features,
)

N_OSC = 16
N_PST = pst_linear_dim(N_OSC)  # 16*3*6 + 12 = 300
DT = 0.05
FREQ_MIN, FREQ_MAX = 0.5, 3.0
N_T3_STEPS = 10


def spike_pattern_to_t3(patterns, labels):
    """
    Demo: synthetic spike patterns → T3 → PST features.
    Show that PST features are CLASSIFIABLE even when spikes are noisy.
    """
    tri = PyTriOscillator(N_OSC, FREQ_MIN, FREQ_MAX)
    tri.set_uniform_coupling(0.3, 0.15, 0.08)
    fv = np.zeros(N_PST, dtype=np.float32)

    # Random projection: N_OSC spikes → T3 drives (3 pops)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0, (3 * N_OSC, N_OSC)).astype(np.float32)

    # FEATURE EXTRACTION: spike patterns → PST vectors
    features = []
    for spikes in patterns:
        drives = (proj @ spikes).astype(np.float32)

        # Inject into T3 (3 populations)
        for p in range(3):
            tri.set_drive(drives[p * N_OSC:(p + 1) * N_OSC], p + 1)

        for _ in range(N_T3_STEPS):
            tri.step(DT)

        pst_linear_features(tri, fv)
        features.append(fv.copy())

    features = np.array(features)

    # LINEAR CLASSIFICATION on PST features (sklearn)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=200)
    scores = cross_val_score(clf, features, labels, cv=5)
    acc = scores.mean()

    return features, acc


def main():
    print("═" * 60)
    print("T3-as-trainer for SNN — Proof of Concept")
    print(f"  {N_OSC} oscillators × 3 pops, {N_PST} PST features")
    print("═" * 60)

    # Generate synthetic spike patterns (3 classes)
    rng = np.random.default_rng(123)
    n_samples = 200
    n_spike_neurons = N_OSC

    patterns = []
    labels = []
    for i in range(n_samples):
        cls = i % 3
        # Base spike pattern per class
        base = np.zeros(n_spike_neurons)
        base[cls * 5:(cls + 1) * 5] = 1.0

        # Add noise
        noise = rng.random(n_spike_neurons) < 0.15
        spikes = np.where(base + noise > 0, 1.0, 0.0).astype(np.float32)

        patterns.append(spikes)
        labels.append(cls)

    # Extract PST features and classify
    features, acc = spike_pattern_to_t3(patterns, labels)

    print(f"\n  Synthetic SNN: {n_samples} patterns, 3 classes")
    print(f"  Spike noise: 15%")
    print(f"  PST features: {features.shape}")
    print(f"  Test accuracy: {acc*100:.0f}%")

    # Show feature separability
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for cls in range(3):
        idx = np.array(labels) == cls
        axes[cls].imshow(features[idx][:5], aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[cls].set_title(f'Class {cls} (5 samples)')
        axes[cls].set_xlabel('PST feature')
    plt.tight_layout()
    plt.savefig('/tmp/t3_snn_trainer.png', dpi=150)
    print(f"\n  Saved /tmp/t3_snn_trainer.png")
    print()

    # Key insight
    print("─" * 60)
    print("KEY INSIGHT")
    print("─" * 60)
    print("  SNN spikes (binary, non-diff)  →  T3 oscillator dynamics  →  PST (continuous, diff)")
    print("                               ↓                               ↓")
    print("                    hard to train                      easy to classify")
    print()
    print("  T3 = differentiable kernel for spike patterns.")
    print("  Gradients flow: loss → NLMS → PST → T3 → drive → spikes (surrogate).")
    print("  SNN weights ← STDP gated by T3 phase, or evolution strategies.")
    print("─" * 60)


if __name__ == "__main__":
    main()
