#!/usr/bin/env python3
"""
T3-SPIKE demo: Surrogate gradient training for SNN via differentiable T3 oscillator.

Demonstrates:
1. Synthetic spike patterns (3 classes, 15% noise)
2. SNN → T3 → PST → decoder gradient flow
3. W_snn learns to separate classes through T3 dynamics

Usage:
    python3 -m t3_spike.demo
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t3_spike import T3SpikeEncoder, T3SpikeTrainer, pst_jacobian, pst_linear_features_np


def generate_synthetic_data(n_samples=50, n_neurons=16, noise_level=0.15, seed=123):
    """Generate synthetic spike patterns for 3 classes."""
    rng = np.random.default_rng(seed)
    patterns = []
    labels = []
    for i in range(n_samples):
        cls = i % 3
        base = np.zeros(n_neurons)
        base[cls * 5:(cls + 1) * 5] = 1.0
        noise = rng.random(n_neurons) < noise_level
        spikes = np.where(base + noise > 0, 1.0, 0.0).astype(np.float32)
        patterns.append(spikes)
        labels.append(float(cls))
    return patterns, labels


def test_jacobian():
    """Test that PST Jacobian is correct via finite differences."""
    print("═" * 60)
    print("TEST: PST Jacobian — finite difference check")
    print("═" * 60)

    n = 4
    eps = 1e-6

    # Random state
    rng = np.random.default_rng(42)
    th1 = rng.uniform(-np.pi, np.pi, n)
    th2 = rng.uniform(-np.pi, np.pi, n)
    th3 = rng.uniform(-np.pi, np.pi, n)
    r1 = rng.uniform(0.1, 2.0, n)
    r2 = rng.uniform(0.1, 2.0, n)
    r3 = rng.uniform(0.1, 2.0, n)
    prev_th1 = th1 - 0.1
    prev_th2 = th2 - 0.15
    prev_th3 = th3 - 0.2

    # Analytical Jacobian
    d_th1, d_th2, d_th3, d_r1, d_r2, d_r3 = pst_jacobian(
        th1, r1, th2, r2, th3, r3, prev_th1, prev_th2, prev_th3, n
    )

    # Finite difference: d(PST)/d(θ₁[0])
    fv_base = pst_linear_features_np(th1, r1, th2, r2, th3, r3, prev_th1, prev_th2, prev_th3, n)

    th1_eps = th1.copy()
    th1_eps[0] += eps
    fv_eps = pst_linear_features_np(th1_eps, r1, th2, r2, th3, r3, prev_th1, prev_th2, prev_th3, n)
    fd = (fv_eps - fv_base) / eps

    analytical = d_th1[:, 0]

    # Compare
    max_err = np.max(np.abs(fd - analytical))
    pst_dim = len(fv_base)
    print(f"  PST dim: {pst_dim}")
    print(f"  Max error d(PST)/d(th1[0]): {max_err:.6e}")
    if max_err < 1e-3:
        print("  ✅ Jacobian checks out!")
    else:
        print("  ❌ Jacobian error too large!")
        # Show top errors
        bad_idx = np.argsort(np.abs(fd - analytical))[-5:]
        for idx in bad_idx:
            print(f"    idx {idx}: analytical={analytical[idx]:.6f}, fd={fd[idx]:.6f}")
        return False
    return True


def test_encoder_gradients():
    """Test that T3SpikeEncoder gradients flow correctly."""
    print("\n" + "═" * 60)
    print("TEST: T3SpikeEncoder — gradient flow")
    print("═" * 60)

    import torch
    from t3_spike.encoder import T3SpikeEncoder

    n_osc = 8
    n_steps = 5
    dt = 0.05

    # Random drives
    rng = np.random.default_rng(42)
    drives = rng.normal(0, 1.0, (2, n_steps, 3 * n_osc)).astype(np.float32)
    drives_t = torch.tensor(drives, requires_grad=True)

    # Forward
    pst = T3SpikeEncoder.apply(
        drives_t, None, n_osc, n_steps, dt,
        10.0, 1.0, 0.5, 3.0, (0.3, 0.15, 0.08), False
    )

    # Simple loss: sum of squares
    loss = pst.pow(2).sum()
    loss.backward()

    grad = drives_t.grad
    has_grad = grad is not None and grad.abs().sum().item() > 0

    print(f"  Batch: 2 × n_osc={n_osc}, steps={n_steps}, pst_dim={pst.shape[-1]}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grad.norm().item():.4f}" if has_grad else "  ❌ No gradient!")
    print(f"  Gradient range: [{grad.min().item():.4f}, {grad.max().item():.4f}]" if has_grad else "")

    if has_grad:
        print("  ✅ Gradients flow through T3!")
    else:
        print("  ❌ No gradient flow!")
    return has_grad


def train_demo():
    """End-to-end T3-SPIKE training demo."""
    print("\n" + "═" * 60)
    print("T3-SPIKE TRAINING DEMO")
    print("  Surrogate gradient through T3 oscillator dynamics")
    print("═" * 60)

    n_osc = 16
    n_neurons = 16
    n_steps = 10
    dt = 0.05

    # Generate data
    patterns, labels = generate_synthetic_data(n_samples=60, n_neurons=n_neurons, noise_level=0.15)
    print(f"\n  Data: {len(patterns)} patterns, 3 classes, {n_neurons} neurons")
    print(f"  Noise level: 15%")

    # Create trainer
    trainer = T3SpikeTrainer(
        n_osc=n_osc,
        n_steps=n_steps,
        dt=dt,
        n_spike_neurons=n_neurons,
        lr_snn=0.05,
        lr_decoder=0.1,
        freq_min=0.5,
        freq_max=3.0,
        coupling=(0.3, 0.15, 0.08),
        seed=42,
    )
    print(f"  SNN weights: {trainer.W_snn.shape}")
    print(f"  PST dim: {trainer.pst_dim}")

    # Pre-training accuracy (random W_snn)
    acc_before = trainer.accuracy(patterns, labels)
    print(f"\n  Pre-training accuracy: {acc_before * 100:.1f}%")

    # Train
    print(f"\n  Training ({n_steps} T3 steps, {n_osc} oscillators)...")
    loss_history = trainer.train_batch(patterns, labels, epochs=10, verbose=True)

    # Post-training accuracy
    acc_after = trainer.accuracy(patterns, labels)
    final_loss = loss_history[-1]

    print(f"\n  ────────────────────────────────────────────")
    print(f"  RESULTS")
    print(f"  ────────────────────────────────────────────")
    print(f"  Pre-training accuracy:  {acc_before * 100:.1f}%")
    print(f"  Post-training accuracy: {acc_after * 100:.1f}%")
    print(f"  Final loss:            {final_loss:.6f}")
    print(f"  Improvement:           {(acc_after - acc_before) * 100:.1f}%")
    print(f"  ────────────────────────────────────────────")

    if acc_after > acc_before:
        print("\n  ✅ T3-SPIKE: SNN learned through differentiable T3 dynamics!")
    else:
        print("\n  ⚠️  No improvement — adjust hyperparameters or check backward pass")

    return acc_before, acc_after, loss_history


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               T3-SPIKE — Proof of Concept              ║")
    print("║  SNN surrogate gradients through differentiable T3    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Run tests
    jac_ok = test_jacobian()
    grad_ok = test_encoder_gradients()

    # Run training demo
    if jac_ok and grad_ok:
        print("\n  All tests passed! Running training demo...")
        train_demo()
    else:
        print("\n  ❌ Tests failed — check backward implementation")
