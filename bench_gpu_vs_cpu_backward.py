#!/usr/bin/env python3
"""Benchmark: CPU vs GPU backward for T2-SPIGRAD.

Usage:
    python bench_gpu_vs_cpu_backward.py

Requires:
    - pip install numpy
    - cargo build --release --features python,cuda  (for GPU)
    - cargo build --release --features python        (for CPU only)
"""

import time
import sys
import numpy as np

# Import Rust module
try:
    import taas_checkmate as t3
except ImportError:
    print("ERROR: Could not import 'taas_checkmate'.")
    print("Build with: cd taas_checkmate && cargo build --release --features python,cuda")
    print("Then set PYTHONPATH=target/release/")
    sys.exit(1)

# Check if CUDA backward is available
HAS_CUDA = hasattr(t3, "t2spigrad_backward_cuda")
if HAS_CUDA:
    print("✅ CUDA backward available")
else:
    print("⚠️  CUDA backward NOT available (re-build with --features python,cuda)")
    print("   Running CPU-only benchmark.")

# ── Parameters ────────────────────────────────────────────
N_OSC = 64
N_STEPS = 1000
DT = 0.05
DRIVE_GAIN = 10.0
ALPHA = 1.0
FREQ_MIN = 0.5
FREQ_MAX = 3.0
W12, W13, W23 = 0.3, 0.15, 0.08
N_WARMUP = 3
N_REPEAT = 5

# ── Generate random drives ────────────────────────────────
rng = np.random.default_rng(42)
drives = rng.normal(0, 1.0, size=(N_STEPS, N_OSC * 3)).astype(np.float32)

print(f"\n{'='*60}")
print(f"T2-SPIGRAD Backward Benchmark")
print(f"{'='*60}")
print(f"  Oscillators: {N_OSC}")
print(f"  Steps:       {N_STEPS}")
print(f"  Drive dim:   {N_STEPS} × {3*N_OSC} = {N_STEPS * 3 * N_OSC:,}")
print(f"  Warmup runs: {N_WARMUP}")
print(f"  Repeat runs: {N_REPEAT}")

# ── Forward (once) ────────────────────────────────────────
print(f"\n  ▶ Running forward pass...")
t0 = time.perf_counter()
pst, traj = t3.t2spigrad_forward(
    N_OSC, N_STEPS, DT, DRIVE_GAIN, ALPHA,
    FREQ_MIN, FREQ_MAX, W12, W13, W23, drives,
)
t_fwd = time.perf_counter() - t0
pst_dim = len(pst)
print(f"  ✓ Forward done in {t_fwd*1000:.1f} ms")
print(f"  ✓ PST dim = {pst_dim}")

# Random gradient w.r.t. PST features
grad_pst = rng.normal(0, 1.0, size=pst_dim).astype(np.float32)

# ── CPU Backward Benchmark ────────────────────────────────
print(f"\n  ▶ CPU backward warmup ({N_WARMUP}x)...")
for _ in range(N_WARMUP):
    cpu_grad = t3.t2spigrad_backward(grad_pst, traj)

print(f"  ▶ CPU backward benchmark ({N_REPEAT}x)...")
cpu_times = []
for _ in range(N_REPEAT):
    t0 = time.perf_counter()
    cpu_grad = t3.t2spigrad_backward(grad_pst, traj)
    cpu_times.append(time.perf_counter() - t0)

cpu_mean = np.mean(cpu_times)
cpu_std = np.std(cpu_times)
cpu_ms = cpu_mean * 1000
cpu_steps_per_sec = N_STEPS / cpu_mean
print(f"  ✓ CPU:  {cpu_ms:.1f} ± {cpu_std*1000:.1f} ms")
print(f"         {cpu_steps_per_sec:.0f} steps/s")

# ── GPU Backward Benchmark ────────────────────────────────
gpu_grad = None
if HAS_CUDA:
    print(f"\n  ▶ GPU backward warmup ({N_WARMUP}x)...")
    for _ in range(N_WARMUP):
        gpu_grad = t3.t2spigrad_backward_cuda(grad_pst, traj)

    print(f"  ▶ GPU backward benchmark ({N_REPEAT}x)...")
    gpu_times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        gpu_grad = t3.t2spigrad_backward_cuda(grad_pst, traj)
        gpu_times.append(time.perf_counter() - t0)

    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    gpu_ms = gpu_mean * 1000
    gpu_steps_per_sec = N_STEPS / gpu_mean

    print(f"  ✓ GPU:  {gpu_ms:.1f} ± {gpu_std*1000:.1f} ms")
    print(f"         {gpu_steps_per_sec:.0f} steps/s")

    # ── Speedup ──────────────────────────────────────────
    speedup = cpu_mean / gpu_mean
    print(f"\n  🚀 Speedup: {speedup:.1f}x")

    # ── Cross-validation ─────────────────────────────────
    print(f"\n  ▶ Cross-validating CPU vs GPU...")
    diff = np.abs(cpu_grad - gpu_grad)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_err = mean_diff / (np.abs(cpu_grad).mean() + 1e-10)
    print(f"  ✓ Max diff:   {max_diff:.10f}")
    print(f"  ✓ Mean diff:  {mean_diff:.10f}")
    print(f"  ✓ Rel err:    {rel_err:.6e}")

    # Run GPU again to check determinism
    gpu_grad2 = t3.t2spigrad_backward_cuda(grad_pst, traj)
    gpu_self_diff = np.abs(gpu_grad - gpu_grad2).max()
    print(f"  ✓ GPU self:   {gpu_self_diff:.10f}")

    if max_diff < 2e-4:
        print(f"  ✅ CPU vs GPU match within tolerance!")
    else:
        print(f"  ⚠️  max_diff={max_diff} exceeds tolerance 2e-4")
else:
    # CPU-only results
    print(f"\n  (GPU not available — CPU-only benchmark)")

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Forward:     {t_fwd*1000:.1f} ms")
print(f"  CPU backward: {cpu_ms:.1f} ms ({cpu_steps_per_sec:.0f} steps/s)")
if HAS_CUDA:
    print(f"  GPU backward: {gpu_ms:.1f} ms ({gpu_steps_per_sec:.0f} steps/s)")
    print(f"  Speedup:      {speedup:.1f}x")
print(f"{'='*60}\n")
