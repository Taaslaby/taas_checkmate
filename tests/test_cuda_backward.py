#!/usr/bin/env python3
"""CUDA backward: сравниваем CPU и GPU backward pass.

    cd taas_checkmate && python3 tests/test_cuda_backward.py
"""

import numpy as np
import taas_checkmate as tcm

N_OSC = 8
N_STEPS = 30
DT = 0.05

rng = np.random.default_rng(42)

# ─── Траектория через CPU forward ─────────────────────
drives = rng.normal(0, 0.05, (N_STEPS, 3 * N_OSC)).astype(np.float32)
pst_cpu, traj = tcm.t2spigrad_forward(
    N_OSC, N_STEPS, DT,
    drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0,
    coupling_w12=0.2, coupling_w13=0.1, coupling_w23=0.05,
    drives=drives,
)

pst_dim = 18 * N_OSC + 12
grad_pst = np.zeros(pst_dim, dtype=np.float32)
grad_pst[0:N_OSC] = 1.0

# ─── CPU backward ─────────────────────────────────────
print("CPU backward...")
grad_drives_cpu = tcm.t2spigrad_backward(grad_pst, traj)
grad_cpu = np.array(grad_drives_cpu, dtype=np.float32)

# ─── CUDA backward ────────────────────────────────────
print("CUDA backward...")
try:
    grad_drives_gpu = tcm.t3.t2spigrad_backward_cuda(grad_pst, traj)
    grad_gpu = np.array(grad_drives_gpu, dtype=np.float32)

    diff = np.abs(grad_cpu - grad_gpu)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    rel_diff = max_diff / (float(np.abs(grad_cpu).max()) + 1e-8)

    print(f"\n─── CPU vs GPU ───")
    print(f"  Max abs diff:  {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")
    print(f"  Rel diff:      {rel_diff:.8f}")
    print(f"  CPU grad norm: {np.linalg.norm(grad_cpu):.4f}")
    print(f"  GPU grad norm: {np.linalg.norm(grad_gpu):.4f}")

    if rel_diff < 0.05 or max_diff < 0.2:
        print("  ✅ PASSED — CPU ≈ GPU")
    else:
        print("  ❌ FAILED — значительное расхождение CPU vs GPU")
        # Покажем топ-5 расхождений
        flat_diff = diff.flatten()
        top5 = np.argsort(flat_diff)[-5:][::-1]
        for idx in top5:
            i, j = divmod(idx, grad_cpu.shape[1])
            print(f"      [{i},{j}]: cpu={grad_cpu[i,j]:.6f} gpu={grad_gpu[i,j]:.6f} diff={flat_diff[idx]:.6f}")

except Exception as e:
    print(f"  ❌ CUDA backward error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ─── CUDA backward with per_step_grad_pst ─────────────
print(f"\n─── CUDA per_step backward ───")
per_step_grad = rng.normal(0, 0.01, (N_STEPS, pst_dim)).astype(np.float32)
try:
    gpu_per_step = tcm.t3.t2spigrad_backward_cuda(grad_pst, traj, per_step_grad)
    print(f"  GPU per_step grad shape: {np.array(gpu_per_step).shape}")
    print(f"  GPU per_step grad norm: {np.linalg.norm(np.array(gpu_per_step)):.4f}")
    print(f"  ✅ CUDA per_step backward OK")
except Exception as e:
    print(f"  ❌ CUDA per_step error: {e}")
    import traceback
    traceback.print_exc()

# ─── CUDA forward + backward ──────────────────────────
print(f"\n─── CUDA full pipeline (forward + backward) ───")
try:
    pst_gpu, traj_gpu = tcm.t3.t2spigrad_forward_cuda(
        N_OSC, N_STEPS, DT,
        10.0, 1.0, 0.5, 3.0, 0.2, 0.1, 0.05,
        drives,
    )
    pst_gpu = np.array(pst_gpu, dtype=np.float32)
    pst_cpu = np.array(pst_cpu, dtype=np.float32)
    pst_diff = np.abs(pst_gpu - pst_cpu).max()
    print(f"  CPU vs GPU PST max diff: {pst_diff:.8f}")

    grad_gpu2 = tcm.t3.t2spigrad_backward_cuda(grad_pst, traj_gpu)
    grad_gpu2 = np.array(grad_gpu2, dtype=np.float32)
    diff2 = np.abs(grad_cpu - grad_gpu2).max()
    print(f"  CPU vs GPU (full GPU pipeline) max diff: {diff2:.8f}")

    if pst_diff < 0.01 and diff2 < 0.2:
        print("  ✅ PASSED — полный CUDA pipeline")
    else:
        print("  ❌ FAILED")
except Exception as e:
    print(f"  ❌ CUDA full pipeline error: {e}")
    import traceback
    traceback.print_exc()
