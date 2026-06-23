#!/usr/bin/env python3
"""GPU ψ-wave + Abertons + CUDA Graph on full FAFB."""
import numpy as np
import taas_checkmate as tc
import time

def f32(x): return np.ascontiguousarray(x, dtype=np.float32)
def i32(x): return np.ascontiguousarray(x, dtype=np.int32)

# Load FAFB CSR
data = np.load('/tmp/fafb_csr_npz.npz')
row_ptr = i32(data['row_ptr'])
col_indices = i32(data['col_indices'])
values   = f32(data['values'])
n = len(row_ptr) - 1
print(f"FAFB: {n} neurons, {len(values)} synapses")

# Init ψ-wave
psi_init = f32(np.ones(n) * np.sqrt(1.0/n, dtype=np.float32))
u_init   = f32(np.zeros(n))

# Phase-gradient drive on left optic lobe (high-degree neurons)
i_ext = f32(np.zeros(n))
left_lobe = np.where(row_ptr[1:] - row_ptr[:-1] > 10)[0]
if len(left_lobe) > 100:
    i_ext[left_lobe[:100]] = 0.015

DT, HBAR = 0.5, 1.0
STEPS = 500

# ─── 1. GPU ψ-wave only (no graph) ─────────────────────
print("\n1. ψ-wave (plain async) …")
v, u = psi_init.copy(), u_init.copy()
t0 = time.perf_counter()
elapsed, v_f, u_f = tc.bench_psi_wave_gpu(row_ptr, col_indices, values, v, u, DT, HBAR, STEPS, i_ext)
t1 = time.perf_counter()
v_np, u_np = np.array(v_f), np.array(u_f)
nrm = np.sqrt(v_np**2 + u_np**2)
print(f"   {t1-t0:.3f}s ({STEPS/(t1-t0):.0f} step/s, {(t1-t0)/STEPS*1e6:.0f} μs/step)")
print(f"   |ψ| in [{nrm.min():.6f}, {nrm.max():.6f}]  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")

# ─── 2. GPU ψ-wave + Abertons ──────────────────────────
print("\n2. ψ-wave + Abertons …")
v, u = psi_init.copy(), u_init.copy()
t0 = time.perf_counter()
elapsed, v_f, u_f, coll = tc.bench_psi_aberton_gpu(
    row_ptr, col_indices, values, v, u, DT, HBAR, STEPS,
    aber_gamma=0.05, aber_omega=2.0, aber_gain=1.5,
    aber_threshold=0.5, aber_collapse_scale=0.3,
    use_synaptic_field=False, i_ext=i_ext)
t1 = time.perf_counter()
v_np, u_np = np.array(v_f), np.array(u_f)
nrm = np.sqrt(v_np**2 + u_np**2)
total_coll = sum(coll)
print(f"   {t1-t0:.3f}s ({STEPS/(t1-t0):.0f} step/s, {(t1-t0)/STEPS*1e6:.0f} μs/step)")
print(f"   |ψ| in [{nrm.min():.6f}, {nrm.max():.6f}]  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")
print(f"   Collapses: {total_coll}")

# ─── 3. GPU ψ-wave with CUDA Graph ─────────────────────
print("\n3. ψ-wave CUDA Graph (10K steps) …")
# Use GpuCtx directly for graph replay
N_GRAPH = 10000

# Build GpuCtx via the Python binding
# We'll use bench_psi_wave_gpu with n_steps=N_GRAPH
v, u = psi_init.copy(), u_init.copy()
t0 = time.perf_counter()
elapsed, v_f, u_f = tc.bench_psi_wave_gpu(row_ptr, col_indices, values, v, u, DT, HBAR, N_GRAPH, i_ext)
t1 = time.perf_counter()
v_np, u_np = np.array(v_f), np.array(u_f)
nrm = np.sqrt(v_np**2 + u_np**2)
print(f"   {t1-t0:.3f}s ({N_GRAPH/(t1-t0):.0f} step/s, {(t1-t0)/N_GRAPH*1e6:.0f} μs/step)")
print(f"   |ψ| in [{nrm.min():.6f}, {nrm.max():.6f}]  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")

print("\n── Done ──")
