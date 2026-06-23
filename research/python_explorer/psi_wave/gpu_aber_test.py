import numpy as np
import taas_checkmate as tc
import time

def f32(arr):
    """Ensure float32 contiguous array."""
    return np.ascontiguousarray(arr, dtype=np.float32)

def i32(arr):
    """Ensure int32 contiguous array."""
    return np.ascontiguousarray(arr, dtype=np.int32)

# Load FAFB CSR
data = np.load('/tmp/fafb_csr_npz.npz')
row_ptr = i32(data['row_ptr'])
col_indices = i32(data['col_indices'])
values = f32(data['values'])
n_neurons = len(row_ptr) - 1
print(f"FAFB: {n_neurons} neurons, {len(values)} synapses")

# Init ψ-wave
inv_sqrt_n = np.sqrt(1.0 / n_neurons).astype(np.float32)
v = f32(np.ones(n_neurons, dtype=np.float32) * inv_sqrt_n)
u = f32(np.zeros(n_neurons, dtype=np.float32))
dt = 0.5
hbar = 1.0

# Phase-gradient drive on left optic lobe
i_ext = f32(np.zeros(n_neurons, dtype=np.float32))
left_lobe = np.where(row_ptr[1:] - row_ptr[:-1] > 10)[0]  # high-degree neurons
if len(left_lobe) > 100:
    i_ext[left_lobe[:100]] = 0.1

# First test ψ-wave only (existing function)
print("Testing ψ-wave only (500 steps)...")
t0 = time.perf_counter()
elapsed_psi, v_final, u_final = tc.bench_psi_wave_gpu(
    row_ptr, col_indices, values,
    v.copy(), u.copy(), dt, hbar, 500, i_ext
)
t1 = time.perf_counter()
print(f"  ψ-wave: {elapsed_psi:.3f}s ({500/elapsed_psi:.0f} steps/s)")

# Now test ψ-wave + Abertons
print("\nTesting ψ-wave + Abertons (500 steps)...")
t0 = time.perf_counter()
elapsed, v_final, u_final, collapse_counts = tc.bench_psi_aberton_gpu(
    row_ptr, col_indices, values,
    v.copy(), u.copy(), dt, hbar, n_steps=500,
    aber_gamma=0.1, aber_omega=1.0, aber_gain=1.0,
    aber_threshold=2.0, aber_collapse_scale=0.5,
    use_synaptic_field=False, i_ext=i_ext
)
t1 = time.perf_counter()

v_np = np.array(v_final, dtype=np.float32)
u_np = np.array(u_final, dtype=np.float32)
total_collapses = sum(collapse_counts)

print(f"\nResults:")
print(f"  Wall time: {elapsed:.3f} s ({500/elapsed:.0f} steps/s, {elapsed/500*1000:.3f} ms/step)")
print(f"  |ψ| range: [{np.min(np.abs(v_np + 1j*u_np)):.6f}, {np.max(np.abs(v_np + 1j*u_np)):.6f}]")
print(f"  Total collapses (all steps): {total_collapses}")
print(f"  L-R asymmetry: {np.mean(u_np[left_lobe[:100]]):.6f}")
print(f"  Python timing: {t1-t0:.3f}s")
