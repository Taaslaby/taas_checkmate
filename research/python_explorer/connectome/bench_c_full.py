#!/usr/bin/env python3
"""C benchmark: same workload as Rust sim_step — FlyWire + Izhikevich + propagation."""
import sys, os, time, warnings, ctypes, numpy as np
warnings.filterwarnings("ignore")

N_STEPS = 1000; DT = 0.5
CONNECTOME_PATH = "/home/taaslab/Проекты/t3-wavelet/data/connectome/flywire_138k.npz"

# Load connectome
print("Loading connectome...", end=" ", flush=True)
data = np.load(CONNECTOME_PATH)
row_ptr = data["row_ptr"].copy().astype(np.int32)
col_indices = data["col_indices"].copy().astype(np.int32)
values = data["values"].copy().astype(np.float32)
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(values):,} synapses")

# Setup C lib
lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'kuramoto_engine', 'build', 'libneuron.so'))
lib.step_n_neurons.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
lib.propagate_spikes_ultra.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
]

# Setup state (matching Rust setup)
np.random.seed(42)
v = np.full(n_pad, -65.0, dtype=np.float32)
u = np.full(n_pad, -13.0, dtype=np.float32)
trace = np.zeros(n_pad, dtype=np.float32)
a = np.full(n_pad, 0.02, dtype=np.float32)
b = np.full(n_pad, 0.2, dtype=np.float32)
c_arr = np.full(n_pad, -65.0, dtype=np.float32)
d_arr = np.full(n_pad, 8.0, dtype=np.float32)
I_buf = np.zeros(n_pad, dtype=np.float32)
spikes = np.zeros(n_pad, dtype=np.int32)
nt_types = np.ones(n_pad, dtype=np.float32)

# Pad CSR
row_ptr_pad = np.zeros(n_pad + 1, dtype=np.int32)
row_ptr_pad[:len(row_ptr)] = row_ptr
row_ptr_pad[len(row_ptr):] = row_ptr[-1]
col_idx_pad = np.zeros(len(col_indices), dtype=np.int32)
col_idx_pad[:] = col_indices
weights_pad = np.zeros(len(values), dtype=np.float32)
weights_pad[:] = values

external_I = np.random.uniform(0, 5, n).astype(np.float32)

def ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
def iptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print(f"Running {N_STEPS} steps (C: step_n_neurons + propagate_spikes_ultra)...", end=" ", flush=True)

total_spikes = 0
t0 = time.perf_counter()
for step in range(N_STEPS):
    # 1. Inject external current (same as Rust sim_step)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * step / 100)
    I_buf[:n] = external_I * modulation
    # Clamp I (same as Rust safety_clamp)
    np.clip(I_buf, -100, 100, out=I_buf)

    # 2. Izhikevich step
    lib.step_n_neurons(
        n_pad, ptr(v), ptr(u), ptr(trace), ptr(a), ptr(b), ptr(I_buf),
        ctypes.c_float(DT), ctypes.c_float(0.5),
        iptr(spikes), ptr(c_arr), ptr(d_arr),
    )

    # 3. Find spikes
    spiked = np.where(spikes[:n] != 0)[0].astype(np.int32)
    if len(spiked) > 0:
        total_spikes += len(spiked)
        # 4. Propagate via C
        lib.propagate_spikes_ultra(
            n, iptr(row_ptr_pad), iptr(col_idx_pad), ptr(weights_pad),
            iptr(spiked), len(spiked),
            ptr(nt_types), ptr(I_buf),
            ctypes.c_float(1.0),
        )

elapsed = time.perf_counter() - t0

print()
print(f"  C (step_n_neurons + propagate_spikes_ultra):")
print(f"    Wall time:      {elapsed:.4f}s")
print(f"    Spikes:         {total_spikes:,}")
print(f"    Real-time:      {N_STEPS*DT/elapsed:.1f}x")
print(f"    Steps/s:        {N_STEPS/elapsed:,.0f}")
