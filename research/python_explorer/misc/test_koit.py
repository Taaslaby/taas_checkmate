#!/usr/bin/env python3
"""Smoke test for koit bindings."""
import sys, os, warnings, importlib.util, shutil
import numpy as np
warnings.filterwarnings("ignore")

D = os.path.dirname(__file__)
for p in ['target/release/libneuron_engine.so', 'target/debug/libneuron_engine.so']:
    s = os.path.join(D, p)
    if os.path.exists(s): so_path = s; break

dest = os.path.join(D, '_ne_koit.so')
shutil.copy2(so_path, dest)
spec = importlib.util.spec_from_file_location('neuron_engine', dest)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test(name, fn, *args):
    print(f"  {name:<25} ...", end=" ", flush=True)
    try:
        fn(*args)
        print("✅")
    except Exception as e:
        print(f"❌ {e}")

# 1. rdrand_fill
print("1. rdrand_fill")
test("rdrand_fill", mod.rdrand_fill, np.zeros(64, dtype=np.float32))

# 2. xor_attention (simplified)
print("2. xor_attention")
n_keys, n_bytes = 8, 8
# Simple query: first key is identical, rest are different
query_bits = np.zeros(n_bytes * 8, dtype=np.uint8)
keys_bits = np.zeros((n_keys, n_bytes * 8), dtype=np.uint8)
for i in range(n_keys):
    keys_bits[i] = np.random.randint(0, 2, n_bytes * 8).astype(np.uint8)
keys_bits[0] = query_bits.copy()  # key 0 = query (identical)
values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
qp = np.packbits(query_bits)
kp = np.packbits(keys_bits.ravel())
test("xor_attention", mod.xor_attention, qp, kp, values, np.zeros(1, dtype=np.float32), n_keys, n_bytes)

# 6. xor_attention with actual data from koit
print("6. xor_attention (random)")
n_keys, n_bytes = 16, 8
query = np.random.randint(0, 2, n_bytes * 8, dtype=np.uint8)
keys = np.random.randint(0, 2, n_keys * n_bytes * 8, dtype=np.uint8)
qp = np.packbits(query)
kp = np.packbits(keys)
vals = np.random.rand(n_keys).astype(np.float32)
out = np.zeros(1, dtype=np.float32)
test("xor_attention random", mod.xor_attention, qp, kp, vals, out, n_keys, n_bytes)
print(f"     result: {out[0]:.6f}")

# 3. xor_attention (exact match test)
print("3. xor_attention (exact)")
n_keys, n_bytes = 8, 8
query_bits = np.zeros(n_bytes * 8, dtype=np.uint8)
keys_bits = np.zeros((n_keys, n_bytes * 8), dtype=np.uint8)
for i in range(n_keys):
    keys_bits[i] = np.random.randint(0, 2, n_bytes * 8).astype(np.uint8)
keys_bits[0] = query_bits.copy()
values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
qp = np.packbits(query_bits)
kp = np.packbits(keys_bits.ravel())
test("xor_attention exact", mod.xor_attention, qp, kp, values, np.zeros(1, dtype=np.float32), n_keys, n_bytes)

os.unlink(dest)
print("\n✅ DONE")
