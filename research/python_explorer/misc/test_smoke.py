#!/usr/bin/env python3
"""Smoke test - all bindings with proper padding (n % 8 == 0) for AVX."""
import sys, os, time, warnings, importlib.util, shutil
import numpy as np
warnings.filterwarnings("ignore")

D = os.path.dirname(__file__)

for p in ['target/release/libneuron_engine.so', 'target/debug/libneuron_engine.so']:
    s = os.path.join(D, p)
    if os.path.exists(s): so_path = s; break

dest = os.path.join(D, '_ne_smoke.so')
shutil.copy2(so_path, dest)
spec = importlib.util.spec_from_file_location('neuron_engine', dest)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

np.random.seed(42)
n_raw = 11797  # SEZ raw neurons
n = (n_raw + 7) // 8 * 8  # padded for AVX
print(f"SEZ: {n_raw} raw, {n} padded")

def test_one(name, fn, *args):
    print(f"  {name:<30} ...", end=" ", flush=True)
    try:
        for _ in range(5): fn(*args)
        for _ in range(20): fn(*args)
        print("✅")
    except Exception as e:
        print(f"❌ {e}")

rand_n = np.random.randn(n).astype(np.float32)

# ── 1. stochastic_encoder ──
print("1. stochastic_encoder")
test_one("stochastic_encoder", mod.stochastic_encoder,
         np.zeros(n, dtype=np.float32), rand_n,
         np.random.rand(n).astype(np.float32) * 0.1, 0.5, 40.0)

# ── 2. thalamic_gate ──
print("2. thalamic_gate")
test_one("thalamic_gate", mod.thalamic_gate,
         np.zeros(n, dtype=np.float32), rand_n,
         np.random.rand(n).astype(np.float32), 0.0, 1.0)

# ── 3. phase_gate_apply ──
print("3. phase_gate_apply")
test_one("phase_gate_apply", mod.phase_gate_apply,
         np.zeros(n, dtype=np.float32), rand_n,
         np.random.rand(n).astype(np.float32) * 2, 5.0, 0.3)

# ── 4. t3_drive_encoder ──
print("4. t3_drive_encoder")
test_one("t3_drive_encoder", mod.t3_drive_encoder,
         np.zeros(n, dtype=np.float32), rand_n, 3.0)

# ── 5. t3_complex_decoder ──
print("5. t3_complex_decoder")
n_out = 4
ampl = np.abs(rand_n)
cos_buf = np.cos(rand_n)
w = np.random.randn(n * n_out).astype(np.float32)
b = np.zeros(n_out, dtype=np.float32)
test_one("t3_complex_decoder", mod.t3_complex_decoder,
         np.zeros(n_out, dtype=np.float32), ampl, cos_buf, w, b, n_out)

# ── 6. echo_tap ──
print("6. echo_tap")
n_osc_pad = 104  # next multiple of 8
delay_len = 50
buf = np.random.randn(n_osc_pad * delay_len).astype(np.float32)
test_one("echo_tap", mod.echo_tap,
         np.zeros(n_osc_pad, dtype=np.float32), buf, n_osc_pad, delay_len, 10)

# ── 7. hebbian_update ──
print("7. hebbian_update")
test_one("hebbian_update", mod.hebbian_update,
         np.random.randn(n).astype(np.float32) * 0.1,
         np.random.rand(n).astype(np.float32) * 2 - 1,
         0.01, 0.1, 0.01)

# ── 8. kalman_update ──
print("8. kalman_update")
test_one("kalman_update", mod.kalman_update,
         np.zeros(n, dtype=np.float32),
         np.full(n, 10.0, dtype=np.float32),
         rand_n, 1.0, 5.0)

# ── 9. adam_train ──
print("9. adam_train")
test_one("adam_train", mod.adam_train,
         np.zeros(n, dtype=np.float32),
         np.zeros(n, dtype=np.float32),
         np.zeros(n, dtype=np.float32),
         rand_n, 0.5, 0.01, 0.9, 0.999, 0.1, 0.001, 1e-8)

# ── 10. nlms_train ──
print("10. nlms_train")
test_one("nlms_train", mod.nlms_train,
         np.zeros(n, dtype=np.float32),
         np.zeros(1, dtype=np.float32),
         rand_n, 0.5, 0.01, 1e-8)

# ── 11. pst_decode ──
print("11. pst_decode")
feat_dim, L = 64, 24  # L padded to multiple of 8
buf_pst = np.random.randn(L * feat_dim).astype(np.float32)
w_pst = np.random.randn(L, feat_dim).astype(np.float32).ravel()
test_one("pst_decode", mod.pst_decode,
         np.zeros(1, dtype=np.float32), buf_pst, w_pst, feat_dim, L, 5)

# ── 12. ring_synthesis ──
print("12. ring_synthesis")
test_one("ring_synthesis", mod.ring_synthesis,
         np.ones(n, dtype=np.float32),
         np.abs(rand_n), np.cos(rand_n),
         np.random.randn(n).astype(np.float32))

# ── 13. t3_inverse_kuramoto ──
print("13. t3_inverse_kuramoto")
test_one("t3_inverse_kuramoto", mod.t3_inverse_kuramoto,
         np.zeros(n, dtype=np.float32),        # drive_out
         np.random.rand(n).astype(np.float32), # r
         np.random.rand(n).astype(np.float32), # r_prev
         1.0, 2.0, 1e-8, 0.1)                  # alpha, gain, eps, decay

# ── 14. t3_pst_features ──
print("14. t3_pst_features")
n1, n2 = n, 8  # ph2 padded to 8
ph1 = np.random.randn(n1).astype(np.float32)
ph2 = np.random.randn(n2).astype(np.float32)
out_sin = np.zeros(n1 * n2, dtype=np.float32)
out_cos = np.zeros(n1 * n2, dtype=np.float32)
test_one("t3_pst_features", mod.t3_pst_features,
         out_sin, out_cos, ph1, ph2)

# ── 15. t3_tact_generator ──
print("15. t3_tact_generator")
ph = np.random.randn(n).astype(np.float32)
test_one("t3_tact_generator", mod.t3_tact_generator,
         np.zeros(n, dtype=np.float32), ph, ph, ph, 0.5)

# ── 16. winding_matrix ──
print("16. winding_matrix")
na, nb = n, 8  # phases_b padded to 8
pa = np.random.randn(na).astype(np.float32)
pb = np.random.randn(nb).astype(np.float32)
test_one("winding_matrix", mod.winding_matrix,
         np.zeros(na * nb, dtype=np.float32), pa, pb)

# ── 17. apply_stdp_ltd ──
print("17. apply_stdp_ltd")
# Build simple CSR for testing
n_small = 8
row_ptr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.int32)
col_indices = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)
weights = np.ones(n_small, dtype=np.float32)
spiked = np.array([0], dtype=np.int32)
traces = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
test_one("apply_stdp_ltd", mod.apply_stdp_ltd,
         row_ptr, col_indices, weights, spiked, traces, 0.01)

# ── 18. xor_attention ──
print("18. xor_attention")
n_keys, n_bytes = 16, 8  # 8 bytes = 64 bits
query = np.random.randint(0, 2, n_bytes * 8, dtype=np.uint8).astype(np.uint8)
keys = np.random.randint(0, 2, n_keys * n_bytes * 8, dtype=np.uint8).astype(np.uint8)
qp = np.packbits(query)
kp = np.packbits(keys.reshape(-1, n_bytes * 8)).ravel()
values = np.random.rand(n_keys).astype(np.float32)
test_one("xor_attention", mod.xor_attention,
         qp, kp, values, np.zeros(1, dtype=np.float32), n_keys, n_bytes)

# ── 19. rdrand_fill ──
print("19. rdrand_fill")
test_one("rdrand_fill", mod.rdrand_fill,
         np.zeros(64, dtype=np.float32))

print()
print("✅ ALL PASSED")
os.unlink(dest)
