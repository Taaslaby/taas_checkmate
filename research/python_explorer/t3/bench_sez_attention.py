#!/usr/bin/env python3
"""
Benchmark: XOR attention vs softmax attention on real SEZ connectome.
Compares speed, output correlation, and attention distribution characteristics.
"""
import os, time, warnings, importlib.util, shutil
import numpy as np
warnings.filterwarnings("ignore")

np.random.seed(42)

# ── Load module ──
d = os.path.dirname(__file__) or "."
for p in ["target/release/libneuron_engine.so", "target/debug/libneuron_engine.so"]:
    s = os.path.join(d, p)
    if os.path.exists(s):
        so_path = s
        break

shutil.copy2(so_path, os.path.join(d, "_ne_sez.so"))
spec = importlib.util.spec_from_file_location("neuron_engine", os.path.join(d, "_ne_sez.so"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ── Load SEZ connectome ──
print("Loading SEZ connectome...", end=" ", flush=True)
data = np.load("/home/taaslab/Проекты/t3-wavelet/data/connectome/sez_connectome.npz")
row_ptr = data["row_ptr"]        # (11798,) int32
col_idx = data["col_indices"]    # (846992,) int32
values = data["values"]          # (846992,) float32
n_neurons = len(row_ptr) - 1     # 11797
print(f"{n_neurons} neurons, {len(col_idx)} synapses")

# ── Hyperparameters ──
N_BITS = 256      # feature size (32 bytes)
N_KEYS = 1024     # attention keys (subset of neurons)
N_ITER = 500      # benchmark iterations

print(f"\nConfig: {N_KEYS} keys, {N_BITS}-bit features, {N_ITER} iterations")

# ── Feature extraction ──
# Hash-based compression: map each target neuron to a random bit position
# This preserves connectivity structure in a compressed binary fingerprint
rng = np.random.default_rng(42)

def make_hash_map(n_bits):
    """Create a hash map for given bit width."""
    return rng.integers(0, n_bits, size=n_neurons).astype(np.uint32)

hash_map = make_hash_map(N_BITS)

def neuron_to_bits(neuron_idx, bits=N_BITS, hmap=None):
    """Convert a neuron's outgoing connectivity to a packed bit vector."""
    if hmap is None:
        hmap = hash_map
    bits_arr = np.zeros(bits, dtype=np.uint8)
    start = row_ptr[neuron_idx]
    end = row_ptr[neuron_idx + 1]
    for t in col_idx[start:end]:
        if t < len(hmap):
            bit_pos = hmap[t]
            if bit_pos < bits:
                bits_arr[bit_pos] = 1
    return np.packbits(bits_arr)

# Build query/key vectors
print("Building feature vectors...", end=" ", flush=True)
query_bits = neuron_to_bits(0)  # neuron 0 as query
n_bytes = N_BITS // 8

keys_packed = np.zeros(N_KEYS * n_bytes, dtype=np.uint8)
for i in range(N_KEYS):
    keys_packed[i * n_bytes:(i + 1) * n_bytes] = neuron_to_bits(i)
print("done")

# Values: normalized total input strength per neuron
values_arr = np.zeros(N_KEYS, dtype=np.float32)
for i in range(N_KEYS):
    start = row_ptr[i]
    end = row_ptr[i + 1]
    if end > start:
        values_arr[i] = np.mean(values[start:end])
    else:
        values_arr[i] = 0.0

v_max = np.max(np.abs(values_arr))
if v_max > 0:
    values_arr /= v_max


# ══════════════════════════════════════════
#  1. SINGLE QUERY BENCHMARK
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  1. SINGLE QUERY BENCHMARK")
print(f"{'='*60}")

# XOR attention
out = np.zeros(1, dtype=np.float32)
for _ in range(50):
    mod.xor_attention(query_bits, keys_packed, values_arr, out, N_KEYS, n_bytes)

t0 = time.perf_counter()
for _ in range(N_ITER):
    mod.xor_attention(query_bits, keys_packed, values_arr, out, N_KEYS, n_bytes)
t_xor = (time.perf_counter() - t0) / N_ITER * 1000
result_xor = out[0]
print(f"  XOR:     {t_xor:.6f} ms/query  → result = {result_xor:.6f}")

# Softmax attention
query_f = np.unpackbits(query_bits).astype(np.float32)[:N_BITS] * 2.0 - 1.0
keys_f = np.unpackbits(keys_packed).astype(np.float32).reshape(N_KEYS, n_bytes * 8)[:, :N_BITS] * 2.0 - 1.0

def softmax_attention(q, k, v):
    scores = np.dot(k, q) / np.sqrt(k.shape[1])
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= np.sum(weights) + 1e-10
    return np.dot(weights, v)

for _ in range(50):
    r = softmax_attention(query_f, keys_f, values_arr)

t0 = time.perf_counter()
for _ in range(N_ITER):
    r = softmax_attention(query_f, keys_f, values_arr)
t_sfm = (time.perf_counter() - t0) / N_ITER * 1000
result_sfm = r
print(f"  Softmax: {t_sfm:.6f} ms/query  → result = {result_sfm:.6f}")

print(f"\n  ▶ Speedup: {t_sfm / t_xor:.1f}x")
print(f"  ▶ Output ratio (XOR/Softmax): {result_xor / result_sfm:.4f}")


# ══════════════════════════════════════════
#  2. ATTENTION DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  2. ATTENTION DISTRIBUTION ANALYSIS")
print(f"{'='*60}")

# XOR weights (computed in Python for distribution)
xor_weights = np.zeros(N_KEYS, dtype=np.float32)
for i in range(N_KEYS):
    k_bits = np.unpackbits(keys_packed[i * n_bytes:(i + 1) * n_bytes])[:N_BITS]
    q_bits = np.unpackbits(query_bits)[:N_BITS]
    dist = np.sum(q_bits != k_bits)
    xor_weights[i] = max(0.0, 1.0 - dist / N_BITS)
xor_weights /= np.sum(xor_weights) + 1e-10

# Softmax distribution
scores = np.dot(keys_f, query_f) / np.sqrt(N_BITS)
scores -= np.max(scores)
sfm_weights = np.exp(scores)
sfm_weights /= np.sum(sfm_weights) + 1e-10

# Statistical comparison
corr = np.corrcoef(xor_weights, sfm_weights)[0, 1]
kl_div = np.sum(sfm_weights * np.log((sfm_weights + 1e-10) / (xor_weights + 1e-10)))

nz_xor = np.sum(xor_weights > 1e-6)
nz_sfm = np.sum(sfm_weights > 1e-6)

print(f"  Correlation (weights):         {corr:.4f}")
print(f"  KL divergence (softmax||xor):  {kl_div:.4f}")
print(f"  Non-zero weights (XOR):        {nz_xor} / {N_KEYS} ({100*nz_xor/N_KEYS:.1f}%)")
print(f"  Non-zero weights (softmax):    {nz_sfm} / {N_KEYS} ({100*nz_sfm/N_KEYS:.1f}%)")

# Top-K overlap
for k in [1, 5, 10, 50]:
    top_xor = set(np.argsort(xor_weights)[-k:])
    top_sfm = set(np.argsort(sfm_weights)[-k:])
    overlap = len(top_xor & top_sfm)
    print(f"  Top-{k:<4} overlap:            {overlap} / {k} ({100*overlap/k:.0f}%)")

# Entropy / sparsity
ent_xor = -np.sum(xor_weights * np.log2(xor_weights + 1e-10))
ent_sfm = -np.sum(sfm_weights * np.log2(sfm_weights + 1e-10))
max_ent = np.log2(N_KEYS)
print(f"  Entropy (XOR):                {ent_xor:.2f} / {max_ent:.2f} bits (sparsity={ent_xor/max_ent:.3f})")
print(f"  Entropy (softmax):            {ent_sfm:.2f} / {max_ent:.2f} bits (sparsity={ent_sfm/max_ent:.3f})")

# Correlation with connectivity strength
conn_strength = np.array([np.sum(values[row_ptr[i]:row_ptr[i+1]]) for i in range(N_KEYS)], dtype=np.float32)
conn_strength /= np.max(conn_strength) + 1e-10
corr_xor_conn = np.corrcoef(xor_weights, conn_strength)[0, 1]
corr_sfm_conn = np.corrcoef(sfm_weights, conn_strength)[0, 1]
print(f"  Corr with connectivity:       XOR={corr_xor_conn:.4f}  Softmax={corr_sfm_conn:.4f}")


# ══════════════════════════════════════════
#  3. SCALE SWEEP: XOR scalability
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  3. SCALE SWEEP (XOR attention)")
print(f"{'='*60}")

scale_configs = [
    ("256 keys × 128b",   256,  128,  2000),
    ("1K keys × 128b",    1024, 128,  1000),
    ("4K keys × 128b",    4096, 128,  500),
    ("12K keys × 128b",   11797, 128,  100),
    ("256 keys × 256b",   256,  256,  2000),
    ("1K keys × 256b",    1024, 256,  1000),
    ("4K keys × 256b",    4096, 256,  200),
    ("12K keys × 256b",   11797, 256,  50),
]

print(f"{'Config':<22} {'XOR (ms)':<12} {'Softmax (ms)':<14} {'Speedup':<10}")
print("-" * 58)

for label, nk, nb, nit in scale_configs:
    nb_bytes = nb // 8
    nk_use = min(nk, n_neurons)

    # Build features with per-config hash map
    hmap = make_hash_map(nb)
    qb = neuron_to_bits(0, nb, hmap)
    kp = np.zeros(nk_use * nb_bytes, dtype=np.uint8)
    for i in range(nk_use):
        kp[i * nb_bytes:(i + 1) * nb_bytes] = neuron_to_bits(i, nb, hmap)

    vals = np.ones(nk_use, dtype=np.float32)

    # XOR
    o = np.zeros(1, dtype=np.float32)
    for _ in range(10):
        mod.xor_attention(qb, kp, vals, o, nk_use, nb_bytes)
    t0 = time.perf_counter()
    for _ in range(nit):
        mod.xor_attention(qb, kp, vals, o, nk_use, nb_bytes)
    tx = (time.perf_counter() - t0) / nit * 1000

    # Softmax
    qf = np.unpackbits(qb).astype(np.float32)[:nb] * 2.0 - 1.0
    kf = np.unpackbits(kp).astype(np.float32).reshape(nk_use, nb_bytes * 8)[:, :nb] * 2.0 - 1.0
    for _ in range(10):
        softmax_attention(qf, kf, vals)
    t0 = time.perf_counter()
    for _ in range(nit):
        softmax_attention(qf, kf, vals)
    ts = (time.perf_counter() - t0) / nit * 1000

    speedup = ts / tx if tx > 0 else float("inf")
    print(f"{label:<22} {tx:<12.6f} {ts:<14.6f} {speedup:<10.1f}x")


# ══════════════════════════════════════════
#  4. MULTI-QUERY BENCHMARK (50 random)
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  4. MULTI-QUERY BENCHMARK (50 queries)")
print(f"{'='*60}")

n_queries = 50
t_xor_total = 0.0
t_sfm_total = 0.0
xor_results = []
sfm_results = []

for q_idx in range(n_queries):
    qb = neuron_to_bits(q_idx)
    qf = np.unpackbits(qb).astype(np.float32)[:N_BITS] * 2.0 - 1.0

    o = np.zeros(1, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        mod.xor_attention(qb, keys_packed, values_arr, o, N_KEYS, n_bytes)
    t_xor_total += (time.perf_counter() - t0) / 100 * 1000
    xor_results.append(o[0])

    t0 = time.perf_counter()
    for _ in range(100):
        r = softmax_attention(qf, keys_f, values_arr)
    t_sfm_total += (time.perf_counter() - t0) / 100 * 1000
    sfm_results.append(r)

xor_results = np.array(xor_results)
sfm_results = np.array(sfm_results)

corr_mq = np.corrcoef(xor_results, sfm_results)[0, 1]
print(f"  Avg XOR:     {t_xor_total/n_queries:.6f} ms/query")
print(f"  Avg Softmax: {t_sfm_total/n_queries:.6f} ms/query")
print(f"  Avg Speedup: {t_sfm_total/t_xor_total:.1f}x")
print(f"  Output correlation across queries: {corr_mq:.4f}")
print(f"  XOR output range: [{xor_results.min():.4f}, {xor_results.max():.4f}]")
print(f"  Softmax output range: [{sfm_results.min():.4f}, {sfm_results.max():.4f}]")


# ══════════════════════════════════════════
#  5. BIT-WIDTH VS QUALITY TRADE-OFF
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  5. BIT-WIDTH VS QUALITY (correlation with softmax)")
print(f"{'='*60}")

for bits in [32, 64, 128, 256, 512]:
    nb = bits // 8
    hmap = make_hash_map(bits)
    qb = neuron_to_bits(0, bits, hmap)
    kp = np.zeros(N_KEYS * nb, dtype=np.uint8)
    for i in range(N_KEYS):
        kp[i * nb:(i + 1) * nb] = neuron_to_bits(i, bits, hmap)

    qf = np.unpackbits(qb).astype(np.float32)[:bits] * 2.0 - 1.0
    kf = np.unpackbits(kp).astype(np.float32).reshape(N_KEYS, nb * 8)[:, :bits] * 2.0 - 1.0

    # XOR weights
    xw = np.zeros(N_KEYS, dtype=np.float32)
    for i in range(N_KEYS):
        kb = np.unpackbits(kp[i * nb:(i + 1) * nb])[:bits]
        qb_u = np.unpackbits(qb)[:bits]
        dist = np.sum(qb_u != kb)
        xw[i] = max(0.0, 1.0 - dist / bits)
    xw /= np.sum(xw) + 1e-10

    # Softmax weights
    sc = np.dot(kf, qf) / np.sqrt(bits)
    sc -= np.max(sc)
    sw = np.exp(sc)
    sw /= np.sum(sw) + 1e-10

    c = np.corrcoef(xw, sw)[0, 1]
    print(f"  {bits:>4} bits ({nb:>2}B): correlation = {c:.4f}")


# ── Cleanup ──
os.unlink(os.path.join(d, "_ne_sez.so"))
print(f"\n{'='*60}")
print(f"  ✅ COMPLETE")
print(f"{'='*60}")
