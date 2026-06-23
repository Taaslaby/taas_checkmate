"""
GPU ψ-wave test на FAFB коннектоме: глаза + мозги.
"""
import sys, os, pickle, time
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/Загрузки")
CSV_PATH = os.path.join(DATA_DIR, "connections_princeton.csv.gz")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")
MAPPING_PATH = "/tmp/root_to_csr.pkl"

DT = 0.001
HBAR = 1.0
N_STEPS = 500
PRECOMPUTE = False  # set True if building CSR fresh

# ─── Column names in FAFB CSV ────────────────────────────
PRE = "pre_root_id"
POST = "post_root_id"
WEIGHT = "syn_count"

# ─── Load mapping ─────────────────────────────────────────
print("Loading mapping...")
if not os.path.exists(MAPPING_PATH) or PRECOMPUTE:
    print("  Building mapping from CSV...")
    chunk_iter = pd.read_csv(CSV_PATH, usecols=[PRE, POST],
                             dtype={PRE: "int64", POST: "int64"},
                             chunksize=1_000_000)
    all_ids = set()
    for chunk in chunk_iter:
        all_ids.update(chunk[PRE].unique())
        all_ids.update(chunk[POST].unique())
    sorted_ids = sorted(all_ids)
    root_to_csr = {rid: i for i, rid in enumerate(sorted_ids)}
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(root_to_csr, f)
    print(f"  Built: {len(root_to_csr)} neurons")
else:
    with open(MAPPING_PATH, "rb") as f:
        root_to_csr = pickle.load(f)
    root_to_csr = {int(k): int(v) for k, v in root_to_csr.items()}
    print(f"  Loaded: {len(root_to_csr)} neurons")

n_neurons = len(root_to_csr)

# ─── Build CSR (if needed) ───────────────────────────────
CSR_CACHE = "/tmp/fafb_csr_npz.npz"
if os.path.exists(CSR_CACHE) and not PRECOMPUTE:
    print("Loading cached CSR...")
    data = np.load(CSR_CACHE)
    row_ptr = data["row_ptr"]
    col_indices = data["col_indices"]
    values = data["values"]
    n_synapses = len(values)
    print(f"  {n_neurons} neurons, {n_synapses} synapses")
else:
    print("Building CSR (this takes a while)...")
    row_counts = np.zeros(n_neurons + 1, dtype=np.int32)
    total_syn = 0
    for chunk in pd.read_csv(CSV_PATH, usecols=[PRE, POST],
                              dtype={PRE: "int64", POST: "int64"},
                              chunksize=1_000_000):
        pre = chunk[PRE].values
        post = chunk[POST].values
        mask = np.isin(pre, list(root_to_csr.keys())) & np.isin(post, list(root_to_csr.keys()))
        pre, post = pre[mask], post[mask]
        if len(pre) == 0:
            continue
        pre_idx = np.array([root_to_csr[p] for p in pre], dtype=np.int32)
        np.add.at(row_counts, pre_idx + 1, 1)
        total_syn += len(pre)

    row_ptr = np.cumsum(row_counts, dtype=np.int32)
    row_ptr[0] = 0
    n_synapses = int(row_ptr[-1])
    print(f"  Total synapses: {n_synapses}")

    col_indices = np.zeros(n_synapses, dtype=np.int32)
    values = np.zeros(n_synapses, dtype=np.float32)
    insert_pos = row_ptr[1:].copy() - 1

    for chunk in pd.read_csv(CSV_PATH, usecols=[PRE, POST, WEIGHT],
                              dtype={PRE: "int64", POST: "int64", WEIGHT: "float32"},
                              chunksize=1_000_000):
        pre, post, w = chunk[PRE].values, chunk[POST].values, chunk[WEIGHT].values
        mask = np.isin(pre, list(root_to_csr.keys())) & np.isin(post, list(root_to_csr.keys()))
        pre, post, w = pre[mask], post[mask], w[mask]
        if len(pre) == 0:
            continue
        pre_idx = np.array([root_to_csr[p] for p in pre], dtype=np.int32)
        post_idx = np.array([root_to_csr[p] for p in post], dtype=np.int32)
        for p, po, val in zip(pre_idx, post_idx, w):
            pos = insert_pos[p]
            col_indices[pos] = po
            values[pos] = val
            insert_pos[p] -= 1

    np.savez_compressed(CSR_CACHE, row_ptr=row_ptr, col_indices=col_indices, values=values)
    print("  CSR cached.")

# ─── Optic lobe classification ───────────────────────────
print("Loading classification...")
class_df = pd.read_csv(CLASS_PATH)
has_side = "side" in class_df.columns
optic = class_df[
    (class_df.get("super_class", class_df.get("flow", "")) == "optic") |
    (class_df.get("flow", "") == "intrinsic")
] if "super_class" in class_df.columns and "flow" in class_df.columns else class_df

if has_side and "super_class" in class_df.columns and "flow" in class_df.columns:
    left_optic = class_df[
        (class_df["super_class"] == "optic") &
        (class_df["flow"] == "intrinsic") &
        (class_df["side"] == "left")
    ]["root_id"].values
    right_optic = class_df[
        (class_df["super_class"] == "optic") &
        (class_df["flow"] == "intrinsic") &
        (class_df["side"] == "right")
    ]["root_id"].values
else:
    all_root_ids = np.array(sorted(root_to_csr.keys()))
    mid = len(all_root_ids) // 2
    left_optic = all_root_ids[:mid]
    right_optic = all_root_ids[mid:]

left_idx = np.array([root_to_csr.get(int(rid), -1) for rid in left_optic])
right_idx = np.array([root_to_csr.get(int(rid), -1) for rid in right_optic])
left_idx = left_idx[left_idx >= 0]
right_idx = right_idx[right_idx >= 0]
print(f"  Left optic: {len(left_idx)}, Right optic: {len(right_idx)}")

# ─── Init ψ-wave ─────────────────────────────────────────
print("Initializing ψ-wave...")
rng = np.random.default_rng(42)
angles = rng.uniform(0, 2 * np.pi, n_neurons).astype(np.float32)
v_init = np.cos(angles).astype(np.float32)
u_init = np.sin(angles).astype(np.float32)
nrm = np.sqrt(v_init**2 + u_init**2)
print(f"  Initial |ψ|: {nrm.min():.6f} – {nrm.max():.6f}")

# ─── Stimulus: phase gradient on left optic ──────────────
print("Creating stimulus...")
i_ext = np.zeros(n_neurons, dtype=np.float32)
if len(left_idx) > 0:
    phase = np.linspace(0, 4 * np.pi, len(left_idx), dtype=np.float32)
    i_ext[left_idx] = 50.0 * np.sin(phase)
if len(right_idx) > 0:
    i_ext[right_idx[:min(len(right_idx), 2000)]] = 5.0

# ─── Run GPU ψ-wave ──────────────────────────────────────
import taas_checkmate as tcm
print(f"Running {N_STEPS} ψ-wave steps on GPU...")
t0 = time.time()

v = np.ascontiguousarray(v_init.copy())
u = np.ascontiguousarray(u_init.copy())
rp = np.ascontiguousarray(row_ptr, dtype=np.int32)
ci = np.ascontiguousarray(col_indices, dtype=np.int32)
vals = np.ascontiguousarray(values, dtype=np.float32)
ie = np.ascontiguousarray(i_ext, dtype=np.float32)

elapsed, v_final, u_final = tcm.bench_psi_wave_gpu(
    rp, ci, vals, v, u, DT, HBAR, N_STEPS, ie
)

v_final = np.array(v_final, dtype=np.float32)
u_final = np.array(u_final, dtype=np.float32)
print(f"  GPU: {elapsed:.3f}s ({N_STEPS/elapsed:.0f} step/s, {elapsed/N_STEPS*1e6:.1f}μs/step)")

# ─── Analysis ─────────────────────────────────────────────
print("\n═ RESULTS ════════════════════════════════════")

nrm_final = np.sqrt(v_final**2 + u_final**2)
print(f"Final |ψ|: {nrm_final.min():.6f} – {nrm_final.max():.6f}")
ok = np.abs(nrm_final - 1.0).max() < 1e-5
print(f"  {'✅ Normalization OK' if ok else '⚠️  Max dev: ' + str(np.abs(nrm_final - 1.0).max())}")

if len(left_idx) > 1 and len(right_idx) > 1:
    lp = np.arctan2(u_final[left_idx], v_final[left_idx])
    rp = np.arctan2(u_final[right_idx], v_final[right_idx])
    def op(ph): return np.abs(np.mean(np.exp(1j * ph)))
    print(f"Coherence L: {op(lp):.4f}  R: {op(rp):.4f}")
    ml = np.arctan2(np.mean(np.sin(lp)), np.mean(np.cos(lp)))
    mr = np.arctan2(np.mean(np.sin(rp)), np.mean(np.cos(rp)))
    print(f"Phase diff L-R: {((ml - mr + np.pi) % (2*np.pi) - np.pi):.4f} rad")
    print(f"Mean u L: {np.mean(u_final[left_idx]):.6f}  R: {np.mean(u_final[right_idx]):.6f}")
    print(f"Asymmetry L-R: {np.mean(u_final[left_idx]) - np.mean(u_final[right_idx]):.6f}")

print(f"\n── Benchmark ──")
print(f"FAFB GPU: {N_STEPS/elapsed:.0f} step/s")
print(f"Est 10K steps: {10000 / (N_STEPS / elapsed):.1f}s")
