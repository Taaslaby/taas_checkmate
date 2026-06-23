#!/usr/bin/env python3
"""
Build merged FAFB+MANC CSR with cross-compartment bridges.

→ 161772 neurons (138584 FAFB + 23188 MANC)
→ ~10.7M total synapses (5.3M FAFB + 5.2M MANC + 168K bridge)
"""
import numpy as np, pandas as pd, pickle, os, time

DATA_DIR = os.path.expanduser("~/Загрузки")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")
MAPPING_PATH = "/tmp/root_to_csr.pkl"
RNG = np.random.default_rng(42)

# ─── 1. FAFB ─────────────────────────────────────────────
print("Loading FAFB...")
with open(MAPPING_PATH, 'rb') as f:
    root_to_csr = pickle.load(f)
fafb_ids = np.array(sorted(root_to_csr.keys(), key=lambda r: root_to_csr[r]))
n_fafb = len(fafb_ids)

fa = np.load('/tmp/fafb_csr_npz.npz')
fa_row = fa['row_ptr'].astype(np.int32)
fa_col = fa['col_indices'].astype(np.int32)
fa_val = fa['values'].astype(np.float32)
n_fa_syn = len(fa_col)
print(f"  {n_fafb} neurons, {n_fa_syn} synapses")

# ─── 2. MANC ─────────────────────────────────────────────
print("Loading MANC...")
conn = pd.read_csv(f'{DATA_DIR}/v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv')
manc_ids = pd.unique(conn[['bodyId_pre','bodyId_post']].values.ravel())
manc_ids.sort()
n_manc = len(manc_ids)
print(f"  {n_manc} neurons, {len(conn)} edges")

n_total = n_fafb + n_manc
manc_to_idx = {bid: i + n_fafb for i, bid in enumerate(manc_ids)}

# Sort MANC edges by pre (then post)
pre_arr = np.array([manc_to_idx[pid] for pid in conn['bodyId_pre']], dtype=np.int32)
post_arr = np.array([manc_to_idx[pid] for pid in conn['bodyId_post']], dtype=np.int32)
w_arr = conn['weight'].values.astype(np.float32)
order = np.lexsort((post_arr, pre_arr))
pre_arr = pre_arr[order]
post_arr = post_arr[order]
w_arr = w_arr[order]

# ─── 3. Cross-compartment bridges ────────────────────────
print("Building cross-compartment bridges...")

clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(fafb_ids)].copy()
fafb_root_to_idx = {rid: i for i, rid in enumerate(fafb_ids)}
clf['csr_idx'] = clf['root_id'].map(fafb_root_to_idx)
desc_idx = clf.loc[clf['super_class'] == 'descending', 'csr_idx'].values.astype(np.int32)
central_idx = clf.loc[clf['super_class'] == 'central', 'csr_idx'].values.astype(np.int32)
sensory_idx = clf.loc[clf['super_class'] == 'sensory', 'csr_idx'].values.astype(np.int32)

manc_prop = pd.read_feather(f'{DATA_DIR}/v1.0_manc-v1.0-neuron-properties.feather')
manc_prop_t = manc_prop[manc_prop['bodyId'].isin(manc_ids)].copy()
manc_prop_t['csr_idx'] = manc_prop_t['bodyId'].map(manc_to_idx)

asc_idx = manc_prop_t.loc[manc_prop_t['class'] == 'ascending neuron', 'csr_idx'].values.astype(np.int32)
sens_asc_idx = manc_prop_t.loc[manc_prop_t['class'] == 'sensory ascending', 'csr_idx'].values.astype(np.int32)
motor_idx = manc_prop_t.loc[manc_prop_t['class'] == 'motor neuron', 'csr_idx'].values.astype(np.int32)
intrinsic_idx = manc_prop_t.loc[manc_prop_t['class'] == 'intrinsic neuron', 'csr_idx'].values.astype(np.int32)

bridge_pre, bridge_post, bridge_w = [], [], []

# Descending: FAFB → MANC
desc_targets = np.unique(np.concatenate([motor_idx, intrinsic_idx]))
if len(desc_targets) > 0 and len(desc_idx) > 0:
    n_t = min(60, len(desc_targets))
    for pre in desc_idx:
        tgt = RNG.choice(desc_targets, n_t, replace=False)
        bridge_pre.extend([pre]*n_t); bridge_post.extend(tgt); bridge_w.extend([1.0]*n_t)

# Ascending: MANC → FAFB
asc_targets = central_idx
if len(asc_targets) > 0 and len(asc_idx) > 0:
    n_t = min(40, len(asc_targets))
    for pre in asc_idx:
        tgt = RNG.choice(asc_targets, n_t, replace=False)
        bridge_pre.extend([pre]*n_t); bridge_post.extend(tgt); bridge_w.extend([1.0]*n_t)

# Sensory ascending: MANC → FAFB (central + sensory)
sa_targets = np.unique(np.concatenate([central_idx, sensory_idx]))
if len(sa_targets) > 0 and len(sens_asc_idx) > 0:
    n_t = min(30, len(sa_targets))
    for pre in sens_asc_idx:
        tgt = RNG.choice(sa_targets, n_t, replace=False)
        bridge_pre.extend([pre]*n_t); bridge_post.extend(tgt); bridge_w.extend([1.0]*n_t)

n_bridge = len(bridge_pre)
print(f"  Bridge edges: {n_bridge} ({len(desc_idx)} desc + {len(asc_idx)} asc + {len(sens_asc_idx)} sens_asc)")

# ─── 4. Build CSR from scratch ───────────────────────────
print("Building combined edge list...")

# Collect all edges: FAFB edges (pre in 0..n_fafb-1, post in 0..n_fafb-1)
# + MANC edges (pre in n_fafb..n_total-1, post in n_fafb..n_total-1)
# + bridge edges (pre can be any, post can be any)

# FAFB edges: expand CSR
print("  Expanding FAFB CSR...")
fa_pre = np.repeat(np.arange(n_fafb, dtype=np.int32), np.diff(fa_row))

# MANC edges already in edge list
ma_pre = pre_arr
ma_post = post_arr
ma_w = w_arr

# Bridge
br_pre = np.array(bridge_pre, dtype=np.int32)
br_post = np.array(bridge_post, dtype=np.int32)
br_w = np.array(bridge_w, dtype=np.float32)

# Merge and sort
all_pre = np.concatenate([fa_pre, ma_pre, br_pre])
all_post = np.concatenate([fa_col, ma_post, br_post])
all_w = np.concatenate([fa_val, ma_w, br_w])

print(f"  Sorting {len(all_pre)} edges by (pre, post)...")
order = np.lexsort((all_post, all_pre))
all_pre = all_pre[order]
all_post = all_post[order]
all_w = all_w[order]

print(f"  Building row_ptr ({n_total+1} entries)...")
row_ptr = np.zeros(n_total + 1, dtype=np.int32)
current = -1
for i in range(len(all_pre)):
    pre = all_pre[i]
    while current < pre:
        current += 1
        row_ptr[current + 1] = i
current += 1
row_ptr[current:] = len(all_pre)  # fill remaining

assert row_ptr[0] == 0
assert row_ptr[-1] == len(all_pre)
print(f"  ✅ row_ptr: [{row_ptr[0]}, ..., {row_ptr[-1]}]")

# ─── 5. Save ─────────────────────────────────────────────
out = '/tmp/fafb_manc_csr_bridged.npz'
np.savez_compressed(out,
    row_ptr=row_ptr, col_indices=all_post, values=all_w,
    n_fafb=n_fafb, n_manc=n_manc)

# Save bridge metadata
bm = {'desc_idx': desc_idx, 'central_idx': central_idx, 'sensory_idx': sensory_idx,
      'asc_idx': asc_idx, 'sens_asc_idx': sens_asc_idx,
      'motor_idx': motor_idx, 'intrinsic_idx': intrinsic_idx}
with open('/tmp/fafb_manc_bridge_meta.pkl', 'wb') as f:
    pickle.dump(bm, f)

print(f"\n{'='*60}")
print(f"  MERGED FAFB+MANC with bridges")
print(f"{'='*60}")
print(f"  Neurons:   {n_total:,} ({n_fafb:,} FAFB + {n_manc:,} MANC)")
print(f"  Synapses:  {len(all_pre):,}")
print(f"    FAFB:    {len(fa_pre):,}")
print(f"    MANC:    {len(ma_pre):,}")
print(f"    Bridge:  {n_bridge:,}")
print(f"  File:      {out} ({os.path.getsize(out)/1024**2:.1f} MB)")
