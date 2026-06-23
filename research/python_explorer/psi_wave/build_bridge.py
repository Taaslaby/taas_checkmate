#!/usr/bin/env python3
"""
Build cross-compartment bridge between FAFB (brain) and MANC (VNC).

Adds synthetic descending (FAFB→MANC) and ascending (MANC→FAFB) connections
to the merged CSR, enabling ψ-wave propagation between brain and nerve cord.
"""
import numpy as np, pandas as pd, pickle, os, time

DATA_DIR = os.path.expanduser("~/Загрузки")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")
BRIDGE_PATH = "/tmp/fafb_manc_bridge.pkl"

# ─── Load merged CSR ──────────────────────────────────────
print("Loading merged CSR...")
data = np.load('/tmp/fafb_manc_csr.npz')
row_ptr = data['row_ptr'].copy()
col_indices = data['col_indices'].copy()
values = data['values'].copy()
n_fafb = int(data['n_fafb'])
n_manc = int(data['n_manc'])
n_total = n_fafb + n_manc
print(f"  {n_total} neurons ({n_fafb} FAFB + {n_manc} MANC), {len(values)} synapses (before bridge)")

with open(BRIDGE_PATH, 'rb') as f:
    bridge = pickle.load(f)
fafb_ids = bridge['fafb_ids']
manc_ids = bridge['manc_ids']

# ─── Load FAFB classification ─────────────────────────────
clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(fafb_ids)].copy()
fafb_root_to_idx = {rid: i for i, rid in enumerate(fafb_ids)}
clf['csr_idx'] = clf['root_id'].map(fafb_root_to_idx)
desc_idx = clf.loc[clf['super_class'] == 'descending', 'csr_idx'].values.astype(np.int32)
central_idx = clf.loc[clf['super_class'] == 'central', 'csr_idx'].values.astype(np.int32)
sensory_idx = clf.loc[clf['super_class'] == 'sensory', 'csr_idx'].values.astype(np.int32)
print(f"  FAFB descending: {len(desc_idx)}")
print(f"  FAFB central: {len(central_idx)}")
print(f"  FAFB sensory: {len(sensory_idx)}")

# ─── Load MANC neuron properties ──────────────────────────
manc_prop = pd.read_feather(f'{DATA_DIR}/v1.0_manc-v1.0-neuron-properties.feather')
manc_prop = manc_prop[manc_prop['bodyId'].isin(manc_ids)].copy()
manc_body_to_idx = {rid: i + n_fafb for i, rid in enumerate(manc_ids)}
manc_prop['csr_idx'] = manc_prop['bodyId'].map(manc_body_to_idx)

# Find ascending neurons in MANC
asc_idx = manc_prop.loc[manc_prop['class'] == 'ascending neuron', 'csr_idx'].values.astype(np.int32)
sens_asc_idx = manc_prop.loc[manc_prop['class'] == 'sensory ascending', 'csr_idx'].values.astype(np.int32)
motor_idx = manc_prop.loc[manc_prop['class'] == 'motor neuron', 'csr_idx'].values.astype(np.int32)
intrinsic_idx = manc_prop.loc[manc_prop['class'] == 'intrinsic neuron', 'csr_idx'].values.astype(np.int32)
print(f"  MANC ascending: {len(asc_idx)}")
print(f"  MANC sensory ascending: {len(sens_asc_idx)}")
print(f"  MANC motor: {len(motor_idx)}")
print(f"  MANC intrinsic: {len(intrinsic_idx)}")

# ─── Build bridge edges ───────────────────────────────────
rng = np.random.default_rng(42)

new_pre_ids = []
new_post_ids = []
new_weights = []

# 1. Descending: FAFB → MANC
# Each descending neuron targets ~60 random MANC motor+intrinsic neurons
desc_targets = np.concatenate([motor_idx, intrinsic_idx])
if len(desc_targets) > 0 and len(desc_idx) > 0:
    n_targets_per_desc = min(60, len(desc_targets))
    for pre in desc_idx:
        targets = rng.choice(desc_targets, size=n_targets_per_desc, replace=False)
        new_pre_ids.extend([pre] * len(targets))
        new_post_ids.extend(targets.tolist())
        new_weights.extend([1.0] * len(targets))
    print(f"  Descending edges: {len(desc_idx)} × {n_targets_per_desc} = {len(desc_idx)*n_targets_per_desc}")

# 2. Ascending: MANC → FAFB
# Each ascending neuron targets ~40 random FAFB central neurons
asc_targets = central_idx
if len(asc_targets) > 0 and len(asc_idx) > 0:
    n_targets_per_asc = min(40, len(asc_targets))
    for pre in asc_idx:
        targets = rng.choice(asc_targets, size=n_targets_per_asc, replace=False)
        new_pre_ids.extend([pre] * len(targets))
        new_post_ids.extend(targets.tolist())
        new_weights.extend([1.0] * len(targets))
    print(f"  Ascending edges: {len(asc_idx)} × {n_targets_per_asc} = {len(asc_idx)*n_targets_per_asc}")

# 3. Sensory Ascending: MANC → FAFB (target central + sensory)
if len(asc_targets) > 0 and len(sens_asc_idx) > 0:
    sa_targets = np.unique(np.concatenate([central_idx, sensory_idx]))
    n_targets_per_sa = min(30, len(sa_targets))
    for pre in sens_asc_idx:
        targets = rng.choice(sa_targets, size=n_targets_per_sa, replace=False)
        new_pre_ids.extend([pre] * len(targets))
        new_post_ids.extend(targets.tolist())
        new_weights.extend([1.0] * len(targets))
    print(f"  Sensory ascending edges: {len(sens_asc_idx)} × {n_targets_per_sa} = {len(sens_asc_idx)*n_targets_per_sa}")

if len(new_pre_ids) == 0:
    print("  No bridge edges created!")
    exit(1)

# ─── Merge into CSR ───────────────────────────────────────
# Build edge list for new bridge edges
n_bridge = len(new_pre_ids)
print(f"\nTotal bridge edges: {n_bridge}")

# We need to insert these edges into the CSR while maintaining sorted order by pre
# Strategy: build edge lists for all three (existing + new), sort, rebuild CSR

print("  1. Collecting existing edges...")
# Existing edges: from CSR
existing_pre = np.repeat(np.arange(n_total), np.diff(row_ptr)).astype(np.int32)
existing_post = col_indices
existing_weights = values

print(f"     Existing: {len(existing_pre)} edges")

# New bridge edges
bridge_pre = np.array(new_pre_ids, dtype=np.int32)
bridge_post = np.array(new_post_ids, dtype=np.int32)
bridge_w = np.array(new_weights, dtype=np.float32)

# Combine and sort by pre (then post)
print("  2. Sorting combined edge list...")
all_pre = np.concatenate([existing_pre, bridge_pre])
all_post = np.concatenate([existing_post, bridge_post])
all_w = np.concatenate([existing_weights, bridge_w])

order = np.lexsort((all_post, all_pre))
all_pre = all_pre[order]
all_post = all_post[order]
all_w = all_w[order]

print(f"     Combined: {len(all_pre)} edges")

# Build new CSR
print("  3. Building new row_ptr...")
new_row_ptr = np.zeros(n_total + 1, dtype=np.int32)
current = -1
for i in range(len(all_pre)):
    pre = all_pre[i]
    while current < pre:
        current += 1
        new_row_ptr[current + 1] = i
while current < n_total:
    current += 1
    new_row_ptr[current] = len(all_pre)
# Check
new_row_ptr[-1] = len(all_pre)  # ensure last is correct

# Verify
assert new_row_ptr[0] == 0
assert new_row_ptr[-1] == len(all_pre)
print(f"     row_ptr[0]={new_row_ptr[0]}, row_ptr[-1]={new_row_ptr[-1]} ✅")

# ─── Save ──────────────────────────────────────────────────
out_path = '/tmp/fafb_manc_csr_bridged.npz'
np.savez_compressed(out_path,
    row_ptr=new_row_ptr, col_indices=all_post, values=all_w,
    n_fafb=n_fafb, n_manc=n_manc)
print(f"\nSaved to {out_path} ({os.path.getsize(out_path)/1024**2:.1f} MB)")
print(f"  Final: {n_total} neurons, {len(all_pre)} synapses")
print(f"  Bridge: {n_bridge} cross-compartment edges ({n_bridge/len(all_pre)*100:.2f}% of total)")
