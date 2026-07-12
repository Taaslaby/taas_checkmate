#!/usr/bin/env python3
"""
Export VNC Ring Oscillator (Cluster 1) from MANC → CSR binary.

Извлекает самый большой кластер реципрокных GABA пар из MANC,
строит CSR (row_ptr, col_indices, values) с NT типами,
экспортирует в бинарный файл для Rust LIF симуляции.

Формат экспорта:
  [n_neurons: u32] [n_synapses: u32]
  [row_ptr: i32 × (n_neurons+1)]
  [col_indices: i32 × n_synapses]
  [values: f32 × n_synapses]
  [nt_types: f32 × n_neurons]  (1.0=exc, -1.0=inh)
  [root_id: u64 × n_neurons]
  [type_str_len: u32] [type_str: utf8] × n_neurons
  [class_str_len: u32] [class_str: utf8] × n_neurons

Использование:
  python3 fly/scripts/export_vnc_cluster1.py

Выход: /tmp/vnc_cluster1.bin
"""

import sys
import os
import csv
import time
import struct
import math
from collections import defaultdict

import pandas as pd

DLOAD = os.environ.get("FLYWIRE_DATA", "/home/taaslab/Загрузки")
CONNS_CSV = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
PROPS_FTR = os.path.join(DLOAD, "v1.0_manc-v1.0-neuron-properties.feather")
OUTPUT = "/tmp/vnc_cluster1.bin"

t0 = time.time()

print("═" * 60)
print("📤 VNC Cluster 1 Exporter")
print("═" * 60)

# ═══════════════════════════════════════════════════════════════
# 1. Load properties
# ═══════════════════════════════════════════════════════════════
print("\n▶ Loading neuron properties...")
props = pd.read_feather(PROPS_FTR)

nt_map = {}
type_map = {}
class_map = {}
for _, row in props.iterrows():
    bid = int(row['bodyId'])
    nt = str(row.get('predictedNt', '')).strip().lower()
    if nt and nt != 'nan' and nt != 'unknown':
        nt_map[bid] = nt
    t = str(row.get('type', '')).strip()
    if t and t != 'nan':
        type_map[bid] = t
    c = str(row.get('class', '')).strip()
    if c and c != 'nan':
        class_map[bid] = c

gaba_set = set(bid for bid, nt in nt_map.items() if nt == 'gaba')
print(f"   {len(nt_map)} neurons with NT, {len(gaba_set)} GABA")

# ═══════════════════════════════════════════════════════════════
# 2. Build GABA adjacency set
# ═══════════════════════════════════════════════════════════════
print("\n▶ Scanning connections for GABA→GABA adjacency...")
gaba_adj = defaultdict(set)
gaba_edge_w = {}
with open(CONNS_CSV) as f:
    r = csv.reader(f)
    header = next(r)
    for i, row in enumerate(r):
        if i % 500000 == 0 and i > 0:
            print(f"   {i:,} lines...", end='\r')
        pre, post, w = int(row[0]), int(row[1]), float(row[2])
        if pre in gaba_set and post in gaba_set:
            gaba_adj[pre].add(post)
            gaba_edge_w[(pre, post)] = w

print(f"\n   GABA→GABA edges: {len(gaba_edge_w):,}")

# ═══════════════════════════════════════════════════════════════
# 3. Find largest connected component
# ═══════════════════════════════════════════════════════════════
print("\n▶ Finding largest connected component (Cluster 1)...")

# Build adjacency (undirected) from reciprocal pairs only
undirected = defaultdict(set)
for (pre, post), w in gaba_edge_w.items():
    if post in gaba_adj.get(pre, set()) and pre in gaba_adj.get(post, set()):
        # Reciprocal pair
        undirected[pre].add(post)
        undirected[post].add(pre)

print(f"   Reciprocal GABA: {sum(len(v) for v in undirected.values())//2} pairs")

# BFS to find all connected components
visited = set()
components = []
for node in undirected:
    if node in visited:
        continue
    queue = [node]
    comp = set()
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        comp.add(n)
        for nb in undirected[n]:
            if nb not in visited:
                queue.append(nb)
    if len(comp) >= 2:
        components.append(comp)

components.sort(key=len, reverse=True)
cluster1 = components[0]
print(f"   {len(components)} components total")
print(f"   Cluster 1: {len(cluster1):,} neurons 🎯")

# ═══════════════════════════════════════════════════════════════
# 4. Extract ALL connections within Cluster 1
# ═══════════════════════════════════════════════════════════════
print("\n▶ Extracting all connections within Cluster 1...")

# Build id → index mapping
ids_sorted = sorted(cluster1)
id_to_idx = {bid: i for i, bid in enumerate(ids_sorted)}
n = len(ids_sorted)

# Scan connections again, filter by cluster 1
# Pre-count edges per neuron for row_ptr
post_counts = defaultdict(int)
edge_list = []  # (pre_idx, post_idx, weight, nt_pre)

with open(CONNS_CSV) as f:
    r = csv.reader(f)
    header = next(r)
    for i, row in enumerate(r):
        if i % 500000 == 0 and i > 0:
            print(f"   Scanning: {i:,} lines...", end='\r')
        pre, post, w = int(row[0]), int(row[1]), float(row[2])
        if pre in cluster1 and post in cluster1:
            pre_idx = id_to_idx[pre]
            post_idx = id_to_idx[post]
            nt_pre = nt_map.get(pre, 'gaba')  # default GABA for VNC
            edge_list.append((pre_idx, post_idx, w, nt_pre))
            post_counts[post_idx] += 1

print(f"\n   Total edges within Cluster 1: {len(edge_list):,}")

# Build CSR (incoming convention: row_ptr[post] → col_indices[pre])
edge_list.sort(key=lambda x: x[1])  # sort by post_idx

row_ptr = [0] * (n + 1)
col_indices = []
values = []
nt_signs = []

# Count edges per post
for post_idx in range(n):
    row_ptr[post_idx + 1] = row_ptr[post_idx] + post_counts.get(post_idx, 0)

# Fill CSR
cursor = [0] * n
for pre_idx, post_idx, w, nt_pre in edge_list:
    pos = row_ptr[post_idx] + cursor[post_idx]
    col_indices.append(pre_idx)
    values.append(w)
    nt_signs.append(1.0 if nt_pre == 'acetylcholine' or nt_pre == 'glutamate' else -1.0)
    cursor[post_idx] += 1

# Compute per-neuron NT types
nt_types = [-1.0] * n  # default inhibitory (GABA dominant in VNC)
nt_votes = [{'exc': 0, 'inh': 0} for _ in range(n)]
for pre_idx, post_idx, w, nt_pre in edge_list:
    if nt_pre == 'acetylcholine' or nt_pre == 'glutamate':
        nt_votes[pre_idx]['exc'] += 1
    else:
        nt_votes[pre_idx]['inh'] += 1

for i in range(n):
    if nt_votes[i]['exc'] > nt_votes[i]['inh']:
        nt_types[i] = 1.0

print(f"   CSR: {n} neurons, {len(values)} synapses")
exc = sum(1 for nt in nt_types if nt > 0)
inh = sum(1 for nt in nt_types if nt < 0)
print(f"   NT: {exc} exc, {inh} inh")

# ═══════════════════════════════════════════════════════════════
# 5. Export to binary
# ═══════════════════════════════════════════════════════════════
print(f"\n▶ Exporting to {OUTPUT}...")

with open(OUTPUT, 'wb') as f:
    # Header
    f.write(struct.pack('<II', n, len(values)))
    # row_ptr
    f.write(struct.pack(f'<{n+1}i', *row_ptr))
    # col_indices
    f.write(struct.pack(f'<{len(col_indices)}i', *col_indices))
    # values
    f.write(struct.pack(f'<{len(values)}f', *values))
    # nt_types
    f.write(struct.pack(f'<{n}f', *nt_types))
    # root_ids
    f.write(struct.pack(f'<{n}Q', *ids_sorted))
    # Neuron types + classes
    for bid in ids_sorted:
        typ = type_map.get(bid, '?')
        cls = class_map.get(bid, '?')
        typ_bytes = typ.encode('utf-8')
        cls_bytes = cls.encode('utf-8')
        f.write(struct.pack('<I', len(typ_bytes)))
        f.write(typ_bytes)
        f.write(struct.pack('<I', len(cls_bytes)))
        f.write(cls_bytes)

import os as _os
size = _os.path.getsize(OUTPUT)
print(f"   ✅ Exported: {size:,} bytes")

# ═══════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"📊 CLUSTER 1 SUMMARY")
print(f"{'═' * 60}")
print(f"   Neurons:   {n:,}")
print(f"   Synapses:  {len(values):,}")
print(f"   Density:   {len(values) / (n * n) * 100:.4f}%")

# Count neuron classes
class_counts = defaultdict(int)
sc_counts = defaultdict(int)
for bid in ids_sorted:
    cls = class_map.get(bid, 'unknown')
    class_counts[cls] += 1

print(f"\n   Neuron classes:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f"     {cls}: {cnt}")

# Check for key types
dn_count = sum(1 for bid in ids_sorted if 'descending' in class_map.get(bid, '').lower() or 'dn' in type_map.get(bid, '').lower())
an_count = sum(1 for bid in ids_sorted if 'ascending' in class_map.get(bid, '').lower() or 'an' in type_map.get(bid, '').lower())
en_count = sum(1 for bid in ids_sorted if 'efferent' in class_map.get(bid, '').lower() or 'en' in type_map.get(bid, '').lower())
in_count = sum(1 for bid in ids_sorted if 'intrinsic' in class_map.get(bid, '').lower() or 'in' in type_map.get(bid, '').lower())

print(f"\n   Key types:")
print(f"     Intrinsic (IN):   {in_count}  ← ринг-осциллятор")
print(f"     Ascending (AN):   {an_count}  ← обратная связь от ног 🦗")
print(f"     Descending (DN):  {dn_count}  ← команды от мозга 🧠")
print(f"     Efferent (EN):    {en_count}  ← выход на мышцы ⚡")

print(f"\n   Output: {OUTPUT}")
print(f"   Time: {time.time()-t0:.1f}s")
print("✅ Done!")
