#!/usr/bin/env python3
"""
VNC Ring Oscillator Finder — ищем CPG в MANC коннектоме.

Алгоритм:
1. Загружаем MANC нейроны + свойства (type, predictedNt, class)
2. Загружаем MANC синапсы (bodyId_pre, bodyId_post, weight)
3. Находим GABAergic нейроны
4. Ищем реципрокные пары: A→B (GABA) AND B→A (GABA)
5. Строим граф и ищем chains/rings

Данные: /home/taaslab/Загрузки/
  v1.0_manc-traced-adjacencies-v1.0_traced-neurons.csv
  v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv
  v1.0_manc-v1.0-neuron-properties.feather
"""

import sys
import os
import csv
import time
from collections import defaultdict
import math

DLOAD = os.environ.get("FLYWIRE_DATA", "/home/taaslab/Загрузки")
NEURONS_CSV = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-neurons.csv")
CONNS_CSV  = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
PROPS_FTR = os.path.join(DLOAD, "v1.0_manc-v1.0-neuron-properties.feather")

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# 1. Load neuron properties
# ═══════════════════════════════════════════════════════════════

print("═" * 60)
print("🔬 VNC Ring Oscillator Finder")
print("═" * 60)
print()

print("▶ Loading neuron properties...")
import pandas as pd
props = pd.read_feather(PROPS_FTR)
print(f"   {len(props):,} neurons, {len(props.columns)} columns [{time.time()-t0:.1f}s]")
print(f"   Columns: {list(props.columns)}")
print()

# Build lookup dicts
nt_map = {}        # bodyId → predictedNt
type_map = {}      # bodyId → type
class_map = {}     # bodyId → class
subclass_map = {}  # bodyId → subclass
group_map = {}     # bodyId → group
modal_map = {}     # bodyId → modality

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
    sc = str(row.get('subclass', '')).strip()
    if sc and sc != 'nan':
        subclass_map[bid] = sc
    g = str(row.get('group', '')).strip()
    if g and g != 'nan':
        group_map[bid] = g
    m = str(row.get('modality', '')).strip()
    if m and m != 'nan':
        modal_map[bid] = m

# ═══════════════════════════════════════════════════════════════
# 2. Count NT types
# ═══════════════════════════════════════════════════════════════

print("▶ Neurotransmitter distribution:")
nt_counts = defaultdict(int)
for bid, nt in nt_map.items():
    nt_counts[nt] += 1
for nt, cnt in sorted(nt_counts.items(), key=lambda x: -x[1]):
    print(f"   {nt}: {cnt:>6,}")
print()

# ═══════════════════════════════════════════════════════════════
# 3. Load connections
# ═══════════════════════════════════════════════════════════════

print("▶ Loading connections (this may take a moment)...")

# Build connection matrix: pre → [(post, weight)]
# Only for GABAergic neurons (ring oscillator = reciprocal inhibition)
gaba_set = set(bid for bid, nt in nt_map.items() if nt == 'gaba')
print(f"   GABAergic neurons: {len(gaba_set):,}")

# First pass: count edges for pre-allocation
total_edges = 0
with open(CONNS_CSV) as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        total_edges += 1
print(f"   Total connections to scan: {total_edges:,}")

# Second pass: filter GABA→GABA edges
print("   Scanning 5.2M connections for GABA→GABA edges...")
t1 = time.time()
# Build adjacency SET for quick lookups
gaba_adj_set = defaultdict(set)  # pre → {post}
gaba_edge_w = {}  # (pre, post) → weight
n_gaba_edges = 0
with open(CONNS_CSV) as f:
    r = csv.reader(f)
    header = next(r)
    for i, row in enumerate(r):
        if i % 500000 == 0 and i > 0:
            print(f"      {i:,} lines scanned, {n_gaba_edges} GABA→GABA edges found [{time.time()-t1:.1f}s]", end='\r')
        pre = int(row[0])
        post = int(row[1])
        weight = float(row[2])
        if pre in gaba_set and post in gaba_set:
            gaba_adj_set[pre].add(post)
            gaba_edge_w[(pre, post)] = weight
            n_gaba_edges += 1

print(f"   GABA→GABA edges: {n_gaba_edges:,} [{time.time()-t1:.1f}s]")
print()

# ═══════════════════════════════════════════════════════════════
# 4. Find reciprocal pairs
# ═══════════════════════════════════════════════════════════════

print("▶ Searching for reciprocal GABA pairs (ring oscillator candidates)...")
t2 = time.time()

# Find reciprocal pairs: A→B AND B→A both exist (using SET for O(1) lookup)
reciprocal_pairs = []  # [(A, w_AB, B, w_BA)]
checked = set()

for pre, posts in gaba_adj_set.items():
    for post in posts:
        if pre >= post:  # only count each pair once
            continue
        
        # O(1) check: does B have A in its posts?
        if pre in gaba_adj_set.get(post, set()):
            w_ab = gaba_edge_w.get((pre, post), 0)
            w_ba = gaba_edge_w.get((post, pre), 0)
            reciprocal_pairs.append((pre, w_ab, post, w_ba))

print(f"   Reciprocal GABA pairs found: {len(reciprocal_pairs):,} [{time.time()-t2:.1f}s]")
print()

# ═══════════════════════════════════════════════════════════════
# 5. Analyze reciprocal pairs
# ═══════════════════════════════════════════════════════════════

if reciprocal_pairs:
    # Sort by geometric mean of weights (strongest reciprocal inhibition first)
    reciprocal_pairs.sort(key=lambda x: math.sqrt(abs(x[1] * x[3])), reverse=True)
    
    print("═" * 60)
    print(f"🏆 TOP 30 RECIPROCAL GABA PAIRS (ring oscillator candidates)")
    print("═" * 60)
    print(f"{'Neuron A':<12} {'type':<25} {'→ B':<10} {'← B':<10} {'Neuron B':<12} {'type':<25} {'class':<20}")
    print("-" * 120)
    
    shown = 0
    for pre, w_ab, post, w_ba in reciprocal_pairs:
        if shown >= 30:
            break
        
        a_type = type_map.get(pre, '?')
        b_type = type_map.get(post, '?')
        a_class = class_map.get(pre, '?')
        b_class = class_map.get(post, '?')
        a_sub = subclass_map.get(pre, '')
        b_sub = subclass_map.get(post, '')
        a_modal = modal_map.get(pre, '')
        b_modal = modal_map.get(post, '')
        
        # Only show leg/ascending/descending related pairs
        a_info = f"{a_type}"
        if a_sub: a_info += f" ({a_sub})"
        if a_modal: a_info += f" [{a_modal}]"
        
        b_info = f"{b_type}"
        if b_sub: b_info += f" ({b_sub})"
        if b_modal: b_info += f" [{b_modal}]"
        
        # Check if either neuron is leg-related
        is_leg = any(kw in str(a_type).lower() or kw in str(b_type).lower() or 
                     kw in str(a_class).lower() or kw in str(b_class).lower()
                     for kw in ['leg', 't1', 't2', 't3', 'coxa', 'femur', 'tibia', 
                                 'tarsus', 'flexor', 'extensor', 'motor', 'mn',
                                 'ascending', 'descending', 'dn', 'an'])
        
        marker = '🦗' if is_leg else '  '
        print(f"{marker} {pre:<10} {a_info:<35} {w_ab:<8.4f} {w_ba:<8.4f} {post:<10} {b_info:<35} {a_class:<25}")
        shown += 1
    
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # 6. Build graph of top reciprocal pairs and find chains/rings
    # ═══════════════════════════════════════════════════════════════
    
    print("═" * 60)
    print("🔗 Building reciprocal inhibition graph (threshold > 1.0)...")
    print("═" * 60)
    
    # Filter: only pairs with strong reciprocal inhibition
    strong_pairs = [(a, w_ab, b, w_ba) for a, w_ab, b, w_ba in reciprocal_pairs 
                    if abs(w_ab) > 1.0 and abs(w_ba) > 1.0]
    print(f"   Strong reciprocal pairs (|w| > 1.0): {len(strong_pairs)}")
    
    # Build adjacency
    adj = defaultdict(set)
    for a, _, b, _ in strong_pairs:
        adj[a].add(b)
        adj[b].add(a)
    
    # Find connected components (potential ring oscillators)
    visited = set()
    components = []
    for node in adj:
        if node in visited:
            continue
        # BFS
        queue = [node]
        comp = set()
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    queue.append(nb)
        if len(comp) >= 2:
            components.append(comp)
    
    components.sort(key=len, reverse=True)
    print(f"   Connected components: {len(components)}")
    print()
    
    print("═" * 60)
    print("🎯 TOP RING OSCILLATOR CANDIDATES (largest reciprocal clusters)")
    print("═" * 60)
    
    for ci, comp in enumerate(components[:10]):
        print()
        print(f"─── Cluster {ci+1}: {len(comp)} neurons ───")
        
        # Show each neuron in the cluster
        for nid in sorted(comp):
            nt = nt_map.get(nid, '?')
            typ = type_map.get(nid, '?')
            cls = class_map.get(nid, '?')
            sub = subclass_map.get(nid, '')
            grp = group_map.get(nid, '')
            modal = modal_map.get(nid, '')
            
            info = f"  {nid:<10} {nt:<6} type={typ:<25}"
            if sub: info += f" sub={sub:<15}"
            if cls: info += f" cls={cls:<15}"
            if grp: info += f" grp={grp:<15}"
            if modal: info += f" modal={modal}"
            print(info)
        
        # Show edges within cluster
        print("  Edges:")
        edge_set = set()
        for nid in sorted(comp):
            for post in gaba_adj_set.get(nid, set()):
                if post in comp:
                    pair = (min(nid, post), max(nid, post))
                    if pair not in edge_set:
                        edge_set.add(pair)
                        w = gaba_edge_w.get((nid, post), 0)
                        direction = "↔"
                        print(f"    {pair[0]} {direction} {pair[1]}  w={w:.4f}")

else:
    print("No reciprocal GABA pairs found!")

print()
print(f"⏱ Total: {time.time()-t0:.1f}s")
print("✅ Done!")
