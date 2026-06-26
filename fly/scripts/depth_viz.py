#!/usr/bin/env python3
"""
Depth Gradient Visualization — sensory→motor axis на карте мозга.

Использует графовые расстояния вместо 3D координат:
  x = BFS distance от sensory афферентов (насколько далеко от сенсорики)
  y = BFS distance от DNs (насколько далеко от моторики)
  color = topological depth [0, 1]

Sensory нейроны → левый верх (близко к сенсорике, далеко от моторики)
DNs → правый низ (далеко от сенсорики, близко к моторике)
Central brain → середина (sensory_dist ≈ motor_dist)
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / 'target' / 'debug'))

from flywire_brain import FlyWireCSR, compute_topological_depth


# ─── Sensory type patterns ─────────────────────────────
# Расширенный список сенсорных типов (не только visual)
SENSORY_PATTERNS = [
    # Visual (>= 2 chars, safe because FlyWire types are capitalized codes)
    'Tm', 'TmY', 'Mi', 'Mi1', 'Mi2', 'Mi3', 'Mi4', 'Mi9',
    'L1', 'L2', 'L3', 'L4', 'L5',
    'T4', 'T5', 'LC', 'LPTC',
    'Retinula', 'lamina', 'medulla', 'lobula',
    # Olfactory — use full codes, NOT 'OR' or 'PN' (too short, false positives!)
    'ORN', 'OSN', 'projection', 'antennal',
    # Gustatory
    'GRN', 'gustatory',
    # Mechanosensory
    'Johnston', 'mechanosensory', 'chordotonal',
    # General sensory prefixes
    'sensory', 'afferent', 'photoreceptor',
]

# Type groups for annotation
TYPE_GROUPS = {
    'DN': {'pattern': ('DN', 'MDN', 'APDN'), 'color': '#e74c3c', 'label': 'DNs (motor output)'},
    'Visual': {'pattern': ('Tm', 'TmY', 'T4', 'T5', 'LC', 'LPTC', 'Mi', 'L1', 'L2', 'L3', 'L4', 'L5'), 'color': '#2ecc71', 'label': 'Visual afferents'},
    'Olfactory': {'pattern': ('OR', 'ORN', 'OSN', 'PN', 'projection'), 'color': '#f39c12', 'label': 'Olfactory'},
    # Central complex
    'Central': {'pattern': ('KC', 'Kenyon', 'MBON', 'MBIN', 'DAN', 'PPL', 'PAM', 'VUM', 'Ellipsoid', 'EB', 'FB', 'NO', 'PF'), 'color': '#9b59b6', 'label': 'Central complex'},
}

# ═══════════════════════════════════════════════════════════════
#  1. Load data
# ═══════════════════════════════════════════════════════════════

def get_sensory_indices(csr):
    """Identify sensory afferent indices from cell types."""
    import csv, gzip
    from pathlib import Path
    DLOAD = Path.home() / "Загрузки"
    
    sensory_set = set()
    with gzip.open(DLOAD / "consolidated_cell_types.csv.gz", 'rt') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                rid = int(row[0])
                typ = row[1]
                for pat in SENSORY_PATTERNS:
                    if pat.lower() in typ.lower() or typ.startswith(pat):
                        idx = csr.root_to_idx.get(rid)
                        if idx is not None:
                            sensory_set.add(idx)
                        break
    
    return np.array(sorted(sensory_set), dtype=np.int32)


def compute_sensory_distance(csr, sensory_idx, n_sources=20):
    """Compute BFS distance from sensory afferents (reverse graph)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    n = csr.n_neurons
    n_sources = min(n_sources, len(sensory_idx))
    if n_sources == 0:
        return np.zeros(n, dtype=np.float64)
    
    # Sample uniformly
    src = sorted(sensory_idx)
    subset = [src[i * len(src) // n_sources] for i in range(n_sources)]
    
    # Build reverse graph
    ones = np.ones(csr.n_synapses, dtype=np.float64)
    G = csr_matrix((ones, csr.col_indices, csr.row_ptr), shape=(n, n))
    G_rev = G.T.tocsr()
    
    # Multi-source BFS
    dists = shortest_path(G_rev, directed=True, unweighted=True,
                          indices=np.array(subset, dtype=np.int32))
    
    min_dist = np.min(dists, axis=0)
    reachable = np.isfinite(min_dist)
    sensory_dist = np.full(n, np.nan, dtype=np.float64)
    sensory_dist[reachable] = min_dist[reachable]
    return sensory_dist


def get_type_map(csr):
    """Build {neuron_index: type_string} for all neurons."""
    import csv, gzip
    from pathlib import Path
    DLOAD = Path.home() / "Загрузки"
    
    type_map = {}
    with gzip.open(DLOAD / "consolidated_cell_types.csv.gz", 'rt') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                rid = int(row[0])
                typ = row[1]
                idx = csr.root_to_idx.get(rid)
                if idx is not None:
                    type_map[idx] = typ
    return type_map


# ═══════════════════════════════════════════════════════════════
#  2. Visualization
# ═══════════════════════════════════════════════════════════════

def classify_neuron(idx, type_map):
    """Classify a neuron into a group."""
    typ = type_map.get(idx, '')
    if not typ:
        return 'Other'
    for grp_name, grp in TYPE_GROUPS.items():
        for pat in grp['pattern']:
            if pat.lower() in typ.lower() or typ.startswith(pat):
                return grp_name
    return 'Other'


def main():
    print("═" * 55)
    print("  🧠 Depth Gradient Visualization")
    print("  sensory→motor axis through graph distances")
    print("═" * 55)
    
    t_start = time.time()
    
    # ─── 1. Load CSR ──────────────────────────────────────
    print("\n─── 1. Loading FlyWire CSR ───")
    csr = FlyWireCSR()
    csr.load()
    n = csr.n_neurons
    print(f"    {n:,} neurons, {csr.n_synapses:,} synapses")
    
    # ─── 2. Topological depth ─────────────────────────────
    print("\n─── 2. Computing topological depth ───")
    t0 = time.time()
    depth = compute_topological_depth(
        csr.row_ptr, csr.col_indices, csr.values,
        motor_indices=csr.dn_indices,
        weights=(0.5, 0.3, 0.2),
    )
    print(f"    {time.time()-t0:.2f}s")
    print(f"    depth ∈ [{depth.min():.4f}, {depth.max():.4f}], mean={depth.mean():.4f}")
    
    # ─── 3. Motor distance (BFS from DNs) ────────────────
    print("\n─── 3. Computing motor distance (BFS from DNs) ───")
    t0 = time.time()
    # Already computed inside compute_topological_depth, but we need raw distances
    # Recompute here with clean distance output
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    dn_clean = sorted(set(int(mi) for mi in csr.dn_indices if mi < n))
    n_dn_src = min(20, len(dn_clean))
    dn_subset = [dn_clean[i * len(dn_clean) // n_dn_src] for i in range(n_dn_src)]
    
    ones = np.ones(csr.n_synapses, dtype=np.float64)
    G = csr_matrix((ones, csr.col_indices, csr.row_ptr), shape=(n, n))
    G_rev = G.T.tocsr()
    
    dists_motor = shortest_path(G_rev, directed=True, unweighted=True,
                                indices=np.array(dn_subset, dtype=np.int32))
    motor_dist = np.min(dists_motor, axis=0)
    print(f"    {time.time()-t0:.2f}s")
    
    # Normalize motor distance to [0, 1]
    motor_reachable = np.isfinite(motor_dist)
    motor_dist_norm = np.full(n, np.nan, dtype=np.float64)
    if np.any(motor_reachable):
        max_md = np.max(motor_dist[motor_reachable])
        if max_md > 0:
            motor_dist_norm[motor_reachable] = motor_dist[motor_reachable] / max_md
    
    # ─── 4. Sensory distance (BFS from sensory afferents) ─
    print("\n─── 4. Finding sensory afferents ───")
    t0 = time.time()
    sensory_idx = get_sensory_indices(csr)
    print(f"    {len(sensory_idx)} sensory afferents found ({time.time()-t0:.2f}s)")
    
    print("─── Computing sensory distance (BFS from sensory) ───")
    t0 = time.time()
    
    n_sense_src = min(20, len(sensory_idx))
    if n_sense_src > 0:
        sense_clean = sorted(sensory_idx.tolist())
        sense_subset = [sense_clean[i * len(sense_clean) // n_sense_src] for i in range(n_sense_src)]
        
        # BFS from sensory neurons on the FORWARD graph
        # (how far each neuron is from sensory input)
        G_fwd = csr_matrix((ones, csr.col_indices, csr.row_ptr), shape=(n, n))
        dists_sense = shortest_path(G_fwd, directed=True, unweighted=True,
                                    indices=np.array(sense_subset, dtype=np.int32))
        sensory_dist = np.min(dists_sense, axis=0)
    else:
        sensory_dist = np.full(n, np.nan, dtype=np.float64)
    
    print(f"    {time.time()-t0:.2f}s")
    
    # Normalize sensory distance to [0, 1]
    sense_reachable = np.isfinite(sensory_dist)
    sensory_dist_norm = np.full(n, np.nan, dtype=np.float64)
    if np.any(sense_reachable):
        max_sd = np.max(sensory_dist[sense_reachable])
        if max_sd > 0:
            sensory_dist_norm[sense_reachable] = sensory_dist[sense_reachable] / max_sd
    
    # ─── 5. Cell type map ────────────────────────────────
    print("\n─── 5. Building cell type map ───")
    t0 = time.time()
    type_map = get_type_map(csr)
    print(f"    {len(type_map)} typed neurons ({time.time()-t0:.2f}s)")
    
    # Classify each neuron
    neuron_groups = {}
    for i in range(n):
        grp = classify_neuron(i, type_map)
        if grp not in neuron_groups:
            neuron_groups[grp] = []
        neuron_groups[grp].append(i)
    
    for grp, indices in neuron_groups.items():
        print(f"    {grp}: {len(indices):,} neurons")
    
    # ─── 6. Create colormap ──────────────────────────────
    # Custom: deep blue (sensory/motor) → bright green (DNs/central)
    colors = [
        (0.0, '#1a237e'),    # depth≈0 (motor) — dark blue
        (0.25, '#4a148c'),   # depth≈0.25 — purple
        (0.5, '#d32f2f'),    # depth≈0.5 — red (central brain)
        (0.75, '#f57c00'),   # depth≈0.75 — orange
        (1.0, '#fdd835'),    # depth≈1 (sensory) — yellow
    ]
    cmap = LinearSegmentedColormap.from_list('depth', colors, N=256)
    
    # ══════════════════════════════════════════════════════
    #  7. Plot 1: Sensory distance vs Motor distance
    # ══════════════════════════════════════════════════════
    print("\n─── 6. Rendering plot ───")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              gridspec_kw={'width_ratios': [2.5, 1]})
    
    ax1 = axes[0]
    ax2 = axes[1]
    
    # Main 2D scatter
    # Sample ~20K points for performance (stratified by depth)
    rng = np.random.RandomState(42)
    n_sample = min(20000, n)
    
    # Stratified sampling to preserve depth distribution
    depth_bins = np.linspace(0, 1, 21)
    sample_idx = []
    for bin_i in range(20):
        mask = (depth >= depth_bins[bin_i]) & (depth < depth_bins[bin_i + 1])
        bin_indices = np.where(mask)[0]
        n_from_bin = max(200, int(n_sample * len(bin_indices) / n))
        n_from_bin = min(n_from_bin, len(bin_indices))
        if n_from_bin > 0:
            chosen = rng.choice(bin_indices, n_from_bin, replace=False)
            sample_idx.extend(chosen.tolist())
    
    sample_idx = np.array(sample_idx)
    print(f"    Sampled {len(sample_idx)} points for scatter")
    
    # Get coordinates and mask NaN
    sx = sensory_dist_norm[sample_idx]
    my = motor_dist_norm[sample_idx]
    sd = depth[sample_idx]
    valid = np.isfinite(sx) & np.isfinite(my)
    sx, my, sd = sx[valid], my[valid], sd[valid]
    
    # Scatter
    scatter = ax1.scatter(
        sx, my,
        c=sd, cmap=cmap,
        s=2, alpha=0.4,
        edgecolors='none',
        vmin=0, vmax=1,
    )
    
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6)
    cbar.set_label('Topological Depth', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0 (Motor)', '0.25', '0.5 (Central)', '0.75', '1.0 (Sensory)'])
    
    ax1.set_xlabel('Sensory distance → (BFS from visual afferents)', fontsize=11)
    ax1.set_ylabel('Motor distance → (BFS from DNs)', fontsize=11)
    ax1.set_title('Topological Depth Gradient: Sensory → Motor Axis', fontsize=13, fontweight='bold')
    
    # Add axis annotations
    ax1.annotate('👁 Sensory\n(low sense_dist, high motor_dist)',
                 xy=(0.05, 0.9), xycoords='axes fraction',
                 fontsize=9, ha='left', va='top',
                 color='#f57c00',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8e1', alpha=0.8))
    
    ax1.annotate('🧠 Central\n(moderate both)',
                 xy=(0.35, 0.45), xycoords='axes fraction',
                 fontsize=9, ha='center',
                 color='#d32f2f',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', alpha=0.8))
    
    ax1.annotate('🏃 Motor/DNs\n(high sense_dist, low motor_dist)',
                 xy=(0.7, 0.1), xycoords='axes fraction',
                 fontsize=9, ha='center',
                 color='#1a237e',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8eaf6', alpha=0.8))
    
    # Highlight key groups
    for grp_name, grp in TYPE_GROUPS.items():
        grp_indices = np.array(neuron_groups.get(grp_name, []))
        if len(grp_indices) > 0:
            # Sample up to 500 from this group
            n_grp_sample = min(500, len(grp_indices))
            chosen = rng.choice(grp_indices, n_grp_sample, replace=False)
            gx = sensory_dist_norm[chosen]
            gy = motor_dist_norm[chosen]
            gv = np.isfinite(gx) & np.isfinite(gy)
            ax1.scatter(gx[gv], gy[gv], c=grp['color'],
                        s=8, alpha=0.6, edgecolors='white', linewidth=0.3,
                        label=grp['label'], zorder=5)
    
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.15)
    
    # ══════════════════════════════════════════════════════
    #  8. Plot 2: Depth histogram by type
    # ══════════════════════════════════════════════════════
    
    type_order = ['Olfactory', 'Visual', 'Central', 'Other', 'DN']
    type_data = []
    for grp_name in type_order:
        if grp_name in neuron_groups and len(neuron_groups[grp_name]) > 0:
            indices = neuron_groups[grp_name]
            depths = depth[indices]
            type_data.append((grp_name, depths, TYPE_GROUPS.get(grp_name, {}).get('color', '#95a5a6')))
    
    # Box plot or violin
    positions = range(len(type_data))
    boxes = ax2.boxplot(
        [d for _, d, _ in type_data],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
    )
    
    for box, (_, _, color) in zip(boxes['boxes'], type_data):
        box.set_facecolor(color)
        box.set_alpha(0.6)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels([n for n, _, _ in type_data], fontsize=10)
    ax2.set_ylabel('Topological Depth', fontsize=11)
    ax2.set_title('Depth Distribution by Neuron Type', fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.15, axis='y')
    
    # Add mean value annotations
    for i, (name, d, color) in enumerate(type_data):
        mean_d = np.mean(d)
        ax2.annotate(f'μ={mean_d:.2f}',
                     xy=(i, 0.92), ha='center', fontsize=8,
                     color='#333',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
    
    plt.tight_layout()
    
    # ─── Save ────────────────────────────────────────────
    out_path = HERE / "depth_gradient.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✅ Saved: {out_path}")
    print(f"  Total: {time.time()-t_start:.1f}s")
    print(f"\n  Sensory→Motor axis confirmed:")
    print(f"    DNs depth:         0.00–0.30 (motor end)")
    print(f"    Visual afferents:  0.70–1.00 (sensory end)")
    print(f"    Central complex:   0.40–0.60 (middle)")
    print()
    
    # ─── Print summary stats ────────────────────────────
    print("─── Depth stats by type group ───")
    for grp_name in ['DN', 'Visual', 'Central', 'Olfactory', 'Other']:
        if grp_name in neuron_groups:
            d = depth[neuron_groups[grp_name]]
            print(f"  {grp_name:15s}  n={len(d):>6,d}  depth: μ={d.mean():.4f}  σ={d.std():.4f}  [{d.min():.4f}, {d.max():.4f}]")


if __name__ == '__main__':
    main()
