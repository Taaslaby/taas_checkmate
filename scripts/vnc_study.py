#!/usr/bin/env python3
"""🧬 VNC Study — исследование VNC через сардану (StructuralTokenizer)

Этап 1: Загрузить VNC Cluster 1 → StructuralTokenizer → срез фич
         → визуализация распределений (depth, degree, community, betweenness)
         → сравнение с sarDNA-generated VNC

Этап 2: Обучить FlowMLP на VNC-распределении → генерация нового VNC
         → Kuramoto CPG тест

Архитектура:
  MANC VNC Cluster 1 (5,681 нейронов, 461K синапсов, 90% GABA)
      ↓
  StructuralTokenizer
      ↓
  State vectors [depth, degree, community, betweenness]
      ↓
  Flow Matching / sarDNA → анализ → генерация
"""

import sys, os, time, math
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
SARDAANA = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM")
SARDANA1 = SARDAANA / "experiments" / "sarDNA_1"
SARDANA2 = SARDAANA / "experiments" / "sardana_2"

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SARDAANA))
sys.path.insert(0, str(SARDANA1))
sys.path.insert(0, str(SARDANA1 / "scripts"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from structural_tokenizer import StructuralTokenizer, load_vnc

# VNC data path
VNC_DIR = BASE / "examples" / "fly" / "archive"


def load_vnc_data():
    """Load VNC Cluster 1 via sardana's load_vnc."""
    print("=" * 55)
    print("  Этап 1: Загрузка VNC Cluster 1")
    print("=" * 55)
    t0 = time.time()
    
    adj, ids, n = load_vnc(VNC_DIR)
    print(f"  ✅ VNC loaded in {time.time()-t0:.1f}s")
    print(f"  Neurons: {n:,}")
    print(f"  IDs: {len(ids):,}")
    print(f"  Sample adj[0]: {adj[0][:3]}...")
    return adj, ids, n


def tokenize_vnc(adj, ids):
    """Tokenize VNC and extract structural features."""
    print("\n" + "=" * 55)
    print("  Этап 2: StructuralTokenizer + Feature Extraction")
    print("=" * 55)
    t0 = time.time()
    
    tokenizer = StructuralTokenizer(adj, ids)
    print(f"  ✅ Tokenizer created in {time.time()-t0:.1f}s")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Neurons: {len(ids)}")
    
    # Feature ranges
    n_neurons = len(ids)
    depths = np.array([tokenizer.depths[i] for i in range(n_neurons)])
    degrees = np.array([tokenizer.degrees[i] for i in range(n_neurons)])
    communities = np.array([tokenizer.communities[i] for i in range(n_neurons)])
    betweenness = np.array([tokenizer.betweenness[i] for i in range(n_neurons)])
    
    print(f"\n  📊 Feature statistics:")
    print(f"    Depth:         mean={depths.mean():.2f}  std={depths.std():.2f}  "
          f"range=[{depths.min():.1f}, {depths.max():.1f}]")
    print(f"    Degree:        mean={degrees.mean():.2f}  std={degrees.std():.2f}  "
          f"range=[{degrees.min():.1f}, {degrees.max():.1f}]")
    print(f"    Community:     mean={communities.mean():.2f}  std={communities.std():.2f}  "
          f"range=[{communities.min():.1f}, {communities.max():.1f}]")
    print(f"    Betweenness:   mean={betweenness.mean():.6f}  std={betweenness.std():.6f}  "
          f"range=[{betweenness.min():.6f}, {betweenness.max():.6f}]")
    
    # Compute graph metrics
    total_edges = sum(len(adj[i]) for i in range(n_neurons))
    density = total_edges / (n_neurons * (n_neurons - 1))
    print(f"\n  🔗 Graph metrics:")
    print(f"    Total edges: {total_edges:,}")
    print(f"    Density: {density:.6f}")
    print(f"    Mean out-degree: {total_edges / n_neurons:.2f}")
    
    # Degree distribution stats
    degree_cv = degrees.std() / max(degrees.mean(), 1e-8)
    print(f"    Degree CV: {degree_cv:.2f}")
    
    # E/I ratio estimate from community structure
    n_communities = len(set(communities))
    print(f"    Communities: {n_communities}")
    
    return tokenizer, depths, degrees, communities, betweenness


def plot_vnc_features(depths, degrees, communities, betweenness, save_path):
    """Plot VNC structural feature distributions."""
    print("\n" + "=" * 55)
    print("  Этап 3: Визуализация")
    print("=" * 55)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("MANC VNC — Structural Features (real connectome)", fontsize=14)
    
    # Depth
    ax = axes[0, 0]
    ax.hist(depths, bins=50, alpha=0.7, color='steelblue')
    ax.set_title(f"Depth (mean={depths.mean():.2f})")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Count")
    
    # Degree
    ax = axes[0, 1]
    ax.hist(degrees, bins=50, alpha=0.7, color='coral')
    ax.set_title(f"Degree (mean={degrees.mean():.1f})")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    
    # Community
    ax = axes[1, 0]
    n_comms = len(set(communities))
    ax.hist(communities, bins=min(50, n_comms), alpha=0.7, color='seagreen')
    ax.set_title(f"Community ({n_comms} clusters)")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Count")
    
    # Betweenness
    ax = axes[1, 1]
    ax.hist(betweenness, bins=50, alpha=0.7, color='purple')
    ax.set_title(f"Betweenness (mean={betweenness.mean():.6f})")
    ax.set_xlabel("Betweenness")
    ax.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"  📈 Plot saved: {save_path}")


def analyze_communities(tokenizer, ids):
    """Analyze community structure of VNC."""
    print("\n" + "=" * 55)
    print("  Этап 4: Анализ сообществ")
    print("=" * 55)
    
    n = len(ids)
    communities = np.array([tokenizer.communities[i] for i in range(n)])
    degrees = np.array([tokenizer.degrees[i] for i in range(n)])
    depths = np.array([tokenizer.depths[i] for i in range(n)])
    
    unique_comms = sorted(set(communities))
    print(f"  Communities: {len(unique_comms)}")
    
    for c in unique_comms[:10]:  # Top 10
        mask = communities == c
        n_c = mask.sum()
        mean_deg = degrees[mask].mean()
        mean_dep = depths[mask].mean()
        print(f"    Community {c:>3d}: {n_c:>5d} neurons  "
              f"mean_deg={mean_deg:>6.1f}  mean_depth={mean_dep:>5.2f}")
    
    if len(unique_comms) > 10:
        print(f"    ... and {len(unique_comms) - 10} more communities")


def main():
    save_dir = BASE / "examples" / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🧬 VNC Study — сардана исследует VNC               ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Step 1: Load VNC
    adj, ids, n = load_vnc_data()
    
    # Step 2: Tokenize
    tokenizer, depths, degrees, communities, betweenness = tokenize_vnc(adj, ids)
    
    # Step 3: Plot
    plot_path = save_dir / "vnc_structural_features.png"
    plot_vnc_features(depths, degrees, communities, betweenness, plot_path)
    
    # Step 4: Community analysis
    analyze_communities(tokenizer, ids)
    
    # Save features for later comparison
    features = np.column_stack([depths, degrees, communities, betweenness])
    feat_path = save_dir / "vnc_structural_features.npy"
    np.save(feat_path, features)
    print(f"\n  💾 Features saved: {feat_path}")
    
    print(f"\n{'='*55}")
    print(f"  ✅ VNC Study — Этап 1 завершён!")
    print(f"  VNC нейронов: {len(ids):,}")
    print(f"  Сообществ: {len(set(communities))}")
    print(f"  Фичи сохранены для сравнения с генерацией")
    print(f"  {'='*55}")
    
    # Compare with sarDNA-generated VNC
    gen_path = BASE / "examples" / "data" / "vnc_generated.taas"
    if gen_path.exists():
        print(f"\n  🔬 Сравнение с sarDNA-generated VNC доступно!")
        print(f"  Запусти: python scripts/vnc_study.py --compare")
    else:
        print(f"\n  🎯 Следующий шаг: сгенерировать VNC через FlowMLP")
        print(f"  и сравнить распределения")


if __name__ == "__main__":
    main()
