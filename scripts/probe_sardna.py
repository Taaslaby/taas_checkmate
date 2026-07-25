#!/usr/bin/env python3
"""
🧬 PROBE SARDNA — Что она знает о связи мухи и GPT?

После fine-tuning control tokens (50 эпох), probing показывает:
  1. PCA всех 23 species embeddings — какие кластеры видит Sardna
  2. Сравнение arch:gpt2 vs arch:fly — чем отличаются?
  3. Token arithmetic — комбинации архитектур
  4. Что Sardna считает "attention" в мухе?

Usage:
  python scripts/probe_sardna.py                          # полный probing
  python scripts/probe_sardna.py --pca-only                # только PCA
  python scripts/probe_sardna.py --quick                   # только arch сравнение
"""

import sys
import os
import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

# ─── Путь к Sardna ────────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_gpt.sardna import SarDNA

# ─── Константы (дублируем из finetune_control_tokens.py чтобы избежать importlib) ──
SPECIES_BIOLOGICAL = 7

CONTROL_TOKENS = {
    "gen_visual":      0,
    "gen_attention":   1,
    "gen_mlp":         2,
    "gen_cpg":         3,
    "gen_lm_head":     4,
    "gen_recurrent":   5,
    "gen_random":      6,
    "arch:gpt2":       7,
    "arch:fly":        8,
    "arch:vnc":        9,
    "arch:mouse":     10,
    "region:optic":   11,
    "region:antennal": 12,
    "region:mushroom": 13,
    "region:motor":   14,
    "region:random":  15,
}

N_CONTROL = len(CONTROL_TOKENS)
SPECIES_TOTAL = SPECIES_BIOLOGICAL + N_CONTROL

CONTROL_PROFILES = {
    "gen_visual":      (0.75, 0.60, 0.20, 0.30),
    "gen_attention":   (0.50, 0.80, 0.70, 0.75),
    "gen_mlp":         (0.40, 0.90, 0.30, 0.20),
    "gen_cpg":         (0.40, 0.50, 0.55, 0.50),
    "gen_lm_head":     (0.15, 0.70, 0.20, 0.10),
    "gen_recurrent":   (0.50, 0.50, 0.50, 0.80),
    "gen_random":      (0.50, 0.50, 0.50, 0.50),
    "arch:gpt2":       (0.60, 0.80, 0.80, 0.65),
    "arch:fly":        (0.70, 0.50, 0.35, 0.40),
    "arch:vnc":        (0.30, 0.40, 0.60, 0.30),
    "arch:mouse":      (0.60, 0.60, 0.50, 0.50),
    "region:optic":    (0.85, 0.65, 0.15, 0.30),
    "region:antennal": (0.75, 0.55, 0.25, 0.35),
    "region:mushroom": (0.55, 0.50, 0.55, 0.55),
    "region:motor":    (0.20, 0.40, 0.70, 0.30),
    "region:random":   (0.50, 0.50, 0.50, 0.50),
}

SARDNA_MODEL_DEFAULT = os.path.join(SARDANA_PATH, "models", "sardna", "sarDNA_v2.pt")

# ═══════════════════════════════════════════════════════════
#  PROBE ENGINE
# ═══════════════════════════════════════════════════════════

class SardnaProbe:
    """
    Probing Sardna's understanding of architecture.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        
        print(f"  📥 Loading Sardna v2 from {model_path}...")
        self.model = SarDNA.load_trained(model_path, config=None, device=device)
        self.model.eval()
        self.config = self.model.config
        
        # ── Expand species + load fine-tuned weights ──
        self._load_with_control()
        
        print(f"  ✅ Sardna loaded: d_model={self.config.d_model}, "
              f"n_layers={self.config.n_layers}")
    
    def _load_with_control(self):
        """Load fine-tuned control embeddings if available."""
        # Expand
        import torch.nn as nn
        old_embed = self.model.embed.species_embed
        old_weight = old_embed.weight.data
        d = self.config.d_model
        
        new_embed = nn.Embedding(SPECIES_TOTAL, d)
        new_embed.weight.data[:SPECIES_BIOLOGICAL] = old_weight
        
        # Init new control embeddings
        base = old_weight.mean(dim=0)
        for token_name, token_id in CONTROL_TOKENS.items():
            species_idx = SPECIES_BIOLOGICAL + token_id
            np.random.seed(42 + token_id * 7)
            direction = torch.from_numpy(np.random.randn(d).astype(np.float32))
            direction = direction / (direction.norm() + 1e-8)
            scale = 0.15 + 0.15 * (token_id / N_CONTROL)
            init_vec = base + scale * direction * old_weight.std().item()
            new_embed.weight.data[species_idx] = init_vec
        
        self.model.embed.species_embed = new_embed
        self.model.config.n_species = SPECIES_TOTAL
        
        # ── Try to load fine-tuned checkpoint ──
        ft_path = "/tmp/sardna_ft_checkpoint.pt"
        if os.path.exists(ft_path):
            print(f"  🔄 Loading fine-tuned checkpoint from {ft_path}")
            try:
                sd = torch.load(ft_path, map_location=self.device, weights_only=True)
                # Load only species_embed weights
                if 'embed.species_embed.weight' in sd:
                    self.model.embed.species_embed.weight.data = sd['embed.species_embed.weight']
                    print(f"  ✅ Fine-tuned control embeddings loaded!")
            except Exception as e:
                print(f"  ⚠️  Could not load fine-tuned checkpoint: {e}")
        else:
            print(f"  ⚠️  No fine-tuned checkpoint found at {ft_path}")
            print(f"     Running with random-init control embeddings")
    
    # ── 1. PCA on species embeddings ──
    def pca_embeddings(self) -> Dict:
        """
        PCA всех 23 species embeddings.
        Показывает как biological и control tokens кластеризуются.
        """
        print(f"\n{'='*60}")
        print(f"  📐 PCA: {SPECIES_TOTAL} species embeddings")
        print(f"{'='*60}")
        
        weights = self.model.embed.species_embed.weight.data.cpu().numpy()  # [23, 256]
        
        # Centering
        mean = weights.mean(axis=0, keepdims=True)
        centered = weights - mean
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pcs = centered @ Vt[:2].T  # [23, 2]
        
        explained = S[:2]**2 / (S**2).sum()
        
        # ── Print table ──
        print(f"\n  PC1 explains {explained[0]:.1%}, PC2 explains {explained[1]:.1%}")
        print(f"\n  {'Token':<22} | {'Type':<12} | {'PC1':>8} | {'PC2':>8} | {'Group':>12}")
        print(f"  {'─'*22}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*12}")
        
        groups = {"biological": [], "gen": [], "arch": [], "region": []}
        
        for i in range(SPECIES_TOTAL):
            if i < SPECIES_BIOLOGICAL:
                name = f"species_{i}"
                tok_type = "biological"
                groups["biological"].append((name, pcs[i]))
            else:
                token_id = i - SPECIES_BIOLOGICAL
                name = list(CONTROL_TOKENS.keys())[token_id]
                if name.startswith("gen_"):
                    tok_type = "control/gen"
                    groups["gen"].append((name, pcs[i]))
                elif name.startswith("arch:"):
                    tok_type = "control/arch"
                    groups["arch"].append((name, pcs[i]))
                else:
                    tok_type = "control/region"
                    groups["region"].append((name, pcs[i]))
            
            print(f"  {name:<22} | {tok_type:<12} | {pcs[i][0]:>8.3f} | {pcs[i][1]:>8.3f}")
        
        # ── Group analysis ──
        print(f"\n  📊 Group centroids:")
        for gname, items in groups.items():
            if items:
                centroid = np.mean([pc for _, pc in items], axis=0)
                spread = np.std([pc for _, pc in items], axis=0).mean()
                print(f"    {gname:<15} centroid=({centroid[0]:.3f}, {centroid[1]:.3f}) "
                      f"spread={spread:.3f}")
        
        return {
            "pcs": pcs,
            "explained": explained,
            "weights": weights,
            "groups": {k: [(n, float(pc[0]), float(pc[1])) for n, pc in v] 
                      for k, v in groups.items()},
        }
    
    # ── 2. Compare arch:gpt2 vs arch:fly ──
    def compare_architectures(self, n_neurons: int = 20, seed: int = 42) -> Dict:
        """
        Сравнивает генерацию разных архитектурных токенов.
        Фокус: arch:gpt2 vs arch:fly, gen_attention vs gen_mlp
        """
        print(f"\n{'='*60}")
        print(f"  🏛️  ARCHITECTURE COMPARISON ({n_neurons} neurons)")
        print(f"{'='*60}")
        
        comparisons = [
            ("arch:gpt2", "arch:fly", "GPT-2 vs Fly"),
            ("gen_attention", "gen_mlp", "Attention vs MLP"),
            ("gen_visual", "region:motor", "Visual vs Motor"),
            ("gen_recurrent", "gen_feedforward", "Recurrent vs Feedforward"),
        ]
        
        results = {}
        
        for tok1, tok2, label in comparisons:
            if tok2 == "gen_feedforward":
                tok2 = "gen_mlp"  # alias
            
            if tok1 not in CONTROL_TOKENS or tok2 not in CONTROL_TOKENS:
                print(f"\n  ⏭️  {label}: skipping (tokens not found)")
                continue
            
            print(f"\n  ── {label} ──")
            
            sp1 = SPECIES_BIOLOGICAL + CONTROL_TOKENS[tok1]
            sp2 = SPECIES_BIOLOGICAL + CONTROL_TOKENS[tok2]
            
            n1 = self._generate_states(sp1, n_neurons, seed)
            n2 = self._generate_states(sp2, n_neurons, seed + 1000)
            
            if len(n1) == 0 or len(n2) == 0:
                print(f"    ⚠️  One or both generated empty sequences")
                continue
            
            m1 = n1.mean(axis=0)
            m2 = n2.mean(axis=0)
            diff = m1 - m2
            
            print(f"  {'Metric':<12} | {'tok1':>8} | {'tok2':>8} | {'diff':>8} | {'interpretation':>20}")
            print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*20}")
            
            metrics = [
                ("Depth", 0, "сенсорность (high=вход)"),
                ("Degree", 1, "связность (high=плотно)"),
                ("Community", 2, "специализация (low=зрение)"),
                ("Btwn", 3, "интеграция (high=hub)"),
            ]
            
            interpretations = []
            for mname, idx, interp in metrics:
                d = diff[idx]
                arrow = "↑" if d > 0 else "↓"
                interp_text = f"{arrow} {interp} ({d:+.3f})"
                interpretations.append(interp_text)
                
                tok1_name = tok1.split(":")[-1].split("_")[-1][:8]
                tok2_name = tok2.split(":")[-1].split("_")[-1][:8]
                
                print(f"  {mname:<12} | {m1[idx]:>8.3f} | {m2[idx]:>8.3f} | "
                      f"{diff[idx]:>+8.3f} | {interp_text:>20}")
            
            results[f"{tok1}_vs_{tok2}"] = {
                "tok1_mean": m1.tolist(),
                "tok2_mean": m2.tolist(),
                "diff": diff.tolist(),
                "interpretation": interpretations,
            }
        
        return results
    
    def _generate_states(self, species_id: int, n_neurons: int, seed: int) -> np.ndarray:
        """Generate states for a species_id."""
        with torch.no_grad():
            tokens = self.model.generate(
                max_len=n_neurons + 4,
                temperature=0.7,
                top_k=6,
                species_id=species_id,
            )
        
        neurons = []
        for t in tokens:
            if len(t) == 4:
                if all(4 <= x <= 13 for x in t):
                    neurons.append(t)
        
        if not neurons:
            return np.zeros((0, 4))
        
        states = np.array([[(d-4)/9.0, (g-4)/9.0, (c-4)/9.0, (b-4)/9.0] 
                          for d, g, c, b in neurons[:n_neurons]], dtype=np.float32)
        return states
    
    # ── 3. Token arithmetic ──
    def token_arithmetic(self, n_neurons: int = 15, seed: int = 42):
        """
        Token arithmetic в embedding space.
        
        Вместо averaging outputs — averaging EMBEDDINGS и генерация.
        Это настоящая token arithmetic как word2vec.
        """
        print(f"\n{'='*60}")
        print(f"  🧮 TOKEN ARITHMETIC (embedding space)")
        print(f"{'='*60}")
        
        expressions = [
            ("gen_attention + arch:fly", ["gen_attention", "arch:fly"], [1.0, 1.0]),
            ("gen_visual + arch:gpt2", ["gen_visual", "arch:gpt2"], [1.0, 1.0]),
            ("arch:gpt2 + region:optic - arch:fly", 
             ["arch:gpt2", "region:optic", "arch:fly"], [1.0, 1.0, -1.0]),
            ("gen_attention - gen_mlp", ["gen_attention", "gen_mlp"], [1.0, -1.0]),
        ]
        
        for expr_str, tokens, signs in expressions:
            print(f"\n  ── [{expr_str}] ──")
            
            # Получаем embeddings токенов
            embed = self.model.embed.species_embed.weight
            d = self.config.d_model
            
            combined = torch.zeros(d, device=self.device)
            
            for tok, sign in zip(tokens, signs):
                if tok not in CONTROL_TOKENS:
                    continue
                sp = SPECIES_BIOLOGICAL + CONTROL_TOKENS[tok]
                vec = embed[sp].detach().clone()
                combined = combined + sign * vec
            
            # Нормализуем (важно для arithmetic)
            combined = combined / (combined.norm() + 1e-8)
            
            # Поиск ближайшего существующего species embedding
            cosines = torch.nn.functional.cosine_similarity(
                combined.unsqueeze(0), embed.detach(), dim=1
            ).cpu().numpy()
            
            top5 = np.argsort(cosines)[::-1][:5]
            
            print(f"    Nearest species to expression:")
            for idx in top5:
                if idx < SPECIES_BIOLOGICAL:
                    name = f"species_{idx} (biological)"
                else:
                    token_id = idx - SPECIES_BIOLOGICAL
                    name = list(CONTROL_TOKENS.keys())[token_id]
                print(f"      {name:<25} cos={cosines[idx]:.4f}")
            
            # Генерируем с арифметическим embedding'ом
            # Подменяем species_embed на лету для одного вызова
            # (если бы Sardna умела принимать произвольный embedding...)
            # Для PoC: используем ближайший control token
            best_idx = top5[0]
            if best_idx >= SPECIES_BIOLOGICAL:
                best_token_id = best_idx - SPECIES_BIOLOGICAL
                best_name = list(CONTROL_TOKENS.keys())[best_token_id]
                print(f"    → Best match: [{best_name}]")
                
                # Генерируем с этим токеном
                states = self._generate_states(best_idx, n_neurons, seed)
                if len(states) > 0:
                    mean = states.mean(axis=0)
                    print(f"    Generated: depth={mean[0]:.3f}, degree={mean[1]:.3f}, "
                          f"comm={mean[2]:.3f}, btwn={mean[3]:.3f}")
    
    # ── 4. All tokens full report ──
    def full_report(self, n_neurons: int = 15, seed: int = 42):
        """Полный отчёт по всем control tokens."""
        print(f"\n{'='*60}")
        print(f"  📊 FULL REPORT: All {N_CONTROL} control tokens")
        print(f"{'='*60}")
        
        print(f"\n  {'Token':<22} | {'Depth':>6} | {'Degree':>7} | {'Comm':>5} | "
              f"{'Btwn':>5} | {'Target':>25} | {'Dist':>6}")
        print(f"  {'─'*22}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*25}─┼─{'─'*6}")
        
        for token_name, token_id in sorted(CONTROL_TOKENS.items(), 
                                           key=lambda x: x[1]):
            species_id = SPECIES_BIOLOGICAL + token_id
            states = self._generate_states(species_id, n_neurons, seed)
            
            if len(states) == 0:
                print(f"  [{token_name:<20}] | {'N/A':>6}")
                continue
            
            mean = states.mean(axis=0)
            target = np.array(CONTROL_PROFILES.get(token_name, [0.5]*4))
            dist = np.sqrt(((mean - target)**2).mean())
            
            target_str = f"d={target[0]:.2f} dg={target[1]:.2f}"
            
            # Эмодзи-статус
            if dist < 0.15:
                status = "✅"
            elif dist < 0.30:
                status = "🟡"
            else:
                status = "❌"
            
            print(f"  [{token_name:<20}] | {mean[0]:6.3f} | {mean[1]:7.3f} | "
                  f"{mean[2]:5.3f} | {mean[3]:5.3f} | {target_str:<25} | "
                  f"{dist:6.4f} {status}")
        
        # ── Вывод: кто ближе всего к target ──
        print(f"\n  🏆 Best matched tokens:")
        best = []
        for token_name, _ in CONTROL_TOKENS.items():
            species_id = SPECIES_BIOLOGICAL + CONTROL_TOKENS[token_name]
            states = self._generate_states(species_id, n_neurons, seed)
            if len(states) > 0:
                target = np.array(CONTROL_PROFILES.get(token_name, [0.5]*4))
                dist = np.sqrt(((states.mean(axis=0) - target)**2).mean())
                best.append((token_name, dist))
        
        best.sort(key=lambda x: x[1])
        for name, dist in best[:5]:
            print(f"    [{name:<20}] distance={dist:.4f}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧬 Probe Sardna")
    parser.add_argument("--model", type=str, default=SARDNA_MODEL_DEFAULT)
    parser.add_argument("--pca-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    probe = SardnaProbe(args.model)
    
    if args.pca_only:
        probe.pca_embeddings()
        return
    
    if args.quick:
        probe.compare_architectures(n_neurons=args.n, seed=args.seed)
        return
    
    # ── Full probing ──
    # 1. PCA
    pca_results = probe.pca_embeddings()
    
    # 2. Architecture comparison
    arch_results = probe.compare_architectures(n_neurons=args.n, seed=args.seed)
    
    # 3. Token arithmetic
    probe.token_arithmetic(n_neurons=args.n, seed=args.seed)
    
    # 4. Full report
    probe.full_report(n_neurons=args.n, seed=args.seed)
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  📋 PROBE SUMMARY")
    print(f"{'='*60}")
    
    # PCA diversity
    if pca_results:
        groups = pca_results["groups"]
        for gname, items in groups.items():
            if items:
                pcs = np.array([[pc1, pc2] for _, pc1, pc2 in items])
                spread = pcs.std(axis=0).mean()
                print(f"  {gname:<12} spread={spread:.3f} ({len(items)} tokens)")
    
    # Architecture differences
    if arch_results:
        print(f"\n  Key architectural differences:")
        for key, data in arch_results.items():
            diff = data["diff"]
            print(f"    {key}: depth={diff[0]:+.3f}, degree={diff[1]:+.3f}, "
                  f"comm={diff[2]:+.3f}, btwn={diff[3]:+.3f}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ Probe complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
