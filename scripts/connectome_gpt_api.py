#!/usr/bin/env python3
"""
🧬 ConnectomeGPT API — Control Tokens for Sardna v2

Философия:
  Sardna умеет генерировать структурные признаки нейронов (depth, degree,
  community, betweenness). Добавляем CONTROL TOKEN'ы как новые species —
  специальные токены которые работают как API команды:

    [gen_visual]     — сгенерировать зрительную цепь
    [gen_attention]  — сгенерировать attention-like структуру
    [gen_cpg]        — сенсор → мотор (CPG)
    [arch:gpt2]      — стиль GPT-2
    [arch:fly]       — стиль мухи
    [region:optic]   — оптическая область

  Token arithmetic:
    [gen_attention] + [arch:fly] - [region:optic]
    = "attention-like connectivity in non-visual fly regions"

Usage:
  python scripts/connectome_gpt_api.py --prompt gen_visual --n 10
  python scripts/connectome_gpt_api.py --compare gen_visual gen_attention
  python scripts/connectome_gpt_api.py --all
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple

# ─── Путь к Sardna ────────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
SARDNA_MODEL = os.path.join(SARDANA_PATH, "models", "sardna", "sarDNA_v2.pt")

if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_gpt.sardna import SarDNA, Config

# ─── Структура вокабуляра Sardna ─────────────────────────
# Embedding: 10 bins + 4 special = 14 токенов
#   PAD  = 0  (padding_idx)
#   BOS  = 1
#   EOS  = 2
#   MASK = 3
#   bins = 4..13 (10 штук для каждой категории: depth, degree, community, btwn)
# Species: 7 биологических (0=FlyWire, 1-6=другие виды)

# ═══════════════════════════════════════════════════════════
#  CONTROL TOKENS VOCABULARY
# ═══════════════════════════════════════════════════════════

# Sardna species: 0=FlyWire, 1..6=другие виды
# Мы добавляем 16 CONTROL TOKENS как species 7..22
# species_id → control token mapping

SPECIES_BIOLOGICAL = 7   # 0..6 — существующие виды

CONTROL_TOKENS = {
    # ── Gen: генерация определённой архитектуры ──
    "gen_visual":      0,   # visual processing circuit
    "gen_attention":   1,   # attention-like connectivity
    "gen_mlp":         2,   # feedforward / expansion layer
    "gen_cpg":         3,   # central pattern generator (sensorimotor loop)
    "gen_lm_head":     4,   # language modeling head (readout)
    "gen_recurrent":   5,   # recurrent / feedback circuit
    "gen_random":      6,   # random connectome (baseline)
    
    # ── Arch: архитектурный стиль ──
    "arch:gpt2":       7,   # GPT-2 transformer style
    "arch:fly":        8,   # Drosophila fly style
    "arch:vnc":        9,   # ventral nerve cord style
    "arch:mouse":     10,   # mouse connectome style
    
    # ── Region: анатомическая область ──
    "region:optic":   11,   # optic lobe
    "region:antennal": 12,  # antennal lobe / olfactory
    "region:mushroom": 13,  # mushroom body / memory
    "region:motor":   14,   # motor output
    "region:random":  15,   # no specific region
}

N_CONTROL = len(CONTROL_TOKENS)  # 16
SPECIES_TOTAL = SPECIES_BIOLOGICAL + N_CONTROL  # 23

CONTROL_NAMES = {v: k for k, v in CONTROL_TOKENS.items()}

# ── Функциональные метки для каждого control token ──
CONTROL_SIGNATURES = {
    "gen_visual":      "👁️  Зрительная цепь — high depth, community=optic",
    "gen_attention":   "🔗 Attention-like — high betweenness, all-to-all connectivity",
    "gen_mlp":         "⚡ MLP/Feedforward — high degree expansion, low betweenness",
    "gen_cpg":         "🔄 CPG — sensorimotor loop, balanced structure",
    "gen_lm_head":     "🎯 LM Head — output layer, high out-degree",
    "gen_recurrent":   "🌀 Recurrent — high betweenness, feedback loops",
    "gen_random":      "🎲 Random baseline — no specific structure",
    "arch:gpt2":       "🤖 GPT-2 style — layered, high-degree expansion",
    "arch:fly":        "🪰 Fly style — biological connectivity patterns",
    "arch:vnc":        "🐛 VNC style — motor-centric, structured",
    "arch:mouse":      "🐭 Mouse style — mammalian connectivity",
    "region:optic":    "👁️  Optic lobe — sensory input, high depth",
    "region:antennal": "👃 Antennal lobe — olfactory processing",
    "region:mushroom": "🧠 Mushroom body — memory, integration",
    "region:motor":    "🦾 Motor region — output, low depth",
    "region:random":   "🌐 Random region — no specific bias",
}


# ═══════════════════════════════════════════════════════════
#  CONTROL-AWARE SARDNA WRAPPER
# ═══════════════════════════════════════════════════════════

class ConnectomeGPT:
    """
    Sardna v2 с расширенным API контрольных токенов.
    
    Использует species_id как механизм control tokens.
    Species 0..6 = биологические виды (0=FlyWire)
    Species 7..22 = control tokens (gen_visual, arch:gpt2, ...)
    """
    
    def __init__(self, model_path: str = SARDNA_MODEL, device: str = "cpu"):
        self.device = torch.device(device)
        
        # ── Загружаем базовую Sardna ──
        print(f"  📥 Loading Sardna v2 from {model_path}...")
        self.model = SarDNA.load_trained(model_path, config=None, device=device)
        self.model.eval()
        self.config = self.model.config
        
        # ── Расширяем species_embed для control tokens ──
        self._expand_species_embedding()
        
        print(f"  ✅ Sardna loaded: d_model={self.config.d_model}, "
              f"n_layers={self.config.n_layers}")
        print(f"  🏷️  Species: 0..6 biological + 7..22 control = {SPECIES_TOTAL} total")
        print(f"  🎮  {N_CONTROL} control tokens available")
    
    def _expand_species_embedding(self):
        """
        Расширяем species_embed с 7 → 23 species.
        Новые species (7..22) = control tokens.
        
        Инициализация: для каждого control token создаём embedding
        как комбинацию ближайших биологических species.
        """
        old_embed = self.model.embed.species_embed
        old_weight = old_embed.weight.data  # [7, d_model]
        d = self.config.d_model
        
        # Новый embedding [23, d_model]
        new_embed = nn.Embedding(SPECIES_TOTAL, d)
        new_embed.weight.data[:SPECIES_BIOLOGICAL] = old_weight
        
        # Инициализируем control embeddings
        # Базовый принцип: разные control tokens → разные регионы embedding space
        # Используем ортогональные направления от биологических species
        base = old_weight.mean(dim=0)  # центроид биологических species
        
        for token_name, token_id in CONTROL_TOKENS.items():
            species_idx = SPECIES_BIOLOGICAL + token_id
            
            # Создаём уникальное направление через token_id
            np.random.seed(42 + token_id * 7)
            direction = torch.from_numpy(np.random.randn(d).astype(np.float32))
            direction = direction / (direction.norm() + 1e-8)
            
            # Масштаб: 0.1-0.3 от std биологических species
            scale = 0.15 + 0.15 * (token_id / N_CONTROL)
            init_vec = base + scale * direction * old_weight.std().item()
            
            new_embed.weight.data[species_idx] = init_vec
        
        self.model.embed.species_embed = new_embed
        self.model.config.n_species = SPECIES_TOTAL
        print(f"  🔧 Species embedding expanded: 7 → {SPECIES_TOTAL}")
    
    def generate(
        self,
        prompt: Optional[List[str]] = None,
        n_neurons: int = 10,
        temperature: float = 0.8,
        top_k: int = 8,
        seed: Optional[int] = None,
    ) -> Tuple[List[Tuple], Dict]:
        """
        Генерация последовательности нейронов с control token промптом.
        
        Args:
            prompt: список control tokens (например ["gen_visual", "arch:fly"])
            n_neurons: количество нейронов
            temperature: температура sampling
            top_k: top-k фильтр
            seed: random seed
            
        Returns:
            (neurons, info)
              neurons: [(depth_bin, degree_bin, comm_bin, btwn_bin), ...]
              info: метаданные (states, prompt_used, means)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # ── Определяем species_id из промпта ──
        # Первый control token → species_id (7..22)
        # Если промпта нет → species=0 (FlyWire)
        species_id = 0
        prompt_used = prompt or []
        
        for token_str in prompt_used:
            token_name = token_str.strip("[]")
            # Прямой lookup
            if token_name in CONTROL_TOKENS:
                species_id = SPECIES_BIOLOGICAL + CONTROL_TOKENS[token_name]
                break
            # С [gen_/arch:/region:] префиксом
            for prefix in ["gen_", "arch:", "region:"]:
                if token_name.startswith(prefix):
                    clean = token_name[len(prefix):]
                    full = token_name
                    if full in CONTROL_TOKENS:
                        species_id = SPECIES_BIOLOGICAL + CONTROL_TOKENS[full]
                        break
            if species_id > 0:
                break
        
        # ── Генерируем через Sardna ──
        with torch.no_grad():
            tokens = self.model.generate(
                max_len=n_neurons + 4,  # +4 safety margin for EOS
                temperature=temperature,
                top_k=top_k,
                species_id=species_id,
            )
        
        # ── Пост-обработка ──
        # Sardna возвращает [(d_bin, g_bin, c_bin, b_bin), ...]
        # Фильтруем только валидные бины (4..13)
        neurons = []
        for t in tokens:
            if len(t) == 4:
                d, g, c, b = t
                # Бин = 4..13 (первые 4 — special tokens)
                if 4 <= d <= 13 and 4 <= g <= 13 and 4 <= c <= 13 and 4 <= b <= 13:
                    neurons.append(t)
        
        neurons = neurons[:n_neurons]
        
        # ── Конвертируем в float состояния [0, 1] ──
        # Бин i (4..13) → float (i-4)/9 = 0..1
        states = np.array([[(d-4)/9.0, (g-4)/9.0, (c-4)/9.0, (b-4)/9.0] 
                          for d, g, c, b in neurons], dtype=np.float32)
        
        info = {
            "prompt": prompt_used,
            "species_id": species_id,
            "species_name": CONTROL_NAMES.get(species_id - SPECIES_BIOLOGICAL, "biological"),
            "n_neurons": len(neurons),
            "states": states,
            "mean_depth": float(np.mean(states[:, 0])) if len(states) > 0 else 0,
            "mean_degree": float(np.mean(states[:, 1])) if len(states) > 0 else 0,
            "mean_community": float(np.mean(states[:, 2])) if len(states) > 0 else 0,
            "mean_btwn": float(np.mean(states[:, 3])) if len(states) > 0 else 0,
        }
        
        return neurons, info
    
    def compare_prompts(
        self,
        prompts: List[str],
        n_neurons: int = 10,
        seed: int = 42,
    ) -> Dict:
        """Сравнить генерацию с разными промптами."""
        results = {}
        for prompt_name in prompts:
            neurons, info = self.generate(
                prompt=[prompt_name],
                n_neurons=n_neurons,
                seed=seed,
            )
            results[prompt_name] = info
        return results
    
    def token_arithmetic(
        self,
        expression: str,
        n_neurons: int = 10,
        temperature: float = 0.9,
        seed: int = 42,
    ) -> Dict:
        """
        Token arithmetic: комбинируем эффекты control tokens.
        
        В v0: weighted average state vectors от каждого компонента.
        В будущем: interpolation в embedding space.
        """
        # Парсим выражение
        terms = expression.replace("+", " +").replace("-", " -").split()
        tokens = []
        sign = 1.0
        
        for term in terms:
            if term == "+":
                sign = 1.0
            elif term == "-":
                sign = -1.0
            else:
                clean = term.strip("[]")
                if clean in CONTROL_TOKENS:
                    tokens.append((clean, sign))
        
        # Нормализуем веса
        total_abs = sum(abs(s) for _, s in tokens)
        if total_abs > 0:
            tokens = [(n, s / total_abs) for n, s in tokens]
        else:
            tokens = [("gen_random", 1.0)]
        
        # Генерируем каждым токеном
        results = {}
        for name, w in tokens:
            neurons, info = self.generate(
                prompt=[name],
                n_neurons=n_neurons,
                temperature=temperature,
                seed=seed,
            )
            results[name] = info
        
        # Комбинируем weighted states
        combined_states = np.zeros((n_neurons, 4), dtype=np.float32)
        for name, w in tokens:
            if name in results:
                states = results[name]["states"]
                if len(states) >= n_neurons:
                    combined_states += w * states[:n_neurons]
                else:
                    pad = np.tile(states.mean(axis=0, keepdims=True),
                                 (n_neurons - len(states), 1))
                    combined_states += w * np.vstack([states, pad]) if len(states) > 0 else 0
        
        return {
            "expression": expression,
            "weights": {n: s for n, s in tokens},
            "component_results": results,
            "combined_mean_depth": float(np.mean(combined_states[:, 0])),
            "combined_mean_degree": float(np.mean(combined_states[:, 1])),
            "combined_mean_community": float(np.mean(combined_states[:, 2])),
            "combined_mean_btwn": float(np.mean(combined_states[:, 3])),
            "combined_states": combined_states,
        }


# ═══════════════════════════════════════════════════════════
#  DEMO & VISUALIZATION
# ═══════════════════════════════════════════════════════════

def print_neurons(neurons: List[Tuple], states: np.ndarray, label: str = ""):
    """Форматированный вывод нейронов."""
    print(f"\n  {'─'*55}")
    print(f"  📋 {label}")
    print(f"  {'─'*55}")
    
    if len(states) == 0:
        print("  (no neurons generated)")
        return
    
    print(f"  {'#':>3} | {'Depth':>6} | {'Degree':>7} | {'Comm':>5} | {'Btwn':>5} | {'Hint':>8}")
    print(f"  {'─'*3}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*8}")
    
    for i, (d, g, c, b) in enumerate(states[:10]):
        # Эмодзи-подсказка на основе community
        if c < 0.25: emoji = "👁️"
        elif c < 0.45: emoji = "👃"
        elif c < 0.65: emoji = "🧠"
        else: emoji = "🦾"
        
        print(f"  {i:3d} | {d:6.3f} | {g:7.3f} | {c:5.3f} | {b:5.3f} | {emoji:>8}")
    
    if len(states) > 10:
        print(f"  {'...':3} | {'...':6} | {'...':7} | {'...':5} | {'...':5} | {'...':>8}")
    
    mean = states.mean(axis=0)
    print(f"  {'─'*3}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*8}")
    print(f"  {'μ':3} | {mean[0]:6.3f} | {mean[1]:7.3f} | {mean[2]:5.3f} | {mean[3]:5.3f}")


def run_comparison(cgpt: ConnectomeGPT, prompts: List[str], n: int = 10):
    """Запустить и сравнить несколько промптов."""
    print(f"\n{'='*65}")
    print(f"  🔬 COMPARISON: {len(prompts)} prompts × {n} neurons")
    print(f"{'='*65}")
    
    results = cgpt.compare_prompts(prompts, n_neurons=n)
    
    # Таблица сравнения
    print(f"\n  {'Control Token':<22} | {'Depth':>6} | {'Degree':>7} | {'Comm':>5} | {'Btwn':>5} | {'Region':>10}")
    print(f"  {'─'*22}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*10}")
    
    # Собираем для bar chart
    names_list = []
    depth_list = []
    degree_list = []
    comm_list = []
    btwn_list = []
    
    for prompt_name in prompts:
        info = results[prompt_name]
        states = info["states"]
        if len(states) == 0:
            print(f"  {prompt_name:<22} | {'N/A':>6}")
            continue
        
        mean = states.mean(axis=0)
        names_list.append(prompt_name)
        depth_list.append(mean[0])
        degree_list.append(mean[1])
        comm_list.append(mean[2])
        btwn_list.append(mean[3])
        
        # Определяем доминирующий регион
        c = mean[2]
        if c < 0.25: region = "👁️ visual"
        elif c < 0.45: region = "👃 olfact"
        elif c < 0.65: region = "🧠 central"
        else: region = "🦾 motor"
        
        print(f"  [{prompt_name:<20}] | {mean[0]:6.3f} | {mean[1]:7.3f} | "
              f"{mean[2]:5.3f} | {mean[3]:5.3f} | {region:>10}")
    
    # ASCII bar chart — depth
    print(f"\n  📊 Depth by prompt:")
    for name, d in zip(names_list, depth_list):
        bar = "█" * int(d * 30)
        print(f"    [{name:<20}] {bar} {d:.3f}")
    
    # ASCII bar chart — degree  
    print(f"\n  📊 Degree by prompt:")
    for name, dg in zip(names_list, degree_list):
        bar = "█" * int(dg * 30)
        print(f"    [{name:<20}] {bar} {dg:.3f}")
    
    # Описательная статистика
    print(f"\n  📝 Анализ:")
    for prompt_name in prompts:
        info = results[prompt_name]
        desc = CONTROL_SIGNATURES.get(prompt_name, "")
        print(f"    [{prompt_name:<20}] {desc}")


def run_arithmetic_demo(cgpt: ConnectomeGPT):
    """Демонстрация token arithmetic."""
    print(f"\n{'='*65}")
    print(f"  🧮 TOKEN ARITHMETIC DEMO")
    print(f"{'='*65}")
    
    expressions = [
        "gen_visual + arch:gpt2",
        "gen_cpg + region:motor",
        "gen_attention + arch:fly",
    ]
    
    for expr in expressions:
        print(f"\n  ── Expression: [{expr}]")
        result = cgpt.token_arithmetic(expr, n_neurons=10)
        
        print(f"     Weights:        {result['weights']}")
        print(f"     Combined:       depth={result['combined_mean_depth']:.3f}, "
              f"degree={result['combined_mean_degree']:.3f}, "
              f"comm={result['combined_mean_community']:.3f}, "
              f"btwn={result['combined_mean_btwn']:.3f}")
        
        for name, comp_info in result['component_results'].items():
            desc = CONTROL_SIGNATURES.get(name, "")
            print(f"       [{name}] depth={comp_info['mean_depth']:.3f}, "
                  f"degree={comp_info['mean_degree']:.3f} — {desc}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧬 ConnectomeGPT API — Control Tokens for Sardna v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/connectome_gpt_api.py --prompt gen_visual
  python scripts/connectome_gpt_api.py --compare gen_visual gen_attention gen_cpg
  python scripts/connectome_gpt_api.py --arithmetic
  python scripts/connectome_gpt_api.py --all
        """
    )
    parser.add_argument("--prompt", type=str, default=None,
                       help="Control token (gen_visual, gen_attention, arch:gpt2, ...)")
    parser.add_argument("--n", type=int, default=10,
                       help="Number of neurons to generate")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                       help="Compare multiple prompts")
    parser.add_argument("--arithmetic", action="store_true",
                       help="Run token arithmetic demo")
    parser.add_argument("--all", action="store_true",
                       help="Run all demos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()
    
    # ── Init ──
    cgpt = ConnectomeGPT()
    
    # ── Single prompt ──
    if args.prompt and not args.all:
        desc = CONTROL_SIGNATURES.get(args.prompt, "")
        print(f"\n  🎯 Generating with prompt: [{args.prompt}]")
        print(f"     {desc}")
        
        neurons, info = cgpt.generate(
            prompt=[args.prompt],
            n_neurons=args.n,
            seed=args.seed,
            temperature=args.temp,
        )
        
        print_neurons(neurons, info["states"],
                     f"Species #{info['species_id']}: {info['species_name']}")
    
    # ── Compare ──
    if args.compare or args.all:
        prompts = args.compare or [
            "gen_visual", "gen_attention", "gen_mlp", "gen_cpg",
            "gen_lm_head", "gen_recurrent", "gen_random",
            "arch:gpt2", "arch:fly",
            "region:optic", "region:antennal", "region:mushroom", "region:motor",
        ]
        run_comparison(cgpt, prompts, n=args.n)
    
    # ── Arithmetic ──
    if args.arithmetic or args.all:
        run_arithmetic_demo(cgpt)
    
    print(f"\n{'='*65}")
    print(f"  ✅ ConnectomeGPT API Demo Complete!")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
