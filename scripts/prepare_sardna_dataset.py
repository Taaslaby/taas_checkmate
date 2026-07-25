#!/usr/bin/env python3
"""
🧬 Prepare Sardna Dataset for Qwen3 Phase 1 Training

Конвертирует Sardna train_data.npy (10K samples × 128 steps × 4 features)
в текстовый формат с Sardna токенами для LoRA fine-tuning.

Формат:
  Вход:  "Generate a brain circuit: <|brain_start|>"
  Выход: "<|depth_3|> <|degree_5|> <|comm_2|> <|btwn_4|> ... <|brain_end|>"

Usage:
  python scripts/prepare_sardna_dataset.py
  python scripts/prepare_sardna_dataset.py --viz  # показать примеры
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Пути
SARDNA_DATASET_DIR = "/home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets"
OUTPUT_DIR = "/home/taaslab/Проекты/asm_engines/taas_checkmate/data/sardna_text"

N_BINS = 10
FEATURE_NAMES = ["depth", "degree", "comm", "btwn"]
TOKEN_NAMES = {feat: [f"<|{feat}_{i}|>" for i in range(N_BINS)] for feat in FEATURE_NAMES}
BRAIN_START = "<|brain_start|>"
BRAIN_END = "<|brain_end|>"

# ═══════════════════════════════════════════════════════════
#  Discretization: float → bin token
# ═══════════════════════════════════════════════════════════

# Data-driven normalization range (computed from train_data)
DATA_MIN = -0.6
DATA_MAX = 1.7
DATA_RANGE = DATA_MAX - DATA_MIN  # 2.3

def float_to_bin(val: float, feat_idx: int = 0) -> str:
    """
    Convert float to Sardna bin token.
    
    Data-driven normalization: [-0.6, 1.7] → [0, 1], then 10 bins.
    """
    if np.isnan(val):
        val = DATA_MIN
    val = np.clip(val, DATA_MIN, DATA_MAX)
    normalized = (val - DATA_MIN) / DATA_RANGE
    normalized = np.clip(normalized, 0.0, 0.999)
    bin_idx = int(normalized * N_BINS)
    bin_idx = min(bin_idx, N_BINS - 1)
    return TOKEN_NAMES[FEATURE_NAMES[feat_idx]][bin_idx]


def sample_to_text(sample: np.ndarray, max_steps: int = 128) -> str:
    """
    Convert one Sardna sample [128, 4] to text format.
    """
    tokens = [BRAIN_START]
    for t in range(min(max_steps, len(sample))):
        for f in range(4):
            tokens.append(float_to_bin(sample[t, f], f))
    tokens.append(BRAIN_END)
    return " ".join(tokens)


def text_to_tensor(text: str, tokenizer) -> dict:
    """Convert text to input_ids for Qwen3 training."""
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)


# ═══════════════════════════════════════════════════════════
#  Dataset creation
# ═══════════════════════════════════════════════════════════

def create_dataset():
    """Создаём текстовый датасет из Sardna train_data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Загружаем Sardna datasets
    print("📥 Loading Sardna datasets...")
    train_data = np.load(os.path.join(SARDNA_DATASET_DIR, "train_data.npy"))       # [10000, 128, 4]
    train_hybrid = np.load(os.path.join(SARDNA_DATASET_DIR, "train_hybrid_data.npy"))  # [20000, 128, 4]
    train_scaled = np.load(os.path.join(SARDNA_DATASET_DIR, "train_scaled_data.npy"))  # [5000, 1024, 4]

    print(f"  train_data:        {train_data.shape}")
    print(f"  train_hybrid_data: {train_hybrid.shape}")
    print(f"  train_scaled_data: {train_scaled.shape}")

    # Создаём текстовые датасеты
    print("\n📝 Converting to text...")

    # train_data — 10K samples, 128 steps
    texts_train = []
    for i in range(len(train_data)):
        texts_train.append(sample_to_text(train_data[i]))
        if (i + 1) % 2000 == 0:
            print(f"  train_data: {i+1}/{len(train_data)}")

    # train_hybrid — 20K samples, 128 steps
    texts_hybrid = []
    for i in range(len(train_hybrid)):
        texts_hybrid.append(sample_to_text(train_hybrid[i]))
        if (i + 1) % 5000 == 0:
            print(f"  hybrid: {i+1}/{len(train_hybrid)}")

    # train_scaled — 5K samples, 1024 steps → stride-slice в 128-step сегменты
    texts_scaled = []
    stride = 64  # 50% overlap
    for i in range(len(train_scaled)):
        for start in range(0, train_scaled.shape[1] - 128 + 1, stride):
            segment = train_scaled[i, start:start+128]
            texts_scaled.append(sample_to_text(segment, max_steps=128))
        if (i + 1) % 1000 == 0:
            print(f"  scaled: {i+1}/{len(train_scaled)}, segments={len(texts_scaled)}")

    # Сохраняем
    print("\n💾 Saving datasets...")

    def save_texts(texts, name):
        path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        with open(path, "w") as f:
            for text in texts:
                f.write(text + "\n")
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {name}: {len(texts)} samples, {size_mb:.1f} MB → {path}")

    save_texts(texts_train, "sardna_train")
    save_texts(texts_hybrid, "sardna_hybrid")
    save_texts(texts_scaled, "sardna_scaled")

    # Конкатенируем всё
    all_texts = texts_train + texts_hybrid + texts_scaled
    save_texts(all_texts, "sardna_all")

    print(f"\n✅ Total: {len(all_texts)} samples in sardna_all.txt")
    return all_texts


# ═══════════════════════════════════════════════════════════
#  Viz: показать примеры
# ═══════════════════════════════════════════════════════════

def show_examples():
    """Показать примеры конвертированных данных."""
    print("=" * 60)
    print("  📊 Sardna Dataset Examples")
    print("=" * 60)

    train_data = np.load(os.path.join(SARDNA_DATASET_DIR, "train_data.npy"))

    for s in range(3):
        text = sample_to_text(train_data[s], max_steps=8)
        print(f"\n  Sample {s} (first 8 timesteps):")
        print(f"  Raw:    depth={train_data[s,0,0]:.3f} degree={train_data[s,0,1]:.3f} "
              f"comm={train_data[s,0,2]:.3f} btwn={train_data[s,0,3]:.3f}")
        # Показываем первые 4 шага
        tokens = [BRAIN_START]
        for t in range(4):
            tokens += [float_to_bin(train_data[s, t, f], f) for f in range(4)]
        tokens.append("...")
        print(f"  Tokens: {' '.join(tokens)}")

    # Показать распределение бинов
    print(f"\n  📊 Bin distribution across all data:")
    all_vals = train_data[:, :, :].reshape(-1, 4)
    for f, feat in enumerate(FEATURE_NAMES):
        vals = all_vals[:, f]
        bins = [float_to_bin(v, f) for v in vals]
        from collections import Counter
        counts = Counter(bins)
        total = len(bins)
        print(f"  {feat}:")
        for i in range(N_BINS):
            token = TOKEN_NAMES[feat][i]
            pct = counts.get(token, 0) / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {token:15s} {pct:5.1f}% {bar}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧬 Prepare Sardna dataset for Qwen3")
    parser.add_argument("--viz", action="store_true", help="Show dataset examples")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion")
    args = parser.parse_args()

    if args.viz:
        show_examples()
        return

    if not args.skip_convert:
        create_dataset()

    print(f"\n  ✅ Ready for Phase 1 training!")
    print(f"  Run: python scripts/train_phase1_sardna.py")


if __name__ == "__main__":
    main()
