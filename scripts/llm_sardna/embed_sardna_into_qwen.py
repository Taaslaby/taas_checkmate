#!/usr/bin/env python3
"""
🧬 Embed Sardna structural tokens into Qwen3 vocabulary

Добавляет 42 новых токена в Qwen3-1.7B:
  <|depth_0|> .. <|depth_9|>    — 10 depth bins
  <|degree_0|> .. <|degree_9|>   — 10 degree bins  
  <|comm_0|> .. <|comm_9|>       — 10 community bins
  <|btwn_0|> .. <|btwn_9|>       — 10 betweenness bins
  <|brain_start|>                 — начать генерацию схемы
  <|brain_end|>                   — закончить генерацию схемы

Инициализация embedding-ов из Sardna structural весов (64-dim → 256-dim).
Сохраняет модель с расширенным вокабуляром для LoRA fine-tuning.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# ─── Пути ────────────────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
QWEN_CACHE = "/home/taaslab/Проекты/asm_engines/taas_checkmate/models/qwen3"
OUTPUT_DIR = "/home/taaslab/Проекты/asm_engines/taas_checkmate/models/qwen3_brain"

if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_gpt.sardna import SarDNA

# ══════════════════════════════════════════════════════════
#  1. Sardna Structural Tokens
# ══════════════════════════════════════════════════════════

N_BINS = 10  # Sardna output head = 10 classes
FEATURES = ["depth", "degree", "comm", "btwn"]

# Генерируем имена токенов
SARDNA_TOKENS = []
for feat in FEATURES:
    for i in range(N_BINS):
        SARDNA_TOKENS.append(f"<|{feat}_{i}|>")
SARDNA_TOKENS += ["<|brain_start|>", "<|brain_end|>"]

N_SARDNA_TOKENS = len(SARDNA_TOKENS)  # 42
print(f"  🧬 Sardna tokens to add: {N_SARDNA_TOKENS}")
for t in SARDNA_TOKENS:
    print(f"     {t}")


# ══════════════════════════════════════════════════════════
#  2. Загрузка Sardna для весов
# ══════════════════════════════════════════════════════════

def load_sardna_embeddings() -> dict:
    """Загружаем Sardna и извлекаем structural embeddings."""
    print(f"\n  📥 Loading Sardna v2...")
    model = SarDNA.load_trained(
        os.path.join(SARDANA_PATH, "models", "sardna", "sarDNA_v2.pt"),
        config=None, device="cpu"
    )
    d_model = model.config.d_model  # 256
    embed_dim = d_model // 4  # 64 (4 features, 256/4=64 each)

    embeddings = {}
    for feat in FEATURES:
        embed_layer = getattr(model.embed, f"{feat}_embed")
        # Sardna embedding: [14, 64] (4 special + 10 bins)
        weights = embed_layer.weight.data  # [14, 64]
        # Берём только bin weights (indices 4..13)
        bin_weights = weights[4:4+N_BINS]  # [10, 64]
        embeddings[feat] = bin_weights
        print(f"     {feat}_embed: {weights.shape} → bin {bin_weights.shape}")

    return embeddings, d_model, embed_dim


# ══════════════════════════════════════════════════════════
#  3. Расширение Qwen3 tokenizer и модели
# ══════════════════════════════════════════════════════════

def extend_qwen3(sardna_embeddings: dict, d_model: int, embed_dim: int,
                 device: str = "cpu"):
    """Загружаем Qwen3, расширяем вокабуляр, инициализируем embedding."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Находим snapshot
    snap_dir = os.path.join(QWEN_CACHE, "models--Qwen--Qwen3-1.7B", "snapshots")
    if not os.path.exists(snap_dir):
        snap_dir = QWEN_CACHE
        print(f"  📍 Looking for model in: {QWEN_CACHE}")
    else:
        snap = sorted(os.listdir(snap_dir))[-1]
        snap_dir = os.path.join(snap_dir, snap)

    print(f"\n  📥 Loading Qwen3-1.7B...")
    tokenizer = AutoTokenizer.from_pretrained(snap_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snap_dir, torch_dtype="auto", device_map=device,
        trust_remote_code=True,
    )
    vocab_size = len(tokenizer)
    embed_dim_qwen = model.config.hidden_size  # 2048
    print(f"     Vocab size: {vocab_size}")
    print(f"     Embed dim: {embed_dim_qwen}")
    print(f"     Device: {model.device}")

    # ── Добавляем новые токены ──
    print(f"\n  ➕ Adding {N_SARDNA_TOKENS} new tokens...")
    tokenizer.add_tokens(SARDNA_TOKENS)
    new_vocab_size = len(tokenizer)
    print(f"     New vocab size: {new_vocab_size}")
    assert new_vocab_size == vocab_size + N_SARDNA_TOKENS, \
        f"Expected {vocab_size + N_SARDNA_TOKENS}, got {new_vocab_size}"

    # ── Resize embeddings (расширяет и embed, и lm_head) ──
    print(f"\n  📏 Resizing embeddings to {new_vocab_size}...")
    model.resize_token_embeddings(new_vocab_size)
    embed = model.get_input_embeddings()
    print(f"     Embed shape: {embed.weight.shape}")

    # ── Инициализация Sardna токенов из структурных весов ──
    print(f"\n  🧬 Initializing Sardna token embeddings...")
    n_sardna = 0

    with torch.no_grad():
        for idx, token_name in enumerate(SARDNA_TOKENS):
            new_idx = vocab_size + idx
            feat_part = token_name.replace("<|", "").replace("|>", "")
            parts = feat_part.split("_")

            if parts[0] in FEATURES:
                # Structural bin token: <|depth_3|>, <|degree_7|>, etc.
                feat_name = parts[0]
                bin_idx = int(parts[1])
                sardna_vec = sardna_embeddings[feat_name][bin_idx]  # [64]

                # Sardna 64-dim → Qwen3 2048-dim
                # Копируем 64-dim в соответствующую четверть, repeat 8× чтобы заполнить 512
                qwen_vec = torch.zeros(embed_dim_qwen, dtype=embed.weight.dtype)
                offset = FEATURES.index(feat_name)  # 0=depth, 1=degree, 2=comm, 3=btwn
                quarter_size = embed_dim_qwen // 4  # 512
                start = offset * quarter_size
                end = start + quarter_size
                # Repeat Sardna vector 8× чтобы заполнить 512 (64×8=512)
                qwen_vec[start:end] = sardna_vec.to(qwen_vec.dtype).repeat(8)

                embed.weight.data[new_idx] = qwen_vec
                n_sardna += 1

            elif feat_part == "brain_start":
                # <|brain_start|> — среднее всех Sardna bins
                avg_vec = torch.zeros(embed_dim_qwen, dtype=embed.weight.dtype)
                for feat in FEATURES:
                    feat_mean = sardna_embeddings[feat].mean(dim=0)
                    offset = FEATURES.index(feat)
                    quarter_size = embed_dim_qwen // 4
                    start = offset * quarter_size
                    avg_vec[start:start+quarter_size] = feat_mean.to(avg_vec.dtype).repeat(8)
                embed.weight.data[new_idx] = avg_vec

            elif feat_part == "brain_end":
                # <|brain_end|> — инициализируем нулями (будет учиться)
                embed.weight.data[new_idx] = torch.zeros(embed_dim_qwen, dtype=embed.weight.dtype)

    print(f"     ✅ Initialized {n_sardna} Sardna bin tokens")
    print(f"     ✅ Initialized <|brain_start|> and <|brain_end|>")

    return model, tokenizer


# ══════════════════════════════════════════════════════════
#  4. Тест: проверка генерации новых токенов
# ══════════════════════════════════════════════════════════

def test_generation(model, tokenizer):
    """Проверяем что Qwen3 понимает новые токены."""
    print(f"\n{'='*55}")
    print(f"  🧪 TEST: Can Qwen3 generate Sardna tokens?")
    print(f"{'='*55}")

    # Тест 1: токенизация
    test_text = "<|brain_start|> <|depth_7|> <|degree_5|> <|comm_2|> <|btwn_3|> <|brain_end|>"
    tokens = tokenizer(test_text, return_tensors="pt")
    decoded = tokenizer.decode(tokens.input_ids[0])
    print(f"\n  Test 1 - Tokenization:")
    print(f"     Input:  {test_text}")
    print(f"     Tokens: {tokens.input_ids[0].tolist()}")
    print(f"     Decoded: {decoded}")
    assert "brain_start" in decoded and "depth_7" in decoded, \
        "Tokenization failed!"

    # Тест 2: forward pass (убеждаемся что нет ошибок размерности)
    print(f"\n  Test 2 - Forward pass:")
    with torch.no_grad():
        outputs = model(**tokens.to(model.device))
    logits = outputs.logits
    print(f"     Logits shape: {logits.shape}")
    assert logits.shape[-1] == len(tokenizer), \
        f"Logits vocab {logits.shape[-1]} != tokenizer vocab {len(tokenizer)}"
    print(f"     ✅ Forward pass OK!")

    # Тест 3: генерация brain последовательности
    print(f"\n  Test 3 - Generate brain tokens:")
    prompt = "Generate a visual motion detection circuit: <|brain_start|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.9,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"     Prompt: {prompt}")
    print(f"     Output: {response}")
    print(f"     ✅ Generation completed!")

    return True


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧬 Embed Sardna structural tokens into Qwen3 vocabulary"
    )
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to load model on")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                       help="Output directory")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip generation test")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Torch dtype: auto, float16, float32")
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f"  🧬 EMBED SARDNA INTO QWEN3")
    print(f"{'='*55}")

    # 1. Load Sardna embeddings
    sardna_embeddings, d_model, embed_dim = load_sardna_embeddings()

    # 2. Extend Qwen3
    model, tokenizer = extend_qwen3(
        sardna_embeddings, d_model, embed_dim, device=args.device
    )

    # 3. Save
    os.makedirs(args.output, exist_ok=True)
    print(f"\n  💾 Saving to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"     ✅ Model and tokenizer saved!")

    # 4. Test
    if not args.skip_test:
        test_generation(model, tokenizer)

    print(f"\n{'='*55}")
    print(f"  ✅ DONE! Qwen3 now speaks Sardna! 🧬🤖")
    print(f"  Model: {args.output}")
    print(f"  Sardna tokens: {N_SARDNA_TOKENS}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"{'='*55}")
    print(f"\n  Next: LoRA fine-tune on FlyWire text data!")
    print(f"  Run: python scripts/train_brain_lora.py")


if __name__ == "__main__":
    main()
