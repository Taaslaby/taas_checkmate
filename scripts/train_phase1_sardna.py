#!/usr/bin/env python3
"""
🎯 Phase 1: LoRA Fine-tune Qwen3 on Sardna dataset

Обучаем Qwen3 генерировать структурные последовательности (d,g,c,b)
как Sardna, используя текстовый датасет с brain-токенами.

Формат: "<|brain_start|> <|depth_3|> <|degree_5|> ... <|brain_end|>"

Strategy:
  - Load Qwen3-brain (расширенный вокабуляр с Sardna токенами)
  - LoRA на attention слои
  - Авторегрессивная генерация brain-токенов
  - Save LoRA adapter (не всю модель)

Usage:
  python scripts/train_phase1_sardna.py
  python scripts/train_phase1_sardna.py --epochs 5
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ─── Пути ─────────────────────────────────────────────────
QWEN_BRAIN_DIR = "/home/taaslab/Проекты/asm_engines/taas_checkmate/models/qwen3_brain"
DATASET_DIR = "/home/taaslab/Проекты/asm_engines/taas_checkmate/data/sardna_text"
OUTPUT_DIR = "/home/taaslab/Проекты/asm_engines/taas_checkmate/models/lora_phase1"


def load_model():
    """Загружаем Qwen3-brain с 4-bit квантизацией (QLoRA)."""
    print(f"  📥 Loading Qwen3-brain from {QWEN_BRAIN_DIR} (4-bit)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 4-bit quantization config — влезает в 8 GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN_BRAIN_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_BRAIN_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Проверяем Sardna токены
    test_tokens = ["<|depth_0|>", "<|brain_start|>", "<|brain_end|>"]
    for t in test_tokens:
        ids = tokenizer.encode(t)
        assert len(ids) == 1, f"Token {t} should be a single token but got {ids}"
    print(f"     Vocab size: {len(tokenizer)}")
    print(f"     Sardna tokens: OK")

    return model, tokenizer


def setup_lora(model):
    """Настраиваем LoRA на Qwen3."""
    from peft import get_peft_model, LoraConfig, TaskType

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dataset(tokenizer, dataset_name="sardna_all",
                 max_length=1024, max_samples=None):
    """Загружаем текстовый датасет и токенизируем."""
    dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.txt")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Устанавливаем pad_token = eos_token (оба уже есть в вокабуляре)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  📂 Loading {dataset_path}...")
    with open(dataset_path, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    if max_samples:
        texts = texts[:max_samples]

    print(f"     {len(texts)} samples")

    # Токенизируем с padding до max_length
    print(f"  🔄 Tokenizing and padding...")
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    print(f"     Done! Shape: {encodings['input_ids'].shape}")
    print(f"     Avg non-pad tokens: {(encodings['input_ids'] != tokenizer.pad_token_id).sum(dim=1).float().mean():.0f}")
    return encodings


def train_epoch(model, dataloader, optimizer, device, epoch, n_epochs, val_dataloader=None):
    """Одна эпоха обучения."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Next-token prediction с игнором PAD
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore PAD

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()
        n_batches += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}/{n_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                  f"Loss: {avg_loss:.4f}")

    train_loss = total_loss / n_batches

    # Validation
    val_loss = None
    if val_dataloader is not None:
        model.eval()
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_total += outputs.loss.item()
                val_batches += 1
        val_loss = val_total / max(val_batches, 1)
        print(f"  📊 Val loss: {val_loss:.4f}")

    return train_loss, val_loss


def generate_test(model, tokenizer, prompt="Generate a brain circuit: <|brain_start|>"):
    """Тестовая генерация."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🎯 Phase 1: LoRA Fine-tune Qwen3 on Sardna")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="sardna_all",
                       help="Dataset: sardna_train, sardna_hybrid, sardna_scaled, sardna_all")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples for quick test")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Max sequence length")
    parser.add_argument("--test-only", action="store_true",
                       help="Run generation test only")
    args = parser.parse_args()

    print("=" * 60)
    print("  🎯 PHASE 1: LoRA Fine-tune Qwen3 on Sardna")
    print("=" * 60)

    # 1. Load model
    model, tokenizer = load_model()

    if args.test_only:
        # Test generation
        for prompt in [
            "Generate a brain circuit: <|brain_start|>",
            "Visual motion detection circuit: <|brain_start|>",
            "Motor pattern: <|brain_start|>",
        ]:
            response = generate_test(model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {response}")
        return

    # 2. Setup LoRA
    print(f"\n  🔧 Setting up LoRA...")
    model = setup_lora(model)
    device = model.device

    # 3. Load dataset
    encodings = load_dataset(tokenizer, args.dataset,
                              max_length=args.max_length,
                              max_samples=args.max_samples)

    # 4. Create dataset + train/val split (один раз, ДО цикла)
    from torch.utils.data import DataLoader, TensorDataset, random_split
    input_ids = encodings["input_ids"]
    attention_mask = encodings.get("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    dataset = TensorDataset(input_ids, attention_mask)

    val_size = max(1, len(dataset) // 5)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_batches_per_epoch = len(train_loader)

    # 5. Optimizer (только LoRA params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * n_batches_per_epoch
    )

    # 6. Gradient checkpointing + clear cache
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()
    print(f"  🔄 Gradient checkpointing enabled, cache cleared")

    # 7. Train
    print(f"\n  🚀 Starting training ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})")
    print(f"     {n_batches_per_epoch} batches per epoch, {len(val_loader)} val batches")
    print()

    for epoch in range(args.epochs):
        train_loss, val_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs, val_loader
        )
        scheduler.step()
        print(f"\n  ✅ Epoch {epoch+1} complete! Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}\n")

        # Test after each epoch
        test_output = generate_test(model, tokenizer)
        print(f"  🧪 Test: {test_output[:200]}...\n")

    # 7. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n  💾 LoRA adapter saved to {OUTPUT_DIR}")

    # 8. Final test
    print(f"\n  🧪 Final generation test:")
    for prompt in [
        "Generate a brain circuit: <|brain_start|>",
        "Optic flow pattern: <|brain_start|>",
        "Motor neuron sequence: <|brain_start|>",
    ]:
        response = generate_test(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {response}")

    print(f"\n{'='*60}")
    print(f"  ✅ Phase 1 complete! Qwen3 now speaks Sardna! 🧬")
    print(f"  Next: python scripts/train_phase2_annotations.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
