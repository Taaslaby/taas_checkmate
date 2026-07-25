#!/usr/bin/env python3
"""
🎯 Fine-tune Control Tokens для Sardna v2

Стратегия:
  1. Загружаем pre-trained Sardna v2
  2. Расширяем species_embed: 7 → 23 (7 biological + 16 control)
  3. Замораживаем ВСЕ веса, КРОМЕ 16 новых control embeddings
  4. Self-supervised data: используем BIOLOGICAL species (0-6)
     для генерации (d,g,c,b) последовательностей
  5. Каждой последовательности присваиваем control token
     на основе статистических свойств (depth, degree, comm, btwn)
  6. Обучаем: control token → forward() → predict next tokens
     Градиент идёт ТОЛЬКО в control embedding

После fine-tuning:
  [gen_visual]   — будет генерировать visual-like (high depth, low community)
  [arch:gpt2]    — high degree, high betweenness
  [region:motor] — low depth, high community
  и т.д.

Usage:
  python scripts/finetune_control_tokens.py --epochs 10
  python scripts/finetune_control_tokens.py --verify  # проверить после обучения
  python scripts/finetune_control_tokens.py --epochs 20 --lr 1e-3
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import time

# ─── Путь к Sardna ────────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
SARDNA_MODEL = os.path.join(SARDANA_PATH, "models", "sardna", "sarDNA_v2.pt")

if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_gpt.sardna import SarDNA, Config

# ═══════════════════════════════════════════════════════════
#  CONTROL TOKENS (same as connectome_gpt_api.py)
# ═══════════════════════════════════════════════════════════

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
SPECIES_TOTAL = SPECIES_BIOLOGICAL + N_CONTROL  # 23

# ── Target profiles: какие (d,g,c,b) распределения должен давать каждый control token
# Профиль = (mean_depth, mean_degree, mean_community, mean_btwn)
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

# ═══════════════════════════════════════════════════════════
#  DATA GENERATOR — self-supervised walks
# ═══════════════════════════════════════════════════════════

class ControlDataGenerator:
    """
    Генерирует training data для fine-tuning control tokens.
    
    Использует биологические species Sardna для создания
    (d,g,c,b) последовательностей, затем назначает
    control token label на основе профиля.
    
    Token layout:
      0 = PAD, 1 = BOS, 2 = EOS, 3 = MASK
      4..13 = structural bins (10 bins per feature)
    """
    
    NUM_BINS = 10  # 10 bins per feature
    SPECIAL = 4    # 4 special tokens
    BIN_START = 4  # first bin index
    
    def __init__(self, model: SarDNA, device: str = "cpu"):
        self.model = model
        self.device = device
        self.seq_len = 32
    
    def generate_sequence(self, species_id: int, seed: int = None) -> List[Tuple]:
        """
        Генерируем последовательность (d,g,c,b) используя Sardna.
        
        Sardna output head = 10 классов (0..9). generate() выдаёт
        значения 0..9, где 0=PAD, 1=BOS, 2=EOS, 3=MASK, 4..9=bins.
        
        Принимаем ВСЕ значения 0..9 — специальные токены тоже валидные
        next-token предсказания модели. Нормализация 0..9 → 0..1.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            tokens = self.model.generate(
                max_len=self.seq_len,
                temperature=0.7,
                top_k=5,
                species_id=species_id,
            )
        
        # Принимаем все значения 0..9 (Sardna output head = 10 классов)
        neurons = []
        for t in tokens:
            if len(t) == 4:
                d, g, c, b = t
                if all(0 <= x <= 9 for x in (d, g, c, b)):
                    neurons.append(t)
        return neurons
    
    def assign_control_token(self, neurons: List[Tuple]) -> str:
        """
        По статистике последовательности определяем
        наиболее подходящий control token.
        
        Используем L2 distance до target профилей.
        
        Sardna output head = 10 классов (0..9). Нормализация: (x - 0) / 9
        """
        if not neurons:
            return "gen_random"
        
        # Средние значения — нормализация 0..9 → 0..1
        states = np.array([(d, g, c, b) for d, g, c, b in neurons], dtype=np.float32)
        states = (states - 0.0) / 9.0
        mean = states.mean(axis=0)  # [depth, degree, comm, btwn]
        
        # Поиск ближайшего профиля
        best_token = "gen_random"
        best_dist = float('inf')
        
        for token_name, profile in CONTROL_PROFILES.items():
            dist = np.sqrt(np.sum((mean - np.array(profile))**2))
            if dist < best_dist:
                best_dist = dist
                best_token = token_name
        
        return best_token
    
    def create_batch(self, batch_size: int = 8, seed: int = 42) -> Dict:
        """
        Создаём батч training данных.
        
        Returns:
          batch = {
            "depths":    [B, seq_len] — int, bin indices (4..13 or 0=pad)
            "degrees":   [B, seq_len]
            "comms":     [B, seq_len]
            "btwns":     [B, seq_len]
            "species":   [B] — control token species_id (7..22)
            "mask":      [B, seq_len] — 1=valid, 0=pad
          }
        """
        depths, degrees, comms, btwns = [], [], [], []
        species_list = []
        
        rng = np.random.RandomState(seed)
        
        for i in range(batch_size):
            # Генерируем случайным biological species или control token
            if rng.rand() < 0.5:
                # Biological species (0..6) — для разнообразия
                s = int(rng.randint(0, SPECIES_BIOLOGICAL))
            else:
                # Control token (0..15) → species (7..22)
                ctrl = int(rng.randint(0, N_CONTROL))
                s = SPECIES_BIOLOGICAL + ctrl
            
            # Генерируем последовательность
            neurons = self.generate_sequence(s, seed=seed + i * 7)
            if len(neurons) < 2:
                continue
            
            # Определяем control token label
            assigned = self.assign_control_token(neurons)
            if assigned in CONTROL_TOKENS:
                label_species = SPECIES_BIOLOGICAL + CONTROL_TOKENS[assigned]
            else:
                label_species = s  # fallback
            
            # Pad/crop до seq_len
            seq = neurons[:self.seq_len - 1]  # reserve 1 for last token
            n = len(seq)
            
            # Создаём последовательность с BOS в конце (для next-token prediction)
            # input:  [t0, t1, ..., t_{n-1}, PAD...PAD]
            # target: [t1, t2, ..., EOS,    PAD...PAD]
            d_seq = [self.BIN_START] + [t[0] for t in seq]  # BOS = BIN_START (index of first bin)
            g_seq = [self.BIN_START] + [t[1] for t in seq]
            c_seq = [self.BIN_START] + [t[2] for t in seq]
            b_seq = [self.BIN_START] + [t[3] for t in seq]
            
            # Pad
            d_seq = d_seq[:self.seq_len]
            g_seq = g_seq[:self.seq_len]
            c_seq = c_seq[:self.seq_len]
            b_seq = b_seq[:self.seq_len]
            
            while len(d_seq) < self.seq_len:
                d_seq.append(0)  # PAD
                g_seq.append(0)
                c_seq.append(0)
                b_seq.append(0)
            
            depths.append(d_seq)
            degrees.append(g_seq)
            comms.append(c_seq)
            btwns.append(b_seq)
            species_list.append(label_species)
        
        if not depths:
            return None
        
        return {
            "depths":  torch.tensor(np.array(depths), dtype=torch.long, device=self.device),
            "degrees": torch.tensor(np.array(degrees), dtype=torch.long, device=self.device),
            "comms":   torch.tensor(np.array(comms), dtype=torch.long, device=self.device),
            "btwns":   torch.tensor(np.array(btwns), dtype=torch.long, device=self.device),
            "species": torch.tensor(species_list, dtype=torch.long, device=self.device),
            "mask":    (torch.tensor(np.array(depths), dtype=torch.long, device=self.device) != 0).float(),
        }


# ═══════════════════════════════════════════════════════════
#  FINE-TUNING ENGINE
# ═══════════════════════════════════════════════════════════

class ControlTokenTrainer:
    """
    Fine-tuning control token embeddings.
    
    Заморожено: все веса Sardna кроме 16 новых species embeddings
    Обучается: только control token embeddings (7..22)
    """
    
    def __init__(self, model_path: str = SARDNA_MODEL, device: str = "cpu",
                 lr: float = 1e-3, data_path: Optional[str] = None):
        self.device = torch.device(device)
        
        # ── Загружаем Sardna ──
        print(f"  📥 Loading Sardna v2 from {model_path}...")
        self.model = SarDNA.load_trained(model_path, config=None, device=device)
        self.model.train()
        self.config = self.model.config
        
        # ── Расширяем species embeddings ──
        self._expand_species()
        
        # ── Замораживаем ВСЁ кроме control embeddings ──
        self._freeze_all_except_control()
        
        # ── Оптимизатор: передаём species_embed.weight целиком,
        #    но biological rows (0..6) получат zero grad после backward ──
        self.optimizer = torch.optim.AdamW(
            [self.control_params],
            lr=lr,
            weight_decay=1e-5,
        )
        
        # ── Загрузка реальных данных ──
        self.loaded_data = None
        if data_path is not None:
            if os.path.exists(data_path):
                print(f"  📂 Loading training data from {data_path}...")
                self.loaded_data = self._load_npz_data(data_path)
                print(f"     {len(self.loaded_data['depths'])} walks loaded")
            else:
                print(f"  ⚠️  Data file not found: {data_path}. Falling back to synthetic data.")
        
        # ── Data generator (fallback if no real data) ──
        self.data_gen = ControlDataGenerator(self.model, device)
        
        # ── История ──
        self.history = {"loss": [], "species": []}
        
        # ── Путь для чекпоинта ──
        self.checkpoint_path = "/tmp/sardna_ft_checkpoint.pt"
        
        # ── Счётчик параметров ──
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  📊 Params: {total:,} total | {trainable:,} trainable | {frozen:,} frozen")
    
    def _expand_species(self):
        """Расширяем species_embed: 7 → 23."""
        old_embed = self.model.embed.species_embed
        old_weight = old_embed.weight.data  # [7, d_model]
        d = self.config.d_model
        
        new_embed = nn.Embedding(SPECIES_TOTAL, d)
        new_embed.weight.data[:SPECIES_BIOLOGICAL] = old_weight
        
        # Инициализируем control embeddings как вариации существующих
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
        print(f"  🔧 Species embedding: 7 → {SPECIES_TOTAL} "
              f"(7 biological + {N_CONTROL} control)")
    
    def _freeze_all_except_control(self):
        """
        Замораживаем все веса, кроме control token части species_embed.
        
        Проблема: weight[7:].requires_grad_(True) НЕ работает в PyTorch,
        потому что срез от Parameter не создаёт отдельный вычислительный граф.
        
        Решение: оставляем весь species_embed.weight requires_grad=True,
        но в оптимизатор передаём только control token часть.
        После backward обнуляем градиенты для biological species (0..6).
        """
        # Замораживаем всё КРОМЕ species_embed
        for name, param in self.model.named_parameters():
            if 'species_embed' in name:
                param.requires_grad = True  # весь weight trainable
            else:
                param.requires_grad = False
        
        # Маска: 1 для control tokens (7..22), 0 для biological (0..6)
        self.control_mask = torch.cat([
            torch.zeros(SPECIES_BIOLOGICAL, 1),
            torch.ones(N_CONTROL, 1),
        ]).to(self.device)
        
        self.control_params = self.model.embed.species_embed.weight
        
        n_ctrl_params = N_CONTROL * self.config.d_model
        print(f"  🔒 Frozen: all weights except {N_CONTROL} control token rows "
              f"({n_ctrl_params} params trainable)")
    
    def _load_npz_data(self, path: str) -> Dict:
        """Load pre-computed training data from .npz file."""
        data = np.load(path)
        return {
            'depths': torch.tensor(data['depths'], dtype=torch.long, device=self.device),
            'degrees': torch.tensor(data['degrees'], dtype=torch.long, device=self.device),
            'comms': torch.tensor(data['comms'], dtype=torch.long, device=self.device),
            'btwns': torch.tensor(data['btwns'], dtype=torch.long, device=self.device),
            'species': torch.tensor(data['species'], dtype=torch.long, device=self.device),
        }
    
    def train_step(self, batch: Dict) -> float:
        """Один шаг обучения."""
        self.optimizer.zero_grad()
        
        depths = batch["depths"]
        degrees = batch["degrees"]
        comms = batch["comms"]
        btwns = batch["btwns"]
        species = batch["species"]
        
        # Forward через Sardna
        logits_d, logits_g, logits_c, logits_b = self.model(
            depths[:, :-1],   # input: all but last
            degrees[:, :-1],
            comms[:, :-1],
            btwns[:, :-1],
            species_ids=species,
        )
        
        # Targets: next token (shifted by 1)
        target_d = depths[:, 1:]
        target_g = degrees[:, 1:]
        target_c = comms[:, 1:]
        target_b = btwns[:, 1:]
        
        # Данные: bins 4..13 (из конвертера) или 0..9 (из Sardna generate)
        # Sardna output heads = 10 classes (0..9).
        # Сдвигаем: bin 4..13 → 0..9; значения 0..9 оставляем как есть.
        # PAD(0)→-100 (игнор)
        def shift_targets(t):
            # Если макс > 9, значит данные в формате 4..13 → shift
            if t.max() > 9:
                return torch.where(t == 0, -100, t - 4)
            else:
                return torch.where(t == 0, -100, t.to(torch.long))
        
        # Cross-entropy loss
        loss_d = F.cross_entropy(
            logits_d.reshape(-1, logits_d.size(-1)),
            shift_targets(target_d).reshape(-1),
            ignore_index=-100,
        )
        loss_g = F.cross_entropy(
            logits_g.reshape(-1, logits_g.size(-1)),
            shift_targets(target_g).reshape(-1),
            ignore_index=-100,
        )
        loss_c = F.cross_entropy(
            logits_c.reshape(-1, logits_c.size(-1)),
            shift_targets(target_c).reshape(-1),
            ignore_index=-100,
        )
        loss_b = F.cross_entropy(
            logits_b.reshape(-1, logits_b.size(-1)),
            shift_targets(target_b).reshape(-1),
            ignore_index=-100,
        )
        
        loss = loss_d + loss_g + loss_c + loss_b
        
        # Backward
        loss.backward()
        
        # Обнуляем градиенты для biological species (0..6)
        # Только control token rows (7..22) сохраняют градиент
        if self.model.embed.species_embed.weight.grad is not None:
            self.model.embed.species_embed.weight.grad *= self.control_mask
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, n_epochs: int = 10, steps_per_epoch: int = 50,
              batch_size: int = 8, verbose: bool = True):
        """Полный цикл fine-tuning."""
        
        print(f"\n{'='*55}")
        print(f"  🎯 FINE-TUNING CONTROL TOKENS")
        if self.loaded_data is not None:
            print(f"  📂 Using REAL FlyWire annotations data")
            print(f"     {len(self.loaded_data['depths'])} walks loaded")
        else:
            print(f"  🔄 Using synthetic data from Sardna")
        print(f"  {n_epochs} epochs × {steps_per_epoch} steps × batch={batch_size}")
        print(f"{'='*55}")
        
        t0 = time.time()
        n_data = len(self.loaded_data['depths']) if self.loaded_data is not None else 10000
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            
            for step in range(steps_per_epoch):
                if self.loaded_data is not None:
                    # Берём батч из загруженных данных
                    batch = self._sample_batch(batch_size, seed=42 + epoch * 1000 + step)
                else:
                    # Генерируем синтетические данные
                    batch = self.data_gen.create_batch(
                        batch_size=batch_size,
                        seed=42 + epoch * 1000 + step,
                    )
                
                if batch is None:
                    continue
                
                loss = self.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / steps_per_epoch
            self.history["loss"].append(avg_loss)
            
            if verbose:
                improvement = ""
                if len(self.history["loss"]) > 1:
                    prev = self.history["loss"][-2]
                    imp = (prev - avg_loss) / prev * 100
                    improvement = f" ({imp:+.1f}%)"
                
                print(f"  Epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f}{improvement}")
        
        # ── Сохраняем чекпоинт ──
        # config не сохраняем — он содержит custom class, ломает weights_only=True
        torch.save({
            'embed.species_embed.weight': self.model.embed.species_embed.weight.data.cpu(),
            'history': self.history,
            'epoch': n_epochs,
        }, self.checkpoint_path)
        
        t = time.time() - t0
        print(f"\n  ⏱  Training: {t:.0f}s ({t/60:.1f}min)")
        print(f"  💾 Checkpoint saved: {self.checkpoint_path}")
        print(f"  🎯 Control embeddings trained for {n_epochs} epochs!")
        
        return self.history
    
    def _sample_batch(self, batch_size: int, seed: int) -> Dict:
        """Sample a batch from pre-loaded data."""
        if self.loaded_data is None:
            return None
        
        rng = np.random.RandomState(seed)
        n = len(self.loaded_data['depths'])
        indices = rng.choice(n, min(batch_size, n), replace=False)
        
        return {
            'depths': self.loaded_data['depths'][indices],
            'degrees': self.loaded_data['degrees'][indices],
            'comms': self.loaded_data['comms'][indices],
            'btwns': self.loaded_data['btwns'][indices],
            'species': self.loaded_data['species'][indices],
        }
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Загрузка обученных control embeddings из чекпоинта."""
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"     Running with randomly initialized control embeddings!")
            return
        
        # PyTorch 2.6+ требует weights_only=True, но старые чекпоинты
        # содержат custom Config объект. Пробуем safe load сначала.
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            print(f"  ⚠️  Safe load failed (likely custom Config object in checkpoint). "
                  f"Trying unsafe load...")
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"  ❌ Cannot load checkpoint: {e}")
                print(f"     Running with randomly initialized control embeddings!")
                return
        
        if 'embed.species_embed.weight' in ckpt:
            saved_weight = ckpt['embed.species_embed.weight']
            if saved_weight.shape == self.model.embed.species_embed.weight.shape:
                self.model.embed.species_embed.weight.data = saved_weight.to(self.device)
                print(f"  ✅ Loaded trained control embeddings from {checkpoint_path}")
                if 'history' in ckpt and 'loss' in ckpt['history']:
                    losses = ckpt['history']['loss']
                    print(f"     Trained for {ckpt.get('epoch', '?')} epochs, "
                          f"final loss={losses[-1]:.4f}")
            else:
                print(f"  ⚠️  Shape mismatch: checkpoint {saved_weight.shape} vs "
                      f"model {self.model.embed.species_embed.weight.shape}. "
                      f"Running with random init.")
        else:
            print(f"  ⚠️  No 'embed.species_embed.weight' in checkpoint. "
                  f"Running with random init.")
    
    def verify(self, n_neurons: int = 10, seed: int = 42):
        """
        Проверяем что control tokens дают РАЗНЫЕ результаты.
        
        Sardna output head = 10 классов (0..9). Нормализация: (x - 0) / 9
        """
        self.model.eval()
        
        print(f"\n{'='*55}")
        print(f"  🔬 VERIFICATION: {len(CONTROL_TOKENS)} control tokens")
        print(f"{'='*55}")
        
        print(f"\n  {'Control Token':<22} | {'Depth':>6} | {'Degree':>7} | "
              f"{'Comm':>5} | {'Btwn':>5} | {'Target':>20}")
        print(f"  {'─'*22}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*20}")
        
        results = {}
        for token_name, token_id in sorted(CONTROL_TOKENS.items(), 
                                           key=lambda x: x[1]):
            species_id = SPECIES_BIOLOGICAL + token_id
            
            neurons = self.data_gen.generate_sequence(species_id, seed=seed)
            if len(neurons) < 3:
                print(f"  [{token_name:<20}] | {'N/A':>6}")
                continue
            
            states = np.array(neurons, dtype=np.float32)
            # Нормализация 0..9 → 0..1
            states = (states - 0.0) / 9.0
            mean = states.mean(axis=0)
            
            target = CONTROL_PROFILES[token_name]
            
            results[token_name] = mean
            
            print(f"  [{token_name:<20}] | {mean[0]:6.3f} | {mean[1]:7.3f} | "
                  f"{mean[2]:5.3f} | {mean[3]:5.3f} | "
                  f"d={target[0]:.2f} g={target[1]:.2f}")
        
        # ── Оценка дифференциации ──
        print(f"\n  📊 Differentiation Analysis:")
        
        # Считаем variance между контрольными токенами
        means = np.array([v for v in results.values()])
        if len(means) > 1:
            var_between = means.var(axis=0).mean()
            var_within = 0
            
            print(f"    Between-token variance: {var_between:.6f}")
            
            if var_between > 0.001:
                print(f"    ✅ Control tokens are DIFFERENTIATED! (var={var_between:.6f})")
            else:
                print(f"    ❌ Control tokens are NOT differentiated yet "
                      f"(var={var_between:.6f})")
                print(f"    → Need more epochs or higher learning rate")
        
        # ── Hit rate: какой % токенов попал в target профиль ──
        hits = 0
        total = 0
        for token_name, mean_vec in results.items():
            target = np.array(CONTROL_PROFILES[token_name])
            dist = np.sqrt(((mean_vec - target)**2).mean())
            if dist < 0.25:
                hits += 1
            total += 1
            print(f"    [{token_name:<20}] dist to target: {dist:.4f}")
        
        if total > 0:
            print(f"\n    Hit rate (<0.25 distance): {hits}/{total} ({hits/total*100:.0f}%)")
        
        return results


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Fine-tune Control Tokens for Sardna v2"
    )
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=50,
                       help="Steps per epoch")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for control embeddings")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to .npz training data (from convert_annotations_to_sardna.py)")
    parser.add_argument("--verify", action="store_true",
                       help="Run verification only (skip training)")
    parser.add_argument("--checkpoint", type=str, default="/tmp/sardna_ft_checkpoint.pt",
                       help="Path to fine-tuned checkpoint for verification")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or cuda)")
    args = parser.parse_args()
    
    # ── Init trainer ──
    trainer = ControlTokenTrainer(lr=args.lr, device=args.device, data_path=args.data)
    
    # ── Verify only ──
    if args.verify:
        trainer._load_checkpoint(args.checkpoint)
        trainer.verify()
        return
    
    # ── Train ──
    trainer.train(
        n_epochs=args.epochs,
        steps_per_epoch=args.steps,
        batch_size=args.batch,
    )
    
    # ── Verify after training ──
    trainer.verify()
    
    print(f"\n{'='*55}")
    print(f"  ✅ Fine-tuning complete!")
    print(f"  Run `python scripts/connectome_gpt_api.py --all` to test")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
