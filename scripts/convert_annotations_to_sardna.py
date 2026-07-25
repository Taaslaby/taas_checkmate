#!/usr/bin/env python3
"""
🔄 Convert FlyWire Annotations → Sardna Training Data

Берёт FlyWire neuron annotations (139K нейронов, 31 колонка):
  root_id, super_class, cell_type, top_nt, flow, hemilineage, ...

И конвертирует в Sardna training format:
  Последовательности (depth, degree, community, betweenness)
  + control token labels (gen_visual, gen_motor, etc.)

Логика маппинга:
  super_class → control token
    optic/lobula/medulla/retina → gen_visual
    motor → gen_motor / region:motor
    sensory_... → gen_... (конкретный тип)
    intrinsic/central → gen_attention (внутренняя обработка)
    
  flow → depth (afferent=high, efferent=low, intrinsic=mid)
  hemilineage → community (эмбриональное происхождение)
  cell_type → degree (приближение)

Usage:
  python scripts/convert_annotations_to_sardna.py
  python scripts/convert_annotations_to_sardna.py --analyze-only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import json
import time

# ─── Пути ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ANNOT_PATH = BASE_DIR / "data" / "connectomes" / "flywire" / "annotations" / "flywire_neuron_annotations.tsv"
GIT_ANNOT_PATH = BASE_DIR / "data" / "connectomes" / "flywire" / "annotations_git" / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
OUTPUT_DIR = BASE_DIR / "data" / "sardna_training"

# ═══════════════════════════════════════════════════════════
#  CONTROL TOKEN CONSTANTS (duplicated from finetune_control_tokens.py)
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

# super_class → control token
SUPERCLASS_TO_TOKEN = {
    "optic": "gen_visual",
    "medulla": "gen_visual",
    "lobula": "gen_visual",
    "lobula_plate": "gen_visual",
    "retina": "gen_visual",
    "lamina": "gen_visual",
    
    "motor": "gen_motor",
    
    "sensory_olfactory": "gen_olfactory",
    "sensory_auditory": "gen_auditory",
    "sensory_thermo": "gen_thermo",
    "sensory_humidity": "gen_humidity",
    "sensory": "gen_sensory",
    
    "central": "gen_attention",
    "intrinsic": "gen_attention",
    "intrinsic_mushroom_body": "gen_memory",
    
    "descending": "gen_motor",
    "ascending": "gen_sensory",
    "endocrine": "gen_modulator",
    "glial": "gen_glia",
}

# flow → depth (нормализовано 0..1)
FLOW_TO_DEPTH = {
    "afferent": 0.85,     # вход → высокая depth
    "efferent": 0.15,     # выход → низкая depth
    "intrinsic": 0.50,    # центр → средняя depth
}

# top_nt → community (нормализовано 0..1)
NT_TO_COMMUNITY = {
    "acetylcholine": 0.20,   # возбуждающий, сенсорные пути
    "glutamate": 0.40,       # возбуждающий, моторные пути
    "gaba": 0.60,            # тормозной, центральная обработка
    "octopamine": 0.70,      # нейромодулятор
    "dopamine": 0.75,        # нейромодулятор
    "serotonin": 0.80,       # нейромодулятор
    "nan": 0.50,             # неизвестно
}

# hemilineage → betweenness (приближение)
HEMILINEAGE_TO_BTWN = {
    "VPN": 0.70,    # ventral projection neurons — хабы
    "DCN": 0.60,    # dorsal cluster neurons
    "LCN": 0.55,    # lateral cluster
    "MCN": 0.50,    # medial cluster
}

# ═══════════════════════════════════════════════════════════
#  MAIN CONVERTER
# ═══════════════════════════════════════════════════════════

class AnnotationConverter:
    """
    Конвертирует FlyWire аннотации в Sardna training data.
    
    Каждый нейрон → (depth_bin, degree_bin, community_bin, btwn_bin) + control_token
    """
    
    # Token layout (same as Sardna)
    NUM_BINS = 10
    BIN_START = 4
    SPECIAL = 4
    
    def __init__(self, annot_path: str = None):
        self.annot_path = annot_path or str(ANNOT_PATH)
        if not os.path.exists(self.annot_path):
            self.annot_path = str(GIT_ANNOT_PATH)
        
        print(f"  📥 Loading annotations from {self.annot_path}")
        self.df = pd.read_csv(self.annot_path, sep='\t', low_memory=False)
        print(f"  ✅ {len(self.df):,} neurons loaded")
        print(f"  📊 Columns: {list(self.df.columns[:15])}")
        
        # Маппинги для быстрого доступа
        self.token_id_map = self._build_token_map()
        self.superclass_counts = self.df['super_class'].value_counts()
    
    def _build_token_map(self) -> Dict:
        """Build control token mapping from available data."""
        # Собираем все уникальные super_class
        tokens = {}
        for sc in self.df['super_class'].unique():
            if isinstance(sc, str):
                tokens[sc] = SUPERCLASS_TO_TOKEN.get(sc, "gen_random")
        return tokens
    
    def neuron_to_features(self, row: pd.Series) -> Tuple[np.ndarray, str]:
        """
        Конвертирует одну строку аннотации в (features, control_token).
        
        features = [depth, degree, community, betweenness] — float 0..1
        """
        # ── depth: из flow ──
        flow = row.get('flow', '')
        if isinstance(flow, str) and flow in FLOW_TO_DEPTH:
            depth = FLOW_TO_DEPTH[flow]
        else:
            depth = 0.50  # default
        
        # ── degree: из cell_type — приближение ──
        # Чем более специфичный тип, тем больше связей
        ct = row.get('cell_type', '')
        if isinstance(ct, str) and ct != 'nan' and ct != '':
            degree = min(1.0, 0.3 + len(ct) * 0.01)  # длина названия → degree
        else:
            degree = 0.30  # default
        
        # ── community: из top_nt / super_class ──
        nt = row.get('top_nt', '')
        if isinstance(nt, str) and nt in NT_TO_COMMUNITY:
            community = NT_TO_COMMUNITY[nt]
        else:
            # Fallback: super_class
            sc = row.get('super_class', '')
            if isinstance(sc, str):
                if 'sensory' in sc or 'optic' in sc or 'retina' in sc:
                    community = 0.20
                elif 'motor' in sc or 'descending' in sc:
                    community = 0.70
                else:
                    community = 0.50
            else:
                community = 0.50
        
        # ── betweenness: из hemilineage / side ──
        hl = row.get('ito_lee_hemilineage', '')
        if isinstance(hl, str) and 'VPN' in hl:
            btwn = 0.70
        elif isinstance(hl, str) and 'DCN' in hl:
            btwn = 0.60
        elif isinstance(hl, str) and 'LCN' in hl:
            btwn = 0.55
        elif isinstance(hl, str) and 'MCN' in hl:
            btwn = 0.50
        else:
            # Определяем по side
            side = row.get('side', '')
            if isinstance(side, str) and side in ('L', 'R'):
                btwn = 0.45  # лево/право
            else:
                btwn = 0.40  # центр
        
        features = np.array([depth, degree, community, btwn], dtype=np.float32)
        
        # ── control token ──
        sc = row.get('super_class', '')
        if isinstance(sc, str):
            control_token = SUPERCLASS_TO_TOKEN.get(sc, "gen_random")
        else:
            control_token = "gen_random"
        
        return features, control_token
    
    def discretize(self, value: float) -> int:
        """Float 0..1 → bin index (BIN_START..BIN_START+NUM_BINS-1)."""
        idx = min(self.NUM_BINS - 1, max(0, int(value * self.NUM_BINS)))
        return self.BIN_START + idx
    
    def convert_all(self, max_neurons: int = None) -> Dict:
        """
        Конвертировать все нейроны в Sardna-формат.
        
        Returns:
          dict с:
            'samples': [(features, token_name), ...]
            'walks': сгенерированные последовательности
            'token_counts': сколько нейронов на каждый control token
        """
        samples = []
        token_counts = defaultdict(int)
        
        df = self.df
        if max_neurons:
            df = df.head(max_neurons)
        
        print(f"\n  🔄 Converting {len(df):,} neurons...")
        t0 = time.time()
        
        for idx, (_, row) in enumerate(df.iterrows()):
            features, token = self.neuron_to_features(row)
            samples.append((features, token))
            token_counts[token] += 1
            
            if (idx + 1) % 20000 == 0:
                dt = time.time() - t0
                print(f"    {idx+1}/{len(df)} ({dt:.0f}s)")
        
        dt = time.time() - t0
        print(f"  ✅ Converted {len(samples)} samples in {dt:.0f}s")
        
        return {
            'samples': samples,
            'token_counts': dict(token_counts),
        }
    
    def generate_walks(
        self, 
        samples: List[Tuple[np.ndarray, str]],
        seq_len: int = 16,
        walks_per_token: int = 100,
    ) -> Dict:
        """
        Генерирует training walks из конвертированных samples.
        
        Каждый walk: последовательность нейронов одного control token,
        оформленная как Sardna training sample.
        """
        # Группируем по control token
        token_groups = defaultdict(list)
        for features, token in samples:
            token_groups[token].append(features)
        
        walks = {
            'depths': [],
            'degrees': [],
            'comms': [],
            'btwns': [],
            'species': [],
        }
        
        rng = np.random.RandomState(42)
        
        for token_name, group in token_groups.items():
            if len(group) < 3:
                continue
            
            # Маппим token_name → species_id (control token)
            if token_name in CONTROL_TOKENS:
                species_id = SPECIES_BIOLOGICAL + CONTROL_TOKENS[token_name]
            else:
                continue
            
            n_walks = min(walks_per_token, len(group) * 10)
            
            for _ in range(n_walks):
                # Случайная подпоследовательность
                if len(group) < seq_len:
                    indices = rng.choice(len(group), seq_len, replace=True)
                else:
                    start = rng.randint(0, len(group) - seq_len)
                    indices = range(start, start + seq_len)
                
                seq = [group[i] for i in indices]
                
                depth_seq = [self.discretize(f[0]) for f in seq]
                degree_seq = [self.discretize(f[1]) for f in seq]
                comm_seq = [self.discretize(f[2]) for f in seq]
                btwn_seq = [self.discretize(f[3]) for f in seq]
                
                walks['depths'].append(depth_seq)
                walks['degrees'].append(degree_seq)
                walks['comms'].append(comm_seq)
                walks['btwns'].append(btwn_seq)
                walks['species'].append(species_id)
        
        print(f"\n  🚶 Generated {len(walks['depths'])} walks across "
              f"{len(set(walks['species']))} control tokens")
        
        return walks
    
    def save_training_data(self, walks: Dict, prefix: str = "flywire_v1"):
        """Сохраняет в Sardna-формат."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        path = OUTPUT_DIR / f"{prefix}_walks.npz"
        np.savez_compressed(
            str(path),
            depths=np.array(walks['depths'], dtype=np.int32),
            degrees=np.array(walks['degrees'], dtype=np.int32),
            comms=np.array(walks['comms'], dtype=np.int32),
            btwns=np.array(walks['btwns'], dtype=np.int32),
            species=np.array(walks['species'], dtype=np.int32),
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  💾 Saved: {path} ({size_mb:.1f} MB)")
        
        # Также сохраняем метаданные
        meta = {
            'prefix': prefix,
            'n_walks': len(walks['depths']),
            'seq_len': len(walks['depths'][0]) if walks['depths'] else 0,
            'n_control_tokens': len(set(walks['species'])),
            'source': 'FlyWire annotations',
            'n_neurons': len(self.df),
            'superclass_distribution': self.superclass_counts.head(20).to_dict(),
        }
        
        meta_path = OUTPUT_DIR / f"{prefix}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  💾 Saved: {meta_path}")
        
        return path
    
    def analyze(self):
        """Анализ распределения аннотаций."""
        print(f"\n{'='*60}")
        print(f"  📊 ANNOTATIONS ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\n  Total neurons: {len(self.df):,}")
        
        # super_class
        print(f"\n  super_class distribution:")
        for sc, count in self.superclass_counts.head(15).items():
            pct = count / len(self.df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {sc:<25} {count:6d} ({pct:5.1f}%) {bar}")
        
        # flow
        print(f"\n  flow distribution:")
        if 'flow' in self.df.columns:
            flow_counts = self.df['flow'].value_counts()
            for flow, count in flow_counts.items():
                if isinstance(flow, str):
                    pct = count / len(self.df) * 100
                    print(f"    {flow:<15} {count:6d} ({pct:5.1f}%)")
        
        # top_nt
        print(f"\n  top_nt (neurotransmitters):")
        if 'top_nt' in self.df.columns:
            nt_counts = self.df['top_nt'].value_counts()
            for nt, count in nt_counts.head(10).items():
                pct = count / len(self.df) * 100
                print(f"    {nt:<15} {count:6d} ({pct:5.1f}%)")
        
        # Контрольные токены
        print(f"\n  Control token distribution (predicted):")
        token_counts = defaultdict(int)
        for _, row in self.df.iterrows():
            _, token = self.neuron_to_features(row)
            token_counts[token] += 1
        
        for token, count in sorted(token_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {token:<20} {count:6d} ({pct:5.1f}%) {bar}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔄 Convert FlyWire annotations → Sardna training data"
    )
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only show analysis, don't convert")
    parser.add_argument("--max-neurons", type=int, default=None,
                       help="Max neurons to process (for testing)")
    parser.add_argument("--walks-per-token", type=int, default=200,
                       help="Number of walks per control token")
    parser.add_argument("--seq-len", type=int, default=16,
                       help="Walk sequence length")
    parser.add_argument("--prefix", type=str, default="flywire_v1",
                       help="Output file prefix")
    args = parser.parse_args()
    
    converter = AnnotationConverter()
    
    if args.analyze_only:
        converter.analyze()
        return
    
    # ── 1. Convert ──
    print(f"\n{'='*60}")
    print(f"  🔄 CONVERTING ANNOTATIONS → SARDNA FORMAT")
    print(f"{'='*60}")
    
    result = converter.convert_all(max_neurons=args.max_neurons)
    
    print(f"\n  📊 Token distribution:")
    for token, count in sorted(result['token_counts'].items(), 
                                key=lambda x: -x[1]):
        print(f"    {token:<20} {count:5d} neurons")
    
    # ── 2. Generate walks ──
    print(f"\n{'='*60}")
    print(f"  🚶 GENERATING TRAINING WALKS")
    print(f"{'='*60}")
    
    walks = converter.generate_walks(
        samples=result['samples'],
        seq_len=args.seq_len,
        walks_per_token=args.walks_per_token,
    )
    
    # ── 3. Save ──
    print(f"\n{'='*60}")
    print(f"  💾 SAVING TRAINING DATA")
    print(f"{'='*60}")
    
    converter.save_training_data(walks, prefix=args.prefix)
    
    print(f"\n{'='*60}")
    print(f"  ✅ Conversion complete!")
    print(f"  📍 Training data: {OUTPUT_DIR}/")
    print(f"  🏃 Next: run finetune_control_tokens.py with --data {OUTPUT_DIR}/{args.prefix}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
