#!/usr/bin/env python3
"""
FlyWire Izhikevich Brain — полная симуляция коннектома 138K нейронов.

Загружает реальный FlyWire коннектом (connections_princeton.csv.gz) в CSR формат,
запускает Izhikevich/LIF/Rate/ψ + CSR propagate через Rust,
извлекает спайки descending нейронов и подаёт rates в VNC контроллер.

Архитектура:
  FlyWire CSR → Izhikevich/LIF/Rate/ψ → DN spikes → rate [0,1] → VNC

Модели:
  --model izhi    Izhikevich (биологичный, ~0.02× RT на CPU)
  --model lif     LIF (Shiu et al. 2024, ~0.01× RT на CPU)
  --model rate    Rate-based ANN (~100× RT)
  --model psi     ψ-wave (экспериментальная)

Запуск:
  python3 fly/scripts/vnc_walk.py --model izhi --duration 30

Зависимости:
  pip install numpy pandas pyarrow scipy
  + cargo build --features python
"""
import sys
import os
import time
import math
import csv
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np

FLYWIRE_DATA = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))
DLOAD = FLYWIRE_DATA
HERE = Path(__file__).parent
PROJ = HERE.parent.parent

# Сначала пробуем release (с CUDA), затем debug как fallback
if str(PROJ / 'target' / 'release') not in sys.path:
    sys.path.insert(0, str(PROJ / 'target' / 'release'))
if str(PROJ / 'target' / 'debug') not in sys.path:
    sys.path.append(str(PROJ / 'target' / 'debug'))
try:
    import taas_checkmate as tcm
except ImportError:
    print("⚠ taas_checkmate не найден. Собери: cd taas_checkmate && cargo build --features python")


# ═══════════════════════════════════════════════════════════════
#  Vision looming sensor
# ═══════════════════════════════════════════════════════════════

VISUAL_TYPE_PATTERNS = {
    'Tm':    0.5,
    'TmY':   0.4,
    'Mi':    0.3,
    'Mi1':   0.3,
    'Mi2':   0.3,
    'Mi3':   0.3,
    'Mi4':   0.3,
    'Mi9':   0.3,
    'L1':    0.2,
    'L2':    0.2,
    'L3':    0.2,
    'L4':    0.2,
    'L5':    0.2,
    'T4':    0.8,
    'T5':    0.8,
    'LC':    0.7,
    'LPTC':  0.7,
    'Retinula': 0.1,
    'lamina': 0.1,
    'medulla': 0.1,
    'lobula': 0.1,
}

# DN type patterns (descending neuron prefixes)
DN_TYPE_PATTERNS = ('DN', 'MDN', 'APDN', 'tdDN')


class LoomingSensor:
    """Virtual walls + looming = exp(-distance/scale)."""

    def __init__(self):
        self.x_min = -0.4
        self.x_max = 0.4
        self.y_min = -0.3
        self.y_max = 0.3
        self.scale = 0.12
        self.looming = 0.0
        self.wall_dir = 0.0
        self.nearest_wall_dist = 1.0
        self.nearest_wall_name = ""

    def compute(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        dists = [
            (x - self.x_min, 'right', math.pi/2),
            (self.x_max - x, 'left', -math.pi/2),
            (y - self.y_min, 'front', 0.0),
            (self.y_max - y, 'back', math.pi),
        ]
        min_idx = min(range(4), key=lambda i: dists[i][0])
        min_dist, self.nearest_wall_name, wall_angle = dists[min_idx]
        self.nearest_wall_dist = min_dist

        looming_dist = math.exp(-min_dist / self.scale) if min_dist > 0 else 1.0

        speed = math.sqrt(vx*vx + vy*vy)
        vel_factor = 1.0
        if speed > 0.001:
            heading = math.atan2(vy, vx)
            angle_diff = abs(heading - wall_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            vel_factor = max(0.0, 1.0 - angle_diff / (math.pi * 0.5))
            vel_factor = vel_factor * vel_factor

        self.looming = min(1.0, looming_dist * (0.5 + 0.5 * vel_factor))


# ═══════════════════════════════════════════════════════════════
#  FlyWire CSR — загрузка коннектома и построение CSR
# ═══════════════════════════════════════════════════════════════

class FlyWireCSR:
    """FlyWire connectome — CSR matrix builder from CSV data.

    Поддерживает два формата:
      V1 (старый): connections_princeton.csv.gz + consolidated_cell_types.csv.gz
      V2 (новый):  connections_princeton(5).csv.gz + neurons(3).csv.gz

    V2 включает VNC нейроны (abdomen, leg, wing) с neuropil и Body Part.
    V2 использует Super Class из neurons файла для нахождения ДН/МН.

    Построение:
      - CSR matrix (row_ptr, col_indices, values)
      - DN, MN, VNC, abdomen, leg, wing индексы
      - Visual afferent индексы
      - root_id → CSR index
    """

    def __init__(self):
        self.n_neurons = 0
        self.n_synapses = 0
        self.row_ptr = None
        self.col_indices = None
        self.values = None
        self.dn_indices = []
        self.dn_root_ids = []
        self.mn_indices = []      # motor neuron indices (Super Class == 'motor')
        self.mn_root_ids = []
        self.visual_afferent_indices = []
        self.root_to_idx = {}
        self.nt_types = None
        self.type_map = {}  # root_id → cell type name (old format) or Primary Cell Type (new)

        # V2 new fields
        self.is_v2 = False
        self.body_part_map = {}        # root_id → body part string
        self.super_class_map = {}      # root_id → super class
        self.flow_map = {}             # root_id → flow (afferent/efferent/intrinsic)
        self.abdomen_indices = []      # abdomen neuron indices (Body Part == 'abdomen')
        self.leg_indices = []          # leg neuron indices
        self.wing_indices = []         # wing neuron indices
        self.vnc_indices = []          # all VNC intrinsic neurons
        self.ascending_indices = []    # ascending neuron indices
        self.neuropil_map = {}         # root_id → primary neuropil region

        # Named DN lookup: {name: (root_id, csr_idx)}
        self.named_dns = {}

    def _find_connections_file(self):
        """Find connections file, preferring new V2 format."""
        candidates = [
            ('connections_princeton(5).csv.gz', True),   # V2: with neuropil!
            ('connections_princeton.csv.gz', False),      # V1: old format
            ('connections_princeton(4).csv.gz', False),
            ('connections_princeton(3).csv.gz', False),
        ]
        for name, is_v2 in candidates:
            p = DLOAD / name
            if p.exists():
                return p, is_v2
        return None, False

    def _load_neurons_v2(self, t0):
        """Load neurons(3).csv.gz — Body Part, NT type, Super Class."""
        neurons_path = DLOAD / "neurons(3).csv.gz"
        if not neurons_path.exists():
            for alt in ['neurons.csv.gz', 'neurons(2).csv.gz']:
                p = DLOAD / alt
                if p.exists():
                    neurons_path = p
                    break
        if not neurons_path.exists():
            return False

        print("  [CSR] Loading neurons metadata (Body Part, NT, Super Class)...")
        n_loaded = 0
        n_bp = 0
        with gzip.open(str(neurons_path), 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid_str = row.get('Root ID', '').strip()
                if not rid_str:
                    continue
                try:
                    rid = int(rid_str)
                except ValueError:
                    continue
                n_loaded += 1

                # Body Part
                bp = row.get('Body Part', '').strip()
                if bp:
                    self.body_part_map[rid] = bp
                    n_bp += 1

                # Super Class (descending, motor, ventral_nerve_cord_intrinsic, etc.)
                sc = row.get('Super Class', '').strip()
                if sc:
                    self.super_class_map[rid] = sc

                # Flow (afferent, efferent, intrinsic)
                fl = row.get('Flow', '').strip()
                if fl:
                    self.flow_map[rid] = fl

                # Primary Cell Type → type_map
                pct = row.get('Primary Cell Type', '').strip()
                if pct:
                    self.type_map[rid] = pct

                # Predicted NT type (ACH, GABA, GLUT, DA, etc.)
                nt = row.get('Predicted NT type', '').strip().upper()
                if nt and nt not in ('', '?'):
                    # Store for use after CSR is built (step 3)
                    if not hasattr(self, '_nt_map_v2'):
                        self._nt_map_v2 = {}
                    if nt == 'GABA':
                        self._nt_map_v2[rid] = -1.0
                    elif nt in ('ACH', 'GLUT', 'DA', 'SER', 'HIST', 'OCT', 'TYR'):
                        self._nt_map_v2[rid] = 1.0

        print(f"    {n_loaded} neurons loaded, {n_bp} with Body Part "
              f"[{time.time()-t0:.1f}s]")
        return True

    def load(self):
        t0 = time.time()

        # ─── 0. Find connections file ───────────────────
        conn_path, self.is_v2 = self._find_connections_file()
        if conn_path is None:
            print(f"  [CSR] ⚠ Файл connections не найден в {DLOAD}")
            return

        # ─── 0b. Load neurons V2 metadata (if new format) ───
        if self.is_v2:
            self._load_neurons_v2(t0)

        # ─── 1. Cell types (V1 fallback) ────────────────
        if not self.type_map:
            types_path = DLOAD / "consolidated_cell_types.csv.gz"
            if types_path.exists():
                print("  [CSR] Loading cell types...")
                with gzip.open(str(types_path), 'rt') as f:
                    r = csv.reader(f)
                    next(r)
                    for row in r:
                        if len(row) >= 2:
                            self.type_map[int(row[0])] = row[1]
                print(f"    {len(self.type_map)} typed neurons [{time.time()-t0:.1f}s]")

        # ─── 2. Connections ─────────────────────────────
        print(f"  [CSR] Loading {conn_path.name}...")

        # First pass: count pre-synaptic partners + NT type votes
        pre_count = defaultdict(int)
        pre_nt_votes = defaultdict(lambda: defaultdict(int))
        total_rows = 0
        conn_len = 0
        with gzip.open(str(conn_path), 'rt') as f:
            r = csv.reader(f)
            try:
                header = next(r)
            except StopIteration:
                header = []
            conn_len = len(header)
            for row in r:
                if len(row) >= min(5, conn_len):
                    pre = int(row[0])
                    pre_count[pre] += 1
                    if len(row) >= 5:
                        pre_nt_votes[pre][row[4].strip().upper()] += 1
                    total_rows += 1

        # Build neuron index (all unique pre-synaptic neurons)
        all_ids = sorted(set(pre_count.keys()))
        self.root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
        self.n_neurons = len(all_ids)
        print(f"    {self.n_neurons:,} neurons, {total_rows:,} raw synapses "
              f"[{time.time()-t0:.1f}s]")

        # Pre-allocate
        self.n_synapses = total_rows

        # Second pass: build CSR
        print("  [CSR] Building CSR matrix...")
        row_ptr = np.zeros(self.n_neurons + 1, dtype=np.int32)
        for rid, cnt in pre_count.items():
            idx = self.root_to_idx[rid]
            row_ptr[idx + 1] = cnt

        row_ptr = np.cumsum(row_ptr, dtype=np.int32)

        col_indices = np.zeros(self.n_synapses, dtype=np.int32)
        values = np.zeros(self.n_synapses, dtype=np.float32)
        # neuropil_collected = {} if has_neuropil else None  # root_id → set of neuropils
        cursor = np.zeros(self.n_neurons, dtype=np.int32)

        with gzip.open(str(conn_path), 'rt') as f:
            r = csv.reader(f)
            if conn_len > 0:
                next(r)
            for row in r:
                if len(row) >= min(3, conn_len):
                    pre = int(row[0])
                    post = int(row[1])
                    weight_col = 3 if conn_len >= 5 else (2 if conn_len >= 3 else None)
                    if weight_col is None:
                        continue
                    weight = float(row[weight_col])
                    pre_idx = self.root_to_idx.get(pre)
                    if pre_idx is None:
                        continue
                    post_idx = self.root_to_idx.get(post)
                    if post_idx is None:
                        continue
                    pos = row_ptr[pre_idx] + cursor[pre_idx]
                    if pos >= len(col_indices):
                        continue
                    col_indices[pos] = np.int32(post_idx)
                    values[pos] = max(weight, 0.0)
                    cursor[pre_idx] += 1

        # Trim to actual synapses + rebuild row_ptr compactly
        actual_n = int(np.sum(cursor))
        if actual_n < self.n_synapses:
            # Pack synapses contiguously per neuron (remove gaps from skipped post_idx)
            new_col_indices = np.empty(actual_n, dtype=np.int32)
            new_values = np.empty(actual_n, dtype=np.float32)
            new_row_ptr = np.zeros(self.n_neurons + 1, dtype=np.int32)
            dst = 0
            for i in range(self.n_neurons):
                cnt = int(cursor[i])
                if cnt > 0:
                    src_start = int(row_ptr[i])
                    new_col_indices[dst:dst+cnt] = col_indices[src_start:src_start+cnt]
                    new_values[dst:dst+cnt] = values[src_start:src_start+cnt]
                new_row_ptr[i] = dst
                dst += cnt
            new_row_ptr[-1] = dst
            col_indices = new_col_indices
            values = new_values
            row_ptr = new_row_ptr
            self.n_synapses = actual_n

        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.values = values
        print(f"    CSR built: {self.n_neurons:,} n, {self.n_synapses:,} syn "
              f"[{time.time()-t0:.1f}s]")

        # ─── 3. Per-neuron NT types ───────────────────
        if self.is_v2 and hasattr(self, '_nt_map_v2') and self._nt_map_v2:
            # V2: NT type из neurons файла (загружен в _load_neurons_v2)
            print("  [CSR] Setting NT types from neurons metadata...")
            nt_sign = np.ones(self.n_neurons, dtype=np.float32)
            for rid, sign in self._nt_map_v2.items():
                idx = self.root_to_idx.get(rid)
                if idx is not None:
                    nt_sign[idx] = sign
            self.nt_types = nt_sign
            # Clean up temp storage
            del self._nt_map_v2
        else:
            # V1: majority vote from connections
            print("  [CSR] Computing per-neuron NT types (majority vote)...")
            nt_sign = np.ones(self.n_neurons, dtype=np.float32)
            for rid, votes in pre_nt_votes.items():
                idx = self.root_to_idx.get(rid)
                if idx is None:
                    continue
                best_nt = max(votes.items(), key=lambda x: x[1])[0]
                if 'GABA' in best_nt or 'GLY' in best_nt:
                    nt_sign[idx] = -1.0
            self.nt_types = nt_sign

        exc = int(np.sum(self.nt_types > 0))
        inh = int(np.sum(self.nt_types < 0))
        print(f"    NT: {exc} exc, {inh} inh [{time.time()-t0:.1f}s]")

        # ─── 4. DN indices ──────────────────────────────
        print("  [CSR] Finding descending neurons (DNs)...")
        self.dn_indices = []
        self.dn_root_ids = []

        if self.is_v2:
            # V2: descending from Super Class
            for rid, sc in self.super_class_map.items():
                if sc == 'descending':
                    idx = self.root_to_idx.get(rid)
                    if idx is not None:
                        self.dn_indices.append(idx)
                        self.dn_root_ids.append(rid)
            # Also check type_map for DN* patterns (catch any missed)
            for rid, idx in sorted(self.root_to_idx.items(), key=lambda x: x[1]):
                typ = self.type_map.get(rid, '')
                if typ.startswith(DN_TYPE_PATTERNS) and rid not in self.dn_root_ids:
                    self.dn_indices.append(idx)
                    self.dn_root_ids.append(rid)
        else:
            # V1: from type_map
            for rid, idx in sorted(self.root_to_idx.items(), key=lambda x: x[1]):
                typ = self.type_map.get(rid, '')
                if typ.startswith(DN_TYPE_PATTERNS):
                    self.dn_indices.append(idx)
                    self.dn_root_ids.append(rid)
        print(f"    {len(self.dn_indices)} DNs [{time.time()-t0:.1f}s]")

        # ─── 4a. Motor neuron (MN) indices (V2 only) ───
        if self.is_v2:
            print("  [CSR] Finding motor neurons (MNs)...")
            self.mn_indices = []
            self.mn_root_ids = []
            for rid, sc in self.super_class_map.items():
                if sc == 'motor':
                    idx = self.root_to_idx.get(rid)
                    if idx is not None:
                        self.mn_indices.append(idx)
                        self.mn_root_ids.append(rid)
            print(f"    {len(self.mn_indices)} MNs [{time.time()-t0:.1f}s]")

            # ─── Ascending neurons ───
            print("  [CSR] Finding ascending neurons...")
            self.ascending_indices = []
            for rid, sc in self.super_class_map.items():
                if sc in ('ascending', 'ascending_neuron'):
                    idx = self.root_to_idx.get(rid)
                    if idx is not None:
                        self.ascending_indices.append(idx)
            print(f"    {len(self.ascending_indices)} ascending [{time.time()-t0:.1f}s]")

            # ─── VNC intrinsic indices ───
            print("  [CSR] Finding VNC neurons...")
            self.vnc_indices = []
            self.abdomen_indices = []
            self.leg_indices = []
            self.wing_indices = []
            for rid, sc in self.super_class_map.items():
                if sc == 'ventral_nerve_cord_intrinsic':
                    idx = self.root_to_idx.get(rid)
                    if idx is not None:
                        self.vnc_indices.append(idx)
            # Also check body parts for more VNC neurons
            for rid, bp in self.body_part_map.items():
                bp_lower = bp.lower()
                idx = self.root_to_idx.get(rid)
                if idx is None:
                    continue
                if 'abdomen' in bp_lower:
                    self.abdomen_indices.append(idx)
                if 'leg' in bp_lower or 'coxa' in bp_lower or 'trochanter' in bp_lower:
                    self.leg_indices.append(idx)
                if 'wing' in bp_lower or 'tegula' in bp_lower or 'haltere' in bp_lower:
                    self.wing_indices.append(idx)
            print(f"    VNC intrinsic: {len(self.vnc_indices)}, "
                  f"abdomen: {len(self.abdomen_indices)}, "
                  f"leg: {len(self.leg_indices)}, "
                  f"wing: {len(self.wing_indices)}"
                  f" [{time.time()-t0:.1f}s]")

        # ─── 4b. Named DN search (DNg100, DNb08, etc.) ──
        print("  [CSR] Searching for named DNs (DNg100, DNb08)...")
        dn_names = ['DNg100', 'DNb08', 'DNg23', 'DNg05', 'DNg12', 'DNg14',
                     'MDN014', 'MDN015', 'MDN016']
        self.named_dns = {}
        for rid, idx in self.root_to_idx.items():
            typ = self.type_map.get(rid, '')
            if typ in dn_names:
                self.named_dns[typ] = (int(rid), int(idx))
            # Also check Primary Cell Type for V2
            if self.is_v2 and typ not in dn_names:
                # type_map already has Primary Cell Type from _load_neurons_v2
                pass
        if self.named_dns:
            for name, (rid, idx) in sorted(self.named_dns.items()):
                print(f"    ✅ {name}: root_id={rid}, csr_idx={idx:,}")
        else:
            print(f"    ⚠ Named DNs не найдены в type_map")
        print(f"      [{time.time()-t0:.1f}s]")

        # ─── 5. Visual afferent indices ─────────────────
        print("  [CSR] Finding visual afferents...")
        self.visual_afferent_indices = []
        for rid, idx in sorted(self.root_to_idx.items(), key=lambda x: x[1]):
            typ = self.type_map.get(rid, '')
            for pat in VISUAL_TYPE_PATTERNS:
                if pat in typ or typ.startswith(pat):
                    self.visual_afferent_indices.append(idx)
                    break
        print(f"    {len(self.visual_afferent_indices)} visual afferents "
              f"[{time.time()-t0:.1f}s]")

        print(f"  [CSR] ✅ Загружено: {self.n_neurons:,} нейронов, "
              f"{self.n_synapses:,} синапсов, "
              f"{len(self.dn_indices)} DNs" +
              (f", {len(self.mn_indices)} MNs" if self.is_v2 else ""))


# ═══════════════════════════════════════════════════════════════
#  Topological depth — sensory→motor axis
# ═══════════════════════════════════════════════════════════════

def compute_topological_depth(row_ptr, col_indices, values,
                               motor_indices=None, weights=(0.5, 0.3, 0.2)):
    """Compute topological depth from motor → sensory axis.

    Uses graph distances to identify sensory, central, and motor regions.

    Args:
        row_ptr, col_indices, values: CSR connectome
        motor_indices: list of motor (DN) neuron indices
        weights: (alpha, beta, gamma) for depth formula

    Returns:
        depth: [n] float32 in [0, 1] — 0=motor, 1=sensory
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    n = len(row_ptr) - 1
    n_syn = len(values)

    # Build forward graph
    ones = np.ones(n_syn, dtype=np.float64)
    G = csr_matrix((ones, col_indices, row_ptr), shape=(n, n))

    # Motor distance (reverse graph from DNs)
    motor_dist = np.full(n, np.inf, dtype=np.float64)
    if motor_indices is not None and len(motor_indices) > 0:
        n_src = min(30, len(motor_indices))
        src = sorted(motor_indices)
        subset = np.array([src[i * len(src) // n_src] for i in range(n_src)], dtype=np.int32)
        G_rev = G.T.tocsr()
        dists = shortest_path(G_rev, directed=True, unweighted=True, indices=subset)
        motor_dist = np.min(dists, axis=0)

    # Sensory distance (forward graph from sources furthest from DNs)
    motor_reachable = np.isfinite(motor_dist)
    sensory_sources = np.where(~motor_reachable)[0]
    sensory_dist = np.full(n, np.inf, dtype=np.float64)
    if len(sensory_sources) > 0:
        n_src = min(30, len(sensory_sources))
        src = sorted(sensory_sources)
        subset = np.array([src[i * len(src) // n_src] for i in range(n_src)], dtype=np.int32)
        dists = shortest_path(G, directed=True, unweighted=True, indices=subset)
        sensory_dist = np.min(dists, axis=0)

    # Normalize
    alpha, beta, gamma = weights

    motor_norm = np.full(n, np.nan, dtype=np.float64)
    sensory_norm = np.full(n, np.nan, dtype=np.float64)

    mr = np.isfinite(motor_dist) & (motor_dist > 0)
    sr = np.isfinite(sensory_dist) & (sensory_dist > 0)

    if np.any(mr):
        max_m = np.max(motor_dist[mr])
        if max_m > 0:
            motor_norm[mr] = motor_dist[mr] / max_m

    if np.any(sr):
        max_s = np.max(sensory_dist[sr])
        if max_s > 0:
            sensory_norm[sr] = sensory_dist[sr] / max_s

    # Depth formula
    depth = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(motor_norm) & np.isfinite(sensory_norm)
    depth[valid] = (alpha * (1 - motor_norm[valid]) +
                    beta * sensory_norm[valid] +
                    gamma * 0.5)

    depth[~valid] = 0.5  # fallback
    depth = np.clip(depth, 0.0, 1.0)
    return depth.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  FlyWire Izhikevich Brain — полный коннектом
# ═══════════════════════════════════════════════════════════════

DEFAULT_IZHI_PARAMS = {
    'a': 0.02,
    'b': 0.2,
    'c': -65.0,
    'd': 8.0,
    'tau_ref': 3.0,
}


class FlyWireBrain:
    """Izhikevich brain — коннектом 138K нейронов через Rust.

    Использует Rust `bench_connectome_cpu_recording` — единый проход
    Izhikevich + CSR propagation в Rust (SIMD f32x8 + rayon).

    Совместимость с BrainBridge:
        .bodyids     — список MANC bodyId
        .get_rates(t) — {MANC bodyId: rate} из dn_spike_counts
        .step_batch(t_mujoco) — Izhikevich на batch шагов (единый Rust вызов)
    """

    def __init__(self, csr: FlyWireCSR, dt_ms: float = 0.5,
                 manc_bodyids: list = None, fw_to_manc: dict = None,
                 vnc_gain: float = 2.0, autotune_path: str = None):
        self.csr = csr
        self.dt_ms = dt_ms
        self.n = csr.n_neurons
        self.n_padded = (self.n + 7) // 8 * 8
        self.manc_bodyids = manc_bodyids or []
        self.fw_to_manc = fw_to_manc or {}
        self.vnc_gain = vnc_gain

        # ─── Izhikevich parameters ───
        # Default: per-type autotune or canonical RS
        if autotune_path and Path(autotune_path).exists():
            data = np.load(autotune_path)
            self.a = np.zeros(self.n_padded, dtype=np.float32)
            self.b = np.zeros(self.n_padded, dtype=np.float32)
            self.c = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['c'], dtype=np.float32)
            self.d = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['d'], dtype=np.float32)
            self.I_ext_bias = np.full(self.n_padded, 10.0, dtype=np.float32)
            for key in ['a', 'b', 'c', 'd', 'I_ext']:
                arr = data[key]
                n_copy = min(len(arr), self.n)
                getattr(self, key if key != 'I_ext' else 'I_ext_bias')[:n_copy] = arr[:n_copy]
            print(f"  [IZHI] Autotune params loaded: {autotune_path}")
        else:
            self.a = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['a'], dtype=np.float32)
            self.b = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['b'], dtype=np.float32)
            self.c = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['c'], dtype=np.float32)
            self.d = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['d'], dtype=np.float32)
            self.I_ext_bias = np.full(self.n_padded, 10.0, dtype=np.float32)

        # ─── Izhikevich state ───
        rng = np.random.RandomState(42)
        self.v = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['c'], dtype=np.float32)
        self.v[:self.n] += rng.uniform(-2.0, 2.0, self.n).astype(np.float32)
        self.u = np.full(self.n_padded, DEFAULT_IZHI_PARAMS['b'] * DEFAULT_IZHI_PARAMS['c'],
                         dtype=np.float32)
        self.trace = np.zeros(self.n_padded, dtype=np.float32)
        self.I_buf = np.zeros(self.n_padded, dtype=np.float32)
        self.spikes = np.zeros(self.n_padded, dtype=np.int32)

        # ─── NT types ───
        self.nt_types = np.ones(self.n_padded, dtype=np.float32)
        if csr.nt_types is not None:
            n_nt = min(len(csr.nt_types), self.n)
            self.nt_types[:n_nt] = csr.nt_types[:n_nt]

        # ─── External input ───
        self.external_I = np.zeros(self.n_padded, dtype=np.float32)

        # ─── Looming sensor ───
        self.looming = LoomingSensor()
        self.looming_enabled = len(csr.visual_afferent_indices) > 0
        self.visual_afferent_indices = csr.visual_afferent_indices
        self.visual_afferent_mask = np.zeros(self.n, dtype=bool)
        n_vis = min(len(self.visual_afferent_indices), self.n // 2)
        self.visual_afferent_mask[self.visual_afferent_indices[:n_vis]] = True
        print(f"  [IZHI] Vision looming: {np.sum(self.visual_afferent_mask)} afferents")

        # ─── Drive mask ───
        rng2 = np.random.RandomState(42)
        self.drive_mask = rng2.rand(self.n) < 0.3

        # ─── DN rate buffer ───
        self.dn_rate = np.zeros(len(csr.dn_indices), dtype=np.float32)
        self._batch_count = 0

        # ─── Injection targets (DNg100, CB0013) — set by VNC controller ───
        self._injection_targets = {}

        # ─── Rust backend check ───
        self.use_rust = hasattr(tcm, 'connectome') and \
            hasattr(tcm.connectome, 'bench_connectome_cpu_recording')
        backend = 'Rust bench_connectome_cpu_recording' if self.use_rust else 'Python fallback'
        if not self.use_rust:
            print("  ⚠ Rust backend недоступен. Использую Python (очень медленно).")

        print(f"  [IZHI] Izhikevich: {self.n:,} нейронов, dt={dt_ms}ms, "
              f"{len(csr.dn_indices)} DN, "
              f"VNC mapped: {len(self.manc_bodyids)}")
        print(f"  [IZHI] Backend: {backend}")

    @property
    def bodyids(self):
        return self.manc_bodyids

    def get_rates(self, t: float) -> dict:
        return self.get_dn_rates()

    def get_dn_rates(self) -> dict:
        manc_rates = defaultdict(list)
        for i, rid in enumerate(self.csr.dn_root_ids):
            if i < len(self.dn_rate):
                rate = float(self.dn_rate[i]) * self.vnc_gain
                if rate > 0.01:
                    manc_bid = self.fw_to_manc.get(int(rid))
                    if manc_bid is not None:
                        manc_rates[manc_bid].append(rate)
        rates = {}
        for bid, rate_list in manc_rates.items():
            avg = sum(rate_list) / len(rate_list)
            if avg > 0.05:
                rates[bid] = min(1.0, avg)
        return rates

    def get_vnc_rates_vector(self) -> np.ndarray:
        vec = np.zeros(len(self.manc_bodyids), dtype=np.float32)
        rates = self.get_dn_rates()
        for i, bid in enumerate(self.manc_bodyids):
            vec[i] = rates.get(bid, 0.0)
        return vec

    def set_sensory_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if not self.looming_enabled:
            return
        self.looming.compute(x, y, vx, vy)

    def _set_sensory_input(self, t: float):
        self.external_I.fill(0.0)
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        bias = self.I_ext_bias[:self.n] if hasattr(self, 'I_ext_bias') else 10.0
        drive = np.where(
            self.drive_mask[:self.n],
            12.0 + mod * 8.0 + bias * 0.5,
            5.0 + mod * 4.0 + bias * 0.5,
        )
        self.external_I[:self.n] = drive.astype(np.float32)
        if self.looming_enabled and self.looming.looming > 0.01:
            aff_indices = np.where(self.visual_afferent_mask[:self.n])[0]
            if len(aff_indices):
                ramp = np.linspace(1.0, 0.3, len(aff_indices), dtype=np.float32)
                self.external_I[aff_indices] += self.looming.looming * 10.0 * ramp

    def step_batch(self, t_mujoco: float):
        """Izhikevich batch: единый Rust вызов bench_connectome_cpu_recording."""
        n_steps = max(1, int(25.0 / self.dt_ms))
        dt = self.dt_ms

        self._set_sensory_input(t_mujoco)
        self._batch_count += 1
        if self._batch_count % 10 == 1:
            noise = np.random.RandomState(self._batch_count).randn(self.n).astype(np.float32) * 0.3
            self.v[:self.n] += noise
            np.clip(self.v[:self.n], -100.0, 50.0, out=self.v[:self.n])

        # ─── Apply persistent injection targets (DNg100, CB0013) ───
        # Must be AFTER _set_sensory_input (which zeros external_I)
        for csr_idx, current in self._injection_targets.items():
            if csr_idx < self.n and current > 0:
                self.external_I[csr_idx] += current

        self.I_buf.fill(0.0)
        self.I_buf[:self.n] = self.external_I[:self.n]

        if self.use_rust:
            spike_buf = np.zeros(n_steps * self.n, dtype=np.int32)
            try:
                tcm.connectome.bench_connectome_cpu_recording(
                    self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                    n_steps, dt,
                    self.v, self.u, self.trace,
                    self.a, self.b, self.c, self.d,
                    self.nt_types, self.external_I, self.I_buf, self.spikes,
                    spike_buf,
                )
            except Exception as e:
                print(f"  ⚠ Rust Izhikevich error: {e}, fallback")
                self._extract_dn_rates(spike_buf, n_steps)
                return
        else:
            # Python fallback
            spike_buf = np.zeros(n_steps * self.n, dtype=np.int32)
            for step in range(n_steps):
                v = self.v[:self.n].copy()
                u = self.u[:self.n]
                for i in range(self.n):
                    dv = 0.04 * v[i] * v[i] + 5 * v[i] + 140 - u[i] + self.I_buf[i]
                    du = self.a[i] * (self.b[i] * v[i] - u[i])
                    v[i] += dt * dv
                    u[i] += dt * du
                    if v[i] >= 30.0:
                        spike_buf[step * self.n + i] = 1
                        v[i] = self.c[i]
                        u[i] += self.d[i]
                self.v[:self.n] = v
                self.u[:self.n] = u
                self.I_buf[:self.n] = self.external_I[:self.n]
                for pre_idx in np.where(spike_buf[step * self.n:(step + 1) * self.n])[0]:
                    s, e = int(self.csr.row_ptr[pre_idx]), int(self.csr.row_ptr[pre_idx + 1])
                    for j in range(s, e):
                        post = int(self.csr.col_indices[j])
                        if post < self.n:
                            self.I_buf[post] += self.csr.values[j] * self.nt_types[pre_idx]

        self._extract_dn_rates(spike_buf, n_steps)
        # Store for V2VNCController
        self._last_spike_buf = spike_buf
        self._last_n_steps = n_steps

    def _extract_dn_rates(self, spike_buf: np.ndarray, n_steps: int):
        dn_indices = self.csr.dn_indices
        n_dn = len(dn_indices)

        dn_spikes = np.zeros(n_dn, dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            for i, dn_idx in enumerate(dn_indices):
                if dn_idx < self.n and spike_buf[offset + dn_idx]:
                    dn_spikes[i] += 1

        batch_time_s = n_steps * self.dt_ms / 1000.0
        max_rate = 100.0
        self.dn_rate = np.clip(
            dn_spikes.astype(np.float32) / (batch_time_s * max_rate),
            0.0, 1.0,
        )

    def get_mn_rates(self) -> np.ndarray:
        """Get motor neuron spike rates [n_mn] from last step_batch.

        Returns array of firing rates in [0, 1] for each MN index.
        Used by V2VNCController for direct MN → MuJoCo control.
        """
        if not hasattr(self, '_last_spike_buf') or self._last_spike_buf is None:
            return np.array([])
        mn_indices = self.csr.mn_indices
        n_mn = len(mn_indices)
        n_steps = self._last_n_steps
        spike_buf = self._last_spike_buf

        mn_spikes = np.zeros(n_mn, dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            for i, mn_idx in enumerate(mn_indices):
                if mn_idx < self.n and spike_buf[offset + mn_idx]:
                    mn_spikes[i] += 1

        batch_time_s = n_steps * self.dt_ms / 1000.0
        max_rate = 100.0
        return np.clip(
            mn_spikes.astype(np.float32) / (batch_time_s * max_rate),
            0.0, 1.0,
        )

    def get_r_histogram(self, bins=20):
        hist, edges = np.histogram(
            np.clip(self.v[:self.n], -70.0, -40.0),
            bins=bins, range=(-70, -40)
        )
        return hist, edges


# ═══════════════════════════════════════════════════════════════
#  FlyWire Rate Brain — rate-based (~100× faster than Izhikevich)
# ═══════════════════════════════════════════════════════════════

class FlyWireRateBrain:
    """Rate-based brain — быстрый ANN аппроксиматор.

    Использует `tcm.sparse.rate_step` для rate-based propagation
    через реальный FlyWire коннектом.

    ~100× быстрее Izhikevich, но менее биологична.
    """

    def __init__(self, csr: FlyWireCSR, dt_ms: float = 0.5,
                 manc_bodyids: list = None, fw_to_manc: dict = None,
                 vnc_gain: float = 2.0):
        self.csr = csr
        self.dt_ms = dt_ms
        self.n = csr.n_neurons
        self.manc_bodyids = manc_bodyids or []
        self.fw_to_manc = fw_to_manc or {}
        self.vnc_gain = vnc_gain

        # Rate state: firing probabilities [0, 1]
        rng = np.random.RandomState(42)
        self.r = np.zeros(self.n, dtype=np.float32)

        # External input
        self.external_r = np.zeros(self.n, dtype=np.float32)

        # NT types
        self.nt_types = np.ones(self.n, dtype=np.float32)
        if csr.nt_types is not None:
            n_nt = min(len(csr.nt_types), self.n)
            self.nt_types[:n_nt] = csr.nt_types[:n_nt]

        # ─── Rate brain: используем Rust rate_step ───
        self.use_rust = hasattr(tcm, 'sparse') and hasattr(tcm.sparse, 'rate_step')
        if not self.use_rust:
            print("  ⚠ Rust rate_step недоступен — использую Python fallback")

        # ─── Looming ───
        self.looming = LoomingSensor()
        self.looming_enabled = len(csr.visual_afferent_indices) > 0
        self.visual_afferent_indices = csr.visual_afferent_indices
        self.visual_afferent_mask = np.zeros(self.n, dtype=bool)
        n_vis = min(len(self.visual_afferent_indices), self.n // 2)
        self.visual_afferent_mask[self.visual_afferent_indices[:n_vis]] = True

        # ─── Drive mask ───
        rng2 = np.random.RandomState(42)
        self.drive_mask = rng2.rand(self.n) < 0.3

        # ─── Injection targets (DNg100, CB0013) — set by VNC controller ───
        self._injection_targets = {}

        # ─── DN rate buffer ───
        self.dn_rate = np.zeros(len(csr.dn_indices), dtype=np.float32)

        print(f"  [RATE] Rate-based: {self.n:,} нейронов, dt={dt_ms}ms, "
              f"{len(csr.dn_indices)} DN, VNC mapped: {len(self.manc_bodyids)}")

    @property
    def bodyids(self):
        return self.manc_bodyids

    def get_rates(self, t: float) -> dict:
        return self.get_dn_rates()

    def get_dn_rates(self) -> dict:
        rates = {}
        for i, dn_idx in enumerate(self.csr.dn_indices):
            if dn_idx < len(self.r):
                rate = float(self.r[dn_idx]) * self.vnc_gain
                if rate > 0.01:
                    rid = self.csr.dn_root_ids[i] if i < len(self.csr.dn_root_ids) else None
                    if rid is not None:
                        manc_bid = self.fw_to_manc.get(int(rid))
                        if manc_bid is not None:
                            rates[manc_bid] = min(1.0, rate)
        return rates

    def get_vnc_rates_vector(self) -> np.ndarray:
        vec = np.zeros(len(self.manc_bodyids), dtype=np.float32)
        rates = self.get_dn_rates()
        for i, bid in enumerate(self.manc_bodyids):
            vec[i] = rates.get(bid, 0.0)
        return vec

    def set_sensory_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if not self.looming_enabled:
            return
        self.looming.compute(x, y, vx, vy)

    def _set_sensory_input(self, t: float):
        self.external_r.fill(0.0)
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        drive = np.where(
            self.drive_mask[:self.n],
            0.3 + mod * 0.2,
            0.1 + mod * 0.1,
        )
        self.external_r[:self.n] = drive.astype(np.float32)
        if self.looming_enabled and self.looming.looming > 0.01:
            aff_indices = np.where(self.visual_afferent_mask[:self.n])[0]
            if len(aff_indices):
                ramp = np.linspace(1.0, 0.3, len(aff_indices), dtype=np.float32)
                self.external_r[aff_indices] += self.looming.looming * 0.5 * ramp

    def step_batch(self, t_mujoco: float):
        """Rate-based batch: много шагов rate_step."""
        n_steps = max(1, int(25.0 / self.dt_ms))
        self._set_sensory_input(t_mujoco)

        # ─── Apply persistent injection targets (DNg100, CB0013) ───
        # Must be AFTER _set_sensory_input (which zeros external_r)
        for csr_idx, current in self._injection_targets.items():
            if csr_idx < self.n and current > 0:
                self.external_r[csr_idx] += current

        I_buf = self.external_r.copy()
        tau = 0.1  # rate time constant

        if self.use_rust:
            dt_sec = self.dt_ms / 1000.0  # ms → sec
            decay = math.exp(-dt_sec / tau)
            inv_decay = 1.0 - decay
            for step in range(n_steps):
                tcm.sparse.rate_step(
                    self.r, I_buf, self.csr.row_ptr, self.csr.col_indices,
                    self.csr.values, self.nt_types, self.external_r,
                    decay, inv_decay,
                )
        else:
            # Python fallback
            for step in range(n_steps):
                dt = self.dt_ms / 1000.0
                # Leaky integration
                dr = (-self.r + np.clip(I_buf, 0.0, 1.0)) / tau
                self.r[:] += dt * dr
                self.r = np.clip(self.r, 0.0, 1.0)

                # Reset I = external
                I_buf[:] = self.external_r[:]

                # Propagate spikes (rate-weighted)
                spiked = self.r > 0.5
                for pre_idx in np.where(spiked)[0]:
                    s = int(self.csr.row_ptr[pre_idx])
                    e = int(self.csr.row_ptr[pre_idx + 1])
                    for j in range(s, e):
                        post = int(self.csr.col_indices[j])
                        if post < self.n:
                            I_buf[post] += self.csr.values[j] * self.nt_types[j] * self.r[pre_idx]

        # DN rates = current rate values
        for i, dn_idx in enumerate(self.csr.dn_indices):
            if dn_idx < len(self.r):
                self.dn_rate[i] = float(self.r[dn_idx])

    def get_mn_rates(self) -> np.ndarray:
        """MN rates = current r values at MN indices."""
        mn_indices = self.csr.mn_indices
        return np.array([float(self.r[idx]) for idx in mn_indices if idx < len(self.r)],
                        dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  FlyWire Psi Brain — ψ-wave (квантовая модель)
# ═══════════════════════════════════════════════════════════════

class FlyWirePsiBrain:
    """ψ-wave brain — квантовая модель нейронов.

    Использует ψ = v + i·u как волновую функцию нейрона,
    гамильтониан H = D⁻¹·A (нормированная Laplacian),
    эволюция: dψ/dt = -i·ħ⁻¹·H·ψ

    Использует Rust `step_psi_wave_full` для CSR propagation.
    """

    def __init__(self, csr: FlyWireCSR, dt_ms: float = 0.5,
                 manc_bodyids: list = None, fw_to_manc: dict = None,
                 vnc_gain: float = 2.0, hbar: float = 0.5):
        self.csr = csr
        self.dt_ms = dt_ms
        self.n = csr.n_neurons
        self.manc_bodyids = manc_bodyids or []
        self.fw_to_manc = fw_to_manc or {}
        self.vnc_gain = vnc_gain
        self.hbar = hbar

        # ψ state
        rng = np.random.RandomState(42)
        self.v = rng.randn(self.n).astype(np.float32) * 0.1
        self.u = rng.randn(self.n).astype(np.float32) * 0.1

        # Hamiltonian buffers (computed once)
        self.h_real = np.zeros(self.n, dtype=np.float32)
        self.h_imag = np.zeros(self.n, dtype=np.float32)
        self._compute_hamiltonian = hasattr(tcm, 'sparse') and \
            hasattr(tcm.sparse, 'compute_hamiltonian')
        self._step_psi = hasattr(tcm, 'sparse') and \
            hasattr(tcm.sparse, 'step_psi_wave_full')

        # External input / speed modulation
        self.external_I = np.zeros(self.n, dtype=np.float32)
        self.speed = np.ones(self.n, dtype=np.float32)

        # ─── Looming ───
        self.looming = LoomingSensor()
        self.looming_enabled = len(csr.visual_afferent_indices) > 0
        self.visual_afferent_indices = csr.visual_afferent_indices
        self.visual_afferent_mask = np.zeros(self.n, dtype=bool)
        n_vis = min(len(self.visual_afferent_indices), self.n // 2)
        self.visual_afferent_mask[self.visual_afferent_indices[:n_vis]] = True

        # ─── Drive mask ───
        rng2 = np.random.RandomState(42)
        self.drive_mask = rng2.rand(self.n) < 0.3

        # ─── Injection targets (DNg100, CB0013) — set by VNC controller ───
        self._injection_targets = {}

        # ─── DN rate buffer ───
        self.dn_rate = np.zeros(len(csr.dn_indices), dtype=np.float32)
        self._batch_count = 0

        print(f"  [PSI] ψ-wave: {self.n:,} нейронов, dt={dt_ms}ms, "
              f"ħ={hbar}, {len(csr.dn_indices)} DN, "
              f"VNC mapped: {len(self.manc_bodyids)}")

    @property
    def bodyids(self):
        return self.manc_bodyids

    def get_rates(self, t: float) -> dict:
        # ψ-wave rates = |ψ|² (probability density)
        prob = self.v**2 + self.u**2
        rates = {}
        for i, dn_idx in enumerate(self.csr.dn_indices):
            if dn_idx < self.n:
                rate = float(prob[dn_idx]) * self.vnc_gain
                if rate > 0.01:
                    rid = self.csr.dn_root_ids[i]
                    manc_bid = self.fw_to_manc.get(int(rid))
                    if manc_bid is not None:
                        rates[manc_bid] = min(1.0, rate)
        return rates

    def get_vnc_rates_vector(self) -> np.ndarray:
        vec = np.zeros(len(self.manc_bodyids), dtype=np.float32)
        rates = self.get_rates(0.0)
        for i, bid in enumerate(self.manc_bodyids):
            vec[i] = rates.get(bid, 0.0)
        return vec

    def set_sensory_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if not self.looming_enabled:
            return
        self.looming.compute(x, y, vx, vy)

    def _set_sensory_input(self, t: float):
        self.external_I.fill(0.0)
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        drive = np.where(
            self.drive_mask[:self.n],
            0.5 + mod * 0.3,
            0.2 + mod * 0.15,
        )
        self.external_I[:self.n] = drive.astype(np.float32)
        if self.looming_enabled and self.looming.looming > 0.01:
            aff_indices = np.where(self.visual_afferent_mask[:self.n])[0]
            if len(aff_indices):
                ramp = np.linspace(1.0, 0.3, len(aff_indices), dtype=np.float32)
                self.external_I[aff_indices] += self.looming.looming * 0.5 * ramp

    def step_batch(self, t_mujoco: float):
        """ψ-wave batch: H·ψ → step ψ."""
        n_steps = max(1, int(25.0 / self.dt_ms))
        self._set_sensory_input(t_mujoco)

        # ─── Apply persistent injection targets (DNg100, CB0013) ───
        # Must be AFTER _set_sensory_input (which zeros external_I)
        for csr_idx, current in self._injection_targets.items():
            if csr_idx < self.n and current > 0:
                self.external_I[csr_idx] += current

        if self._compute_hamiltonian and self._step_psi:
            try:
                # Compute Hamiltonian: H = D⁻¹ · A (once per batch)
                tcm.sparse.compute_hamiltonian(
                    self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                    self.h_real, self.h_imag,
                )
                for _ in range(n_steps):
                    tcm.sparse.step_psi_wave_full(
                        self.v, self.u,
                        self.h_real, self.h_imag,
                        self.external_I, self.speed,
                        self.dt_ms / 1000.0, self.hbar,
                    )
            except Exception as e:
                print(f"  ⚠ Rust PSI error: {e}")
        else:
            # Python ψ-wave fallback
            dt = self.dt_ms / 1000.0
            for _ in range(n_steps):
                # Simple rotation in complex plane per neuron
                omega = 2.0 * math.pi * 5.0  # 5 Hz base
                phase = omega * dt
                cos_p = math.cos(phase * self.hbar)
                sin_p = math.sin(phase * self.hbar)
                v_new = self.v * cos_p - self.u * sin_p + self.external_I * dt
                u_new = self.v * sin_p + self.u * cos_p
                self.v[:] = v_new
                self.u[:] = u_new

        # ψ-wave DN rates = |ψ|² for DNs
        prob = self.v[:self.n]**2 + self.u[:self.n]**2
        for i, dn_idx in enumerate(self.csr.dn_indices):
            if dn_idx < self.n:
                self.dn_rate[i] = min(1.0, float(prob[dn_idx]))

    def get_mn_rates(self) -> np.ndarray:
        """ψ-wave MN rates = |ψ|² at MN indices."""
        prob = self.v[:self.n]**2 + self.u[:self.n]**2
        mn_indices = self.csr.mn_indices
        return np.array([min(1.0, float(prob[idx])) for idx in mn_indices if idx < self.n],
                        dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  FlyWire LIF Brain — LIF с параметрами Shiu et al. 2024
# ═══════════════════════════════════════════════════════════════

class FlyWireLIFBrain:
    """LIF brain с валидированными параметрами Shiu et al. 2024.

    Использует Rust `lif_connectome.lif_sim_cpu_recording` — единый проход
    LIF + CSR propagation в Rust (SIMD f32x8), без Python↔Rust crossings
    между шагами.

    Совместимость с BrainBridge:
        .bodyids     — список MANC bodyId
        .get_rates(t) — {MANC bodyId: rate} из dn_spike_counts
        .step_batch(t_mujoco) — LIF на batch шагов (единый Rust вызов)
    """

    def __init__(self, csr: FlyWireCSR, dt_ms: float = 0.5,
                 manc_bodyids: list = None, fw_to_manc: dict = None,
                 vnc_gain: float = 2.0):
        self.csr = csr
        self.dt_ms = dt_ms
        self.n = csr.n_neurons
        self.n_padded = (self.n + 7) // 8 * 8
        self.manc_bodyids = manc_bodyids or []
        self.fw_to_manc = fw_to_manc or {}
        self.vnc_gain = vnc_gain

        # ─── LIF parameters (Shiu et al. 2024, Nature) ───
        # τ_m = 20ms (Kakaria & de Bivort 2017), C=0.002µF, R=10MΩ
        self.tau_m = np.full(self.n_padded, 20.0, dtype=np.float32)
        self.v_rest = np.full(self.n_padded, -52.0, dtype=np.float32)
        self.v_thresh = np.full(self.n_padded, -45.0, dtype=np.float32)
        self.v_reset = np.full(self.n_padded, -52.0, dtype=np.float32)

        # ─── LIF state ───
        rng = np.random.RandomState(42)
        self.v = np.full(self.n_padded, -52.0, dtype=np.float32)
        self.v[:self.n] += rng.uniform(-1.0, 1.0, self.n).astype(np.float32)
        self.I_buf = np.zeros(self.n_padded, dtype=np.float32)
        self.spikes = np.zeros(self.n_padded, dtype=np.int32)

        # ─── NT types (pre-neuron) ───
        self.nt_types = np.ones(self.n_padded, dtype=np.float32)
        if csr.nt_types is not None:
            self.nt_types[:self.n] = csr.nt_types[:self.n]

        # ─── External input ───
        self.external_I = np.zeros(self.n_padded, dtype=np.float32)

        # ─── Alpha-synapse + delay (Shiu et al. 2024) ───
        # Synaptic conductance: g_syn[i] *= (1 - dt/τ_syn)
        self.tau_syn = 5.0  # alpha-synapse time constant (ms)
        self.g_syn = np.zeros(self.n_padded, dtype=np.float32)
        # Synaptic delay: 1.8ms ≈ 4 steps at dt=0.5ms
        self.delay_steps = 4
        self.delay_buf = np.zeros(self.n_padded * self.delay_steps, dtype=np.float32)
        self.delay_pos = np.zeros(1, dtype=np.int32)

        # ─── Poisson generator ───
        self._poisson_rng = np.random.RandomState(0)

        # ─── Looming sensor ───
        self.looming = LoomingSensor()
        self.looming_enabled = len(csr.visual_afferent_indices) > 0
        self.visual_afferent_indices = csr.visual_afferent_indices
        self.visual_afferent_mask = np.zeros(self.n, dtype=bool)
        n_vis = min(len(self.visual_afferent_indices), self.n // 2)
        self.visual_afferent_mask[self.visual_afferent_indices[:n_vis]] = True
        print(f"  [LIFBrain] Vision looming: {np.sum(self.visual_afferent_mask)} afferents")

        # ─── Drive mask (30% of neurons get strong input) ───
        rng2 = np.random.RandomState(42)
        self.drive_mask = rng2.rand(self.n) < 0.3

        # ─── DN rate buffer ───
        self.dn_rate = np.zeros(len(csr.dn_indices), dtype=np.float32)
        self._batch_count = 0

        # ─── Named DN monitoring (DNg100, DNb08) ───
        self.named_dn_indices = {}
        for name in ['DNg100', 'DNb08']:
            entry = csr.named_dns.get(name)
            if entry is not None:
                rid, idx = entry
                self.named_dn_indices[name] = idx
        if self.named_dn_indices:
            dn_list = ', '.join(f'{n}(idx={idx})' for n, idx in self.named_dn_indices.items())
            print(f"  [LIFBrain] Named DN monitoring: {dn_list}")

        # ─── Persistent injection targets (set by VNC controller) ───
        # Applied inside step_batch AFTER _set_sensory_input (which zeros external_I)
        self._injection_targets = {}  # {csr_idx: current_pA}

        # ─── Backend check: CPU Rust by default, GPU probe happens on first step_batch ───
        self.use_rust = hasattr(tcm, 'lif_connectome') and \
            hasattr(tcm.lif_connectome, 'lif_sim_cpu_recording')
        self.use_gpu = False  # Real GPU probe in step_batch
        self._gpu_probed = False  # First step probes GPU once
        if self.use_rust:
            backend = 'Rust CPU lif_sim_cpu_recording'
        else:
            self.use_rust_fallback = hasattr(tcm, 'step_lif') and \
                hasattr(tcm, 'sparse') and hasattr(tcm.sparse, 'propagate_spikes')
            if not self.use_rust_fallback:
                print("  ⚠ Rust backend недоступен. Использую Python (очень медленно).")
            backend = 'Python fallback'

        # ─── Cached MN rates (avoids recomputing from spike_buf every MuJoCo step) ───
        self._cached_mn_rates = np.array([])

        print(f"  [LIFBrain] LIF: {self.n:,} нейронов, dt={dt_ms}ms, "
              f"{len(csr.dn_indices)} DN, "
              f"VNC mapped: {len(self.manc_bodyids)}")
        print(f"  [LIFBrain] Параметры Shiu et al. 2024: "
              f"V_rest={self.v_rest[0]:.0f}, V_thresh={self.v_thresh[0]:.0f}, "
              f"τ_m={self.tau_m[0]:.0f}ms")
        print(f"  [LIFBrain] Backend: {backend}")

    @property
    def bodyids(self):
        return self.manc_bodyids

    def get_rates(self, t: float) -> dict:
        return self.get_dn_rates()

    def get_dn_rates(self) -> dict:
        manc_rates = defaultdict(list)
        for i, rid in enumerate(self.csr.dn_root_ids):
            if i < len(self.dn_rate):
                rate = float(self.dn_rate[i]) * self.vnc_gain
                if rate > 0.01:
                    manc_bid = self.fw_to_manc.get(int(rid))
                    if manc_bid is not None:
                        manc_rates[manc_bid].append(rate)
        rates = {}
        for bid, rate_list in manc_rates.items():
            avg = sum(rate_list) / len(rate_list)
            if avg > 0.05:
                rates[bid] = min(1.0, avg)
        return rates

    def get_vnc_rates_vector(self) -> np.ndarray:
        vec = np.zeros(len(self.manc_bodyids), dtype=np.float32)
        rates = self.get_dn_rates()
        for i, bid in enumerate(self.manc_bodyids):
            vec[i] = rates.get(bid, 0.0)
        return vec

    def set_sensory_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        if not self.looming_enabled:
            return
        self.looming.compute(x, y, vx, vy)

    def _set_sensory_input(self, t: float):
        """Poisson spike input (Shiu et al. 2024).

        Вместо DC тока используем Poisson спайки с частотой ~100-180 Hz.
        Pulse amplitude ~120 pA — хватает для пересечения порога за 1 шаг.
        """
        self.external_I.fill(0.0)
        mod = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)
        # Poisson rates (Hz): τ_m=20ms → нужно выше для drive
        # Brian2 reference: r_poi=150Hz uniform, w_syn*f_poi=68.75mV
        # Удваиваем для компенсации sparser Poisson vs Brian2 uniform
        rate = np.where(
            self.drive_mask[:self.n],
            200.0 + mod * 160.0,  # 200-360 Hz
            100.0 + mod * 80.0,   # 100-180 Hz
        )
        spike_prob = rate * self.dt_ms / 1000.0
        # Poisson draw — vectorized
        spiking = self._poisson_rng.rand(self.n) < spike_prob
        # Pulse amplitude: ~30 pA (нужно ~5 Poisson spikes для порога)
        #   dt=0.5ms, tau_m=10ms → dv = 30 * 0.5/10 = 1.5 mV/spike
        #   Порог: -52 → -45 = 7mV → ~5 Poisson spikes в окне ~20ms
        #   Это даёт плавную интеграцию вместо дёрганых движений
        pulse_amp = 30.0
        self.external_I[:self.n] = np.where(spiking, pulse_amp, 0.0).astype(np.float32)
        if self.looming_enabled and self.looming.looming > 0.01:
            aff_indices = np.where(self.visual_afferent_mask[:self.n])[0]
            if len(aff_indices):
                ramp = np.linspace(1.0, 0.3, len(aff_indices), dtype=np.float32)
                # Looming adds Poisson rate boost, not direct current
                looming_rate = self.looming.looming * 50.0
                looming_spike = self._poisson_rng.rand(len(aff_indices)) < looming_rate * self.dt_ms / 1000.0
                self.external_I[aff_indices[looming_spike]] += pulse_amp * 0.5

    def step_batch(self, t_mujoco: float):
        """LIF batch: единый Rust вызов lif_sim_cpu_recording.

        Rust делает 50 шагов LIF + CSR propagation за ОДИН проход.
        """
        n_steps = max(1, int(25.0 / self.dt_ms))
        dt = self.dt_ms

        self._set_sensory_input(t_mujoco)
        self._batch_count += 1
        if self._batch_count % 10 == 1:
            noise = np.random.RandomState(self._batch_count).randn(self.n).astype(np.float32) * 0.3
            self.v[:self.n] += noise
            np.clip(self.v[:self.n], -100.0, 50.0, out=self.v[:self.n])

        # ─── Apply persistent injection targets (DNg100, CB0013, etc.) ───
        # Must be AFTER _set_sensory_input (which zeros external_I)
        for csr_idx, current in self._injection_targets.items():
            if csr_idx < self.n and current > 0:
                self.external_I[csr_idx] += current

        # Initialize I = external_I (Rust продолжит: reset + propagate per step)
        self.I_buf.fill(0.0)
        self.I_buf[:self.n] = self.external_I[:self.n]

        spike_buf = np.zeros(n_steps * self.n, dtype=np.int32)
        gpu_ok = False

        # ─── Probe GPU once on first step_batch ───
        if not self._gpu_probed:
            self._gpu_probed = True
            if hasattr(tcm, 'lif_connectome') and hasattr(tcm.lif_connectome, 'lif_sim_gpu_recording'):
                try:
                    tcm.lif_connectome.lif_sim_gpu_recording(
                        self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                        n_steps, dt,
                        self.v, self.tau_m, self.v_rest, self.v_thresh, self.v_reset,
                        self.nt_types, self.external_I,
                        spike_buf,
                    )
                    self.use_gpu = True
                    gpu_ok = True
                    # GPU succeeded — v and spike_buf contain valid state
                except Exception:
                    self.use_gpu = False
                    print("  [LIFBrain] GPU недоступен, использую CPU Rust")

        if self.use_gpu and not gpu_ok:
            # GPU didn't work in this step, try again (but label said GPU)
            try:
                tcm.lif_connectome.lif_sim_gpu_recording(
                    self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                    n_steps, dt,
                    self.v, self.tau_m, self.v_rest, self.v_thresh, self.v_reset,
                    self.nt_types, self.external_I,
                    spike_buf,
                )
                gpu_ok = True
            except Exception:
                self.use_gpu = False  # Больше не пробовать GPU

        if not gpu_ok and self.use_rust:
            try:
                tcm.lif_connectome.lif_sim_cpu_recording(
                    self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                    n_steps, dt,
                    self.v, self.tau_m, self.v_rest, self.v_thresh, self.v_reset,
                    self.nt_types, self.external_I, self.I_buf, self.spikes,
                    spike_buf,
                    self.g_syn, self.tau_syn,
                    self.delay_buf, self.delay_steps, self.delay_pos,
                )
            except Exception as e:
                print(f"  ⚠ Rust CPU LIF error: {e}")
                # continue with empty spike_buf

        if not gpu_ok and not self.use_rust:
            # ─── Fallback: per-step Rust step_lif + propagate_spikes ───
            if hasattr(self, 'use_rust_fallback') and self.use_rust_fallback:
                self.I_buf[:self.n] = self.external_I[:self.n]
                for step in range(n_steps):
                    tcm.step_lif(
                        self.v, self.I_buf, self.spikes, dt,
                        self.tau_m, self.v_rest, self.v_thresh, self.v_reset,
                    )
                    spiked_mask = self.spikes[:self.n] > 0
                    for i in range(self.n):
                        if spiked_mask[i]:
                            spike_buf[step * self.n + i] = 1
                    self.I_buf[:self.n] = self.external_I[:self.n]
                    spiked_idx = np.where(spiked_mask)[0]
                    if len(spiked_idx) > 0:
                        tcm.sparse.propagate_spikes(
                            self.csr.row_ptr, self.csr.col_indices, self.csr.values,
                            spiked_idx.astype(np.int32),
                            self.csr.nt_types, self.I_buf, 1.0,
                        )
            else:
                # Pure Python fallback (very slow)
                v = self.v.copy()[:self.n]
                for step in range(n_steps):
                    alpha = dt / 10.0
                    v = v * (1 - alpha) + self.I_buf[:self.n] * alpha + self.v_rest[0] * alpha
                    spiked = v >= self.v_thresh[0]
                    v[spiked] = self.v_reset[0]
                    spike_buf[step * self.n:step * self.n + self.n] = spiked.astype(np.int32)
                    self.I_buf[:self.n] = self.external_I[:self.n]
                    for pre_idx in np.where(spiked)[0]:
                        s, e = int(self.csr.row_ptr[pre_idx]), int(self.csr.row_ptr[pre_idx + 1])
                        for j in range(s, e):
                            post = int(self.csr.col_indices[j])
                            if post < self.n:
                                self.I_buf[post] += self.csr.values[j] * self.csr.nt_types[pre_idx]
                self.v[:self.n] = v

        # ─── DN + MN rates из spike_buffer ───
        self._extract_dn_rates(spike_buf, n_steps)
        self._cache_mn_rates(spike_buf, n_steps)
        # Store for V2VNCController
        self._last_spike_buf = spike_buf
        self._last_n_steps = n_steps

        # ─── Named DN monitoring (DNg100, DNb08) ───
        if self.named_dn_indices:
            dn_mon = []
            for name, csr_idx in self.named_dn_indices.items():
                spike_count = 0
                for step in range(n_steps):
                    offset = step * self.n
                    if csr_idx < self.n:
                        pos = offset + csr_idx
                        if pos < len(spike_buf) and spike_buf[pos]:
                            spike_count += 1
                rate = spike_count / (n_steps * self.dt_ms / 1000.0)
                dn_mon.append(f'{name}: {rate:.1f}Hz')
            print(f"    [{self._batch_count}] DN: {' | '.join(dn_mon)}")

    def _extract_dn_rates(self, spike_buf: np.ndarray, n_steps: int):
        """DN rate = spike_count / (n_steps * dt) normalized to [0, 1]."""
        dn_indices = self.csr.dn_indices
        n_dn = len(dn_indices)

        dn_spikes = np.zeros(n_dn, dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            for i, dn_idx in enumerate(dn_indices):
                if dn_idx < self.n and spike_buf[offset + dn_idx]:
                    dn_spikes[i] += 1

        batch_time_s = n_steps * self.dt_ms / 1000.0
        max_rate = 100.0
        self.dn_rate = np.clip(
            dn_spikes.astype(np.float32) / (batch_time_s * max_rate),
            0.0, 1.0,
        )

    def _cache_mn_rates(self, spike_buf: np.ndarray, n_steps: int):
        """Vectorized MN rate extraction — обновляет кеш.

        Использует numpy fancy indexing вместо Python double loop.
        """
        mn_indices = self.csr.mn_indices
        n_mn = len(mn_indices)
        if n_mn == 0 or self.n == 0:
            self._cached_mn_rates = np.array([])
            return

        mn_idx_arr = np.array(mn_indices, dtype=np.int32)
        valid = mn_idx_arr < self.n
        if not np.any(valid):
            self._cached_mn_rates = np.zeros(n_mn, dtype=np.float32)
            return

        # Vectorized: sum spikes across all steps at MN indices
        # spike_buf shape: (n_steps * n,) → reshape to (n_steps, n)
        # Then count spikes at mn_indices across all steps
        mn_spikes = np.zeros(n_mn, dtype=np.int32)
        for step in range(n_steps):
            offset = step * self.n
            step_spikes = spike_buf[offset:offset + self.n]
            mn_spikes[valid] += step_spikes[mn_idx_arr[valid]].astype(np.int32)

        batch_time_s = n_steps * self.dt_ms / 1000.0
        max_rate = 100.0
        self._cached_mn_rates = np.clip(
            mn_spikes.astype(np.float32) / (batch_time_s * max_rate),
            0.0, 1.0,
        )

    def get_mn_rates(self) -> np.ndarray:
        """Get motor neuron spike rates [n_mn] from cache.

        Кеш обновляется в step_batch. Не пересчитывает каждый раз.
        """
        if len(self._cached_mn_rates) == 0 and hasattr(self, '_last_spike_buf') and self._last_spike_buf is not None:
            # Fallback: compute on demand (shouldn't happen normally)
            self._cache_mn_rates(self._last_spike_buf, self._last_n_steps)
        return self._cached_mn_rates

    def get_r_histogram(self, bins=20):
        hist, edges = np.histogram(
            np.clip(self.v[:self.n], -70.0, -40.0),
            bins=bins, range=(-70, -40)
        )
        return hist, edges
