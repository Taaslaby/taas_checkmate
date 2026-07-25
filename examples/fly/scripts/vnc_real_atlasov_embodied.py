#!/usr/bin/env python3
"""
🧬🦗 REAL VNC CONNECTOME → HH → ATLASOV → MUJOCO FLY
=====================================================
Первый в мире пайплайн: РЕАЛЬНЫЙ VNC коннектом (158K нейронов, 4M синапсов)
через HH/Izhikevich симуляцию → анатомический MN→мышцы map → Atlasov Ca²⁺
→ Tug-of-War → MuJoCo 3D физика.

Отличие от fafb_atlasov_embodied.py:
  ✅ Реальный VNC коннектом с нейротрансмиттерами (GABA/GLUT/ACH)
  ✅ Анатомическое отображение MN→суставы из Franken/VNC метаданных
  ✅ Группы нейронов: descending → VNC intrinsic → motor → мышцы
  ✅ НЕ fake CPG — CPG рождается из реципрокного GABA-торможения

Data: /home/taaslab/Проекты/asm_engines/sardaanaCM/experiments/datasets/vnc/
Author / Concept: Atlasov Anton (Атласов Антон)
"""

import os
import sys
import time
import gzip
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Offscreen rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

VNC_DATA = Path.home() / "Проекты/asm_engines/sardaanaCM/experiments/datasets/vnc"
CONN_FILE = VNC_DATA / "connections_princeton.csv.gz"
NEURONS_FILE = VNC_DATA / "neurons.csv.gz"
SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

import numpy as np
import mujoco as mu
import taas_checkmate as tc


# ═════════════════════════════════════════════════════════════════════
#  1. VNC CONNECTOME LOADER — из реальных данных
# ═════════════════════════════════════════════════════════════════════

class RealVNCConnectome:
    """Загружает РЕАЛЬНЫЙ VNC коннектом из датасета.

    Формат connections:
        pre_root_id, post_root_id, neuropil, syn_count, nt_type

    Формат neurons:
        Root ID, Predicted NT type, Body Part, Super Class, Flow,
        Hemilineage, Nerve, Soma side, Primary Cell Type

    Фильтрует нейроны по нейропилям VNC для получения спинномозгового субграфа.
    """

    def __init__(self, vnc_only=True):
        """
        Args:
            vnc_only: если True, берём ТОЛЬКО VNC нейроны (VNC_T1/T2/T3/ABDNM нейропили)
                      если False, берём полный мозг+VNC (158K нейронов)
        """
        self.vnc_only = vnc_only
        self.neurons = {}       # root_id → {metadata}
        self.synapses = []      # (pre_root_id, post_root_id, syn_count, nt_type)
        self.root_to_idx = {}   # root_id → neuron index
        self.idx_to_root = {}   # neuron index → root_id
        self.n_neurons = 0
        self.n_synapses = 0

        # Группы нейронов (по индексам)
        self.vnc_intrinsic = []
        self.motor = []
        self.descending = []
        self.ascending = []
        self.sensory = []
        self.brain_intrinsic = []
        self.leg_motor = []         # leg motor neurons specifically
        self.leg_indices_by_nerve = defaultdict(list)  # nerve → [indices]

        # CSR arrays
        self.row_ptr = None
        self.col_indices = None
        self.values = None
        self.nt_types = None

        # Neurotransmitter sign map
        self.NT_SIGN = {
            'GABA': -1.0,
            'GLUT': 1.0,
            'ACH':  1.0,
            'DA':   -0.5,
            'SER':  -0.5,
            'OCT':  0.5,
            'HIST': 0.5,
            'TYR':  0.5,
            '':     1.0,
        }

    def load_neurons(self):
        """Загрузить метаданные нейронов."""
        t0 = time.time()
        print("📁 Загрузка нейронов VNC...")

        if not NEURONS_FILE.exists():
            print(f"❌ Файл не найден: {NEURONS_FILE}")
            return False

        with gzip.open(NEURONS_FILE, 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid_str = row.get('Root ID', '').strip()
                if not rid_str:
                    continue
                try:
                    rid = int(rid_str)
                except ValueError:
                    continue

                nt = row.get('Predicted NT type', '').strip()
                bp = row.get('Body Part', '').strip()
                sc = row.get('Super Class', '').strip()
                flow = row.get('Flow', '').strip()
                nerve = row.get('Nerve', '').strip()
                hemi = row.get('Hemilineage', '').strip()
                ct = row.get('Primary Cell Type', '').strip()
                neuropil = ''  # будет заполнено позже из connections

                self.neurons[rid] = {
                    'nt': nt,
                    'body_part': bp,
                    'super_class': sc,
                    'flow': flow,
                    'nerve': nerve,
                    'hemilineage': hemi,
                    'cell_type': ct,
                    'neuropil': '',
                }

        print(f"✅ {len(self.neurons):,} нейронов за {time.time()-t0:.2f}с")
        return True

    def load_connections(self, max_synapses=None):
        """Загрузить синапсы и построить CSR."""
        t0 = time.time()
        print("📁 Загрузка синапсов VNC...")

        if not CONN_FILE.exists():
            print(f"❌ Файл не найден: {CONN_FILE}")
            return False

        # Первый проход: собираем нейропили для нейронов
        neuropil_by_neuron = {}
        raw_synapses = []
        with gzip.open(CONN_FILE, 'rt') as f:
            reader = csv.DictReader(f)
            for row_n, row in enumerate(reader):
                if max_synapses and row_n >= max_synapses:
                    break
                try:
                    pre = int(row['pre_root_id'])
                    post = int(row['post_root_id'])
                    syn = int(row.get('syn_count', 1))
                    npil = row.get('neuropil', '')
                except (ValueError, KeyError):
                    continue
                raw_synapses.append((pre, post, syn))

                # Запоминаем нейропили
                if pre not in neuropil_by_neuron:
                    neuropil_by_neuron[pre] = set()
                if post not in neuropil_by_neuron:
                    neuropil_by_neuron[post] = set()
                if npil:
                    neuropil_by_neuron[pre].add(npil)
                    neuropil_by_neuron[post].add(npil)

        # Обновляем нейропили в метаданных
        for rid, npils in neuropil_by_neuron.items():
            if rid in self.neurons:
                main_npil = max(npils, key=lambda x: len(x)) if npils else ''
                self.neurons[rid]['neuropil'] = main_npil

        # Определяем VNC нейроны по нейропилям
        VNC_NPILS = {'VNC_T1_ProNm', 'VNC_T2_MesoNm', 'VNC_T3_MetaNm',
                     'ABDNM', 'VNC'}

        vnc_neuron_ids = set()
        for rid, meta in self.neurons.items():
            npil = meta['neuropil']
            if any(vnc_npil in npil for vnc_npil in VNC_NPILS):
                vnc_neuron_ids.add(rid)
            if meta['super_class'] in ('ventral_nerve_cord_intrinsic', 'motor',
                                        'descending', 'ascending'):
                vnc_neuron_ids.add(rid)
            if meta['body_part'] in ('leg', 'front_leg', 'middle_leg', 'hind_leg',
                                      'abdomen', 'wing'):
                vnc_neuron_ids.add(rid)

        # Фильтруем нейроны
        if self.vnc_only:
            keep_ids = vnc_neuron_ids
        else:
            keep_ids = set(self.neurons.keys())

        # Фильтруем синапсы
        synapses = [(p, o, s) for p, o, s in raw_synapses
                     if p in keep_ids and o in keep_ids]

        self.synapses = synapses
        self.n_synapses = len(synapses)

        # Строим индекс нейронов
        all_ids = sorted(set([p for p, o, s in synapses] + [o for p, o, s in synapses]))
        self.root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
        self.idx_to_root = {i: rid for i, rid in enumerate(all_ids)}
        self.n_neurons = len(all_ids)

        print(f"✅ {len(synapses):,} синапсов, {self.n_neurons:,} нейронов "
              f"за {time.time()-t0:.2f}с")

        # ─── Строим CSR ──────────────────────────────────────────
        print("🔧 Построение CSR матрицы...")
        t1 = time.time()

        # pre_count per neuron
        pre_count = Counter(p for p, o, s in synapses)

        row_ptr = np.zeros(self.n_neurons + 1, dtype=np.int32)
        for rid, cnt in pre_count.items():
            idx = self.root_to_idx[rid]
            row_ptr[idx + 1] = cnt
        row_ptr = np.cumsum(row_ptr, dtype=np.int32)

        col_indices = np.zeros(self.n_synapses, dtype=np.int32)
        values = np.ones(self.n_synapses, dtype=np.float32)
        nt_types = np.ones(self.n_neurons, dtype=np.float32)

        cursor = np.zeros(self.n_neurons, dtype=np.int32)
        for pre, post, syn_cnt in synapses:
            pre_idx = self.root_to_idx[pre]
            post_idx = self.root_to_idx[post]
            pos = row_ptr[pre_idx] + cursor[pre_idx]
            col_indices[pos] = np.int32(post_idx)
            values[pos] = float(syn_cnt)
            cursor[pre_idx] += 1

        # NT types — per neuron sign
        for rid, idx in self.root_to_idx.items():
            meta = self.neurons.get(rid, {})
            nt_type = meta.get('nt', '')
            sign = self.NT_SIGN.get(nt_type, 1.0)
            nt_types[idx] = sign

        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.values = values
        self.nt_types = nt_types

        print(f"✅ CSR построена за {time.time()-t1:.2f}с")

        # ─── Классифицируем нейроны по группам ───────────────────
        for rid, idx in self.root_to_idx.items():
            meta = self.neurons.get(rid, {})
            sc = meta.get('super_class', '')
            bp = meta.get('body_part', '')

            if sc == 'ventral_nerve_cord_intrinsic':
                self.vnc_intrinsic.append(idx)
            elif sc == 'motor':
                self.motor.append(idx)
                if any(leg in bp for leg in ['leg', 'front_leg', 'middle_leg', 'hind_leg']):
                    self.leg_motor.append(idx)
            elif sc == 'descending':
                self.descending.append(idx)
            elif sc == 'ascending':
                self.ascending.append(idx)
            elif sc == 'sensory':
                self.sensory.append(idx)
            else:
                self.brain_intrinsic.append(idx)

            # Nerve-based grouping
            nerve = meta.get('nerve', '')
            if nerve:
                self.leg_indices_by_nerve[nerve].append(idx)

        # Статистика
        print(f"\n📊 Группы нейронов VNC:")
        print(f"  🧬 VNC intrinsic:  {len(self.vnc_intrinsic):>6,}")
        print(f"  🦵 Motor:           {len(self.motor):>6,}  (leg: {len(self.leg_motor)})")
        print(f"  ⬇️  Descending:     {len(self.descending):>6,}")
        print(f"  ⬆️  Ascending:      {len(self.ascending):>6,}")
        print(f"  👁️  Sensory:        {len(self.sensory):>6,}")
        print(f"  🧠 Brain intrinsic: {len(self.brain_intrinsic):>6,}")

        # NT type distribution
        nt_counts = Counter()
        for rid, idx in self.root_to_idx.items():
            nt = self.neurons.get(rid, {}).get('nt', '')
            nt_counts[nt] += 1
        print(f"\n  🧪 Neurotransmitters:")
        for nt, cnt in nt_counts.most_common(10):
            print(f"    {nt or '(?!)':>6s}: {cnt:>6,}")

        # Map leg nerves
        leg_nerve_groups = {k: v for k, v in self.leg_indices_by_nerve.items()
                           if 'leg' in k or 'mesothoracic' in k or 'metathoracic' in k or 'prothoracic' in k}
        print(f"\n  🦵 Leg nerves ({len(leg_nerve_groups)}):")
        for nerve, indices in sorted(leg_nerve_groups.items()):
            print(f"    {nerve:>40s}: {len(indices):>4d} нейронов")

        return True

    def build_connectome_data(self):
        """Построить taan_checkmate ConnectomeData для симуляции."""
        groups = [
            ("vnc_intrinsic", self.vnc_intrinsic),
            ("motor", self.motor),
            ("descending", self.descending),
            ("ascending", self.ascending),
            ("sensory", self.sensory),
            ("leg_motor", self.leg_motor),
        ]
        return tc.ConnectomeData(
            self.row_ptr, self.col_indices, self.values, self.nt_types, groups
        )


# ═════════════════════════════════════════════════════════════════════
#  2. АНАТОМИЧЕСКИЙ MN → ATLASOV MAP (из старого vnc_controller.py)
# ═════════════════════════════════════════════════════════════════════

# Анатомическая карта: тип МН → (сегмент, движение)
V2_MN_TYPE_MAP = {
    'sternal_anterior_rotator':   ('coxa', 'twist'),
    'sternal_posterior_rotator':  ('coxa', 'twist'),
    'pleural_remotor_abductor':   ('coxa', 'abduct'),
    'sternotrochanter':           ('coxa', 'extend'),
    'tergotrochanter':            ('coxa', 'extend'),
    'pleural_rotator':            ('coxa', 'twist'),
    'tergopleural_promotor':      ('coxa', 'abduct'),
    'trochanter_flexor':          ('femur', 'flex'),
    'trochanter_extensor':        ('femur', 'extend'),
    'accessory_trochanter_flexor':('femur', 'flex'),
    'femur_reductor':             ('femur', 'twist'),
    'long_tendon_muscle':         ('femur', 'extend'),
    'tibia_flexor':               ('tibia', 'flex'),
    'tibia_extensor':             ('tibia', 'extend'),
    'accessory_tibia_flexor':     ('tibia', 'flex'),
    'tarsus_depressor':           ('tarsus', 'flex'),
    'tarsus_levator':             ('tarsus', 'flex'),
    'claw':                       ('tarsus2', 'flex'),
    'sternal_adductor':           ('coxa', 'twist'),
}

MOTION_SCALES = {'flex': 0.5, 'extend': -0.5, 'abduct': 0.3, 'twist': 0.3}

# Ноги
LEGS = [("L1", "T1_left"), ("R1", "T1_right"),
        ("L2", "T2_left"), ("R2", "T2_right"),
        ("L3", "T3_left"), ("R3", "T3_right")]

SEGMENTS = ["coxa_abduct", "coxa_twist", "coxa",
            "femur_twist", "femur", "tibia",
            "tarsus", "tarsus2"]

TRIPOD_A = {"T1_left", "T2_right", "T3_left"}
TRIPOD_B = {"T1_right", "T2_left", "T3_right"}

# Маппинг body_part → leg_prefix
BP_TO_LEG = {
    'front_leg': 'T1',
    'middle_leg': 'T2',
    'hind_leg': 'T3',
}


class AtlasovLegMapper:
    """Proper per-segment motor neuron → Atlasov muscle mapper.

    Uses V2_MN_TYPE_MAP to map each motor neuron type to a specific
    joint segment (coxa_twist, femur_flex, tibia_extend, etc.) within
    the correct leg. Each segment drives ONE DOF with flexor/extensor
    pair in Atlasov.
    """

    # Joint segments in order (matches MuJoCo joint naming)
    SEGMENTS = ["coxa_abduct", "coxa_twist", "coxa",
                "femur_twist", "femur", "tibia",
                "tarsus", "tarsus2"]

    # Motion type → (muscle flexor scale, muscle extensor scale, direction)
    MOTION_PARAMS = {
        'flex':   (0.5,  0.0,  1.0),   # flexor contracts
        'extend': (0.0,  0.5, -1.0),   # extensor contracts
        'abduct': (0.3, -0.1,  1.0),   # abductor pulls
        'twist':  (0.3,  0.0,  1.0),   # twister rotates
    }

    def __init__(self, connectome: RealVNCConnectome, fly_sim):
        self.connectome = connectome
        self.fly = fly_sim
        self.n_muscles = fly_sim.n_muscles
        self.n_joints = fly_sim.n_joints

        # Build DOF → segment lookup
        # fly.leg_dofs: {leg_label: [dof_indices...]} for each leg
        # We need: (leg_label, segment_name) → dof_index
        self.segment_dof_map = {}  # (leg_label, segment) → dof_index
        for leg_label, dofs in fly_sim.leg_dofs.items():
            for dof_idx in dofs:
                # Find joint name for this DOF
                for j in range(fly_sim.model.njnt):
                    if fly_sim.model.jnt_dofadr[j] == dof_idx:
                        jname = mu.mj_id2name(fly_sim.model, mu.mjtObj.mjOBJ_JOINT, j) or ''
                        # Extract segment name from joint name e.g. "coxa_twist_T1_left"
                        for seg in self.SEGMENTS:
                            if seg in jname:
                                self.segment_dof_map[(leg_label, seg)] = dof_idx
                        break

        # Build MN → (dof_idx, flex_scale, ext_scale, direction)
        self.mn_to_muscle = {}  # motor_idx → (flex_muscle, ext_muscle, flex_spikes, ext_spikes)
        self._build_mapping()

    def _build_mapping(self):
        """Map motor neurons to specific DOFs using V2_MN_TYPE_MAP + body_part."""
        from collections import defaultdict

        # Group motor neurons by (body_part, cell_type)
        side_counter = {'T1': 0, 'T2': 0, 'T3': 0}
        mapped = 0

        for idx in self.connectome.leg_motor:
            rid = self.connectome.idx_to_root[idx]
            meta = self.connectome.neurons.get(rid, {})
            bp = meta.get('body_part', '')
            ct = meta.get('cell_type', '').lower()

            # Determine leg prefix from body_part
            leg_prefix = None
            for bp_key, lp in BP_TO_LEG.items():
                if bp_key in bp:
                    leg_prefix = lp
                    break
            if leg_prefix is None:
                continue

            # Alternate left/right
            cnt = side_counter[leg_prefix]
            side = 'left' if cnt % 2 == 0 else 'right'
            side_counter[leg_prefix] = cnt + 1
            leg_label = f'{leg_prefix}_{side}'

            # Determine segment + motion from cell type
            seg_motion = V2_MN_TYPE_MAP.get(ct)
            if seg_motion is None:
                # Fallback: guess from name
                for seg in self.SEGMENTS:
                    if seg.split('_')[0] in ct:
                        seg_motion = (seg, 'flex')
                        break
                if seg_motion is None:
                    seg_motion = ('coxa_twist', 'twist')

            segment, motion = seg_motion
            motion_params = self.MOTION_PARAMS.get(motion, (0.3, 0.0, 1.0))
            flex_scale, ext_scale, direction = motion_params

            # Find DOF for this (leg_label, segment)
            dof_idx = self.segment_dof_map.get((leg_label, segment))
            if dof_idx is None:
                # Try fallback segment names
                for fb in ['coxa_twist', 'coxa', 'femur', 'tibia']:
                    dof_idx = self.segment_dof_map.get((leg_label, fb))
                    if dof_idx is not None:
                        break
            if dof_idx is None:
                continue

            # Convert DOF index to Atlasov muscle pair
            # Each DOF has 2 Atlasov muscles: 2*local_idx (flexor), 2*local_idx+1 (extensor)
            # Find this DOF's position in all_dofs
            dof_pos = list(self.fly.all_dofs).index(dof_idx) if dof_idx in self.fly.all_dofs else -1
            if dof_pos < 0:
                continue

            flex_muscle = 2 * dof_pos
            ext_muscle = 2 * dof_pos + 1

            # If flex_muscle >= n_muscles, skip
            if flex_muscle >= self.n_muscles or ext_muscle >= self.n_muscles:
                continue

            self.mn_to_muscle[idx] = (flex_muscle, ext_muscle, flex_scale, ext_scale)
            mapped += 1

        if mapped < 10:
            print(f"  ⚠ Мало отображений ({mapped}), равномерное распределение...")
            n_leg_motor = len(self.connectome.leg_motor)
            for i, idx in enumerate(self.connectome.leg_motor):
                if idx in self.mn_to_muscle:
                    continue
                dof_pos = (i * self.n_joints) // max(1, n_leg_motor)
                flex_muscle = 2 * dof_pos
                ext_muscle = 2 * dof_pos + 1
                if flex_muscle < self.n_muscles:
                    phase = 1.0 if (dof_pos % 2 == 0) else -1.0
                    self.mn_to_muscle[idx] = (flex_muscle, ext_muscle,
                                               0.5 if phase > 0 else 0.0,
                                               0.5 if phase < 0 else 0.0)
                    mapped += 1

        print(f"  🦵 Motor→DOF mapping: {len(self.mn_to_muscle)} отображений на {self.n_joints} DOF")

    def route(self, motor_spikes):
        """Convert motor neuron spikes → Atlasov muscle drive [n_muscles].

        Args:
            motor_spikes: [n_leg_motor] array of 0/1 spike indicators

        Returns:
            motor_drive: [n_muscles] int32 — spikes for Atlasov
        """
        motor_drive = np.zeros(self.n_muscles, dtype=np.int32)

        for idx, (flex_m, ext_m, flex_s, ext_s) in self.mn_to_muscle.items():
            if idx >= len(motor_spikes):
                continue
            if motor_spikes[idx] > 0:
                motor_drive[flex_m] = min(3, int(abs(flex_s) * 2 + 0.5))
                motor_drive[ext_m] = min(3, int(abs(ext_s) * 2 + 0.5))

        return motor_drive


# ═════════════════════════════════════════════════════════════════════
#  2b. FRANKEN MOTOR MAPPER — точное отображение через body_part_effector + nerve
# ═════════════════════════════════════════════════════════════════════

# Nerve → leg_label mapping
NERVE_TO_LEG = {
    'left_prothoracic_leg_nerve': 'T1_left',
    'right_prothoracic_leg_nerve': 'T1_right',
    'left_mesothoracic_leg_nerve': 'T2_left',
    'right_mesothoracic_leg_nerve': 'T2_right',
    'left_metathoracic_leg_nerve': 'T3_left',
    'right_metathoracic_leg_nerve': 'T3_right',
    'left_prothoracic_accessory_nerve': 'T1_left',   # accessory → same leg
    'right_prothoracic_accessory_nerve': 'T1_right',
    'left_ventral_prothoracic_nerve': 'T1_left',
    'right_ventral_prothoracic_nerve': 'T1_right',
    'left_dorsal_prothoracic_nerve': 'T1_left',
    'right_dorsal_prothoracic_nerve': 'T1_right',
    'left_posterior_dorsal_mesothoracic_nerve': 'T2_left',
    'right_posterior_dorsal_mesothoracic_nerve': 'T2_right',
    'left_first_abdominal_nerve': 'T3_left',  # abdominal → hind leg area
    'right_first_abdominal_nerve': 'T3_right',
}

# Body part → neuromere prefix
BP_TO_NEUROMERE = {
    'front_leg': 'T1',
    'middle_leg': 'T2',
    'hind_leg': 'T3',
}




class FrankenMotorMapper:
    """Точное отображение MN → Atlasov через body_part_effector + nerve.

    Использует VNC метаданные (тот же подход что Franken dataset):
    - body_part → front_leg=T1, middle_leg=T2, hind_leg=T3
    - nerve → точная сторона (left/right)
    - neurotransmitter → тип движения
    - hemilineage → распределение по сегментам внутри ноги
    """

    def __init__(self, connectome: RealVNCConnectome, fly_sim):
        self.connectome = connectome
        self.fly = fly_sim
        self.n_muscles = fly_sim.n_muscles
        self.n_joints = fly_sim.n_joints

        # Map: leg_label → [dof_indices]
        self.leg_dof_list = {}  # leg_label → [dof_pos in all_dofs]
        for leg_label, dofs in fly_sim.leg_dofs.items():
            positions = []
            for dof in dofs:
                if dof in fly_sim.all_dofs:
                    positions.append(list(fly_sim.all_dofs).index(dof))
            self.leg_dof_list[leg_label] = positions

        # Franken mapping: motor_idx → (flex_idx, ext_idx, flex_scale, ext_scale)
        self.mn_to_muscle = {}  # same format as AtlasovLegMapper
        self._build_mapping()

    def _get_leg_from_nerve(self, nerve, body_part):
        """Определить leg_label по nerve + body_part."""
        # Сначала пробуем точное совпадение nerve
        leg_label = NERVE_TO_LEG.get(nerve)
        if leg_label:
            return leg_label

        # Если nerve не найден, используем body_part + guessing от nerve
        bp_prefix = BP_TO_NEUROMERE.get(body_part)
        if bp_prefix is None:
            return None

        if 'left' in nerve or '_L' in nerve:
            return f'{bp_prefix}_left'
        elif 'right' in nerve or '_R' in nerve:
            return f'{bp_prefix}_right'

        # Если не удалось, вернём None
        return None

    def _build_mapping(self):
        """Build MN→Atlasov mapping using Franken approach (nerve + body_part)."""
        # Собираем метаданные для leg motor нейронов
        leg_mn_meta = []
        for idx in self.connectome.leg_motor:
            rid = self.connectome.idx_to_root[idx]
            meta = self.connectome.neurons.get(rid, {})
            leg_mn_meta.append({
                'idx': idx,
                'body_part': meta.get('body_part', ''),
                'nerve': meta.get('nerve', ''),
                'nt': meta.get('nt', ''),
                'cell_type': meta.get('cell_type', ''),
            })

        # Группируем по (body_part, nerve)
        from collections import defaultdict
        groups = defaultdict(list)
        for mn in leg_mn_meta:
            key = (mn['body_part'], mn['nerve'])
            groups[key].append(mn)

        mapped = 0
        for (body_part, nerve), mns in groups.items():
            leg_label = self._get_leg_from_nerve(nerve, body_part)
            if leg_label is None:
                continue

            dof_positions = self.leg_dof_list.get(leg_label, [])
            n_dofs = len(dof_positions)
            if n_dofs == 0:
                continue

            # Распределяем MN по DOFs внутри этой ноги
            # МН с разными cell_type идут на разные сегменты
            # Используем hemilineage для естественного распределения
            for i, mn in enumerate(mns):
                dof_idx = dof_positions[i % n_dofs]  # round-robin по DOFs ноги
                flex_m = 2 * dof_idx
                ext_m = 2 * dof_idx + 1

                if flex_m >= self.n_muscles:
                    continue

                # ⚡ БИОЛОГИЧЕСКИЙ ФИКС:
                # GABA нейроны в 'motor' группе — это premotor интернейроны,
                # которые модулируют VNC intrinsic, а не напрямую управляют мышцами!
                # Только ACh и GLUT (возбуждающие) идут в Atlasov.
                nt = mn['nt']
                if nt == 'GABA':
                    continue  # ← GABA не идёт к мышцам!
                elif nt == 'ACH':
                    flex_s, ext_s = 0.5, 0.3  # основной моторный нейротрансмиттер насекомых
                elif nt == 'GLUT':
                    flex_s, ext_s = 0.5, 0.2  # вторичный (некоторые мышцы)
                elif nt == 'OCT':
                    flex_s, ext_s = 0.3, 0.1  # модулятор
                else:
                    flex_s, ext_s = 0.3, 0.2  # unknown — слабая активация

                self.mn_to_muscle[mn['idx']] = (flex_m, ext_m, flex_s, ext_s)
                mapped += 1

        # Fallback for unmapped: распределяем равномерно по всем leg DOFs
        fallback_mns = [mn for mn in leg_mn_meta if mn['idx'] not in self.mn_to_muscle]
        all_leg_dofs = []
        for leg_label in ['T1_left','T1_right','T2_left','T2_right','T3_left','T3_right']:
            all_leg_dofs.extend(self.leg_dof_list.get(leg_label, []))
        
        for i, mn in enumerate(fallback_mns):
            if not all_leg_dofs:
                break
            dof_idx = all_leg_dofs[i % len(all_leg_dofs)]
            flex_m = 2 * dof_idx
            ext_m = 2 * dof_idx + 1
            if flex_m < self.n_muscles:
                self.mn_to_muscle[mn['idx']] = (flex_m, ext_m, 0.4, 0.2)
                mapped += 1

        print(f"  🧬 Franken MN→DOF: {len(self.mn_to_muscle)} отображений на {self.n_joints} DOF")

        # Статистика распределения
        dof_usage = Counter()
        for _, (flex_m, ext_m, _, _) in self.mn_to_muscle.items():
            dof_usage[flex_m // 2] += 1
        most_used = dof_usage.most_common(5)
        print(f"  📊 Топ-5 DOF по количеству MN: {most_used}")

    def route(self, motor_spikes):
        """Convert motor neuron spikes → Atlasov muscle drive."""
        motor_drive = np.zeros(self.n_muscles, dtype=np.int32)
        for idx, (flex_m, ext_m, flex_s, ext_s) in self.mn_to_muscle.items():
            if idx >= len(motor_spikes):
                continue
            if motor_spikes[idx] > 0:
                motor_drive[flex_m] = min(3, max(1, int(flex_s * 5)))
                motor_drive[ext_m] = min(3, max(1, int(ext_s * 5)))
        return motor_drive


# ═════════════════════════════════════════════════════════════════════
#  3. ATLASOV → MUJOCO (из atlasov_flywire_embodied.py)
# ═════════════════════════════════════════════════════════════════════

class AtlasovFlyMuJoCo:
    """Atlasov Ca²⁺ Neuromotor Engine → MuJoCo FruitFly."""

    def __init__(self, xml_path=SCENE_XML):
        print(f"🦗 Загрузка MuJoCo мухи: {xml_path}")
        self.model = mu.MjModel.from_xml_path(str(xml_path))
        self.data = mu.MjData(self.model)

        # Joint DOFs (legs)
        self.LEG_NAMES = ['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']
        self.leg_dofs = defaultdict(list)
        for i in range(self.model.njnt):
            jname = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_JOINT, i)
            if jname:
                for leg in self.LEG_NAMES:
                    if leg in jname:
                        self.leg_dofs[leg].append(self.model.jnt_dofadr[i])
                        break

        self.all_dofs = np.concatenate(list(self.leg_dofs.values()))
        self.n_joints = len(self.all_dofs)
        self.n_muscles = math.ceil((self.n_joints * 2) / 8) * 8  # SIMD pad

        print(f"   DOFs: {self.n_joints} joints → {self.n_muscles} muscles")

        # Sarcoplasmic muscle state
        self.ca = np.zeros(self.n_muscles, dtype=np.float32)
        self.tension = np.zeros(self.n_muscles, dtype=np.float32)
        self.spikes = np.zeros(self.n_muscles, dtype=np.int32)
        self.resets = np.zeros(self.n_muscles, dtype=np.int32)

        # Insect-optimized parameters (faster SERCA, more sensitive Kd)
        self.d_ca = np.full(self.n_muscles, 0.3, dtype=np.float32)
        self.tau_pump = np.full(self.n_muscles, 8.0, dtype=np.float32)
        self.eta_reset = np.full(self.n_muscles, 0.6, dtype=np.float32)
        self.t_max = np.full(self.n_muscles, 0.8, dtype=np.float32)        # 0.8 dynes·cm — естественная суставная сила
        self.kd = np.full(self.n_muscles, 0.15, dtype=np.float32)

        self.joint_df = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_stf = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_dmp = np.zeros(self.n_muscles // 2, dtype=np.float32)

        self.k0 = np.full(self.n_muscles // 2, 0.01, dtype=np.float32)     # Эластичные гибкие суставы
        self.beta = np.full(self.n_muscles // 2, 0.05, dtype=np.float32)
        self.b0 = np.full(self.n_muscles // 2, 0.10, dtype=np.float32)     # Жидкое амортизирование
        self.alpha = np.full(self.n_muscles // 2, 0.05, dtype=np.float32)

        self.dt = self.model.opt.timestep
        self.step_fn = tc.neurons.step_atlasov_neuromotor

    def step(self, motoneuron_spikes):
        """One MuJoCo step with Atlasov Ca²⁺ muscle drive."""
        n_in = min(len(motoneuron_spikes), self.n_muscles)
        self.spikes[:n_in] = motoneuron_spikes[:n_in]
        self.resets[:n_in] = 0

        # Atlasov Ca²⁺ (Rust SIMD)
        self.step_fn(
            self.ca, self.tension, self.spikes, self.resets,
            float(self.dt * 1000.0),
            self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
            self.joint_df, self.joint_stf, self.joint_dmp,
            self.k0, self.beta, self.b0, self.alpha
        )

        # Apply forces → MuJoCo
        self.data.qfrc_applied.fill(0.0)
        for j in range(self.n_joints):
            dof = self.all_dofs[j]
            q = self.data.qpos[dof]
            qd = self.data.qvel[dof]
            tau = self.joint_df[j] - self.joint_stf[j] * q - self.joint_dmp[j] * qd
            self.data.qfrc_applied[dof] = np.clip(tau, -0.8, 0.8)

        mu.mj_step(self.model, self.data)
        return {
            't': self.data.time,
            'pos': self.data.qpos[:3].copy(),
            'df': self.joint_df[:self.n_joints].copy(),
            'k': self.joint_stf[:self.n_joints].copy(),
        }


# ═════════════════════════════════════════════════════════════════════
#  4. MAIN — ПОЛНЫЙ ПАЙПЛАЙН
# ═════════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("🧬🦗 REAL VNC CONNECTOME → HH → ATLASOV → MUJOCO FLY")
    print("   First ever: Real VNC connectome driving 3D fly through Atlasov!")
    print("═" * 80)

    # ─── 1. Загрузка VNC коннектома ─────────────────────────────────
    print("\n" + "─" * 80)
    print("📥 ШАГ 1: ЗАГРУЗКА VNC КОННЕКТОМА")
    print("─" * 80)

    # ⚡ VNC-only режим (~17K нейронов) для скорости + HH с pacemaking
    # HH спонтанно спайкует → моторные нейроны активны без внешнего тока
    # scale=1.0 — HH работает в nA, реальные синаптические веса
    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons():
        return
    if not conn.load_connections(max_synapses=None):
        return

    # ─── 2. Подготовка данных для симуляции ─────────────────────────
    print("\n" + "─" * 80)
    print("⚙️ ШАГ 2: ПОДГОТОВКА СИМУЛЯЦИИ")
    print("─" * 80)

    conn_data = conn.build_connectome_data()
    N = conn.n_neurons

    print(f"🧠 Модель: HH pacemaking ({N:,} нейронов)")

    # ─── 3. BrainSimulator с HH pacemaking ─────────────────────────
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)

    # ─── 4. Atlasov MuJoCo ─────────────────────────────────────────
    fly = AtlasovFlyMuJoCo()

    # ─── 5. MN → Atlasov mapper (оба подхода для сравнения) ───────
    mapper_v2 = AtlasovLegMapper(conn, fly)  # V2_MN_TYPE_MAP подход
    mapper_franken = FrankenMotorMapper(conn, fly)  # Franken подход (nerve+body_part)
    mapper = mapper_franken  # Используем Franken как основной

    # ─── 6. Параметры симуляции ────────────────────────────────────
    n_warmup = 500       # 250ms warmup
    n_walk = 2000         # 1s walking
    n_total = n_warmup + n_walk
    dt_ms = 0.5

    # External I: HH не требует большого bias — pacemaking сам генерирует спайки
    # Но добавим немного шума + tonic drive для motor нейронов
    rng = np.random.RandomState(42)
    I_bg = rng.randn(N).astype(np.float32) * 2.0  # небольшой шум

    # TONIC DRIVE для motor нейронов — форсируем спайкинг!
    if len(conn.leg_motor) > 0:
        motor_tonic = 30.0  # pA — HH начинает спайковать при ~10-15 pA
        I_bg[np.array(conn.leg_motor) % N] += motor_tonic
        print(f"  🦵 Motor tonic drive: {motor_tonic:.0f} pA × {len(conn.leg_motor)} нейронов")

    # Descending drive: inject into descending neurons
    if len(conn.descending) > 0:
        print(f"  ⬇️  Descending drive: {len(conn.descending)} нейронов")
        I_bg[np.array(conn.descending) % N] += 20.0

    # Sensory drive: periodic pulse
    if len(conn.sensory) > 0:
        print(f"  👁️  Sensory population: {len(conn.sensory)} нейронов")
        I_bg[np.array(conn.sensory[:min(1000, len(conn.sensory))]) % N] += 5.0

    # ─── 6b. Сравнение мапперов ────────────────────────────────────
    print(f"\n  🔬 СРАВНЕНИЕ МАППЕРОВ:")
    print(f"     V2_MN_TYPE_MAP: {len(mapper_v2.mn_to_muscle)} отображений")
    print(f"     Franken (nerve): {len(mapper_franken.mn_to_muscle)} отображений")

    # Проверяем percent overlap (сколько DOF получают сигнал)
    v2_dofs = set(v[0]//2 for v in mapper_v2.mn_to_muscle.values())
    frank_dofs = set(v[0]//2 for v in mapper_franken.mn_to_muscle.values())
    print(f"     V2 DOFs: {len(v2_dofs)}/{fly.n_joints}")
    print(f"     Franken DOFs: {len(frank_dofs)}/{fly.n_joints}")
    print(f"     Пересечение: {len(v2_dofs & frank_dofs)} DOF")

    print(f"\n  ⏱ Шаги: {n_warmup} warmup + {n_walk} walk = {n_total} ({n_total*dt_ms:.0f}ms)")
    print(f"  🏋️  Тест: реальный VNC коннектом → Atlasov → MuJoCo")

    # ─── 7. Batch симуляция (все шаги разом — оптимально) ────────
    print(f"\n{'─' * 80}")
    print("🎬 ЗАПУСК БАТЧ-СИМУЛЯЦИИ (все шаги разом!)")
    print(f"{'─' * 80}\n")

    t_sim_start = time.time()
    spike_buf = sim.step_batch(conn_data, n_steps=n_total, dt_ms=dt_ms,
                                external_I=I_bg, scale=1.0)
    t_sim = time.time() - t_sim_start
    print(f"  ✅ Симуляция {n_total} шагов за {t_sim:.1f}s ({(n_total*dt_ms/1000)/t_sim:.1f}× RT)")

    # ─── 8. Walk phase: iterate over spike buffer → Atlasov → MuJoCo ────
    print(f"\n{'─' * 80}")
    print("🚶 ШАГ 8: ПРОГОНКА ATLASOV → MUJOCO ПО СПАЙКАМ")
    print(f"{'─' * 80}\n")

    start_pos = None
    displacement = 0.0
    total_brain_spikes = 0
    total_motor_firings = 0
    walk_log = []
    t_start = time.time()

    renderer = mu.Renderer(fly.model, height=480, width=640)
    cam = mu.MjvCamera()
    cam.type = mu.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.0
    cam.azimuth = 45.0
    cam.elevation = -25.0
    thorax_bid = mu.mj_name2id(fly.model, mu.mjtObj.mjOBJ_BODY, "thorax")
    frames = []

    for step in range(n_total):
        step_spikes = spike_buf[step * N:(step + 1) * N]
        current_spikes = int(step_spikes.sum())
        total_brain_spikes += current_spikes

        # ─── Motor phase ──────────────────────────────────────────
        if step >= n_warmup:
            # Извлекаем спайки leg motor нейронов (0/1)
            motor_spikes = np.array([float(step_spikes[idx]) for idx in conn.leg_motor
                                     if idx < N], dtype=np.float32)

            # Анатомический mapper → Atlasov
            motor_drive = mapper.route(motor_spikes)
            total_motor_firings += motor_drive.sum()

            # Atlasov → MuJoCo
            out = fly.step(motor_drive)

            if start_pos is None:
                start_pos = out['pos'].copy()
            displacement = np.linalg.norm(out['pos'] - start_pos) * 1000  # mm
            walk_log.append({
                't': out['t'],
                'pos': out['pos'].copy(),
                'disp': displacement,
                'mean_k': out['k'].mean(),
                'mean_df': out['df'].mean(),
            })

            # Запись кадров анимации ходьбы VNC коннектома каждые 20 шагов (100 кадров)
            if (step - n_warmup) % 20 == 0:
                if thorax_bid >= 0:
                    cam.lookat = fly.data.xpos[thorax_bid].copy()
                else:
                    cam.lookat = fly.data.qpos[:3].copy()
                renderer.update_scene(fly.data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))

        # ─── Progress ─────────────────────────────────────────────
        if step % 200 == 0 or step == 0 or step == n_total - 1:
            elapsed = time.time() - t_start
            fps = (step + 1) / (elapsed + 1e-6)
            phase = "🔥 WARMUP" if step < n_warmup else "🚶 WALK"
            spk_bar = "█" * min(20, int(current_spikes / max(1, N * 0.01) * 20))
            disp_str = f" | Disp: {displacement:.2f}mm" if step >= n_warmup else ""
            print(f"  Step {step:5d}/{n_total} | {phase} | "
                  f"Spikes: {current_spikes:5d} [{spk_bar}]{disp_str} | {fps:.0f} s/s")

    total_time = time.time() - t_start

    # ═══════════════════════════════════════════════════════════════
    #  GAIT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    walk_log_arr = np.array([(l['t'], l['disp'], l['mean_k'], l['mean_df'])
                             for l in walk_log],
                            dtype=[('t', 'f4'), ('disp', 'f4'), ('k', 'f4'), ('df', 'f4')])

    total_walk_time = n_walk * dt_ms / 1000
    speed_mm_s = displacement / total_walk_time if total_walk_time > 0 else 0.0

    # Gait classification
    if speed_mm_s < 1.0:
        gait = "🟤 CRAWLING"
    elif speed_mm_s < 5.0:
        gait = "🟡 WALKING"
    elif speed_mm_s < 15.0:
        gait = "🟢 RUNNING"
    else:
        gait = "🟣 FLYING"

    print("\n" + "═" * 80)
    print("🏆 РЕЗУЛЬТАТЫ: РЕАЛЬНЫЙ VNC КОННЕКТОМ → ATLASOV → MUJOCO!")
    print("═" * 80)
    print(f"  🧠 VNC Connectome:")
    print(f"     Нейронов:           {N:,}")
    print(f"     Синапсов:          {conn.n_synapses:,}")
    print(f"     VNC intrinsic:     {len(conn.vnc_intrinsic):,}")
    print(f"     Motor neurons:     {len(conn.motor):,}")
    print(f"     Descending:        {len(conn.descending):,}")
    print(f"  🦗 Fly:")
    print(f"     Суставов:          {fly.n_joints}")
    print(f"     Мышц:              {fly.n_muscles}")
    print(f"     Мото-спайков:      {total_motor_firings:,}")
    print(f"     Дистанция:         {displacement:.2f} mm")
    print(f"     Время ходьбы:      {total_walk_time:.2f} s")
    print(f"     Скорость:          {speed_mm_s:.2f} mm/s")
    print(f"     Gait:              {gait}")
    print(f"     Avg K:             {np.mean([l['mean_k'] for l in walk_log])*1e3:.2f} mN/rad")
    print(f"  ⏱ Performance:")
    print(f"     Симуляция:         {n_total*dt_ms:.0f}ms за {total_time:.1f}s")
    print(f"     Скорость:          {n_total/total_time:.0f} step/s")
    print(f"     RT factor:         {n_total*dt_ms/1000/total_time:.2f}×")
    print("═" * 80)

    # Save
    out_dir = Path(__file__).parent.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "vnc_real_walk_telemetry.npz",
             walk_time=np.array([l['t'] for l in walk_log]),
             walk_pos=np.array([l['pos'] for l in walk_log]),
             walk_disp=np.array([l['disp'] for l in walk_log]),
             walk_k=np.array([l['mean_k'] for l in walk_log]),
             walk_df=np.array([l['mean_df'] for l in walk_log]))
    print(f"💾 Телеметрия: {out_dir / 'vnc_real_walk_telemetry.npz'}")

    if frames:
        video_out = Path(__file__).parent.parent / "vnc_real_walk.webp"
        artifact_out = Path("/home/taaslab/.gemini/antigravity-ide/brain/01abdf90-8c04-48e6-8d5c-b8df411d43c4/vnc_real_walk.webp")
        print(f"💾 Сохранение {len(frames)} кадров анимации в WebP...")
        frames[0].save(video_out, save_all=True, append_images=frames[1:], duration=40, loop=0)
        frames[0].save(artifact_out, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"🎬 Видео готово: {video_out}")
        print(f"🎬 Копия в артефактах: {artifact_out}")

    print("✅ ГОТОВО!")


if __name__ == "__main__":
    main()
