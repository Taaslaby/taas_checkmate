#!/usr/bin/env python3
"""
MANC Motor Neurons → MuJoCo flybody Actuators Mapping.

Для каждой ноги (T1/T2/T3, left/right) сматчиваем моторные нейроны
из MANC с актуаторами из fruitfly.xml.

Запуск:
    cd taas_checkmate
    python3 fly/scripts/mapping_mn_to_actuators.py

Зависимости: pip install pandas pyarrow
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path

import pandas as pd

DLOAD = Path(os.path.expanduser("~/Загрузки"))
FLY_XML = "models/fruitfly.xml"
FLY_DIR = Path(__file__).parent.parent  # fly/


# ═══════════════════════════════════════════════════════════
#  1. Парсинг MuJoCo актуаторов
# ═══════════════════════════════════════════════════════════

def parse_leg_actuators(xml_path: str) -> list[dict]:
    """Извлечь все leg-актуаторы из fruitfly.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    actuators = []

    for act in root.findall('.//actuator//general'):
        name = act.get('name', '')
        joint = act.get('joint', '')
        tendon = act.get('tendon', '')

        # Только ноги (не голова, крылья, антенны, брюшко)
        if not any(x in name for x in ['_T1_', '_T2_', '_T3_']):
            continue

        # Определяем сегмент, ногу, сторону
        # name format: <segment>_[action_]<leg>_<side>
        # e.g. "coxa_abduct_T1_left", "femur_T1_left", "tibia_T1_left"
        seg = name.split('_')[0]  # coxa, femur, tibia, tarsus, tarsus2

        # Определяем leg (T1, T2, T3)
        leg = '?'
        for l in ['T1', 'T2', 'T3']:
            if f'_{l}_' in name:
                leg = l
                break

        # Определяем side (left, right)
        side = 'right' if '_right' in name else 'left'

        # Тип движения
        if 'abduct' in name:
            motion = 'abduct/adduct'
        elif 'twist' in name:
            motion = 'twist/rotate'
        elif 'tarsus2' in name:
            motion = 'tarsus_flex'
            seg = 'tarsus2'
        else:
            motion = 'extend/flex'

        actuators.append({
            'name': name,
            'segment': seg,
            'leg': leg,
            'side': side,
            'motion': motion,
            'joint': joint,
            'tendon': tendon,
        })

    return actuators


# ═══════════════════════════════════════════════════════════
#  2. Загрузка MANC моторных нейронов
# ═══════════════════════════════════════════════════════════

def load_motor_neurons() -> pd.DataFrame:
    """Загрузить моторные нейроны из MANC."""
    path = DLOAD / "v1.0_manc-v1.0-neuron-properties.feather"
    if not path.exists():
        alt = DLOAD / "v1.0_manc-v1.0-neuron-properties(1).feather"
        if alt.exists():
            path = alt
        else:
            print(f"✗ MANC not found at {path}")
            return pd.DataFrame()

    df = pd.read_feather(path)
    mot = df[df['class'] == 'motor neuron'].copy()
    print(f"  Загружено {len(mot)} моторных нейронов")

    return mot


# ═══════════════════════════════════════════════════════════
#  3. Heuristic mapping: MN type → segment/motion
# ═══════════════════════════════════════════════════════════

# Биологический моторный тип → сегмент ноги + тип движения
MN_TYPE_MAP = {
    # Coxa (CMG) — тазобедренный
    'Sternal anterior rotator MN':     ('coxa', 'twist/rotate'),
    'Sternal posterior rotator MN':    ('coxa', 'twist/rotate'),
    'Pleural remotor/abductor MN':     ('coxa', 'abduct/adduct'),
    'Sternotrochanter MN':             ('coxa', 'extend/flex'),
    'Tergotr. MN':                     ('coxa', 'extend/flex'),
    'Pleural rotator MN':              ('coxa', 'twist/rotate'),

    # Femur (TrFe) — бедренный
    'Tr flexor MN':                    ('femur', 'extend/flex'),
    'Tr extensor MN':                  ('femur', 'extend/flex'),
    'Acc. tr flexor MN':               ('femur', 'extend/flex'),
    'Fe reductor MN':                  ('femur', 'twist/rotate'),
    'ltm2-femur MN':                   ('femur', 'extend/flex'),

    # Tibia (Ti) — голень
    'Ti flexor MN':                    ('tibia', 'extend/flex'),
    'Ti extensor MN':                  ('tibia', 'extend/flex'),
    'Acc. ti flexor MN':               ('tibia', 'extend/flex'),
    'ltm1-tibia MN':                   ('tibia', 'extend/flex'),

    # Tarsus (Ta) — лапка
    'Ta depressor MN':                 ('tarsus', 'extend/flex'),
    'Uncrossed ta levator MN':         ('tarsus', 'extend/flex'),
    'Crossed ta levator MN':           ('tarsus', 'extend/flex'),

    # Claw — коготь
    'ltm MN':                          ('tarsus2', 'extend/flex'),
    'Claw MN':                         ('tarsus2', 'extend/flex'),
}

# Лег в аббревиатуру для поиска в outputRois
LEG_ROI_MAP = {
    'T1': 'LegNp(T1)',
    'T2': 'LegNp(T2)',
    'T3': 'LegNp(T3)',
}

SIDE_ROI_MAP = {
    'left': '(L)',
    'right': '(R)',
}


def match_mn_to_actuator(mn_type: str, mn_rois: str,
                          leg: str, side: str) -> str | None:
    """Match motor neuron type + leg/side → actuator name."""
    if mn_type not in MN_TYPE_MAP:
        return None

    segment, motion = MN_TYPE_MAP[mn_type]

    # Ищем подходящий актуатор для этого сегмента + ноги + стороны
    # Актуаторы в XML: <segment>_[action_]<leg>_<side>
    # e.g. coxa_abduct_T1_left, femur_T1_left, tibia_T1_left

    candidates = []
    for act_name in ACTUATOR_BY_LEG_SIDE.get((leg, side), []):
        act_seg = act_name.split('_')[0]
        if act_seg == segment or (segment == 'tarsus2' and act_seg == 'tarsus' and 'tarsus2' in act_name):
            # Check sub-segment for multi-actuator segments
            if 'abduct' in motion and 'abduct' in act_name:
                candidates.append(act_name)
            elif 'twist' in motion and 'twist' in act_name:
                candidates.append(act_name)
            elif 'extend/flex' in motion and 'abduct' not in act_name and 'twist' not in act_name:
                candidates.append(act_name)

    # Prefer the actuator with matching motion type
    for c in candidates:
        if 'abduct' in c and 'abduct' in motion:
            return c
        if 'twist' in c and 'twist' in motion:
            return c
        if 'extend/flex' in motion and 'abduct' not in c and 'twist' not in c:
            # For tarsus2 (claw), must match tarsus2_* specifically
            if segment == 'tarsus2' and 'tarsus2' not in c:
                continue
            # For femur/tibia/tarsus, match non-{abduct,twist} actuator
            return c

    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════
#  4. Main mapping logic
# ═══════════════════════════════════════════════════════════

def build_actuator_index(actuators: list[dict]):
    """Build lookup: (leg, side) → actuator names."""
    idx = {}
    for a in actuators:
        key = (a['leg'], a['side'])
        if key not in idx:
            idx[key] = []
        idx[key].append(a['name'])
    return idx


def classify_mn_leg_side(mn: pd.Series) -> list[tuple[str, str]]:
    """Определить к какой ноге относится моторный нейрон по outputRois."""
    rois = str(mn['outputRois'])
    legs_sides = []
    for leg, roi in LEG_ROI_MAP.items():
        for side, sroi in SIDE_ROI_MAP.items():
            if roi + sroi in rois:
                legs_sides.append((leg, side))
    return legs_sides if legs_sides else [('unknown', 'unknown')]


def main():
    print("═" * 60)
    print("  MANC Motor Neurons → MuJoCo Actuators Mapping")
    print("═" * 60)

    # 1. Парсим актуаторы
    print("\n─── 1. MuJoCo Leg Actuators ───")
    xml_path = str(FLY_DIR / FLY_XML)
    act_list = parse_leg_actuators(xml_path)
    print(f"  Найдено {len(act_list)} leg-актуаторов\n")

    for a in act_list:
        print(f"  {a['name']:40s}  [{a['leg']} {a['side']:5s}]  {a['segment']:7s}  {a['motion']}")

    global ACTUATOR_BY_LEG_SIDE
    ACTUATOR_BY_LEG_SIDE = build_actuator_index(act_list)

    # 2. Загружаем MANC моторные нейроны
    print("\n─── 2. MANC Motor Neurons ───")
    mot = load_motor_neurons()
    if len(mot) == 0:
        return

    # 3. Строим маппинг
    print("\n─── 3. Mapping MN → Actuator ───")
    print(f"\n{'MN Type':45s} {'BodyId':>7s} {'Leg':4s} {'Side':5s} {'→ Actuator':40s}")
    print('-' * 110)

    mapped_count = 0
    unmatched_types = set()

    for _, mn in mot.iterrows():
        mn_type = mn.get('type', '?')
        mn_rois = str(mn.get('outputRois', ''))

        legs_sides = classify_mn_leg_side(mn)

        for leg, side in legs_sides:
            actuator = match_mn_to_actuator(mn_type, mn_rois, leg, side)
            bid = int(mn['bodyId'])

            if actuator:
                mapped_count += 1
                print(f"  {mn_type:45s} {bid:>7} {leg:4s} {side:5s} → {actuator:40s}")
            else:
                if mn_type not in unmatched_types:
                    unmatched_types.add(mn_type)

    print(f"\n  Сматчено: {mapped_count} из {len(mot)} моторных нейронов")

    # 4. Итоговая таблица: тип MN → актуатор(ы)
    print("\n─── 4. Итоговое соответствие ───")
    print(f"\n{'MN Type':45s} {'→ Actuator(s)':50s}")
    print('-' * 95)

    type_to_act = {}
    for _, mn in mot.iterrows():
        mn_type = mn.get('type', '?')
        legs_sides = classify_mn_leg_side(mn)
        for leg, side in legs_sides:
            actuator = match_mn_to_actuator(mn_type, '', leg, side)
            if actuator:
                if mn_type not in type_to_act:
                    type_to_act[mn_type] = set()
                type_to_act[mn_type].add(actuator)

    for mn_type in sorted(type_to_act.keys()):
        acts = ', '.join(sorted(type_to_act[mn_type]))
        print(f"  {mn_type:45s} → {acts:50s}")

    # 5. Non-matched types
    if unmatched_types:
        print(f"\n  ❌ Типы без маппинга ({len(unmatched_types)}):")
        for t in sorted(unmatched_types):
            count = len(mot[mot['type'] == t])
            print(f"    {t:45s} ({count} нейронов)")

    # 6. Сводка по ноге
    print("\n─── 5. Сводка по ногам ───")
    for leg in ['T1', 'T2', 'T3']:
        for side in ['left', 'right']:
            leg_mns = mot[mot['outputRois'].apply(
                lambda x: f'LegNp({leg})' + ('(L)' if side == 'left' else '(R)') in str(x)
            )]
            leg_acts = [a for a in act_list if a['leg'] == leg and a['side'] == side]
            print(f"  {leg} {side:5s}: {len(leg_mns):3d} MN → {len(leg_acts)} actuators")


if __name__ == "__main__":
    main()
