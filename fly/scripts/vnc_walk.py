#!/usr/bin/env python3
"""
VNC (Ventral Nerve Cord) — ходьба мухи через реальный MANC коннектом.

Вместо искусственных синусоид (CPG), ногами управляет real MANC connectome:
  DN input → MANC direct weights → MN activation → MuJoCo актуаторы

Запуск:
  cd taas_checkmate
  python3 fly/scripts/vnc_walk.py

Управление: 1/2=freq  3/4=amp  R=reset
Зависимости: pip install mujoco numpy glfw pandas pyarrow
"""

import sys
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import glfw

import os

DLOAD = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))
FLY_DIR = Path(__file__).parent.parent
XML_PATH = FLY_DIR / "models" / "scene.xml"

# ═══════════════════════════════════════════════════════════════
#  Анатомия
# ═══════════════════════════════════════════════════════════════

LEGS = [("L1","T1_left"),("R1","T1_right"),
        ("L2","T2_left"),("R2","T2_right"),
        ("L3","T3_left"),("R3","T3_right")]

LEG_ROIS = {
    "T1_left":("LegNp(T1)","(L)"), "T1_right":("LegNp(T1)","(R)"),
    "T2_left":("LegNp(T2)","(L)"), "T2_right":("LegNp(T2)","(R)"),
    "T3_left":("LegNp(T3)","(L)"), "T3_right":("LegNp(T3)","(R)"),
}

SEGMENTS = ["coxa_abduct","coxa_twist","coxa",
            "femur_twist","femur","tibia",
            "tarsus","tarsus2"]

# ─── Non-leg body parts ────────────────────────────
HEAD_SEGMENTS = ["head_abduct","head_twist","head"]
ANTENNA_SEGMENTS = [
    "antenna_abduct_left","antenna_twist_left","antenna_left",
    "antenna_abduct_right","antenna_twist_right","antenna_right",
]
ABDOMEN_SEGMENTS = ["abdomen_abduct","abdomen"]
WING_SEGMENTS = [
    "wing_yaw_left","wing_roll_left","wing_pitch_left",
    "wing_yaw_right","wing_roll_right","wing_pitch_right",
]

# New ROIs for non-leg motor outputs
# В MANC: ANM (abdomen), WTCT/HTCT/AMNP (wings/halteres), CV (neck/head)
TRIPOD_A = {"T1_left","T2_right","T3_left"}
TRIPOD_B = {"T1_right","T2_left","T3_right"}


# ═══════════════════════════════════════════════════════════════
#  V2 Walking: V2VNCController — чистый нейронный drive без CPG
# ═══════════════════════════════════════════════════════════════
#  Ритм ходьбы генерируется VNC коннектомом через LIF-мозг.
#  Никаких hand-coded синусоид — только MN rates из real коннектома.


# ═══════════════════════════════════════════════════════════════
#  V2 Cell Type Map — Primary Cell Type → (segment, motion)
# ═══════════════════════════════════════════════════════════════
#  Маппинг названий мышц из V2 neurons.csv на MuJoCo сегменты.
#  Используется V2VNCController для прямого MN → actuator контроля.

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
    'long_tendon_muscle':         ('femur', 'extend'),  # ltm2-femur
    'tibia_flexor':               ('tibia', 'flex'),
    'tibia_extensor':             ('tibia', 'extend'),
    'accessory_tibia_flexor':     ('tibia', 'flex'),
    'tarsus_depressor':           ('tarsus', 'flex'),
    'tarsus_levator':             ('tarsus', 'flex'),  # uncrossed/crossed ta levator
    'claw':                       ('tarsus2', 'flex'),
    'sternal_adductor':           ('coxa', 'twist'),
}

# Sub Class → leg label
V2_LEG_MAP = {
    'front_leg_motor_neuron': 'T1',
    'middle_leg_motor_neuron': 'T2',
    'hind_leg_motor_neuron': 'T3',
}


class V2VNCController:
    """V2 VNC Controller — управляет MuJoCo через MN rates напрямую из мозга.

    БЕЗ MANC загрузки, БЕЗ Kuramoto CPG.
    Моторные нейроны из V2 датасета напрямую дюжат actuator'ами.

    Совместимость с GUI: имеет .dn_act, .turn_bias, .base_freq, .cpg заглушку.
    """

    def __init__(self, model, csr, brain):
        import mujoco
        self.csr = csr
        self.brain = brain
        self.time = 0.0
        self.v2_mode = True

        # ─── GUI compatibility ───
        self.dn_act = np.zeros(1, dtype=np.float32)
        self.dn_freq = 2.0
        self.dn_amp = 1.0
        self.turn_bias = 0.0

        # ─── CPG stub (poll_keys совместимость: 3/4 меняют base_freq) ───
        class CPGCompat:
            def __init__(self, controller):
                self.base_freq = 2.0
            def reset(self):
                pass  # VNC нейроны самовосстанавливаются
        self.cpg = CPGCompat(self)

        # ─── DNg100 injection ───
        self._dng100_idx = None
        dng100_info = csr.named_dns.get('DNg100')
        if dng100_info is not None:
            self._dng100_idx = dng100_info[1]  # csr_idx
        # τ_m=20ms → нужно больше drive. Brian2 reference использует
        # r_poi=150Hz uniform + Poisson напрямую в v (68.75mV).
        # Наш Poisson идёт через I_buf с pulse_amp=30, так что поднимаем injection.
        self._injection_current = 80.0  # pA (усилен — VNC connectome сам генерит ритм, нужно больше drive)

        # ─── CB0013 injection (Central Complex loop partner) ───
        self._cb0013_idx = None
        for rid, idx in csr.root_to_idx.items():
            typ = csr.type_map.get(rid, '')
            if typ == 'CB0013':
                self._cb0013_idx = idx
                print(f"  [V2VNC] CB0013 найден: csr_idx={idx}")
                break
        self._injection_current_cb0013 = 5.0  # pA (weaker — already gets DNg100 via weight 33)

        # ─── Touch sensor addresses (contact feedback) ───
        self._touch_addr = np.full(6, -1, dtype=np.int32)
        for i, (_, lname) in enumerate(LEGS):
            sn = f"touch_claw_{lname}"
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sn)
            if sid >= 0:
                self._touch_addr[i] = model.sensor_adr[sid]

        # ─── MuJoCo actuator map ───
        self.actuator_map = {}
        self.adhere = []
        for _, l in LEGS:
            for s in SEGMENTS:
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{l}")
                if i >= 0:
                    self.actuator_map[(l, s)] = i
        self._adhere_leg_idx = {}  # adhere_act_idx → leg_index
        for idx, (_, l) in enumerate(LEGS):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"adhere_claw_{l}")
            if i >= 0:
                self.adhere.append(i)
                self._adhere_leg_idx[i] = idx

        # Non-leg actuators
        self.non_leg_actuators = {}
        for seg in HEAD_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self.non_leg_actuators[('head', seg)] = i
        for seg in ANTENNA_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self.non_leg_actuators[('antenna', seg)] = i
        for seg in ABDOMEN_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self.non_leg_actuators[('abdomen', seg)] = i
        for seg in WING_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self.non_leg_actuators[('wing', seg)] = i

        n_leg = len(self.actuator_map)
        n_nl = len(self.non_leg_actuators)
        print(f"  [V2VNC] MuJoCo: {n_leg} leg DOFs, {n_nl} non-leg")

        # ─── Build MN → actuator mapping ───
        self.mn_to_ctrl = {}  # {mn_root_id: [(act_idx, scale), ...]}
        n_mapped = 0
        self._actuator_leg_idx = {}  # act_idx → leg_index

        # V2 body parts НЕ содержат left/right — только front_leg, middle_leg, hind_leg.
        # Распределяем MNs по сторонам round-robin (чётный → left, нечётный → right).
        _leg_side_counter = {'T1': 0, 'T2': 0, 'T3': 0}

        for i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '')
            ct = csr.type_map.get(rid, '').lower()

            # Determine leg label
            leg_label = None
            for bp_key, leg_prefix in [('front_leg', 'T1'), ('middle_leg', 'T2'), ('hind_leg', 'T3')]:
                if bp_key in bp:
                    # V2 data has no left/right — use round-robin
                    cnt = _leg_side_counter[leg_prefix]
                    side = 'left' if cnt % 2 == 0 else 'right'
                    _leg_side_counter[leg_prefix] = cnt + 1
                    leg_label = f'{leg_prefix}_{side}'
                    break

            if leg_label is None:
                continue
            if ct not in V2_MN_TYPE_MAP:
                continue

            segment, motion = V2_MN_TYPE_MAP[ct]
            act_idx = self.actuator_map.get((leg_label, segment))
            if act_idx is None:
                for fb in [f'{segment}_abduct', f'{segment}_twist', segment]:
                    act_idx = self.actuator_map.get((leg_label, fb))
                    if act_idx is not None:
                        break

            if act_idx is not None:
                motion_scales = {'flex': 0.5, 'extend': -0.5, 'abduct': 0.3, 'twist': 0.3}
                scale = motion_scales.get(motion, 0.3)
                if rid not in self.mn_to_ctrl:
                    self.mn_to_ctrl[rid] = []
                self.mn_to_ctrl[rid].append((act_idx, scale, motion))
                n_mapped += 1

                # Leg index for balance gain
                for leg_i, (_, lname) in enumerate(LEGS):
                    if lname == leg_label:
                        self._actuator_leg_idx[act_idx] = leg_i
                        break

        print(f"  [V2VNC] Mapped {n_mapped} MN→actuator ({len(self.mn_to_ctrl)} unique MNs)")

        # ─── Abdomen count ───
        abd = sum(1 for rid in csr.mn_root_ids if 'abdomen' in csr.body_part_map.get(rid, ''))
        print(f"  [V2VNC] Abdomen MNs: {abd}")

        # ─── Leg activation summary for GUI ───
        self._leg_act = np.zeros(6, dtype=np.float32)
        self._display_gains = np.ones(6)
        self._actuator_gain = 3.0  # scale MN rates → actuator drive (усилен для neural-only walking)

        # ─── Left/right actuator mirror pairs for symmetrization ───
        # Symmetric actuator pairs: (segment, motion) → same on both sides
        # Build by matching actuator names: {segment}_T{1,2,3}_{left,right}
        self._mirror_pairs = []  # [(left_act_idx, right_act_idx), ...]
        left_acts = {}  # (segment, t_num) → act_idx
        right_acts = {}
        for (lname, seg), act_idx in self.actuator_map.items():
            # Parse T1_left → (T1, left)
            parts = lname.split('_')
            if len(parts) == 2:
                t_num, side = parts
                key = (seg, t_num)
                if side == 'left':
                    left_acts[key] = act_idx
                else:
                    right_acts[key] = act_idx
        for key in set(left_acts.keys()) & set(right_acts.keys()):
            self._mirror_pairs.append((left_acts[key], right_acts[key]))
        print(f"  [V2VNC] {len(self._mirror_pairs)} mirror pairs for L/R symmetrization")

        # ─── MN rate interpolation (smooth between brain updates) ───
        self._brain_dt = 0.025  # brain update interval (seconds)
        self._prev_mn_rates = None
        self._cur_mn_rates = None
        self._last_brain_time = -0.025

        # ─── Pure neural drive: никакого hand-coded CPG! ───
        # Ритм ходьбы генерируется VNC коннектомом через MN rates из LIF-мозга.
        # Никаких синусоид — только реальные нейроны.


    def reset(self):
        self.time = 0.0
        self._leg_act.fill(0.0)

    def standby(self, data):
        """Standing posture — удерживает муху на ногах во время settle.

        Пока мозг ещё не запущен, задаёт лёгкое разгибание всех ног
        чтобы муха не провалилась под пол.
        """
        # Light extension всех ног (отрицательный ctrl = разгибание)
        for act_idx in self.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        # Adhere = grip floor
        for i in self.adhere:
            data.ctrl[i] = 1.0

    def _compute_balance_gain(self, data, dt):
        gain = np.ones(6, dtype=np.float64)
        if data is not None and len(data.qpos) >= 7:
            qw, qx, qy, qz = data.qpos[3:7]
            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (qw * qy - qz * qx)
            sinp = max(-1.0, min(1.0, sinp))
            pitch = math.asin(sinp)
            if not hasattr(self, '_prev_roll'):
                self._prev_roll = 0.0; self._prev_pitch = 0.0
            rv = (roll - self._prev_roll) / max(dt, 1e-6)
            pv = (pitch - self._prev_pitch) / max(dt, 1e-6)
            self._prev_roll, self._prev_pitch = roll, pitch
            rc = 3.0 * roll + 0.3 * rv
            pc = 2.0 * pitch + 0.2 * pv
            gain[0] += rc; gain[2] += rc; gain[4] += rc
            gain[1] -= rc; gain[3] -= rc; gain[5] -= rc
            gain[0] += pc; gain[1] += pc
            gain[4] -= pc; gain[5] -= pc
        t = self.turn_bias
        if abs(t) > 0.01:
            gain[0] += t*0.4; gain[1] -= t*0.4
            gain[2] += t*0.3; gain[3] -= t*0.3
            gain[4] += t*0.2; gain[5] -= t*0.2
        return np.clip(gain, 0.2, 2.5)

    def step(self, dt, data):
        self.time += dt
        mn_rates_raw = self.brain.get_mn_rates()

        # ─── 0. MN rate interpolation (smooth between brain updates) ───
        if len(mn_rates_raw) > 0:
            if self._cur_mn_rates is None:
                self._cur_mn_rates = mn_rates_raw.copy()
                self._prev_mn_rates = mn_rates_raw.copy()
            else:
                diff = float(np.max(np.abs(mn_rates_raw - self._cur_mn_rates)))
                if diff > 0.001:
                    self._prev_mn_rates = self._cur_mn_rates.copy()
                    self._cur_mn_rates = mn_rates_raw.copy()
                    self._last_brain_time = self.time
            elapsed = self.time - self._last_brain_time
            alpha = min(1.0, elapsed / self._brain_dt)
            if self._prev_mn_rates is not None and alpha < 1.0 and \
               len(self._prev_mn_rates) == len(self._cur_mn_rates):
                mn_rates = (1.0 - alpha) * self._prev_mn_rates + alpha * self._cur_mn_rates
            else:
                mn_rates = self._cur_mn_rates if self._cur_mn_rates is not None else mn_rates_raw
        else:
            mn_rates = mn_rates_raw

        # ─── 1. Contact feedback (механосенсоры → boost на ногу) ───
        contact_boost = np.zeros(6, dtype=np.float64)
        if data is not None and hasattr(data, 'sensordata') and data.sensordata is not None:
            sd = data.sensordata
            for i in range(6):
                addr = int(self._touch_addr[i])
                if 0 <= addr < len(sd):
                    contact = abs(float(sd[addr])) > 1e-7
                    if contact:
                        # Нога на земле → дополнительный drive (post-inhibitory rebound)
                        contact_boost[i] = 0.3

        # ─── 2. Balance gain ───
        balance_gain = self._compute_balance_gain(data, dt)
        self._display_gains = balance_gain.copy()

        # ─── 3. MN rates → actuator drive (нормализованное среднее) ───
        # НЕ обнуляем actuator'ы — step 7 перезаписывает их CPG baseline + MN drive.
        # НЕЙРОННЫЙ CPG: VNC коннектом сам генерирует ритм из DNg100 injection.
        # MN rates — результат LIF propagation через реальные синапсы VNC.
        # Actuator drive = MEAN(MN rate × balance_gain + contact_boost).
        #
        # ВАЖНО: нормализуем по числу MNs на actuator, а не суммируем.
        # Если 10 MNs управляют tibia flexor, их сумма = 10×0.5×0.8 = 4.0 → clipped.
        # Среднее = (10×0.5×0.8)/10 = 0.4 — правильная пропорциональная активация.
        act_drive = defaultdict(float)
        act_count = defaultdict(int)
        for i, rid in enumerate(self.csr.mn_root_ids):
            if i >= len(mn_rates):
                break
            rate = float(mn_rates[i])
            ctrl_list = self.mn_to_ctrl.get(rid)
            if ctrl_list is None:
                continue
            for act_idx, scale, motion in ctrl_list:
                leg_i = self._actuator_leg_idx.get(act_idx, -1)
                if leg_i < 0:
                    continue

                # Чистый MN rate × balance + contact boost
                val = rate * float(balance_gain[leg_i]) + float(contact_boost[leg_i])
                act_drive[act_idx] += scale * max(-1.0, min(1.0, val))
                act_count[act_idx] += 1

        # Normalize: среднее вместо суммы (пропорционально доле активных MNs)
        for act_idx in act_drive:
            if act_count.get(act_idx, 0) > 1:
                act_drive[act_idx] /= act_count[act_idx]

        # ─── 5. Symmetrize L/R actuator drive ───
        # Brain produces asymmetric MN rates due to random initial conditions
        # and asymmetric connectome structure. Average left/right for symmetric walking.
        for left_idx, right_idx in self._mirror_pairs:
            lv = act_drive.get(left_idx, 0.0)
            rv = act_drive.get(right_idx, 0.0)
            avg = (lv + rv) * 0.5
            if avg != 0.0:
                act_drive[left_idx] = avg
                act_drive[right_idx] = avg

        # ─── 6. Apply pure neural MN drive (БЕЗ hand-coded CPG!) ───
        # Ритм ходьбы генерируется VNC коннектомом через LIF-мозг.
        # Движение каждой ноги = weighted sum MN rates из real коннектома.
        self._leg_act.fill(0.0)
        for (lname, seg), act_idx in self.actuator_map.items():
            li = None
            for i, (_, ln) in enumerate(LEGS):
                if ln == lname:
                    li = i
                    break
            if li is None:
                continue
            ctrl_val = np.clip(act_drive.get(act_idx, 0.0) * self._actuator_gain, -1.0, 1.0)
            data.ctrl[act_idx] = ctrl_val
            self._leg_act[li] += abs(float(ctrl_val))

        # Non-leg: abdomen fallback
        sp = getattr(self, '_sez_phase', 0.0)
        sp += dt * 2.0 * math.pi * 1.0
        self._sez_phase = sp
        for seg in ABDOMEN_SEGMENTS:
            idx = self.non_leg_actuators.get(('abdomen', seg))
            if idx is not None:
                data.ctrl[idx] = 0.06 * math.sin(sp)

        # Adhesion: contact-based (БЕЗ hand-coded CPG фазы!)
        # Grip when leg touches ground, release during swing.
        # Используем touch sensor напрямую — чисто сенсомоторный рефлекс.
        if data is not None and hasattr(data, 'sensordata') and data.sensordata is not None:
            sd = data.sensordata
            for i in self.adhere:
                li = self._adhere_leg_idx.get(i)
                if li is None:
                    data.ctrl[i] = 1.0
                    continue
                addr = int(self._touch_addr[li])
                if 0 <= addr < len(sd):
                    contact = abs(float(sd[addr])) > 1e-7
                    if contact:
                        data.ctrl[i] = 3.0  # full grip during stance
                    else:
                        data.ctrl[i] = 0.0  # release during swing
                else:
                    data.ctrl[i] = 1.0
        else:
            for i in self.adhere:
                data.ctrl[i] = 1.0





# ═══════════════════════════════════════════════════════════════
#  Клавиши
# ═══════════════════════════════════════════════════════════════

_arrow_state = {'left': False, 'right': False}
_last_keys = {}
def poll_keys(window, vnc):
    keys = {
        '1': glfw.get_key(window,glfw.KEY_1)==glfw.PRESS,
        '2': glfw.get_key(window,glfw.KEY_2)==glfw.PRESS,
        '3': glfw.get_key(window,glfw.KEY_3)==glfw.PRESS,
        '4': glfw.get_key(window,glfw.KEY_4)==glfw.PRESS,
        '5': glfw.get_key(window,glfw.KEY_5)==glfw.PRESS,
        '6': glfw.get_key(window,glfw.KEY_6)==glfw.PRESS,
        '7': glfw.get_key(window,glfw.KEY_7)==glfw.PRESS,
        '8': glfw.get_key(window,glfw.KEY_8)==glfw.PRESS,
        '9': glfw.get_key(window,glfw.KEY_9)==glfw.PRESS,
        '0': glfw.get_key(window,glfw.KEY_0)==glfw.PRESS,
        'R': glfw.get_key(window,glfw.KEY_R)==glfw.PRESS,
    }
    # Arrow keys for turn (edge-triggered)
    left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    if left and not _arrow_state['left']:
        vnc.turn_bias = max(-1.0, vnc.turn_bias - 0.1)
    if right and not _arrow_state['right']:
        vnc.turn_bias = min(1.0, vnc.turn_bias + 0.1)
    _arrow_state['left'] = left
    _arrow_state['right'] = right

    for k,p in keys.items():
        was = _last_keys.get(k,False)
        if p and not was:
            if k=='1': vnc.dn_freq = max(0.5, vnc.dn_freq-0.25)
            elif k=='2': vnc.dn_freq = min(10.0, vnc.dn_freq+0.25)
            elif k=='3': vnc.cpg.base_freq = max(0.5, vnc.cpg.base_freq-0.25)
            elif k=='4': vnc.cpg.base_freq = min(8.0, vnc.cpg.base_freq+0.25)
            elif k=='5': vnc.turn_bias = max(-1.0, vnc.turn_bias-0.1)
            elif k=='6': vnc.turn_bias = min(1.0, vnc.turn_bias+0.1)
            elif k=='7': pass  # unused
            elif k=='8': pass  # unused
            elif k=='9': pass  # unused
            elif k=='0': vnc.turn_bias = 0.0
            elif k=='R':
                vnc.dn_freq=2.0; vnc.dn_amp=1.0; vnc.turn_bias=0.0
                vnc.cpg.reset()
        _last_keys[k] = p


# ═══════════════════════════════════════════════════════════════
#  Симуляция
# ═══════════════════════════════════════════════════════════════

def run_simulation(duration=15.0, record=False, looming=True,
                   model='lif', hbar=0.5):
    from flywire_brain import FlyWireCSR, FlyWireLIFBrain
    import mujoco as mu

    print(f"  Модель: {XML_PATH}")
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.03

    # ─── CSR ───
    print()
    print("  ═══ Full brain+VNC connectome ═══")
    csr = FlyWireCSR()
    csr.load()
    if not csr.is_v2:
        print("  ⚠ V2 dataset required!")
        return

    # ─── Brain ───
    if model == 'lif':
        brain = FlyWireLIFBrain(csr)
    else:
        from flywire_brain import FlyWireBrain, FlyWirePsiBrain, FlyWireRateBrain
        brain_map = {'rate': FlyWireRateBrain, 'psi': FlyWirePsiBrain, 'izhi': FlyWireBrain}
        brain_cls = brain_map.get(model, FlyWireLIFBrain)
        kwargs = {'hbar': hbar} if model == 'psi' else {}
        brain = brain_cls(csr, **kwargs)

    print(f"\n  [V2] {model.upper()} brain: {csr.n_neurons:,} нейронов, "
          f"{len(csr.dn_indices)} DN, {len(csr.mn_indices)} MN")

    # ─── VNC ───
    print()
    vnc = V2VNCController(mj_model, csr, brain)

    # GUI
    if not glfw.init():
        print("  glfw.init() failed"); return
    window = glfw.create_window(1200, 800, "🧠 Fly VNC Walk", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    scene = mu.MjvScene(mj_model, maxgeom=10000)
    cam = mu.MjvCamera()
    mu.mjv_defaultFreeCamera(mj_model, cam)
    cam.distance=0.30; cam.azimuth=45; cam.elevation=-20
    cam.lookat[:] = [0,0.01,-0.005]
    opt=mu.MjvOption(); pert=mu.MjvPerturb()
    ctx=mu.MjrContext(mj_model, mu.mjtFontScale.mjFONTSCALE_150)

    # Stand the fly up before settle (legs support body weight)
    data.qpos[2] = 0.03  # start higher so fly settles ON the floor, not below
    vnc.standby(data)
    print("\n  Settle (0.2s)...")
    for _ in range(2000):
        # Keep legs stiff during settle (override zeroing in standby)
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)
    print(f"  z={data.qpos[2]:.4f} contacts={data.ncon}")

    # Previous fly position for velocity computation
    prev_x, prev_y = float(data.qpos[0]), float(data.qpos[1])
    prev_time = 0.0

    vnc.reset()
    print(f"\n  VNC ходьба ({duration:.0f}s)")
    print("  1/2=freq  3/4=baseFreq  5/6=turn  ←/→=turn  0=resetTurn  R=resetAll")
    if brain is not None:
        print("  [FlyWire Izhikevich] brain обновляется каждые ~25ms")
        if looming and brain is not None and hasattr(brain, 'looming') and brain.looming_enabled:
            print(f"  [Vision looming] walls at ±{brain.looming.x_max:.2f}m, scale={brain.looming.scale:.2f}m")

    sim_t = 0.0; steps = 0; frames = 0; t0 = time.time()
    rec_frames = [] if record else None
    brain_step_interval = max(1, int(0.025 / mj_model.opt.timestep))  # ~250 шагов

    while not glfw.window_should_close(window):
        dt = mj_model.opt.timestep

        # FlyWire brain: Izhikevich каждые ~25ms
        if brain is not None and steps % brain_step_interval == 0:
            # Передаём позицию мухи для looming
            if looming:
                sx, sy = float(data.qpos[0]), float(data.qpos[1])
                dt_sim = sim_t - prev_time if sim_t > prev_time else mj_model.opt.timestep
                svx = (sx - prev_x) / dt_sim if dt_sim > 0 else 0.0
                svy = (sy - prev_y) / dt_sim if dt_sim > 0 else 0.0
                prev_x, prev_y, prev_time = sx, sy, sim_t
                brain.set_sensory_state(sx, sy, svx, svy)
            # V2 режим: DNg100 + CB0013 injection через brain._injection_targets
            # (применяется внутри step_batch ПОСЛЕ _set_sensory_input,
            #  который обнуляет external_I — старый код через external_I терял injection!)
            if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None:
                brain._injection_targets[vnc._dng100_idx] = vnc._injection_current
            if hasattr(vnc, '_cb0013_idx') and vnc._cb0013_idx is not None:
                brain._injection_targets[vnc._cb0013_idx] = vnc._injection_current_cb0013
            brain.step_batch(sim_t)

        vnc.step(dt, data)
        mu.mj_step(mj_model, data)
        sim_t += dt; steps += 1
        poll_keys(window, vnc)

        frames += 1
        if frames % 2 == 0:
            mu.mjv_updateScene(mj_model,data,opt,pert,cam,mu.mjtCatBit.mjCAT_ALL,scene)
            vp = mu.MjrRect(0,0,1200,800)
            mu.mjr_render(vp, scene, ctx)

            is_v2 = getattr(vnc, 'v2_mode', False)
            if is_v2:
                act_dn = len(vnc.csr.dn_indices)  # show total DNs
                act_mn = int(vnc._leg_act.sum())  # active leg actuation
                contacts = np.zeros(6, dtype=bool)
                # Show leg activation instead of phase
                leg_strs = []
                for leg_i, (label, lname) in enumerate(LEGS):
                    act = vnc._leg_act[leg_i]
                    bar = '█' * min(10, int(act * 10))
                    leg_strs.append(f"{label} {bar} {act:.2f}")
                contact_line = '  '.join(leg_strs)
            else:
                act_dn = np.sum(vnc.dn_act>0.1)
                act_mn = np.sum(vnc.cpg.mn_act>0.1)
                contacts = vnc.cpg.get_contact_states(data)
                phases = vnc.cpg.phases
                phase_strs = []
                for leg_i, (label, lname) in enumerate(LEGS):
                    c = '⬤' if contacts[leg_i] else '○'
                    phi = phases[leg_i]
                    phase_deg = phi * 180 / math.pi
                    ext = vnc.cpg.ext_act[leg_i]
                    flx = vnc.cpg.flx_act[leg_i]
                    phase_strs.append(f"{c}{label}{phase_deg:.0f}°({ext:.2f}/{flx:.2f})")
                contact_line = '  '.join(phase_strs)
            # Looming overlay
            looming_line = ''
            if brain is not None and hasattr(brain, 'looming') and looming:
                l = brain.looming
                if l.looming > 0.05:
                    bars = '░' * int(l.looming * 10) + '▓' * int((1-l.looming) * 10)
                    looming_line = f"\n⚠ LOOMING {l.looming:.2f}  [{bars}]  d={l.nearest_wall_dist:.3f}m {l.nearest_wall_name}"
            # Balance gains per leg (stored from step() to avoid recompute)
            bal = getattr(vnc, '_display_gains', np.ones(6))
            bal_strs = [f"{label[0]}={bal[i]:.1f}" for i, (label, _) in enumerate(LEGS)]
            if is_v2:
                title_line = f"🧠 V2 t={sim_t:.1f}s  turn={vnc.turn_bias:.2f}"
                info_line = f"DNs={act_dn}  MNs={act_mn}  x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}\n"
                info_line += f"Bal: {'  '.join(bal_strs)}\n"
                info_line += f"Legs: {contact_line}{looming_line}"
            else:
                title_line = f"🧠 VNC t={sim_t:.1f}s  freq={vnc.cpg.base_freq:.1f}Hz  turn={vnc.turn_bias:.2f}"
                info_line = f"DNs={act_dn}  MNs={act_mn}  x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}\n"
                info_line += f"Bal: {'  '.join(bal_strs)}\n"
                info_line += f"CPG: {contact_line}{looming_line}"

            mu.mjr_overlay(mu.mjtFontScale.mjFONTSCALE_150,
                mu.mjtGridPos.mjGRID_TOPLEFT, vp,
                title_line, info_line, ctx)
            glfw.swap_buffers(window)

        glfw.poll_events()
        if record and frames%2==0 and rec_frames is not None:
            arr=np.zeros((800,1200,3),dtype=np.uint8)
            mu.mjr_readPixels(arr,None,mu.MjrRect(0,0,1200,800),ctx)
            rec_frames.append(arr[::-1].copy())
        if sim_t >= duration: break

    elapsed = time.time()-t0
    final_x = data.qpos[0]
    print(f"\n  {sim_t:.1f}s за {elapsed:.1f}s (RT={sim_t/elapsed:.1f}×)")
    print(f"  Смещение: x={final_x:.3f}  y={data.qpos[1]:.3f}")
    glfw.destroy_window(window); glfw.terminate()
    if record and rec_frames:
        _save_video(rec_frames, "fly_vnc_walk.mp4")
    print("  ✓ Готово!")

def _save_video(frames, path):
    try:
        import cv2
        h,w,_=frames[0].shape
        out=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),30.0,(w,h))
        for f in frames: out.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  Видео: {path}")
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(frames[0]).save("vnc_first_frame.png")
        except ImportError: pass

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧠 Fly VNC Walk — neural V2 walker")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--model", type=str, choices=['izhi', 'psi', 'rate', 'lif'], default='lif',
        help="Модель нейронов: lif (LIF, default), izhi (Izhikevich), psi (ψ-wave), rate (ANN)")
    parser.add_argument("--hbar", type=float, default=0.5,
        help="ħ для ψ-wave")
    args = parser.parse_args()

    print("═"*55)
    print("  🧠 Fly VNC — Full Brain+Cord Connectome (V2)")
    print("═"*55)

    if not XML_PATH.exists():
        print(f"  Нет файла: {XML_PATH}"); sys.exit(1)
    run_simulation(duration=args.duration, record=args.record,
                   model=args.model, hbar=args.hbar)

if __name__ == "__main__":
    main()
