#!/usr/bin/env python3
"""
FlyWire Bridge — Rust Refiner → Python MuJoCo

Загружает очищенные веса из Rust fly_sensorimotor (fly_refined_weights.bin),
применяет их к полному 138K CSR и запускает MuJoCo симуляцию.

Использование:
  python3 fly/scripts/fly_bridge_refined.py [--duration 15] [--model lif]

Формат бинарного файла (из Rust):
  [n_edges: u32] [n_neurons: u32]
  [pre_root_id: u64] [post_root_id: u64] [weight: f32] × n_edges
  [root_id: u64] [flow_str_len: u32] [flow_str: utf8] × n_neurons
"""
import sys
import os
import struct
import time
import math
import argparse
from pathlib import Path
import numpy as np

# Пути
FLY_DIR = Path(__file__).parent.parent
PROJ = FLY_DIR.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# Настройка FLYWIRE_DATA для загрузки коннектома
os.environ.setdefault("FLYWIRE_DATA", str(Path.home() / "Загрузки"))

# Импортируем FlyWire компоненты
sys.path.insert(0, str(FLY_DIR / "scripts"))
from flywire_brain import FlyWireCSR, FlyWireLIFBrain
from fly.archive.vnc_controller import V2VNCController, LEGS, SEGMENTS, V2_MN_TYPE_MAP, TRIPOD_A, TRIPOD_B, HEAD_SEGMENTS, ANTENNA_SEGMENTS
from fly.scripts.vnc_kuramoto_cpg import VNCKuramotoCPG

# ═══════════════════════════════════════════════════════════════
#  Загрузка очищенных весов из Rust
# ═══════════════════════════════════════════════════════════════

def load_refined_weights(path: str):
    """Load refined weights from Rust binary export.

    Returns:
        edges: list of (pre_root_id, post_root_id, weight)
        neurons: list of (root_id, flow_str)
    """
    with open(path, 'rb') as f:
        data = f.read()

    offset = 0
    n_edges = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    n_neurons = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    print(f"  [Bridge] Loaded header: {n_edges} edges, {n_neurons} neurons")

    # Read edges: (pre_root_id: u64, post_root_id: u64, weight: f32)
    edges = []
    for _ in range(n_edges):
        pre = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        post = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        w = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        edges.append((pre, post, w))

    # Read neurons: (root_id: u64, flow_len: u32, flow_str: utf8)
    neurons = {}
    for _ in range(n_neurons):
        rid = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        s_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        flow = data[offset:offset + s_len].decode('utf-8', errors='replace')
        offset += s_len
        neurons[rid] = flow

    print(f"  [Bridge] Loaded {len(edges)} refined edges, {len(neurons)} neuron flows")
    return edges, neurons


# ═══════════════════════════════════════════════════════════════
#  Применение очищенных весов к полному CSR
# ═══════════════════════════════════════════════════════════════

def apply_refined_weights(csr: FlyWireCSR, edges: list):
    """Apply refined weights to the full FlyWire CSR.

    Для каждого refined edge (pre_root_id, post_root_id, weight):
      1. Находим pre_root_id в root_to_idx → pre_idx
      2. Находим post_root_id в root_to_idx → post_idx
      3. Сканируем row_ptr[post_idx] → col_indices[idx] для matching pre_idx
      4. Обновляем значение в csr.values[idx]

    Returns: (n_updated, n_skipped)
    """
    n_updated = 0
    n_skipped = 0

    row_ptr = csr.row_ptr
    col_indices = csr.col_indices
    values = csr.values
    root_to_idx = csr.root_to_idx

    t0 = time.time()
    for pre_root_id, post_root_id, refined_w in edges:
        pre_idx = root_to_idx.get(pre_root_id)
        post_idx = root_to_idx.get(post_root_id)

        if pre_idx is None or post_idx is None:
            n_skipped += 1
            continue

        # Scan post's row for matching pre (incoming convention)
        start = int(row_ptr[post_idx])
        end = int(row_ptr[post_idx + 1])
        found = False
        for idx in range(start, end):
            if col_indices[idx] == pre_idx:
                values[idx] = refined_w
                n_updated += 1
                found = True
                break

        if not found:
            n_skipped += 1

    elapsed = time.time() - t0
    print(f"  [Bridge] Applied: {n_updated} updated, {n_skipped} skipped in {elapsed:.2f}s")
    return n_updated, n_skipped


# ═══════════════════════════════════════════════════════════════
#  Применение original (pre-refinement) весов для сравнения
# ═══════════════════════════════════════════════════════════════

def load_original_weights(path: str):
    """Load the ORIGINAL (unrefined, initial epoch) weights from the same binary.

    Same format as refined, but contains pre-refinement weights.
    For now, we re-run the refiner with --export-original or store both.
    But for simplicity: this function reads the same file structure
    and returns the original (pre-refinement) weights.
    """
    # The binary only stores REFINED weights (after refinement).
    # To get original weights, we need to re-export them separately.
    # For now, return None and use CSR original values.
    return None


# ═══════════════════════════════════════════════════════════════
#  Сравнение ходьбы: original vs refined
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  DirectMNController — мозг напрямую в ноги, без CPG
# ═══════════════════════════════════════════════════════════════

class DirectMNController:
    """MN rates → actuator drive with brain-controlled tripod gait."""

    def __init__(self, model, csr, brain):
        import mujoco
        self.csr = csr
        self.brain = brain
        self.time = 0.0

        # GUI compatibility (для poll_keys)
        self.dn_freq = 2.0
        self.dn_amp = 1.0
        self.turn_bias = 0.0
        self._cpg_freq = 8.0
        self._cpg_amp = 0.5
        self._grooming = False
        class CPGCompat:
            def __init__(self): self.base_freq = 2.0
            def reset(self): pass
        self.cpg = CPGCompat()

        # DNg100 injection
        self._dng100_idx = None
        self._dng100_dn_pos = -1
        dng100_info = csr.named_dns.get('DNg100')
        if dng100_info is not None:
            self._dng100_idx = dng100_info[1]
            for i, dn_idx in enumerate(csr.dn_indices):
                if dn_idx == self._dng100_idx:
                    self._dng100_dn_pos = i
                    break
        self._injection_current = 80.0
        self._dng_mod = 1.0

        # CB0013
        self._cb0013_idx = None
        for rid, idx in csr.root_to_idx.items():
            if csr.type_map.get(rid, '') == 'CB0013':
                self._cb0013_idx = idx
                break
        self._injection_current_cb0013 = 5.0

        # ─── Grooming DNs (T1 + head) для brain-driven груминга ───
        self._grooming_t1_dn_indices = []
        self._grooming_head_dn_indices = []
        self._find_grooming_dns(csr)

        # Touch sensors
        self._touch_addr = np.full(6, -1, dtype=np.int32)
        for i, (_, lname) in enumerate(LEGS):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"touch_claw_{lname}")
            if sid >= 0:
                self._touch_addr[i] = model.sensor_adr[sid]

        # Head actuators (for grooming)
        self._head_act = {}
        for seg in HEAD_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self._head_act[seg] = i
        # Antenna actuators (for grooming)
        self._antenna_act = {}
        for seg in ANTENNA_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self._antenna_act[seg] = i

        # Wing actuators (for flight)
        WING_SEGMENTS = [
            "wing_yaw_left", "wing_roll_left", "wing_pitch_left",
            "wing_yaw_right", "wing_roll_right", "wing_pitch_right",
        ]
        self._wing_act = {}
        for seg in WING_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self._wing_act[seg] = i
        self._thorax_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')
        self._flight_enabled = False
        self._wing_k = 0.000002  # lift coefficient
        self._prev_pitch_l = 0.0
        self._prev_pitch_r = 0.0

        # ─── Wing MN → actuator mapping (brain-driven flight) ───
        WING_MN_TYPE_MAP = {
            'dlm1-4': 'pitch', 'dlm5': 'pitch',
            'dvm1a-c': 'pitch', 'dvm2a-b': 'pitch',
            'tp1': 'pitch', 'tp2': 'pitch', 'tpn': 'pitch',
            'b1': 'yaw', 'b2': 'yaw', 'b3': 'roll',
            'i1': 'roll', 'i2': 'roll',
            'iii1': 'yaw', 'iii3': 'yaw', 'iii4': 'roll',
            'ps1': 'roll', 'ps2': 'roll', 'psi': 'yaw',
        }
        self._wing_mn_map = []  # [(mn_i, side, dof)]
        wing_type_counter = {}
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'wing' not in bp:
                continue
            ct = csr.type_map.get(rid, '').lower()
            dof = WING_MN_TYPE_MAP.get(ct)
            if dof is None:
                continue
            cnt = wing_type_counter.get(ct, 0)
            wing_type_counter[ct] = cnt + 1
            side = 'left' if cnt % 2 == 0 else 'right'
            self._wing_mn_map.append((mn_i, side, dof))
        n_wing_mn = len(self._wing_mn_map)
        n_wing_act = len(self._wing_act)
        if n_wing_mn > 0:
            print(f"  [DirectMN] Wing drive: {n_wing_mn} MNs → {n_wing_act} actuators (brain-driven)")
        else:
            print(f"  [DirectMN] Wing drive: no wing MNs found")

        # Actuator map
        self.actuator_map = {}
        self.adhere = []
        for _, l in LEGS:
            for s in SEGMENTS:
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{l}")
                if i >= 0:
                    self.actuator_map[(l, s)] = i
        self._adhere_leg_idx = {}
        for idx, (_, l) in enumerate(LEGS):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"adhere_claw_{l}")
            if i >= 0:
                self.adhere.append(i)
                self._adhere_leg_idx[i] = idx

        # MN → actuator mapping (same as V2VNCController)
        self.mn_to_ctrl = {}
        self._actuator_leg_idx = {}
        _leg_side_counter = {'T1': 0, 'T2': 0, 'T3': 0}

        for i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '')
            ct = csr.type_map.get(rid, '').lower()

            leg_label = None
            for bp_key, leg_prefix in [('front_leg', 'T1'), ('middle_leg', 'T2'), ('hind_leg', 'T3')]:
                if bp_key in bp:
                    cnt = _leg_side_counter[leg_prefix]
                    side = 'left' if cnt % 2 == 0 else 'right'
                    _leg_side_counter[leg_prefix] = cnt + 1
                    leg_label = f'{leg_prefix}_{side}'
                    break

            if leg_label is None or ct not in V2_MN_TYPE_MAP:
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

                for leg_i, (_, lname) in enumerate(LEGS):
                    if lname == leg_label:
                        self._actuator_leg_idx[act_idx] = leg_i
                        break

        self._leg_act = np.zeros(6, dtype=np.float32)
        self._actuator_gain = 3.0

        # Flat mapping for fast loop
        self._mn_map = []  # [(mn_i, act_idx, scale, leg_i)]
        for mn_i, rid in enumerate(self.csr.mn_root_ids):
            ctrl_list = self.mn_to_ctrl.get(rid)
            if ctrl_list is None:
                continue
            for act_idx, scale, _ in ctrl_list:
                leg_i = self._actuator_leg_idx.get(act_idx, -1)
                if leg_i >= 0:
                    self._mn_map.append((mn_i, act_idx, scale, leg_i))

        # ─── Head & antenna MN→actuator maps (brain-driven) ───
        self._head_mn_map = []   # [(mn_i, act_idx, scale)]
        self._antenna_mn_map = []
        self._build_head_antenna_maps(csr)

        print(f"  [DirectMN] {len(self._mn_map)} MN→actuator mappings, "
              f"{len(set(a for _,a,_,_ in self._mn_map))} unique actuators")
        print(f"  [DirectMN] Head: {len(self._head_mn_map)} neck MN→actuator, "
              f"Antenna: {len(self._antenna_mn_map)} mappings")
        print(f"  [DirectMN] Wings: {len(self._wing_act)} actuators (F to toggle flight)")

    def _build_head_antenna_maps(self, csr):
        """Build brain-driven mappings from neck/antenna MNs to head/antenna actuators.

        Neck MNs (41) → head_abduct, head_twist, head
        Antenna MNs (4) → antenna_abduct_L/R, antenna_twist_L/R
        """
        # Find neck MN indices (all neck MNs → all head actuators)
        neck_mn_indices = []
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'neck' in bp:
                neck_mn_indices.append(mn_i)
        # Find antenna MN indices
        antenna_mn_indices = []
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'antenna' in bp or 'scape' in bp:
                antenna_mn_indices.append(mn_i)
        if not neck_mn_indices and not antenna_mn_indices:
            return
        # Map neck MNs to head actuators (different scales per actuator)
        head_scales = {'head_abduct': 0.4, 'head_twist': 0.2, 'head': 0.3}
        for seg, scale in head_scales.items():
            act_idx = self._head_act.get(seg)
            if act_idx is not None:
                # Split neck MNs: first half for abduct, second half for twist
                n_neck = len(neck_mn_indices)
                mn_subset = neck_mn_indices[:n_neck//2] if seg in ('head_abduct', 'head') else neck_mn_indices[n_neck//2:]
                for mn_i in mn_subset:
                    self._head_mn_map.append((mn_i, act_idx, scale))
        # Map antenna MNs to antenna actuators
        ant_scales = {
            'antenna_abduct_left': -0.25, 'antenna_abduct_right': 0.25,
            'antenna_twist_left': 0.25, 'antenna_twist_right': -0.25,
        }
        for seg, scale in ant_scales.items():
            act_idx = self._antenna_act.get(seg)
            if act_idx is not None:
                for mn_i in antenna_mn_indices:
                    self._antenna_mn_map.append((mn_i, act_idx, scale))

    def _find_grooming_dns(self, csr):
        """Find DNs that project to T1 MNs and head/antenna for brain-driven grooming."""
        t1_patterns = ('DNg105', 'DNg108', 'DNg93', 'DNg74')
        head_patterns = ('DNg24', 'DNge104', 'DNg91')
        for i, rid in enumerate(csr.dn_root_ids):
            typ = csr.type_map.get(rid, '')
            for pat in t1_patterns:
                if pat in typ:
                    self._grooming_t1_dn_indices.append(i)
                    break
            for pat in head_patterns:
                if pat in typ:
                    self._grooming_head_dn_indices.append(i)
                    break
        n_t1 = len(self._grooming_t1_dn_indices)
        n_head = len(self._grooming_head_dn_indices)
        if n_t1 > 0 or n_head > 0:
            print(f"  [Grooming] brain-driven DNs: {n_t1} T1 grooming, {n_head} head")

    def standby(self, data):
        for act_idx in self.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in self.adhere:
            data.ctrl[i] = 1.0

    def reset(self):
        self.time = 0.0
        self._leg_act.fill(0.0)

    def step(self, dt, data):
        """MN rates → actuator drive напрямую, БЕЗ CPG."""
        self.time += dt
        mn_rates = self.brain.get_mn_rates()
        if len(mn_rates) == 0:
            return

        # Balance gain
        balance_gain = np.ones(6, dtype=np.float64)
        if data is not None and len(data.qpos) >= 7:
            qw, qx, qy, qz = data.qpos[3:7]
            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (qw * qy - qz * qx)
            sinp = max(-1.0, min(1.0, sinp))
            pitch = math.asin(sinp)
            if not hasattr(self, '_prev_roll'):
                self._prev_roll = 0.0
                self._prev_pitch = 0.0
            rv = (roll - self._prev_roll) / max(dt, 1e-6)
            pv = (pitch - self._prev_pitch) / max(dt, 1e-6)
            self._prev_roll, self._prev_pitch = roll, pitch
            rc = 3.0 * roll + 0.3 * rv
            pc = 2.0 * pitch + 0.2 * pv
            balance_gain[0] += rc; balance_gain[2] += rc; balance_gain[4] += rc
            balance_gain[1] -= rc; balance_gain[3] -= rc; balance_gain[5] -= rc
            balance_gain[0] += pc; balance_gain[1] += pc
            balance_gain[4] -= pc; balance_gain[5] -= pc
        t = self.turn_bias
        if abs(t) > 0.01:
            balance_gain[0] += t*0.4; balance_gain[1] -= t*0.4
            balance_gain[2] += t*0.3; balance_gain[3] -= t*0.3
            balance_gain[4] += t*0.2; balance_gain[5] -= t*0.2
        balance_gain = np.clip(balance_gain, 0.2, 2.5)

        # Contact boost
        contact_boost = np.zeros(6, dtype=np.float64)
        if data is not None and hasattr(data, 'sensordata') and data.sensordata is not None:
            sd = data.sensordata
            for i in range(6):
                addr = int(self._touch_addr[i])
                if 0 <= addr < len(sd) and abs(float(sd[addr])) > 1e-7:
                    contact_boost[i] = 0.3

        # MN → actuator (direct, NO CPG)
        self._leg_act.fill(0.0)
        act_drive = {}
        for mn_i, act_idx, scale, leg_i in self._mn_map:
            if mn_i >= len(mn_rates):
                continue
            rate = float(mn_rates[mn_i])
            val = rate * float(balance_gain[leg_i]) + float(contact_boost[leg_i])
            val = max(-1.0, min(1.0, val)) * scale
            if act_idx in act_drive:
                act_drive[act_idx] = (act_drive[act_idx] + val) * 0.5
            else:
                act_drive[act_idx] = val
            self._leg_act[leg_i] += abs(float(val))

        # Tripod gait overlay — brain DNg100 controls speed, pattern coordinates
        dng_rate = 0.0
        if self._dng100_dn_pos >= 0 and hasattr(self.brain, 'dn_rate'):
            if self._dng100_dn_pos < len(self.brain.dn_rate):
                dng_rate = float(self.brain.dn_rate[self._dng100_dn_pos]) * 100.0
        drive_level = min(1.0, dng_rate / 150.0)
        if drive_level > 0.05:
            freq = 5.0 + drive_level * 7.0  # 5-12 Hz
            TRIPOD_PHASES = [0, math.pi, math.pi, 0, 0, math.pi]
            for leg_i, (_, lname) in enumerate(LEGS):
                phi = TRIPOD_PHASES[leg_i]
                t_cycle = self.time * freq * 2 * math.pi + phi
                for seg in ['coxa_twist', 'coxa_abduct', 'femur']:
                    act_idx = self.actuator_map.get((lname, seg))
                    if act_idx is None or act_idx >= len(data.ctrl):
                        continue
                    if seg == 'coxa_twist':
                        pat = math.sin(t_cycle) * 0.4
                    elif seg == 'coxa_abduct':
                        pat = max(0.0, math.sin(t_cycle + math.pi)) * 0.3
                    else:
                        pat = -math.sin(t_cycle + 0.75 * math.pi) * 0.3
                    pat *= drive_level * 0.5
                    existing = act_drive.get(act_idx, 0.0)
                    act_drive[act_idx] = existing * 0.3 + pat * 0.7

        # Apply to actuators
        for act_idx, val in act_drive.items():
            if act_idx < len(data.ctrl):
                data.ctrl[act_idx] = np.clip(val * self._actuator_gain, -1.0, 1.0)

        # Adhesion
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
                    data.ctrl[i] = 3.0 if contact else 0.0
                else:
                    data.ctrl[i] = 1.0
        else:
            for i in self.adhere:
                data.ctrl[i] = 1.0

        # ─── FULLY AUTONOMOUS brain-driven grooming mode ───
        # Grooming DNs injected → brain fires MNs → we read MN rates → drive actuators
        # NO hardcoded values! Accumulate all MN contributions per actuator.
        if getattr(self, '_grooming', False):
            mn_rates = self.brain.get_mn_rates()
            groom_drive = {}  # act_idx → summed value
            # Head brain-driven: neck MN rates → head actuators
            for mn_i, act_idx, scale in self._head_mn_map:
                if mn_i < len(mn_rates):
                    groom_drive[act_idx] = groom_drive.get(act_idx, 0.0) + float(mn_rates[mn_i]) * scale * 3.0
            # Antenna brain-driven: antenna MN rates → antenna actuators
            for mn_i, act_idx, scale in self._antenna_mn_map:
                if mn_i < len(mn_rates):
                    groom_drive[act_idx] = groom_drive.get(act_idx, 0.0) + float(mn_rates[mn_i]) * scale * 3.0
            # Apply accumulated actuator values
            for act_idx, val in groom_drive.items():
                data.ctrl[act_idx] = np.clip(val, -1.0, 1.0)

        # ─── Flight: brain-driven wing CPG + lift ───
        if self._thorax_body_id >= 0:
            data.xfrc_applied[self._thorax_body_id, 2] = 0.0
        if self._flight_enabled and len(self._wing_act) >= 6:
            # Brain-driven amplitude
            pitch_amp = 0.1  # baseline
            yaw_off = 0.0
            roll_off = 0.0
            if hasattr(self, '_wing_mn_map') and self._wing_mn_map and len(mn_rates) > 0:
                left_power = 0.0
                right_power = 0.0
                n_left = n_right = 0
                for mn_i, side, dof in self._wing_mn_map:
                    if mn_i < len(mn_rates):
                        r = float(mn_rates[mn_i])
                        if dof == 'pitch':
                            if side == 'left':
                                left_power += r; n_left += 1
                            else:
                                right_power += r; n_right += 1
                        elif dof == 'yaw':
                            yaw_off += (r if side == 'left' else -r) * 0.3
                        elif dof == 'roll':
                            roll_off += (r if side == 'left' else -r) * 0.3
                avg = 0.0
                if n_left > 0:
                    avg += left_power / n_left
                if n_right > 0:
                    avg += right_power / n_right
                if n_left > 0 or n_right > 0:
                    div = (1 if n_left > 0 else 0) + (1 if n_right > 0 else 0)
                    avg /= div
                pitch_amp = 0.1 + avg * 0.9
                yaw_off = max(-0.3, min(0.3, yaw_off))
                roll_off = max(-0.3, min(0.3, roll_off))
            wing_freq = 150.0
            t = self.time
            pitch_l = math.sin(2 * math.pi * wing_freq * t) * pitch_amp
            pitch_r = math.sin(2 * math.pi * wing_freq * t + roll_off * 0.5) * pitch_amp
            roll = math.sin(2 * math.pi * wing_freq * 2 * t + 0.5) * 0.3 + roll_off
            yaw = math.sin(2 * math.pi * wing_freq * t + 1.2) * 0.2 + yaw_off
            data.ctrl[self._wing_act['wing_pitch_left']] = pitch_l
            data.ctrl[self._wing_act['wing_roll_left']] = roll
            data.ctrl[self._wing_act['wing_yaw_left']] = yaw
            data.ctrl[self._wing_act['wing_pitch_right']] = pitch_r
            data.ctrl[self._wing_act['wing_roll_right']] = -roll
            data.ctrl[self._wing_act['wing_yaw_right']] = -yaw
            if self._thorax_body_id >= 0:
                v_l = (pitch_l - self._prev_pitch_l) / max(dt, 1e-8)
                v_r = (pitch_r - self._prev_pitch_r) / max(dt, 1e-8)
                lift = self._wing_k * (v_l * abs(v_l) + v_r * abs(v_r))
                data.xfrc_applied[self._thorax_body_id, 2] += lift
                self._prev_pitch_l = pitch_l
                self._prev_pitch_r = pitch_r


# ═══════════════════════════════════════════════════════════════
#  VNCKuramotoController — Biological CPG via Kuramoto on VNC connectome
# ═══════════════════════════════════════════════════════════════

class VNCKuramotoController:
    """VNC Kuramoto CPG controller — заменяет math.sin() на биологический ринг-осциллятор.

    Использует VNC Cluster 1 (5,681 GABA нейронов) с Kuramoto динамикой
    для генерации антифазного ритма ходьбы.

    tripod_A = sin(θ_A) → {T1_left, T2_right, T3_left}
    tripod_B = sin(θ_B) → {T1_right, T2_left, T3_right}
    """

    def __init__(self, model, csr, brain, vnc_bin="/tmp/vnc_cluster1.bin",
                 dt_s=0.005, omega_base=8.0, coupling_scale=30.0):
        import mujoco
        self.csr = csr
        self.brain = brain
        self.time = 0.0

        # ─── VNC Kuramoto CPG ───
        print(f"  [VNC.KUR] Loading Kuramoto CPG from {vnc_bin}...")
        self.vnc_cpg = VNCKuramotoCPG(vnc_bin, dt_s=dt_s, omega_base=omega_base,
                                        coupling_scale=coupling_scale)
        self.vnc_bin = vnc_bin
        self.dt_s = dt_s
        self.omega_base = omega_base
        self._kuramoto_accum = 0.0  # accumulated Kuramoto time

        # GUI compatibility (для poll_keys)
        self.dn_freq = 2.0
        self.dn_amp = 1.0
        self.turn_bias = 0.0
        self._cpg_freq = omega_base / (2 * math.pi)  # Hz ≈ 1.3
        self._cpg_amp = 0.5
        self._grooming = False  # G key toggles grooming mode
        self._kur_sin_a = 0.0
        self._kur_sin_b = 0.0
        self._kur_cos_a = 1.0
        self._kur_cos_b = 1.0
        self._kur_phase_diff = 0.0
        class CPGCompat:
            def __init__(self): self.base_freq = 2.0
            def reset(self): pass
        self.cpg = CPGCompat()

        # DNg100 injection
        self._dng100_idx = None
        dng100_info = csr.named_dns.get('DNg100')
        if dng100_info is not None:
            self._dng100_idx = dng100_info[1]
        self._injection_current = 80.0
        self._dng_mod = 1.0

        # CB0013
        self._cb0013_idx = None
        for rid, idx in csr.root_to_idx.items():
            if csr.type_map.get(rid, '') == 'CB0013':
                self._cb0013_idx = idx
                break
        self._injection_current_cb0013 = 5.0

        # Touch sensors
        self._touch_addr = np.full(6, -1, dtype=np.int32)
        for i, (_, lname) in enumerate(LEGS):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"touch_claw_{lname}")
            if sid >= 0:
                self._touch_addr[i] = model.sensor_adr[sid]

        # Head actuators (for grooming)
        self._head_act = {}
        for seg in HEAD_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self._head_act[seg] = i
        # Antenna actuators (for grooming)
        self._antenna_act = {}
        for seg in ANTENNA_SEGMENTS:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, seg)
            if i >= 0: self._antenna_act[seg] = i

        # ─── Grooming DNs (T1 + head) для brain-driven груминга ───
        self._grooming_t1_dn_indices = []
        self._grooming_head_dn_indices = []
        self._find_grooming_dns(csr)

        # Actuator map (same as V2VNCController)
        self.actuator_map = {}
        self.adhere = []
        for _, l in LEGS:
            for s in SEGMENTS:
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_{l}")
                if i >= 0:
                    self.actuator_map[(l, s)] = i
        self._adhere_leg_idx = {}
        for idx, (_, l) in enumerate(LEGS):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"adhere_claw_{l}")
            if i >= 0:
                self.adhere.append(i)
                self._adhere_leg_idx[i] = idx

        # Tripod → leg mapping
        self._tripod_legs = {}
        for leg_i, (label, lname) in enumerate(LEGS):
            if lname in TRIPOD_A:
                self._tripod_legs[leg_i] = 'A'
            elif lname in TRIPOD_B:
                self._tripod_legs[leg_i] = 'B'
            else:
                self._tripod_legs[leg_i] = 'A'

        # Actuator groups per leg
        self._leg_twist = {}    # leg_i → coxa_twist actuator idx
        self._leg_abduct = {}   # leg_i → coxa_abduct actuator idx
        self._leg_femur = {}    # leg_i → femur actuator idx
        for leg_i, (_, lname) in enumerate(LEGS):
            for s in SEGMENTS:
                act_idx = self.actuator_map.get((lname, s))
                if act_idx is None:
                    continue
                if 'twist' in s:
                    self._leg_twist[leg_i] = act_idx
                elif 'abduct' in s:
                    self._leg_abduct[leg_i] = act_idx
                elif 'femur' in s or s == 'tibia':
                    self._leg_femur[leg_i] = act_idx

        # ─── T1 MN mapping for brain-driven grooming ───
        self._t1_mn_map = []  # [(mn_i, act_idx, scale, leg_i)] — only T1
        _t1_side_counter = {'T1': 0}
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'front_leg' not in bp:
                continue
            ct = csr.type_map.get(rid, '').lower()
            if ct not in V2_MN_TYPE_MAP:
                continue
            cnt = _t1_side_counter['T1']
            side = 'left' if cnt % 2 == 0 else 'right'
            _t1_side_counter['T1'] = cnt + 1
            tl_label = f'T1_{side}'
            segment, motion = V2_MN_TYPE_MAP[ct]
            act_idx = self.actuator_map.get((tl_label, segment))
            if act_idx is None:
                for fb in [f'{segment}_abduct', f'{segment}_twist', segment]:
                    act_idx = self.actuator_map.get((tl_label, fb))
                    if act_idx is not None:
                        break
            if act_idx is not None:
                motion_scales = {'flex': 0.5, 'extend': -0.5, 'abduct': 0.3, 'twist': 0.3}
                scale = motion_scales.get(motion, 0.3)
                leg_i = 0 if side == 'left' else 1
                self._t1_mn_map.append((mn_i, act_idx, scale, leg_i))

        print(f"  [VNC.KUR] Tripod: A={self.vnc_cpg._tripod_a_idx} B={self.vnc_cpg._tripod_b_idx}")
        print(f"  [VNC.KUR] {len(self._leg_twist)} twist, {len(self._leg_abduct)} abduct, "
              f"{len(self._leg_femur)} femur actuators")
        print(f"  [VNC.KUR] T1 MN map: {len(self._t1_mn_map)} mappings")
        print(f"  [VNC.KUR] Warming Kuramoto (100 steps)...")
        self.vnc_cpg.step(1.0, n_steps=100)
        sa = self.vnc_cpg.get_sin(self.vnc_cpg._tripod_a_idx) if self.vnc_cpg._tripod_a_idx else 0.0
        sb = self.vnc_cpg.get_sin(self.vnc_cpg._tripod_b_idx) if self.vnc_cpg._tripod_b_idx else 0.0
        pa = self.vnc_cpg.get_phase(self.vnc_cpg._tripod_a_idx) if self.vnc_cpg._tripod_a_idx else 0.0
        pb = self.vnc_cpg.get_phase(self.vnc_cpg._tripod_b_idx) if self.vnc_cpg._tripod_b_idx else 0.0
        self._kur_sin_a = sa
        self._kur_sin_b = sb
        self._kur_cos_a = math.cos(pa)
        self._kur_cos_b = math.cos(pb)
        self._kur_phase_diff = min(abs(pa - pb), 2*math.pi - abs(pa - pb))
        print(f"  [VNC.KUR] Warm: A={sa:.3f} B={sb:.3f} Δφ={self._kur_phase_diff:.2f} rad")

        # ─── Head & antenna MN→actuator maps (brain-driven) ───
        self._head_mn_map = []   # [(mn_i, act_idx, scale)]
        self._antenna_mn_map = []
        self._build_head_antenna_maps(csr)
        print(f"  [VNC.KUR] Head: {len(self._head_mn_map)} neck MN→actuator, "
              f"Antenna: {len(self._antenna_mn_map)} mappings")

    def _build_head_antenna_maps(self, csr):
        """Build brain-driven mappings from neck/antenna MNs to head/antenna actuators."""
        neck_mn_indices = []
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'neck' in bp:
                neck_mn_indices.append(mn_i)
        antenna_mn_indices = []
        for mn_i, rid in enumerate(csr.mn_root_ids):
            bp = csr.body_part_map.get(rid, '').lower()
            if 'antenna' in bp or 'scape' in bp:
                antenna_mn_indices.append(mn_i)
        if not neck_mn_indices and not antenna_mn_indices:
            return
        head_scales = {'head_abduct': 0.4, 'head_twist': 0.2, 'head': 0.3}
        for seg, scale in head_scales.items():
            act_idx = self._head_act.get(seg)
            if act_idx is not None:
                n_neck = len(neck_mn_indices)
                mn_subset = neck_mn_indices[:n_neck//2] if seg in ('head_abduct', 'head') else neck_mn_indices[n_neck//2:]
                for mn_i in mn_subset:
                    self._head_mn_map.append((mn_i, act_idx, scale))
        ant_scales = {
            'antenna_abduct_left': -0.25, 'antenna_abduct_right': 0.25,
            'antenna_twist_left': 0.25, 'antenna_twist_right': -0.25,
        }
        for seg, scale in ant_scales.items():
            act_idx = self._antenna_act.get(seg)
            if act_idx is not None:
                for mn_i in antenna_mn_indices:
                    self._antenna_mn_map.append((mn_i, act_idx, scale))

    def _find_grooming_dns(self, csr):
        """Find DNs that project to T1 MNs and head/antenna for brain-driven grooming."""
        t1_patterns = ('DNg105', 'DNg108', 'DNg93', 'DNg74')
        head_patterns = ('DNg24', 'DNge104', 'DNg91')
        for i, rid in enumerate(csr.dn_root_ids):
            typ = csr.type_map.get(rid, '')
            for pat in t1_patterns:
                if pat in typ:
                    self._grooming_t1_dn_indices.append(i)
                    break
            for pat in head_patterns:
                if pat in typ:
                    self._grooming_head_dn_indices.append(i)
                    break
        n_t1 = len(self._grooming_t1_dn_indices)
        n_head = len(self._grooming_head_dn_indices)
        if n_t1 > 0 or n_head > 0:
            print(f"  [Grooming] brain-driven DNs: {n_t1} T1 grooming, {n_head} head")

    def standby(self, data):
        for act_idx in self.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in self.adhere:
            data.ctrl[i] = 1.0

    def reset(self):
        self.time = 0.0
        # Re-init Kuramoto phases
        self.vnc_cpg.phases = np.random.RandomState(42).uniform(
            0, 2 * math.pi, self.vnc_cpg.n).astype(np.float32)

    def step(self, dt, data):
        """Run VNC Kuramoto CPG and map to actuators.

        Phase offsets matching V2VNCController sin() CPG:
          coxa_twist  = sin(phase) * amp
          coxa_abduct = max(0, sin(phase + π)) * amp * 0.6  (lift during swing)
          femur       = -sin(phase + 0.75π) * amp * 0.6     (flex during swing)
        """
        self.time += dt

        # ─── Accumulate Kuramoto time ───
        self._kuramoto_accum += dt
        n_steps = int(self._kuramoto_accum / self.dt_s)
        if n_steps > 0:
            self._kuramoto_accum -= n_steps * self.dt_s
            dn_drive = getattr(self, 'dn_freq', 2.0) / 10.0
            self.vnc_cpg.step(dn_drive, n_steps=n_steps)

            # Cache tripod sin + cos for phase-shifted outputs
            if self.vnc_cpg._tripod_a_idx is not None:
                ph_a = self.vnc_cpg.phases[self.vnc_cpg._tripod_a_idx]
                ph_b = self.vnc_cpg.phases[self.vnc_cpg._tripod_b_idx]
                self._kur_sin_a = math.sin(ph_a)
                self._kur_sin_b = math.sin(ph_b)
                self._kur_cos_a = math.cos(ph_a)
                self._kur_cos_b = math.cos(ph_b)
                pa = float(ph_a) % (2.0 * math.pi)
                pb = float(ph_b) % (2.0 * math.pi)
                self._kur_phase_diff = min(abs(pa - pb), 2.0 * math.pi - abs(pa - pb))

        sin_a = self._kur_sin_a
        sin_b = self._kur_sin_b
        cos_a = self._kur_cos_a
        cos_b = self._kur_cos_b

        # ─── CPG actuator values (matching V2VNCController) ───
        amp = getattr(self, '_cpg_amp', 0.5)

        # Coxa twist: sin(phase) * amp (matching V2VNCController)
        twist_a = sin_a * amp
        twist_b = sin_b * amp

        # Coxa abduct: max(0, sin(phase + π)) = max(0, -sin(phase)) — lift during swing
        # sin(phase + π) = -sin(phase). clamp to [0, 1] for positive-only lift
        abduct_a = max(0.0, -sin_a) * amp * 0.6
        abduct_b = max(0.0, -sin_b) * amp * 0.6

        # Femur: -sin(phase + 0.75π) = sin(phase + 0.75π + π) = sin(phase + 1.75π)
        # -sin(phase + 0.75π) = -(sin·cos(0.75π) + cos·sin(0.75π))
        # cos(0.75π) = -0.707, sin(0.75π) = 0.707
        # = -(-0.707·sin + 0.707·cos) = 0.707·(sin - cos)
        femur_a = (sin_a * 0.707 - cos_a * 0.707) * amp * 0.6
        femur_b = (sin_b * 0.707 - cos_b * 0.707) * amp * 0.6

        # ─── Apply to legs ───
        for leg_i in range(6):
            is_a = self._tripod_legs.get(leg_i, 'A') == 'A'
            twist_v = twist_a if is_a else twist_b
            abduct_v = abduct_a if is_a else abduct_b
            femur_v = femur_a if is_a else femur_b

            # Turn bias
            t = getattr(self, 'turn_bias', 0.0)
            side = -1 if leg_i % 2 == 0 else 1
            twist_v += t * side * 0.3

            if leg_i in self._leg_twist:
                data.ctrl[self._leg_twist[leg_i]] = np.clip(twist_v, -1.0, 1.0)
            if leg_i in self._leg_abduct:
                data.ctrl[self._leg_abduct[leg_i]] = np.clip(abduct_v, -1.0, 1.0)
            if leg_i in self._leg_femur:
                data.ctrl[self._leg_femur[leg_i]] = np.clip(femur_v, -1.0, 1.0)

        # Adhesion (always 1.0 for now)
        for i in self.adhere:
            data.ctrl[i] = 1.0

        # ─── FULLY AUTONOMOUS brain-driven grooming mode ───
        # Grooming DNs injected → brain fires MNs → we read MN rates → drive actuators
        # NO hardcoded values! Accumulate all MN contributions per actuator.
        if getattr(self, '_grooming', False):
            mn_rates = self.brain.get_mn_rates()
            groom_drive = {}  # act_idx → summed value
            # T1 brain-driven (bypass CPG for T1)
            for mn_i, act_idx, scale, leg_i in self._t1_mn_map:
                if mn_i < len(mn_rates):
                    groom_drive[act_idx] = groom_drive.get(act_idx, 0.0) + float(mn_rates[mn_i]) * scale * 3.0
            # Head brain-driven (neck MNs → head actuators)
            for mn_i, act_idx, scale in self._head_mn_map:
                if mn_i < len(mn_rates):
                    groom_drive[act_idx] = groom_drive.get(act_idx, 0.0) + float(mn_rates[mn_i]) * scale * 3.0
            # Antenna brain-driven (antenna MNs → antenna actuators)
            for mn_i, act_idx, scale in self._antenna_mn_map:
                if mn_i < len(mn_rates):
                    groom_drive[act_idx] = groom_drive.get(act_idx, 0.0) + float(mn_rates[mn_i]) * scale * 3.0
            # Apply accumulated actuator values
            for act_idx, val in groom_drive.items():
                data.ctrl[act_idx] = np.clip(val, -1.0, 1.0)


def run_comparison(duration=15.0, model='lif', headless=False, brain_hz=2.0, direct_mn=False, vnc_cpg=False):
    """Run MuJoCo once with original CSR, once with refined weights."""
    import mujoco as mu

    XML_PATH = FLY_DIR / "models" / "scene.xml"
    if not XML_PATH.exists():
        print(f"  ❌ Нет файла: {XML_PATH}")
        return

    print()
    print("═" * 55)
    print("  🧠 FlyWire Bridge — Rust Refiner → MuJoCo")
    print("═" * 55)
    print()
    print(f"  • Refined weights: /tmp/fly_refined_weights.bin")
    print(f"  • MuJoCo model: {XML_PATH}")
    print(f"  • Model: {model.upper()}")
    print(f"  • Brain update rate: {brain_hz:.1f} Hz")
    print(f"  • Duration: {duration:.0f}s")
    print(f"  • Headless: {headless}")
    print(f"  • VNC Kuramoto CPG: {'🧠 VNC' if vnc_cpg else '🦗 sin()'}")
    print()

    # ─── Load CSR once ───
    print("  ═══ Loading full FlyWire connectome ═══")
    csr = FlyWireCSR()
    csr.load()
    if not csr.is_v2:
        print("  ❌ V2 dataset required!")
        return
    print(f"  CSR: {csr.n_neurons:,} neurons, {csr.n_synapses:,} synapses")
    print()

    # ─── Load & apply refined weights (optional) ───
    weights_path = "/tmp/fly_refined_weights.bin"
    if Path(weights_path).exists():
        print("  ═══ Applying refined weights ═══")
        refined_edges, _ = load_refined_weights(weights_path)
        n_upd, n_sk = apply_refined_weights(csr, refined_edges)
        flow_counts = {}
        for pre_rid, post_rid, _ in refined_edges:
            for rid in (pre_rid, post_rid):
                flow = csr.flow_map.get(rid, 'unknown')
                flow_counts[flow] = flow_counts.get(flow, 0) + 1
        print("  Edge distribution by flow:")
        for flow, cnt in sorted(flow_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {flow}: {cnt}")
        print()
    else:
        print("  ═══ Skipping refinement (no weights file) ═══")
        print()

    # ─── Summary ───
    if Path(weights_path).exists():
        print(f"  ✅ Refined weights applied to full CSR!")
        print(f"     Updated: {n_upd} / {len(refined_edges)} edges")
        print(f"     Skipped: {n_sk} (neurons not in full CSR)")
        print()

    # ─── Run MuJoCo simulation ───
    print("  ═══ Starting MuJoCo simulation ═══")
    print(f"  Brain update rate: {brain_hz:.1f} Hz")
    print(f"  Motor control: {'🧠 DIRECT MN (no CPG)' if direct_mn else '🦗 VNC + Kuramoto CPG' if vnc_cpg else '🦗 VNC + sin() CPG'}")
    print()
    if not headless:
        try:
            run_simulation_mujoco(csr, duration=duration, model=model, brain_hz=brain_hz, direct_mn=direct_mn, vnc_cpg=vnc_cpg)
        except KeyboardInterrupt:
            print("\n  ⏹ Simulation stopped by user")
    else:
        print("  Headless mode: skipping GUI (use --headless for auto-run)")
        run_simulation_headless(csr, duration=duration, model=model)


# ═══════════════════════════════════════════════════════════════
#  MuJoCo симуляция (адаптирована из vnc_walk.py)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  Горячие клавиши (из vnc_walk.py)
# ═══════════════════════════════════════════════════════════════

_arrow_state = {'left': False, 'right': False}
_last_keys = {}
def poll_keys(window, vnc):
    import glfw
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
        'F': glfw.get_key(window,glfw.KEY_F)==glfw.PRESS,
    }
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
            elif k=='7': vnc._cpg_freq = max(1.0, vnc._cpg_freq-1.0)
            elif k=='8': vnc._cpg_freq = min(20.0, vnc._cpg_freq+1.0)
            elif k=='9': vnc._cpg_amp = max(0.05, vnc._cpg_amp-0.05)
            elif k=='0': vnc.turn_bias = 0.0; vnc._cpg_amp = 0.5; vnc._cpg_freq = 8.0
            elif k=='R':
                vnc.dn_freq=2.0; vnc.dn_amp=1.0; vnc.turn_bias=0.0
                vnc._cpg_amp = 0.5; vnc._cpg_freq = 8.0
                vnc.cpg.reset()
            elif k=='F':
                if hasattr(vnc, '_flight_enabled'):
                    vnc._flight_enabled = not vnc._flight_enabled
        _last_keys[k] = p

    # Grooming toggle (G key)
    g = glfw.get_key(window, glfw.KEY_G) == glfw.PRESS
    g_was = _last_keys.get('G', False)
    if g and not g_was:
        if hasattr(vnc, '_grooming'):
            vnc._grooming = not vnc._grooming
    _last_keys['G'] = g


def quat_to_yaw(qw, qx, qy, qz):
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)

def run_simulation_mujoco(csr, duration=15.0, model='lif', brain_hz=2.0, direct_mn=False, vnc_cpg=False):
    """Full MuJoCo simulation with GUI, adapted from vnc_walk.py."""
    import mujoco as mu
    import glfw

    XML_PATH = FLY_DIR / "models" / "scene.xml"
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.03

    # ─── Brain ───
    if model == 'lif':
        brain = FlyWireLIFBrain(csr)
    elif model == 'hh':
        from flywire_brain import FlyWireHHBrain
        brain = FlyWireHHBrain(csr)
    else:
        from flywire_brain import FlyWireBrain, FlyWirePsiBrain, FlyWireRateBrain
        brain_map = {'rate': FlyWireRateBrain, 'psi': FlyWirePsiBrain, 'izhi': FlyWireBrain}
        brain_cls = brain_map.get(model, FlyWireLIFBrain)
        brain = brain_cls(csr)

    backend_str = 'GPU CUDA' if getattr(brain, '_use_gpu', False) else 'Rust CPU'
    print(f"  [V2] {model.upper()} brain: {csr.n_neurons:,} neurons, "
          f"{len(csr.dn_indices)} DN, {len(csr.mn_indices)} MN")
    print(f"  [V2] Backend: {backend_str}")

    # ─── Food ───
    food_geom_id = mu.mj_name2id(mj_model, mu.mjtObj.mjOBJ_GEOM, 'food')
    tracking_food = food_geom_id >= 0

    # ─── Controller: VNC+CPG, VNC Kuramoto, или прямая MN → actuator ───
    if direct_mn:
        vnc = DirectMNController(mj_model, csr, brain)
        print(f"  [Controller] 🧠 Direct MN → actuators (NO CPG)")
    elif vnc_cpg:
        vnc = VNCKuramotoController(mj_model, csr, brain,
                                     vnc_bin="/tmp/vnc_cluster1.bin",
                                     omega_base=8.0, coupling_scale=30.0)
        print(f"  [Controller] 🧬 VNC Kuramoto CPG (Cluster1, 5,681 neurons)")
    else:
        vnc = V2VNCController(mj_model, csr, brain)
        print(f"  [Controller] 🦗 VNC + CPG ({vnc._cpg_blend*100:.0f}% CPG blend)")

    # Warmup HH brain offline before body coupling
    if model == 'hh' and hasattr(brain, 'warmup'):
        # Set DNg100 injection during warmup so network reaches correct state
        if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None:
            brain._injection_targets[vnc._dng100_idx] = vnc._injection_current
            print(f"  [V2] Warmup with DNg100 injection ({vnc._injection_current}pA)")
        print("  [V2] Warming up HH brain (500ms offline)...")
        brain.warmup(500.0)
        print("  [V2] Warmup done.")

    # ─── GUI ───
    if not glfw.init():
        print("  glfw.init() failed")
        return
    window = glfw.create_window(1200, 800, "🧠 Fly VNC Walk (Refined Weights)", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    # Camera
    scene = mu.MjvScene(mj_model, maxgeom=10000)
    cam = mu.MjvCamera()
    thorax_id = mu.mj_name2id(mj_model, mu.mjtObj.mjOBJ_BODY, 'thorax')
    cam.type = mu.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = thorax_id
    cam.distance = 0.35
    cam.azimuth = 135
    cam.elevation = -25
    opt = mu.MjvOption()
    pert = mu.MjvPerturb()
    ctx = mu.MjrContext(mj_model, mu.mjtFontScale.mjFONTSCALE_150)

    # Settle
    data.qpos[2] = 0.03
    vnc.standby(data)
    print("\n  Settle (0.2s)...")
    for _ in range(2000):
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)
    print(f"  z={data.qpos[2]:.4f} contacts={data.ncon}")

    # ─── Main loop ───
    prev_x, prev_y = float(data.qpos[0]), float(data.qpos[1])
    prev_time = 0.0

    vnc.reset()
    control_mode = '🧠 DIRECT MN' if direct_mn else '🧬 VNC KURAMOTO' if vnc_cpg else '🦗 VNC+CPG'
    print(f"\n  ✅ Running MuJoCo with REFINED weights ({duration:.0f}s)")
    print(f"  Control: {control_mode}")
    print("  Press R to reset, arrows to turn, 1/2 for DN freq, 3/4 for CPG freq, F toggle flight")
    has_gpu = getattr(brain, '_use_gpu', False) or getattr(brain, 'use_gpu', False)
    print(f"  Backend: {'⚡GPU' if has_gpu else '💻CPU'}")

    sim_t = 0.0
    steps = 0
    frames = 0
    t0 = time.time()
    brain_step_interval = max(1, int(1.0 / brain_hz / mj_model.opt.timestep))
    _render_ms = 33

    # Track displacement
    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])

    while not glfw.window_should_close(window):
        dt = mj_model.opt.timestep

        if brain is not None and steps % brain_step_interval == 0:
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            svx = (sx - prev_x) / max(dt, 1e-6)
            svy = (sy - prev_y) / max(dt, 1e-6)
            prev_x, prev_y, prev_time = sx, sy, sim_t

            food_x = food_y = None
            if tracking_food:
                food_x = float(data.geom_xpos[food_geom_id][0])
                food_y = float(data.geom_xpos[food_geom_id][1])

            brain.set_sensory_state(sx, sy, svx, svy, food_x=food_x, food_y=food_y)

            if tracking_food and food_x is not None:
                fdx = food_x - sx
                fdy = food_y - sy
                food_dist = math.sqrt(fdx*fdx + fdy*fdy)
                if food_dist > 0.001:
                    world_bearing = math.atan2(fdy, fdx)
                    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
                    fly_yaw = quat_to_yaw(qw, qx, qy, qz)
                    rel_bearing = world_bearing - fly_yaw
                    while rel_bearing > math.pi: rel_bearing -= 2.0 * math.pi
                    while rel_bearing < -math.pi: rel_bearing += 2.0 * math.pi
                    brain.food_bearing = rel_bearing
                    brain.food_distance = food_dist

            # DNg100 injection (walking drive)
            if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None:
                drive_mod = 1.0
                if hasattr(brain, 'looming') and brain.looming.looming > 0.05:
                    drive_mod = max(0.0, 1.0 - brain.looming.looming * 1.8)
                brain._injection_targets[vnc._dng100_idx] = vnc._injection_current * drive_mod
                brain._dng_mod = drive_mod
            if hasattr(vnc, '_cb0013_idx') and vnc._cb0013_idx is not None:
                brain._injection_targets[vnc._cb0013_idx] = vnc._injection_current_cb0013

            # ─── Grooming DN injection (brain-driven) ───
            if getattr(vnc, '_grooming', False):
                # T1 grooming DNs (DNg105, DNg108, DNg93, DNg74) — fire T1 MNs
                for dn_list_idx in getattr(vnc, '_grooming_t1_dn_indices', []):
                    if dn_list_idx < len(csr.dn_indices):
                        brain._injection_targets[csr.dn_indices[dn_list_idx]] = 80.0
                # Head grooming DNs (DNg24, DNge104, DNg91) — fire head/antenna
                for dn_list_idx in getattr(vnc, '_grooming_head_dn_indices', []):
                    if dn_list_idx < len(csr.dn_indices):
                        brain._injection_targets[csr.dn_indices[dn_list_idx]] = 40.0
                # Reduce walking drive during grooming
                if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None:
                    brain._injection_targets[vnc._dng100_idx] = 10.0

            brain.step_batch(sim_t)

        if steps % 2 == 0:
            vnc.step(dt * 2.0, data)

        mu.mj_step(mj_model, data)
        sim_t += dt
        steps += 1
        poll_keys(window, vnc)

        # Render
        now = time.time()
        if (now - t0) * 1000 >= _render_ms * frames:
            frames += 1
            mu.mjv_updateScene(mj_model, data, opt, pert, cam, mu.mjtCatBit.mjCAT_ALL, scene)
            vp = mu.MjrRect(0, 0, 1200, 800)
            mu.mjr_render(vp, scene, ctx)

            # HUD
            dng_drive = vnc._injection_current * getattr(brain, '_dng_mod', 1.0)
            walk_status = '🏃' if dng_drive > 20 else ('🚶' if dng_drive > 5 else '🧘')
            dx = data.qpos[0] - start_x
            dy = data.qpos[1] - start_y
            dist = math.sqrt(dx*dx + dy*dy)

            grooming_icon = '🧼 ' if getattr(vnc, '_grooming', False) else ''
            title = f"{grooming_icon}{walk_status} REFINED t={sim_t:.1f}s  DNg={dng_drive:.0f}pA  dist={dist:.3f}m"
            if direct_mn:
                info = f"🧠 DIRECT MN  legs={" ".join(f'{a:.2f}' for a in vnc._leg_act)}\n"
                info += f"x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}"
            elif vnc_cpg:
                sin_a = getattr(vnc, '_kur_sin_a', 0.0)
                sin_b = getattr(vnc, '_kur_sin_b', 0.0)
                pd = getattr(vnc, '_kur_phase_diff', 0.0)
                info = f"🧬 VNC KUR  A={sin_a:.2f}  B={sin_b:.2f}  Δφ={pd:.2f}\n"
                info += f"x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}"
            else:
                cpg_info = f'CPG {vnc._cpg_freq:.1f}Hz ×{vnc._cpg_amp:.2f}'
                info = f"DNs={len(csr.dn_indices)}  {cpg_info}\n"
                info += f"x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}"

            mu.mjr_overlay(mu.mjtFontScale.mjFONTSCALE_150,
                mu.mjtGridPos.mjGRID_TOPLEFT, vp, title, info, ctx)
            glfw.swap_buffers(window)

        glfw.poll_events()

        if sim_t >= duration:
            break

    elapsed = time.time() - t0
    dx = data.qpos[0] - start_x
    dy = data.qpos[1] - start_y
    dist = math.sqrt(dx*dx + dy*dy)
    print(f"\n  ✅ Simulation complete!")
    print(f"  Duration: {sim_t:.1f}s real, {elapsed:.1f}s wall (RT={sim_t/elapsed:.2f}x)")
    print(f"  Displacement: ({dx:.3f}, {dy:.3f}) = {dist:.3f}m")
    print(f"  Final z={data.qpos[2]:.4f}")

    glfw.destroy_window(window)
    glfw.terminate()


def run_simulation_headless(csr, duration=5.0, model='lif'):
    """Headless MuJoCo for benchmark comparison."""
    import mujoco as mu
    from flywire_brain import FlyWireRateBrain

    XML_PATH = FLY_DIR / "models" / "scene.xml"
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.03

    # Rate-based for speed
    brain = FlyWireRateBrain(csr)
    vnc = V2VNCController(mj_model, csr, brain)

    # Settle
    for _ in range(2000):
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)

    vnc.reset()
    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    sim_t = 0.0
    steps = 0
    t0 = time.time()
    brain_step_interval = max(1, int(0.025 / mj_model.opt.timestep))

    while sim_t < duration:
        dt = mj_model.opt.timestep
        if steps % brain_step_interval == 0:
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            brain.set_sensory_state(sx, sy)
            brain._injection_targets = {vnc._dng100_idx: vnc._injection_current} if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None else {}
            brain.step_batch(sim_t)
        if steps % 2 == 0:
            vnc.step(dt * 2.0, data)
        mu.mj_step(mj_model, data)
        sim_t += dt
        steps += 1

    elapsed = time.time() - t0
    dx = data.qpos[0] - start_x
    dy = data.qpos[1] - start_y
    dist = math.sqrt(dx*dx + dy*dy)
    print(f"\n  ✅ Headless run complete!")
    print(f"  Duration: {sim_t:.1f}s real, {elapsed:.1f}s wall (RT={sim_t/elapsed:.2f}x)")
    print(f"  Displacement: ({dx:.3f}, {dy:.3f}) = {dist:.3f}m")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧠 FlyWire Rust Refiner → MuJoCo Bridge")
    parser.add_argument("--weights", type=str, default="/tmp/fly_refined_weights.bin",
                        help="Path to refined weights from Rust refiner")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--model", type=str, choices=['izhi', 'psi', 'rate', 'lif', 'hh'], default='lif')
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (for benchmark)")
    parser.add_argument("--brain-hz", type=float, default=2.0,
                        help="Brain update frequency in Hz (default: 2.0)")
    parser.add_argument("--direct-mn", action="store_true",
                        help="Direct MN → actuators, no CPG")
    parser.add_argument("--vnc-cpg", action="store_true",
                        help="Use VNC Kuramoto CPG instead of sin() CPG")
    args = parser.parse_args()

    has_weights = Path(args.weights).exists()
    if not has_weights:
        print(f"⚠ Refined weights not found: {args.weights}")
        print("   Running with original CSR (unrefined)")

    run_comparison(duration=args.duration, model=args.model, headless=args.headless,
                    brain_hz=args.brain_hz, direct_mn=args.direct_mn, vnc_cpg=args.vnc_cpg)


if __name__ == "__main__":
    main()
