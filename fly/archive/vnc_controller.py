#!/usr/bin/env python3
"""
V2VNC Controller — archived MN→actuator controller.

Архивирован при переходе на полный FlyWire connectome.
Может быть полезен для:
- Роботов (MN → actuator mapping)
- Отладки отдельных ног
- Сравнения neural drive vs hand-coded CPG

Для использования:
  from fly.archive.vnc_controller import V2VNCController
"""

import math
from collections import defaultdict

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Анатомия (из vnc_walk.py)
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

TRIPOD_A = {"T1_left","T2_right","T3_left"}
TRIPOD_B = {"T1_right","T2_left","T3_right"}

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

V2_LEG_MAP = {
    'front_leg_motor_neuron': 'T1',
    'middle_leg_motor_neuron': 'T2',
    'hind_leg_motor_neuron': 'T3',
}


class V2VNCController:
    """V2 VNC Controller — управляет MuJoCo через MN rates напрямую из мозга.

    Архивирован. Использовать только для роботов или отладки.
    """

    def __init__(self, model, csr, brain):
        import mujoco
        self.csr = csr
        self.brain = brain
        self.time = 0.0
        self.v2_mode = True

        # GUI compatibility
        self.dn_act = np.zeros(1, dtype=np.float32)
        self.dn_freq = 2.0
        self.dn_amp = 1.0
        self.turn_bias = 0.0

        class CPGCompat:
            def __init__(self, controller):
                self.base_freq = 2.0
            def reset(self):
                pass
        self.cpg = CPGCompat(self)

        # DNg100 injection
        self._dng100_idx = None
        dng100_info = csr.named_dns.get('DNg100')
        if dng100_info is not None:
            self._dng100_idx = dng100_info[1]
        self._injection_current = 80.0

        # CB0013 injection
        self._cb0013_idx = None
        for rid, idx in csr.root_to_idx.items():
            typ = csr.type_map.get(rid, '')
            if typ == 'CB0013':
                self._cb0013_idx = idx
                break
        self._injection_current_cb0013 = 5.0

        # Touch sensor addresses
        self._touch_addr = np.full(6, -1, dtype=np.int32)
        for i, (_, lname) in enumerate(LEGS):
            sn = f"touch_claw_{lname}"
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sn)
            if sid >= 0:
                self._touch_addr[i] = model.sensor_adr[sid]

        # MuJoCo actuator map
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

        self._flight_enabled = False
        self._wing_k = 0.00005
        self._prev_pitch_l = 0.0
        self._prev_pitch_r = 0.0
        self._thorax_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')
        # Wing MN → actuator mapping
        WING_MN_TYPE_MAP = {
            'dlm1-4': 'pitch', 'dlm5': 'pitch',
            'dvm1a-c': 'pitch', 'dvm2a-b': 'pitch',
            'tp1': 'pitch', 'tp2': 'pitch', 'tpn': 'pitch',
            'b1': 'yaw', 'b2': 'yaw', 'b3': 'roll',
            'i1': 'roll', 'i2': 'roll',
            'iii1': 'yaw', 'iii3': 'yaw', 'iii4': 'roll',
            'ps1': 'roll', 'ps2': 'roll', 'psi': 'yaw',
        }
        self._wing_mn_map = []
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
        print(f"  [V2VNC] Wing drive: {n_wing_mn} MNs mapped (brain-driven amplitude)")
        n_leg = len(self.actuator_map)
        n_nl = len(self.non_leg_actuators)

        # Build MN → actuator mapping
        self.mn_to_ctrl = {}
        n_mapped = 0
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

                for leg_i, (_, lname) in enumerate(LEGS):
                    if lname == leg_label:
                        self._actuator_leg_idx[act_idx] = leg_i
                        break

        # Leg activation summary
        self._leg_act = np.zeros(6, dtype=np.float32)
        self._display_gains = np.ones(6)
        self._actuator_gain = 3.0

        # Mirror pairs
        self._mirror_pairs = []
        left_acts = {}
        right_acts = {}
        for (lname, seg), act_idx in self.actuator_map.items():
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

        # ─── Rust VNC: flat mapping arrays ───
        self._rust_map_mn_idx = []
        self._rust_map_act_idx = []
        self._rust_map_scale = []
        self._rust_map_leg_idx = []
        for i, rid in enumerate(self.csr.mn_root_ids):
            ctrl_list = self.mn_to_ctrl.get(rid)
            if ctrl_list is None:
                continue
            for act_idx, scale, _ in ctrl_list:
                leg_i = self._actuator_leg_idx.get(act_idx, -1)
                if leg_i < 0:
                    continue
                self._rust_map_mn_idx.append(i)
                self._rust_map_act_idx.append(act_idx)
                self._rust_map_scale.append(scale)
                self._rust_map_leg_idx.append(leg_i)
        self._rust_map_mn_idx = np.array(self._rust_map_mn_idx, dtype=np.int32)
        self._rust_map_act_idx = np.array(self._rust_map_act_idx, dtype=np.int32)
        self._rust_map_scale = np.array(self._rust_map_scale, dtype=np.float32)
        self._rust_map_leg_idx = np.array(self._rust_map_leg_idx, dtype=np.int32)
        # Mirror pairs → flat arrays
        self._rust_mirror_left = np.array([p[0] for p in self._mirror_pairs], dtype=np.int32)
        self._rust_mirror_right = np.array([p[1] for p in self._mirror_pairs], dtype=np.int32)
        # Cached ctrl_out buffer — размер от максимального actuator индекса
        all_act_idxs = list(self.actuator_map.values()) + list(self.non_leg_actuators.values()) + self.adhere
        self._n_actuators = max(all_act_idxs) + 1 if all_act_idxs else 0
        self._rust_ctrl_out = np.zeros(self._n_actuators, dtype=np.float64)

        # Rust VNC check
        self._tcm = None
        self._use_rust_vnc = False
        try:
            import taas_checkmate as _tcm
            if hasattr(_tcm, 'vnc') and hasattr(_tcm.vnc, 'vnc_step'):
                self._use_rust_vnc = True
                self._tcm = _tcm
        except ImportError:
            pass

        # MN rate interpolation
        self._brain_dt = 0.025
        self._prev_mn_rates = None
        self._cur_mn_rates = None
        self._last_brain_time = -0.025

        # ─── CPG (Central Pattern Generator) for walking ───
        self._cpg_time = 0.0
        self._cpg_freq = 8.0     # Hz — частота шагания (Drosophila: 20-30Hz, но real-time ≈8-12)
        self._cpg_amp = 0.5      # coxa twist amplitude (radians)
        self._cpg_blend = 0.9    # 1.0 = pure CPG, 0.0 = pure brain (high=clean gait, low=noisy)
        self._cpg_enabled = True
        self._cpg_leg_phases = {}  # leg_label → phase offset
        for _, lname in LEGS:
            in_tripod_a = lname in TRIPOD_A
            self._cpg_leg_phases[lname] = 0.0 if in_tripod_a else math.pi
        # Brain baseline (saved before CPG to avoid recursive blend)
        self._brain_ctrl_base = {}

    def reset(self):
        self.time = 0.0
        self._cpg_time = 0.0
        self._leg_act.fill(0.0)

    def standby(self, data):
        """Standing posture — удерживает муху на ногах во время settle."""
        for act_idx in self.actuator_map.values():
            data.ctrl[act_idx] = -0.15
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
            rc = 1.0 * roll + 0.1 * rv
            pc = 1.0 * pitch + 0.1 * pv
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
        """MN rates → actuator drive (один шаг MuJoCo)."""
        self.time += dt
        mn_rates_raw = self.brain.get_mn_rates()

        # ─── 0. MN rate interpolation ───
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

        # ─── 1. Contact feedback ───
        contact_boost = np.zeros(6, dtype=np.float64)
        if data is not None and hasattr(data, 'sensordata') and data.sensordata is not None:
            sd = data.sensordata
            for i in range(6):
                addr = int(self._touch_addr[i])
                if 0 <= addr < len(sd):
                    contact = abs(float(sd[addr])) > 1e-7
                    if contact:
                        contact_boost[i] = 0.3

        # ─── 2. Balance gain ───
        balance_gain = self._compute_balance_gain(data, dt)
        self._display_gains = balance_gain.copy()

        # ─── 3. MN rates → actuator drive (Rust) ───
        if self._use_rust_vnc and len(mn_rates) > 0 and len(self._rust_map_mn_idx) > 0:
            # Zero ctrl_out first (Rust only writes mapped actuators)
            self._rust_ctrl_out.fill(0.0)
            self._tcm.vnc.vnc_step(
                mn_rates.astype(np.float32, copy=False),
                balance_gain.astype(np.float64, copy=False),
                contact_boost.astype(np.float64, copy=False),
                float(self._actuator_gain),
                self._rust_map_mn_idx,
                self._rust_map_act_idx,
                self._rust_map_scale,
                self._rust_map_leg_idx,
                self._rust_mirror_left,
                self._rust_mirror_right,
                self._rust_ctrl_out,
            )
            # Apply to actuators + leg_act + save brain baseline
            self._leg_act.fill(0.0)
            self._brain_ctrl_base.clear()
            for (lname, seg), act_idx in self.actuator_map.items():
                li = None
                for i, (_, ln) in enumerate(LEGS):
                    if ln == lname:
                        li = i
                        break
                if li is None:
                    continue
                ctrl_val = self._rust_ctrl_out[act_idx]
                data.ctrl[act_idx] = ctrl_val
                self._brain_ctrl_base[(lname, seg)] = ctrl_val
                self._leg_act[li] += abs(float(ctrl_val))
        else:
            # Python fallback (если Rust VNC недоступен)
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
                    val = rate * float(balance_gain[leg_i]) + float(contact_boost[leg_i])
                    act_drive[act_idx] += scale * max(-1.0, min(1.0, val))
                    act_count[act_idx] += 1

            for act_idx in act_drive:
                if act_count.get(act_idx, 0) > 1:
                    act_drive[act_idx] /= act_count[act_idx]

            # Symmetrize L/R
            for left_idx, right_idx in self._mirror_pairs:
                lv = act_drive.get(left_idx, 0.0)
                rv = act_drive.get(right_idx, 0.0)
                avg = (lv + rv) * 0.5
                if avg != 0.0:
                    act_drive[left_idx] = avg
                    act_drive[right_idx] = avg

            # Apply to actuators + save brain baseline
            self._leg_act.fill(0.0)
            self._brain_ctrl_base.clear()
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
                self._brain_ctrl_base[(lname, seg)] = ctrl_val
                self._leg_act[li] += abs(float(ctrl_val))

        # ─── CPG drive (additive to brain baseline — не рекурсивный blend) ───
        if self._cpg_enabled:
            self._cpg_time += dt
            cpg_base_phase = self._cpg_time * 2.0 * math.pi * self._cpg_freq
            for leg_i, (_, lname) in enumerate(LEGS):
                leg_phase_offset = self._cpg_leg_phases.get(lname, 0.0)
                leg_phase = cpg_base_phase + leg_phase_offset

                # Coxa twist: forward/backward sweep
                # Используем fresh brain baseline, не рекурсивный data.ctrl (который уже содержит CPG)
                twist_act = self.actuator_map.get((lname, 'coxa_twist'))
                if twist_act is not None:
                    brain_val = self._brain_ctrl_base.get((lname, 'coxa_twist'), 0.0)
                    cpg_val = math.sin(leg_phase) * self._cpg_amp
                    data.ctrl[twist_act] = (1.0 - self._cpg_blend) * brain_val + self._cpg_blend * cpg_val

                # Coxa abduct: lift during swing
                abduct_act = self.actuator_map.get((lname, 'coxa_abduct'))
                if abduct_act is not None:
                    brain_val = self._brain_ctrl_base.get((lname, 'coxa_abduct'), 0.0)
                    cpg_val = max(0.0, math.sin(leg_phase + math.pi)) * self._cpg_amp * 0.6
                    data.ctrl[abduct_act] = (1.0 - self._cpg_blend) * brain_val + self._cpg_blend * cpg_val

                # Femur: flex during swing (lift leg for ground clearance)
                for seg_name in ['femur', 'femur_twist']:
                    femur_act = self.actuator_map.get((lname, seg_name))
                    if femur_act is not None:
                        brain_val = self._brain_ctrl_base.get((lname, seg_name), 0.0)
                        cpg_val = -math.sin(leg_phase + math.pi * 0.75) * self._cpg_amp * 0.6
                        data.ctrl[femur_act] = (1.0 - self._cpg_blend) * brain_val + self._cpg_blend * cpg_val

                # Tibia: extend during stance (push ground), flex during swing
                tibia_act = self.actuator_map.get((lname, 'tibia'))
                if tibia_act is not None:
                    brain_val = self._brain_ctrl_base.get((lname, 'tibia'), 0.0)
                    cpg_val = -math.sin(leg_phase + math.pi * 0.5) * self._cpg_amp * 0.8
                    data.ctrl[tibia_act] = (1.0 - self._cpg_blend) * brain_val + self._cpg_blend * cpg_val


        # Non-leg: abdomen
        sp = getattr(self, '_sez_phase', 0.0)
        sp += dt * 2.0 * math.pi * 1.0
        self._sez_phase = sp
        for seg in ABDOMEN_SEGMENTS:
            idx = self.non_leg_actuators.get(('abdomen', seg))
            if idx is not None:
                data.ctrl[idx] = 0.06 * math.sin(sp)

        # Wings: brain-driven CPG oscillator + lift
        if getattr(self, '_thorax_body_id', -1) >= 0:
            data.xfrc_applied[self._thorax_body_id, 2] = 0.0
        if getattr(self, '_flight_enabled', True):
            # Brain-driven amplitude
            pitch_amp = 0.1
            yaw_off = 0.0
            roll_off = 0.0
            wm = getattr(self, '_wing_mn_map', [])
            if wm and len(mn_rates) > 0:
                left_power = right_power = 0.0
                n_left = n_right = 0
                for mn_i, side, dof in wm:
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
                if n_left > 0: avg += left_power / n_left
                if n_right > 0: avg += right_power / n_right
                div = (1 if n_left > 0 else 0) + (1 if n_right > 0 else 0)
                if div > 0: avg /= div
                pitch_amp = 0.1 + avg * 0.9
                yaw_off = max(-0.3, min(0.3, yaw_off))
                roll_off = max(-0.3, min(0.3, roll_off))
            wing_freq = 150.0
            t = self.time
            pitch_l = math.sin(2 * math.pi * wing_freq * t) * pitch_amp
            pitch_r = math.sin(2 * math.pi * wing_freq * t + roll_off * 0.5) * pitch_amp
            roll = math.sin(2 * math.pi * wing_freq * 2 * t + 0.5) * 0.3 + roll_off
            yaw = math.sin(2 * math.pi * wing_freq * t + 1.2) * 0.2 + yaw_off
            for side, sign in [('left', 1), ('right', -1)]:
                idx_pitch = self.non_leg_actuators.get(('wing', f'wing_pitch_{side}'))
                idx_roll = self.non_leg_actuators.get(('wing', f'wing_roll_{side}'))
                idx_yaw = self.non_leg_actuators.get(('wing', f'wing_yaw_{side}'))
                if idx_pitch is not None: data.ctrl[idx_pitch] = pitch_l if side == 'left' else pitch_r
                if idx_roll is not None: data.ctrl[idx_roll] = (roll + (roll_off if side == 'left' else -roll_off)) * sign
                if idx_yaw is not None: data.ctrl[idx_yaw] = yaw * sign
            if hasattr(self, '_thorax_body_id') and self._thorax_body_id >= 0:
                v_l = (pitch_l - getattr(self, '_prev_pitch_l', 0.0)) / max(dt, 1e-8)
                v_r = (pitch_r - getattr(self, '_prev_pitch_r', 0.0)) / max(dt, 1e-8)
                k = getattr(self, '_wing_k', 0.00005)
                lift = k * (v_l * abs(v_l) + v_r * abs(v_r))
                data.xfrc_applied[self._thorax_body_id, 2] += lift
                self._prev_pitch_l = pitch_l
                self._prev_pitch_r = pitch_r

        # Adhesion: contact-based
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
