#!/usr/bin/env python3
"""
🧬🦗 VNC CPG → ATLASOV → MUJOCO FLY
======================================
Биологически точный Central Pattern Generator на основе Franken/MANC данных.

Архитектура:
  [Brain: GO/STOP] → Descending cholinergic (2,620 нейронов)
       ↓
  [VNC GABA CPG Ring Oscillators] — 13K VNC intrinsic (37.6% GABA)
       ↕ reciprocal inhibition
  [Glutamatergic Motor Neurons] — 396 leg_motor (359 glutamat!)
       ↓ flexor/extensor alternation
  [Atlasov Ca²⁺ Sarcoplasmic Engine]
       ↓ ΔF - Kq - Bq̇
  [MuJoCo 3D Physics — 66 joints]

Tripod Gait: L1+R2+L3 vs R1+L2+R3
Каждый грудной сегмент (T1/T2/T3) управляет своей парой ног.

Author / Concept: Atlasov Anton (Атласов Антон)
Data: Franken/FlyWire/MANC
"""

import sys
import time
import math
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

import mujoco as mu
import taas_checkmate as tc


# ═════════════════════════════════════════════════════════════════════
#  VNC CPG — Biological Central Pattern Generator
# ═════════════════════════════════════════════════════════════════════

class VNCCPG:
    """VNC Central Pattern Generator с GABA reciprocal inhibition.
    
    Биология (из Franken data):
    - VNC intrinsic: 13K нейронов, 37.6% GABA, 40.6% ACh, 21.6% Glu
    - Motor neurons: 396 leg_motor, 359 glutamate (90.7%)
    - 3 thoracic segments (T1/T2/T3), каждая пара ног
    
    CPG архитектура:
    - 6 half-center oscillators (по одному на ногу)
    - GABA-ингибирование между flexor/extensor группами
    - Glutamatergic motor neurons как выход
    - Tripod coupling: L1↔R2↔L3, R1↔L2↔R3
    """
    
    def __init__(self, freq_hz=6.0):
        # Thoracic segments and legs
        self.legs = ['T1L', 'T1R', 'T2L', 'T2R', 'T3L', 'T3R']
        self.n_legs = 6
        
        # Tripod groups
        self.tripod_A = [0, 3, 4]  # L1(0), R2(3), L3(4)
        self.tripod_B = [1, 2, 5]  # R1(1), L2(2), R3(5)
        
        # CPG parameters (биологические)
        self.freq = freq_hz  # Hz
        self.omega = 2.0 * math.pi * freq_hz  # rad/s
        self.dt = 0.0005  # 0.5ms (совпадает с brain dt)
        
        # GABA coupling strength (reciprocal inhibition)
        self.gaba_strength = 3.0  # strength of reciprocal inhibition
        self.tripod_coupling = 0.5  # cross-segment coupling
        
        # Half-center oscillator state
        # Каждый leg имеет flexor (0) и extensor (1) фазу
        self.phases = np.zeros(self.n_legs, dtype=np.float64)  # [0, 2π)
        self.omegas = np.full(self.n_legs, self.omega, dtype=np.float64)
        
        # Motor neuron state (glutamatergic output)
        self.motor_flexor = np.zeros(self.n_legs, dtype=np.float32)
        self.motor_extensor = np.zeros(self.n_legs, dtype=np.float32)
        
        # Joint kinematics per leg
        self.joint_names = {
            'T1L': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
            'T1R': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
            'T2L': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
            'T2R': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
            'T3L': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
            'T3R': ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur',
                    'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5'],
        }
        
        # Phase offsets for natural gait
        # Tripod A: L1(0°), R2(180°), L3(0°)  — swing phase
        # Tripod B: R1(180°), L2(0°), R3(180°) — stance phase
        self.phase_offsets = np.array([0.0, math.pi, math.pi, 0.0, 0.0, math.pi])
        
        # Descending drive (from brain)
        self.descending_drive = 1.0  # 0.0 = STOP, 1.0 = GO
        self.step_count = 0
        
        # Knee joint kinematics
        # For each leg, define which joints are "flexor" vs "extensor" dominant
        # coxa_abduct = swing (lateral), femur = lift (vertical), tibia = extend
        # tarsus = ground contact (passive)
        self.joint_kinematics = {
            'coxa_abduct': {'amp': 0.3, 'phase': 0.0, 'type': 'swing'},
            'coxa_twist':  {'amp': 0.1, 'phase': 0.0, 'type': 'swing'},
            'coxa':        {'amp': 0.2, 'phase': 0.0, 'type': 'swing'},
            'femur_twist': {'amp': 0.1, 'phase': 0.5, 'type': 'lift'},
            'femur':       {'amp': 0.5, 'phase': 0.0, 'type': 'lift'},
            'tibia':       {'amp': 0.6, 'phase': 0.5, 'type': 'extend'},
            'tarsus':      {'amp': 0.05, 'phase': 0.0, 'type': 'passive'},
            'tarsus2':     {'amp': 0.05, 'phase': 0.0, 'type': 'passive'},
            'tarsus3':     {'amp': 0.05, 'phase': 0.0, 'type': 'passive'},
            'tarsus4':     {'amp': 0.05, 'phase': 0.0, 'type': 'passive'},
            'tarsus5':     {'amp': 0.05, 'phase': 0.0, 'type': 'passive'},
        }
    
    def reset(self):
        """Reset CPG state."""
        rng = np.random.RandomState(42)
        self.phases = rng.uniform(0, 2*math.pi, self.n_legs).astype(np.float64)
        self.motor_flexor.fill(0.0)
        self.motor_extensor.fill(0.0)
        self.step_count = 0
    
    def set_drive(self, drive: float):
        """Set descending drive from brain (0.0=STOP, 1.0=GO)."""
        self.descending_drive = np.clip(drive, 0.0, 1.0)
    
    def step(self):
        """One CPG step: Kuramoto-like GABA reciprocal inhibition."""
        phases = self.phases
        n = self.n_legs
        dt = self.dt
        
        # ── Kuramoto phase dynamics with GABA coupling ──
        dphases = self.omegas.copy() * self.descending_drive
        
        for i in range(n):
            # Reciprocal inhibition: GABA → negative coupling → anti-phase
            for j in range(n):
                if i == j:
                    continue
                # Tripod coupling: same group = inhibitory, opposite = excitatory
                in_same_tripod = ((i in self.tripod_A and j in self.tripod_A) or
                                  (i in self.tripod_B and j in self.tripod_B))
                if in_same_tripod:
                    # Within same tripod: GABA mutual inhibition → synchronize
                    dphases[i] += self.gaba_strength * math.sin(phases[j] - phases[i])
                else:
                    # Cross tripod: push to anti-phase
                    dphases[i] += self.gaba_strength * 0.3 * math.sin(phases[j] - phases[i] + math.pi)
        
        # Update phases
        self.phases = (phases + dt * dphases) % (2.0 * math.pi)
        
        # ── Phase offsets (tripod-specific) ──
        for i in range(n):
            # Each leg has its own offset based on tripod
            offset_phase = (self.phases[i] + self.phase_offsets[i]) % (2.0 * math.pi)
            self.motor_flexor[i] = max(0.0, math.sin(offset_phase))
            self.motor_extensor[i] = max(0.0, -math.sin(offset_phase))
        
        self.step_count += 1
    
    def get_joint_positions(self, leg_idx: int) -> np.ndarray:
        """Get 11 joint angles for a given leg from CPG phase."""
        phase = self.phases[leg_idx] + self.phase_offsets[leg_idx]
        joints = np.zeros(11, dtype=np.float32)
        
        for j, (jname, kin) in enumerate(self.joint_kinematics.items()):
            if kin['type'] == 'passive':
                joints[j] = 0.0
            else:
                # Phase-shifted sine for each joint
                p = phase + kin['phase'] * math.pi
                joints[j] = kin['amp'] * math.sin(p)
        
        return joints
    
    def get_leg_motor_drive(self, leg_idx: int) -> (float, float):
        """Get flexor/extensor drive for Atlasov."""
        return self.motor_flexor[leg_idx], self.motor_extensor[leg_idx]
    
    def coherence(self) -> float:
        """Kuramoto order parameter R ∈ [0, 1]."""
        n = self.n_legs
        sin_sum = np.sin(self.phases).sum()
        cos_sum = np.cos(self.phases).sum()
        return math.sqrt(sin_sum**2 + cos_sum**2) / n


# ═════════════════════════════════════════════════════════════════════
#  Atlasov → MuJoCo Embodied Sim
# ═════════════════════════════════════════════════════════════════════

class AtlasovFlyMuJoCo:
    """Atlasov Neuromotor Engine driving MuJoCo FruitFly."""
    
    LEG_NAMES = ['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']
    N_JOINTS_PER_LEG = 11
    
    def __init__(self, xml_path=SCENE_XML):
        print(f"🦗 Загрузка MuJoCo мухи: {xml_path}")
        self.model = mu.MjModel.from_xml_path(str(xml_path))
        self.data = mu.MjData(self.model)
        
        # Map leg names to DOF indices
        self.leg_dofs = {}
        for leg in self.LEG_NAMES:
            dofs = []
            for i in range(self.model.njnt):
                jname = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_JOINT, i)
                if jname and leg in jname:
                    dofs.append(self.model.jnt_dofadr[i])
            self.leg_dofs[leg] = np.array(dofs, dtype=np.int32)
        
        self.n_legs = 6
        self.n_joints = sum(len(d) for d in self.leg_dofs.values())
        self.n_muscles = math.ceil((self.n_joints * 2) / 8) * 8  # SIMD pad
        
        print(f"   {self.n_legs} legs, {self.n_joints} total joints, {self.n_muscles} muscles")
        
        # Atlasov muscle state
        self.ca = np.zeros(self.n_muscles, dtype=np.float32)
        self.tension = np.zeros(self.n_muscles, dtype=np.float32)
        self.spikes = np.zeros(self.n_muscles, dtype=np.int32)
        self.resets = np.zeros(self.n_muscles, dtype=np.int32)
        
        self.d_ca = np.full(self.n_muscles, 0.6, dtype=np.float32)
        self.tau_pump = np.full(self.n_muscles, 20.0, dtype=np.float32)
        self.eta_reset = np.full(self.n_muscles, 0.8, dtype=np.float32)
        self.t_max = np.full(self.n_muscles, 1.2, dtype=np.float32)        # 1.2 dynes·cm — деликатная нежная мышца
        self.kd = np.full(self.n_muscles, 0.3, dtype=np.float32)
        
        self.joint_df = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_stf = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_dmp = np.zeros(self.n_muscles // 2, dtype=np.float32)
        
        # Органическая мягкость и амортизация (гибкие суставы без бетонирования)
        self.k0 = np.full(self.n_muscles // 2, 0.01, dtype=np.float32)    # Мягкие эластичные суставы
        self.beta = np.full(self.n_muscles // 2, 0.05, dtype=np.float32)
        self.b0 = np.full(self.n_muscles // 2, 0.10, dtype=np.float32)    # Жидкое вязкое гашение вибраций
        self.alpha = np.full(self.n_muscles // 2, 0.05, dtype=np.float32)
        
        self.dt = self.model.opt.timestep
        self.step_fn = tc.neurons.step_atlasov_neuromotor
        self.step_count = 0
        
        # Build flat joint index list
        self.all_dofs = np.concatenate(list(self.leg_dofs.values()))
        
    def step(self, joint_angles):
        """Apply CPG-derived joint angles to MuJoCo fly.
        
        joint_angles: [66] array — непрерывные углы для всех 66 суставов
        (6 ног × 11 joints). Использует direct position control (qfrc_applied).
        """
        # Map joint angles to muscle spikes for Atlasov
        # Каждый сустав → flexor (0) + extensor (1) мышца
        # Если угол > 0 → flexor spike, если < 0 → extensor spike
        self.spikes.fill(0)
        self.resets.fill(0)
        
        for j_idx in range(min(len(joint_angles), self.n_joints)):
            angle = joint_angles[j_idx]
            # Flexor muscle at even index, extensor at odd index
            flex_idx = 2 * j_idx
            ext_idx = 2 * j_idx + 1
            if flex_idx < self.n_muscles:
                self.spikes[flex_idx] = 1 if angle > 0.05 else 0
            if ext_idx < self.n_muscles:
                self.spikes[ext_idx] = 1 if angle < -0.05 else 0
        
        # Atlasov Ca²⁺ step (Rust SIMD)
        self.step_fn(
            self.ca, self.tension, self.spikes, self.resets,
            float(self.dt * 1000.0),
            self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
            self.joint_df, self.joint_stf, self.joint_dmp,
            self.k0, self.beta, self.b0, self.alpha
        )
        
        # Apply joint torque → MuJoCo
        self.data.qfrc_applied.fill(0.0)
        for j_idx in range(min(len(self.joint_df), self.n_joints)):
            if j_idx < len(self.all_dofs):
                dof = self.all_dofs[j_idx]
                q = self.data.qpos[dof]
                qd = self.data.qvel[dof]
                # Use joint_df from Atlasov + impedance modulation
                tau = self.joint_df[j_idx] - self.joint_stf[j_idx] * q - self.joint_dmp[j_idx] * qd
                self.data.qfrc_applied[dof] = np.clip(tau, -1.5, 1.5)
        
        mu.mj_step(self.model, self.data)
        self.step_count += 1
        
        return {
            't': self.data.time,
            'pos': self.data.qpos[:3].copy(),
            'df_mean': float(np.mean(np.abs(self.joint_df[:self.n_joints]))),
            'k_mean': float(np.mean(self.joint_stf[:self.n_joints])) * 1e3,
        }


# ═════════════════════════════════════════════════════════════════════
#  MAIN — VNC CPG → Atlasov → MuJoCo
# ═════════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("🧬🦗 VNC CPG → ATLASOV → MUJOCO FLY")
    print("   Biological Central Pattern Generator with GABA reciprocal inhibition")
    print("   Data: Franken/MANC — 37.6% GABA VNC, 359 glutamatergic leg motor neurons")
    print("═" * 80)
    
    # ── 1. VNC CPG ────────────────────────────────────────────────
    cpg = VNCCPG(freq_hz=6.0)
    print(f"\n🧬 VNC CPG создан:")
    print(f"   6 half-center oscillators (GABA reciprocal inhibition)")
    print(f"   Tripod: {cpg.tripod_A} vs {cpg.tripod_B}")
    print(f"   Frequency: {cpg.freq} Hz")
    
    # ── 2. Atlasov MuJoCo ─────────────────────────────────────────
    fly = AtlasovFlyMuJoCo()
    
    # ── 3. Parameters ─────────────────────────────────────────────
    n_steps = 10000  # 5 seconds
    dt_ms = 0.5
    
    cpg.dt = dt_ms / 1000.0
    cpg.reset()
    
    print(f"\n{'─' * 80}")
    print(f"🎬 ЗАПУСК СИМУЛЯЦИИ: {n_steps} шагов ({n_steps*dt_ms:.0f}ms)")
    print(f"   Все 66 суставов получают непрерывный CPG драйв")
    print(f"{'─' * 80}\n")
    
    # Tracking
    start_pos = None
    displacement = 0.0
    walk_log = []
    
    t_start = time.time()
    
    for step in range(n_steps):
        # ── A. VNC CPG step ───────────────────────────────────────
        cpg.set_drive(1.0)  # GO!
        cpg.step()
        
        # ── B. Build joint angles from CPG ────────────────────────
        # Каждая нога = 11 суставов с кинематикой
        # All 66 joints = concatenated from 6 legs
        joint_angles = np.concatenate([
            cpg.get_joint_positions(i) for i in range(6)
        ])  # [66] continuous values in [-1, 1]
        
        # ── C. Atlasov → MuJoCo step ──────────────────────────────
        out = fly.step(joint_angles)
        
        if start_pos is None:
            start_pos = out['pos'].copy()
        displacement = np.linalg.norm(out['pos'] - start_pos) * 1000
        
        walk_log.append({
            't': out['t'],
            'pos': out['pos'].copy(),
            'disp': displacement,
            'k': out['k_mean'],
            'coh': cpg.coherence(),
        })
        
        # ── D. Progress ────────────────────────────────────────────
        if step % 500 == 0 or step == 0 or step == n_steps - 1:
            elapsed = time.time() - t_start
            fps = (step + 1) / (elapsed + 1e-6)
            coh = cpg.coherence()
            phases_deg = np.degrees(cpg.phases) % 360
            
            print(f"⏱ Step {step:5d}/{n_steps} | "
                  f"Coh: {coh:.3f} | "
                  f"ϕ: [{phases_deg[0]:3.0f}° {phases_deg[1]:3.0f}° {phases_deg[2]:3.0f}° {phases_deg[3]:3.0f}° {phases_deg[4]:3.0f}° {phases_deg[5]:3.0f}°] | "
                  f"Disp: {displacement:.2f}mm | {fps:.0f} s/s")
    
    total_time = time.time() - t_start
    total_walk_time = n_steps * dt_ms / 1000
    speed_mm_s = displacement / total_walk_time if total_walk_time > 0 else 0.0
    
    # ═══════════════════════════════════════════════════════════════
    #  GAIT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("🏆 РЕЗУЛЬТАТЫ VNC CPG → ATLASOV → MUJOCO!")
    print("═" * 80)
    
    # Per-segment speed
    seg_size = 2000
    seg_speeds = []
    print(f"\n📊 ПОСЕГМЕНТНЫЙ АНАЛИЗ СКОРОСТИ (каждые 1 сек):")
    print(f"{'Сектор':<10} | {'Disp (mm)':<12} | {'Скорость (mm/s)':<18} | {'Coh':<8}")
    print("-" * 55)
    for s in range(5):
        idx = min((s + 1) * seg_size, len(walk_log)) - 1
        d = walk_log[idx]['disp']
        prev_d = walk_log[max(0, s * seg_size - 1)]['disp']
        seg_speed = (d - prev_d) / (seg_size * dt_ms / 1000)
        coh = walk_log[idx]['coh']
        seg_speeds.append(seg_speed)
        print(f"  {s+1:<8} | {d:<10.2f}  | {seg_speed:<16.2f}  | {coh:<.3f}")
    
    avg_speed = np.mean(seg_speeds)
    speed_cv = np.std(seg_speeds) / (np.mean(seg_speeds) + 1e-10)
    
    # Gait classification
    if avg_speed < 1.0:
        gait = "🟤 ПОЛЗЁТ (crawling)"
    elif avg_speed < 5.0:
        gait = "🟡 ИДЁТ (walking)"
    elif avg_speed < 15.0:
        gait = "🟢 БЕЖИТ (running)"
    else:
        gait = "🟣 ЛЕТИТ (flying)"
    
    if speed_cv < 0.3:
        stability = "🟢 СТАБИЛЬНЫЙ"
    elif speed_cv < 0.8:
        stability = "🟡 НЕСТАБИЛЬНЫЙ"
    else:
        stability = "🔴 РЫВКАМИ"
    
    print(f"\n📊 ИТОГИ:")
    print(f"  🧬 VNC CPG:")
    print(f"     Coherence: {walk_log[-1]['coh']:.3f}")
    print(f"     Фазы: {np.degrees(cpg.phases) % 360}")
    print(f"  🦗 Муха:")
    print(f"     Дистанция:     {displacement:.2f} mm")
    print(f"     Время:         {total_walk_time:.2f} sec")
    print(f"     Средняя скорость: {speed_mm_s:.2f} mm/s")
    print(f"     Классификация: {gait}")
    print(f"     Стабильность:  {stability}")
    print(f"  ⏱ Производительность:")
    print(f"     Реальное время: {total_time:.2f} s")
    print(f"     Скорость:       {n_steps/total_time:.0f} step/s")
    print(f"     Симуляция/реальность: {n_steps*dt_ms/1000/total_time:.2f}×")
    
    # ── Compare with FAFB random router ──
    print(f"\n📊 СРАВНЕНИЕ С ПРЕДЫДУЩИМ ЗАПУСКОМ:")
    print(f"     Random router (FAFB): 7.51mm за 5s = 1.50 mm/s")
    print(f"     VNC CPG (биологический): {displacement:.2f}mm за 5s = {speed_mm_s:.2f} mm/s")
    ratio = speed_mm_s / 1.50 if 1.50 > 0 else float('inf')
    print(f"     Улучшение: {ratio:.1f}×" if ratio > 1 else f"     Отношение: {ratio:.1f}×")
    print("═" * 80)
    
    # Save telemetry
    out_dir = Path(__file__).parent.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "vnc_cpg_walk_telemetry.npz",
             time=np.array([l['t'] for l in walk_log]),
             pos=np.array([l['pos'] for l in walk_log]),
             disp=np.array([l['disp'] for l in walk_log]),
             k=np.array([l['k'] for l in walk_log]),
             coh=np.array([l['coh'] for l in walk_log]))
    print(f"💾 Телеметрия: {out_dir / 'vnc_cpg_walk_telemetry.npz'}")


if __name__ == "__main__":
    main()
