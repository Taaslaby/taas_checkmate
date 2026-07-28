#!/usr/bin/env python3
"""
CookieCutter Brain — Эволюция весов через GA.

Исправление v3: Pure Python LIF simulator (Rust SpikeQueue cap=256 не вмещает 1024 нейрона).

1. Загружаем cookiecutter граф (GMM + edge predictor)
2. Конвертируем в .taas через Python API (save_connectome)
3. Эволюция: GA по весам синапсов через Python LIF
4. Fitness: спайковая динамика (diversity, firing rate, temporal variety, balance)
"""

import os, sys, time, json
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

BASE = Path('/home/taaslab/Проекты/asm_engines/taas_checkmate')
COOKIE_DIR = Path('/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/sardana_2')

# ── Проверка taas_checkmate (PyO3 — всё что нужно) ──
try:
    import taas_checkmate as tcm
    HAS_TCM = True
    print("  ✅ taas_checkmate: ConnectomeData, BrainSimulator, save_connectome OK")
except ImportError:
    print("  ⚠ taas_checkmate не загружен. Собери: cd target/release && pip install -e .")
    HAS_TCM = False

np.random.seed(42)

# ═══════════════════════════════════════════════════
#  1. ЗАГРУЗКА COOKIECUTTER ГРАФА
# ═══════════════════════════════════════════════════

def load_cookie_graph() -> dict:
    """Загрузить готовый cookiecutter граф или сгенерировать случайный."""
    npz_path = COOKIE_DIR / 'data' / 'cookiecutter_graph.npz'
    
    if npz_path.exists():
        data = np.load(npz_path)
        print(f"  ✅ Загружен cookiecutter граф: {npz_path}")
    else:
        print(f"  ⚠ CookieCutter граф не найден. Генерация случайного...")
        n = 1024
        deg = np.random.poisson(8, n).clip(1, 50)
        row_ptr = np.zeros(n + 1, dtype=np.int32)
        cols = []
        for i in range(n):
            targets = set()
            while len(targets) < deg[i]:
                t = np.random.randint(0, n)
                if t != i:
                    targets.add(t)
            cols.extend(sorted(targets))
            row_ptr[i+1] = len(cols)
        data = {
            'row_ptr': row_ptr,
            'col_indices': np.array(cols, dtype=np.int32),
            'out_deg': deg,
        }
    
    row_ptr = data['row_ptr'].astype(np.int32)
    col_indices = data['col_indices'].astype(np.int32)
    out_deg = np.diff(row_ptr) if 'out_deg' not in data else data['out_deg']
    n_neurons = len(out_deg)
    n_edges = len(col_indices)
    
    print(f"    {n_neurons} нейронов, {n_edges} синапсов, "
          f"avg deg={np.mean(out_deg):.1f}")
    
    return {
        'row_ptr': row_ptr,
        'col_indices': col_indices,
        'out_deg': out_deg,
        'n_neurons': n_neurons,
        'n_edges': n_edges,
    }


def assign_ei_and_weights(
    row_ptr: np.ndarray,
    col_indices: np.ndarray,
    n_neurons: int,
    ei_ratio: float = 4.0,
    target_mean_abs: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """E/I assignment + log-normal weights."""
    n_exc = int(n_neurons * ei_ratio / (ei_ratio + 1.0))
    n_exc = max(1, min(n_exc, n_neurons - 1))
    
    nt_types = np.ones(n_neurons, dtype=np.float32)
    nt_types[n_exc:] = -1.0
    np.random.shuffle(nt_types)
    
    # Log-normal weights
    values = np.zeros(len(col_indices), dtype=np.float32)
    for i in range(n_neurons):
        start, end = row_ptr[i], row_ptr[i+1]
        n_syn = end - start
        if n_syn == 0:
            continue
        sign = 1.0 if nt_types[i] > 0 else -1.0
        z = np.random.randn(n_syn)
        mag = np.exp(z * 0.8 - 0.5).clip(0.01, 2.0)
        values[start:end] = sign * mag
    
    # Scale to target mean absolute
    mean_abs = np.abs(values).mean()
    if mean_abs > 0:
        values *= target_mean_abs / mean_abs
    
    print(f"  E: {(nt_types > 0).sum()}, I: {(nt_types < 0).sum()}")
    print(f"  Веса: mean={values.mean():.4f}, |mean|={np.abs(values).mean():.4f}, "
          f"std={values.std():.4f}")
    
    return nt_types, values


def identify_groups(
    row_ptr: np.ndarray,
    col_indices: np.ndarray,
    n_neurons: int,
) -> list:
    """Определить группы по in/out degree."""
    out_deg = np.diff(row_ptr)
    in_deg = np.zeros(n_neurons, dtype=np.int32)
    for i in range(n_neurons):
        start, end = row_ptr[i], row_ptr[i+1]
        np.add.at(in_deg, col_indices[start:end], 1)
    
    sensory = np.where((in_deg < 3) & (out_deg > 0))[0]
    mn = np.where((out_deg < 3) & (in_deg > 0))[0]
    dn = np.where((out_deg >= 5) & (in_deg >= 5))[0][:max(1, n_neurons // 50)]
    
    groups = []
    if len(sensory) > 0:
        groups.append(("sensory", sensory.tolist()))
    if len(mn) > 0:
        groups.append(("mn", mn.tolist()))
    if len(dn) > 0:
        groups.append(("dn", dn.tolist()))
    
    print(f"  Группы: sensory={len(sensory)}, mn={len(mn)}, dn={len(dn)}")
    return groups


# ═══════════════════════════════════════════════════
#  3. ANT CHASE FITNESS (MuJoCo)
# ═══════════════════════════════════════════════════

ANT_XML = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="ant_chase">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.005" integrator="Euler"/>
  <custom>
    <numeric name="frame_skip" data="5"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" contype="1" friction="0.5 0.005 0.0001" material="geom"/>
    <motor ctrlrange="-1.0 1.0" ctrllimited="true"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="127" mark="cross" markrgb="1 1 1" name="texgeom" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <material name="geom" shininess="0.2" specular="0.2" texuniform="true" texture="texgeom"/>
    <material name="target" rgba="1 0.2 0.2 0.8"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="0.8 0.8 0.8" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular="0.1 0.1 0.1"/>
    <geom conaffinity="1" condim="3" material="geom" name="floor" pos="0 0 0" size="40 40 0.25" type="plane"/>
    <geom conaffinity="1" condim="3" name="floor_side" pos="0 0 0.01" rgba="0.9 0.9 0.9 1" size="40 0.02 0.01" type="plane"/>
    
    <!-- Target sphere -->
    <body name="target" pos="0 0 0.5">
      <geom material="target" pos="0 0 0" size="0.15" type="sphere"/>
      <site name="target_site" pos="0 0 0" size="0.01" type="sphere"/>
    </body>
    
    <!-- Torso -->
    <body name="torso" pos="0 0 0.75">
      <geom density="5" group="1" name="torso_geom" pos="0 0 0" size="0.25 0.25 0.1" type="ellipsoid"/>
      <site name="torso_site" pos="0 0 0" size="0.02" type="sphere"/>
      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" type="free"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.05 0.05 0.1"/>
      
      <!-- Front Left Leg -->
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0 0 0 0.2 0.2 0" name="front_left_leg_geom" size="0.04" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="front_left_hip" pos="0 0 0" range="-30 30" type="hinge"/>
        <body name="front_left_ankle" pos="0.2 0.2 0">
          <geom fromto="0 0 0 0.4 0.4 0" name="front_left_ankle_geom" size="0.03" type="capsule"/>
          <joint axis="-1 1 0" limited="true" name="front_left_ankle" pos="0 0 0" range="-70 70" type="hinge"/>
          <site name="front_left_foot" pos="0.4 0.4 0" size="0.01" type="sphere"/>
          <geom conaffinity="1" condim="3" name="front_left_foot_geom" pos="0.4 0.4 0" size="0.04" type="sphere"/>
        </body>
      </body>
      
      <!-- Front Right Leg -->
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0 0 0 0.2 -0.2 0" name="front_right_leg_geom" size="0.04" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="front_right_hip" pos="0 0 0" range="-30 30" type="hinge"/>
        <body name="front_right_ankle" pos="0.2 -0.2 0">
          <geom fromto="0 0 0 0.4 -0.4 0" name="front_right_ankle_geom" size="0.03" type="capsule"/>
          <joint axis="1 1 0" limited="true" name="front_right_ankle" pos="0 0 0" range="-70 70" type="hinge"/>
          <site name="front_right_foot" pos="0.4 -0.4 0" size="0.01" type="sphere"/>
          <geom conaffinity="1" condim="3" name="front_right_foot_geom" pos="0.4 -0.4 0" size="0.04" type="sphere"/>
        </body>
      </body>
      
      <!-- Back Left Leg -->
      <body name="back_left_leg" pos="0 0 0">
        <geom fromto="0 0 0 -0.2 0.2 0" name="back_left_leg_geom" size="0.04" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="back_left_hip" pos="0 0 0" range="-30 30" type="hinge"/>
        <body name="back_left_ankle" pos="-0.2 0.2 0">
          <geom fromto="0 0 0 -0.4 0.4 0" name="back_left_ankle_geom" size="0.03" type="capsule"/>
          <joint axis="1 -1 0" limited="true" name="back_left_ankle" pos="0 0 0" range="-70 70" type="hinge"/>
          <site name="back_left_foot" pos="-0.4 0.4 0" size="0.01" type="sphere"/>
          <geom conaffinity="1" condim="3" name="back_left_foot_geom" pos="-0.4 0.4 0" size="0.04" type="sphere"/>
        </body>
      </body>
      
      <!-- Back Right Leg -->
      <body name="back_right_leg" pos="0 0 0">
        <geom fromto="0 0 0 -0.2 -0.2 0" name="back_right_leg_geom" size="0.04" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="back_right_hip" pos="0 0 0" range="-30 30" type="hinge"/>
        <body name="back_right_ankle" pos="-0.2 -0.2 0">
          <geom fromto="0 0 0 -0.4 -0.4 0" name="back_right_ankle_geom" size="0.03" type="capsule"/>
          <joint axis="-1 -1 0" limited="true" name="back_right_ankle" pos="0 0 0" range="-70 70" type="hinge"/>
          <site name="back_right_foot" pos="-0.4 -0.4 0" size="0.01" type="sphere"/>
          <geom conaffinity="1" condim="3" name="back_right_foot_geom" pos="-0.4 -0.4 0" size="0.04" type="sphere"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="150" joint="front_left_hip" name="front_left_hip"/>
    <motor gear="150" joint="front_left_ankle" name="front_left_ankle"/>
    <motor gear="150" joint="front_right_hip" name="front_right_hip"/>
    <motor gear="150" joint="front_right_ankle" name="front_right_ankle"/>
    <motor gear="150" joint="back_left_hip" name="back_left_hip"/>
    <motor gear="150" joint="back_left_ankle" name="back_left_ankle"/>
    <motor gear="150" joint="back_right_hip" name="back_right_hip"/>
    <motor gear="150" joint="back_right_ankle" name="back_right_ankle"/>
  </actuator>
</mujoco>
'''


# ═══════════════════════════════════════════════════
#  3a. CONNECTOME BRAIN EVALUATOR (PyO3)
# ═══════════════════════════════════════════════════
#
# Использует taas_checkmate.BrainSimulator вместо старого ctypes gpu_lif.
# Работает на CPU (всегда доступно). Для GPU используйте
# tcm.BrainSimulator(..., model="lif").step_batch_gpu() с флагом --features cuda.

class ConnectomeBrain:
    """Обёртка над PyO3 BrainSimulator для эволюции.
    
    Создаётся один раз и переиспользуется на все эпизоды.
    """
    
    def __init__(
        self,
        row_ptr, col_indices, values, nt_types,
        n_neurons: int,
        dt_ms: float = 1.0,
        tau_m_val: float = 20.0,
        v_rest_val: float = -70.0,
        v_thresh_val: float = -52.0,
        v_reset_val: float = -70.0,
    ):
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.tau_m_val = tau_m_val
        self.v_rest_val = v_rest_val
        self.v_thresh_val = v_thresh_val
        self.v_reset_val = v_reset_val
        
        if HAS_TCM:
            # ConnectomeData — один раз
            self.conn = tcm.ConnectomeData(
                row_ptr.astype(np.int32),
                col_indices.astype(np.int32),
                values.astype(np.float32),
                nt_types.astype(np.float32),
            )
            # BrainSimulator — один раз, переиспользуется через reset()
            self.sim = tcm.BrainSimulator(n_neurons, "lif")
            self._ready = True
        else:
            self.conn = None
            self.sim = None
            self._ready = False
    
    def step_n(self, external_I: np.ndarray, n_steps: int) -> np.ndarray:
        """Прогнать n_steps с external_I и вернуть мембранные потенциалы."""
        if not self._ready:
            return np.full(self.n_neurons, -70.0, dtype=np.float32)
        
        self.sim.step_batch(
            self.conn, n_steps=n_steps,
            dt_ms=self.dt_ms,
            external_I=external_I.astype(np.float32),
        )
        return self.sim.v()
    
    def reset(self):
        """Сбросить состояние симуляции."""
        if self._ready:
            self.sim.reset()
    
    @property
    def v(self) -> np.ndarray:
        """Текущие мембранные потенциалы."""
        if not self._ready:
            return np.full(self.n_neurons, -70.0, dtype=np.float32)
        return self.sim.v()


class AntChaseFitness:
    """MuJoCo Ant chase task fitness.
    
    Задача: Ant следует за случайно блуждающей целью.
    Fitness = 1 / (1 + mean_distance_to_target) за эпизод.
    
    Архитектура управления:
        obs (18) → W_in → sensory_neurons → LIF dynamics → motor_neurons → W_out → actions (8)
    
    obs = [joint_positions(8), joint_velocities(8), target_dir_x, target_dir_y]
    """
    
    def __init__(
        self,
        row_ptr, col_indices, nt_types,
        n_sensory: int = 200,
        n_motor: int = 300,
        obs_dim: int = 18,
        act_dim: int = 8,
        episode_len: int = 200,
        dt_ms: float = 1.0,
        n_internal_steps: int = 5,
        seed: int = 42,
    ):
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.nt_types = nt_types
        self.n_sensory = n_sensory
        self.n_motor = n_motor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episode_len = episode_len
        self.dt_ms = dt_ms
        self.n_internal = n_internal_steps
        self.n_neurons = len(nt_types)
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)
        
        # W_in: obs → sensory neurons (partial connectivity)
        self.w_in_scale = 0.3
        self.w_in = self._init_projection(obs_dim, n_sensory, 0.3)
        
        # W_out: motor neurons → actions
        self.w_out = self._init_projection(n_motor, act_dim, 0.3)
        
        # Remember sensory/motor indices
        self.sensory_idx = self._select_neurons(n_sensory, low_in=True)
        self.motor_idx = self._select_neurons(n_motor, low_out=True)
        
        # Brain simulator (создаётся при первом evaluate, когда известны веса)
        self._brain = None
        
        print(f"  AntChase: obs={obs_dim}→sens={len(self.sensory_idx)}, "
              f"mot={len(self.motor_idx)}→act={act_dim}")
        print(f"  Episode: {episode_len} steps, {n_internal_steps} internal/step")
    
    def _init_projection(self, n_in: int, n_out: int, sparsity: float = 0.3):
        """Случайная разреженная проекционная матрица."""
        w = np.zeros((n_in, n_out), dtype=np.float32)
        for i in range(n_in):
            n_conn = max(1, int(n_out * sparsity))
            idx = self.rng.choice(n_out, n_conn, replace=False)
            w[i, idx] = self.rng.normal(0, 0.1, n_conn).astype(np.float32)
        return w
    
    def _select_neurons(self, n_target: int, low_in: bool = True, low_out: bool = False) -> np.ndarray:
        """Выбрать нейроны по степени."""
        n = self.n_neurons
        out_deg = np.diff(self.row_ptr)
        in_deg = np.zeros(n, dtype=np.int32)
        for i in range(n):
            s, e = self.row_ptr[i], self.row_ptr[i+1]
            np.add.at(in_deg, self.col_indices[s:e], 1)
        
        if low_in:
            candidates = np.where(in_deg < 3)[0]
        elif low_out:
            candidates = np.where(out_deg < 3)[0]
        else:
            candidates = np.arange(n)
        
        if len(candidates) < n_target:
            extra = np.setdiff1d(np.arange(n), candidates)
            n_extra = min(n_target - len(candidates), len(extra))
            candidates = np.concatenate([candidates, extra[:n_extra]])
        
        return candidates[:n_target]
    
    def evaluate(self, params: np.ndarray, n_seeds: int = 2) -> float:
        """Оценить fitness: Ant chase performance.
        
        По-умолчанию среднее по 2 эпизодам с разными сидами
        для уменьшения noise в fitness landscape.
        
        params = flattened [connectome_weights, w_in_flat, w_out_flat]
        """
        try:
            # Deterministic seed per params (within process)
            p_hash = abs(hash(params.tobytes())) % 100000
            fits = []
            for s in range(n_seeds):
                rng = np.random.default_rng(self._base_seed + p_hash + s * 1000)
                fits.append(self._evaluate_ant(params, rng))
            return float(np.mean(fits))
        except Exception as e:
            print(f"    ⚠ Ant error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0.001
    
    def _deconstruct_params(self, flat: np.ndarray):
        """Разобрать плоский вектор на компоненты."""
        n_conn = len(self.row_ptr) - 1  # = n_neurons
        n_edges = len(self.col_indices)
        n_w_in = self.obs_dim * self.n_sensory
        n_w_out = self.n_motor * self.act_dim
        n_total = n_edges + n_w_in + n_w_out
        
        if len(flat) < n_total:
            # Совместимость: только connectome weights
            return flat[:n_edges], self.w_in, self.w_out
        
        conn_w = flat[:n_edges]
        w_in_f = flat[n_edges:n_edges+n_w_in].reshape(self.obs_dim, self.n_sensory)
        w_out_f = flat[n_edges+n_w_in:n_edges+n_w_in+n_w_out].reshape(self.n_motor, self.act_dim)
        return conn_w, w_in_f, w_out_f
    
    def _build_obs(self, data) -> np.ndarray:
        """Построить наблюдение из MuJoCo state.
        
        obs = [qpos[7:] (joints, 6), qvel (8), target_dir (2)]
        total: 6 + 8 + 2 = 16
        """
        # Joint positions (hinge joints = qpos[7:] for free joint)
        qpos = data.qpos  # [0:7] = free joint (quat+pos), [7:] = joints
        qvel = data.qvel  # [0:6] = free joint vel, [6:] = joint vel
        
        n_joints = len(qpos) - 7
        joint_pos = np.array(qpos[7:], dtype=np.float32)
        joint_vel = np.array(qvel[6:], dtype=np.float32) if len(qvel) > 6 else np.zeros(n_joints, dtype=np.float32)
        
        # Ant torso position
        ant_x = float(qpos[4]) if len(qpos) > 4 else 0.0
        ant_y = float(qpos[5]) if len(qpos) > 5 else 0.0
        
        # Target position
        target_pos = data.body('target').xpos[:2]
        target_dir = np.array([target_pos[0] - ant_x, target_pos[1] - ant_y], dtype=np.float32)
        # Normalize direction
        dist = np.linalg.norm(target_dir)
        if dist > 0.01:
            target_dir = target_dir / dist
        
        obs = np.concatenate([joint_pos, joint_vel, target_dir]).astype(np.float32)
        
        # Pad if needed
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        return obs[:self.obs_dim]
    
    def _evaluate_ant(self, params: np.ndarray, rng: np.random.Generator = None) -> float:
        """Запустить MuJoCo эпизод и измерить chase performance (PyO3 BrainSimulator)."""
        import mujoco
        
        if rng is None:
            rng = self.rng
        
        conn_w, w_in, w_out = self._deconstruct_params(params)
        
        # ── Создаём ConnectomeBrain с новыми весами (не кэшируем — веса меняются!) ──
        n = self.n_neurons
        brain = ConnectomeBrain(
            self.row_ptr, self.col_indices,
            conn_w.astype(np.float32),
            self.nt_types.astype(np.float32),
            n, dt_ms=self.dt_ms,
        )
        
        # ── MuJoCo модель ──
        model = mujoco.MjModel.from_xml_string(ANT_XML)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        
        brain.reset()
        
        total_dist = 0.0
        n_contacts = 0
        
        # Цель: случайное блуждание
        target_vel = np.array([0.0, 0.0], dtype=np.float32)
        target_pos = np.array([2.0, 0.0], dtype=np.float32)
        data.body('target').xpos[:2] = target_pos
        
        for step in range(self.episode_len):
            # ── Target random walk ──
            target_vel += rng.uniform(-0.5, 0.5, 2) * 0.1
            target_vel *= 0.95
            target_pos += target_vel * 0.02
            target_pos = np.clip(target_pos, -8, 8)
            data.body('target').xpos[:2] = target_pos
            
            # ── Build observation ──
            obs = self._build_obs(data)
            
            # ── Build external_I from observation ──
            external_I = np.zeros(n, dtype=np.float32)
            for oi in range(min(len(obs), self.obs_dim)):
                for si, sidx in enumerate(self.sensory_idx[:len(w_in[oi])]):
                    external_I[sidx] += obs[oi] * w_in[oi, si]
            
            # ── Internal dynamics via PyO3 BrainSimulator ──
            ext_in = external_I * 0.5 + 0.1
            v_all = brain.step_n(ext_in, self.n_internal)
            
            # ── Read motor neurons → actions ──
            actions = np.zeros(self.act_dim, dtype=np.float32)
            for a in range(self.act_dim):
                for mi, midx in enumerate(self.motor_idx[:len(w_out)]):
                    v_norm = (v_all[midx] + 70.0) / 30.0
                    actions[a] += v_norm * w_out[mi, a]
                actions[a] = np.tanh(actions[a])
            
            # ── Apply actions and step physics ──
            for a in range(min(self.act_dim, model.nu)):
                data.ctrl[a] = actions[a]
            for _ in range(5):
                mujoco.mj_step(model, data)
            
            # ── Distance to target ──
            ant_pos = data.body('torso').xpos[:2]
            dist = np.linalg.norm(ant_pos - target_pos)
            total_dist += dist
            
            if data.body('torso').xpos[2] < 0.2:
                n_contacts = step + 1
                break
            n_contacts = step + 1
        
        mean_dist = total_dist / max(n_contacts, 1)
        fitness = 1.0 / (1.0 + mean_dist)
        return float(max(fitness, 0.001))


# ═══════════════════════════════════════════════════
#  4. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
# ═══════════════════════════════════════════════════

class GeneticEvolution:
    """(μ+λ)-Evolution Strategy with adaptive mutation rate.
    
    Ключевые отличия от предыдущего GA:
    1. (μ+λ): родители НЕ вымирают — конкурируют с детьми
    2. Adaptive mutation: rate растёт при застое, падает при прогрессе
    3. Нет tournament selection — чистая ранговая селекция
    """
    
    def __init__(
        self,
        base_weights: np.ndarray,
        evaluator,  # AntChaseFitness instance
        mu: int = 30,             # число родителей
        lambda_ratio: float = 1.5,  # λ = mu * lambda_ratio детей
        mut_rate: float = 0.15,
        mut_scale: float = 0.3,
        crossover_rate: float = 0.7,
        adapt_window: int = 3,     # окно для adaptive mutation
    ):
        self.base = base_weights.copy()
        self.n_genes = len(base_weights)
        self.evaluator = evaluator
        
        # (μ+λ) параметры
        self.mu = mu
        self.lamb = max(1, int(mu * lambda_ratio))
        self.crossover_rate = crossover_rate
        
        # Mutation (адаптивные)
        self.base_mut_rate = mut_rate
        self.base_mut_scale = mut_scale
        self.mut_rate = mut_rate
        self.mut_scale = mut_scale
        self.mut_rate_min = mut_rate * 0.3
        self.mut_rate_max = mut_rate * 3.0
        self.mut_scale_min = mut_scale * 0.3
        self.mut_scale_max = mut_scale * 3.0
        self.adapt_window = adapt_window
        self._stagnation_counter = 0
        
        self.rng = np.random.default_rng(42)
        
        self.population = []  # [(weights, fitness)] — всегда top μ
        self.history = {
            'gen': [], 'best_fitness': [], 'mean_fitness': [],
            'diversity': [], 'mut_rate': [],
        }
        self._init_population()
    
    def _init_population(self):
        """Создать начальную популяцию."""
        print(f"\n  Инициализация популяции (μ={self.mu}, λ={self.lamb})...", flush=True)
        for i in range(self.mu):
            noise_scale = 0.3 + self.rng.random() * 0.5
            weights = self.base + self.rng.normal(0, noise_scale, self.n_genes)
            weights = np.clip(weights, -3.0, 3.0)
            self.population.append((weights, 0.0))
        
        self._evaluate_list(self.population)
        self.population.sort(key=lambda x: x[1], reverse=True)
        self.population = self.population[:self.mu]
        
        best = max(self.population, key=lambda x: x[1])
        mean_f = np.mean([f for _, f in self.population])
        print(f"  Начальная популяция: best={best[1]:.4f}, mean={mean_f:.4f}", flush=True)
    
    def _evaluate_list(self, pop):
        """Оценить fitness списка особей."""
        for i in range(len(pop)):
            w, _ = pop[i]
            pop[i] = (w, self.evaluator.evaluate(w))
            if (i + 1) % 10 == 0:
                print(f"    оценка {i+1}/{len(pop)}", flush=True)
    
    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        """Гауссова мутация с адаптивной силой."""
        w = weights.copy()
        mask = self.rng.random(self.n_genes) < self.mut_rate
        noise = self.rng.normal(0, self.mut_scale, self.n_genes)
        w[mask] += noise[mask]
        return np.clip(w, -3.0, 3.0)
    
    def _crossover(self, w1: np.ndarray, w2: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover — оба ребёнка."""
        mask = self.rng.random(self.n_genes) < 0.5
        return np.where(mask, w1, w2), np.where(mask, w2, w1)
    
    def _adapt_mutation(self):
        """Adaptive mutation rate (эвристика 1/5 success).
        
        Если best fitness рос последние adapt_window поколений →
          уменьшаем mut_rate (эксплуатируем ландшафт)
        Если не рос (застой) →
          увеличиваем mut_rate (исследуем)
        """
        window = self.adapt_window
        if len(self.history['best_fitness']) < window + 1:
            return  # ещё нет данных
        
        recent = self.history['best_fitness'][-window:]
        old_val = recent[0]
        new_val = recent[-1]
        improvement = new_val - old_val
        
        if improvement > 1e-4:
            # Прогресс: эксплуатируем (меньше мутаций)
            self.mut_rate = max(self.mut_rate_min, self.mut_rate * 0.85)
            self.mut_scale = max(self.mut_scale_min, self.mut_scale * 0.85)
            self._stagnation_counter = 0
        else:
            # Застой: исследуем (больше мутаций)
            self._stagnation_counter += 1
            if self._stagnation_counter >= window:
                self.mut_rate = min(self.mut_rate_max, self.mut_rate * 1.4)
                self.mut_scale = min(self.mut_scale_max, self.mut_scale * 1.4)
    
    def step(self) -> dict:
        """Один шаг (μ+λ)-ES.
        
        1. Генерируем λ детей из μ родителей
        2. Оцениваем всех детей
        3. Выбираем top μ из (родители + дети) — вот что даёт (μ+λ)!
        4. Адаптируем mutation rate
        """
        # 1. Генерация λ детей
        children = []
        while len(children) < self.lamb:
            # Случайный родитель из top-μ (ранговая селекция)
            rank = self.rng.geometric(p=0.4) - 1  # exp распределение: top чаще
            rank = min(rank, self.mu - 1)
            p1 = self.population[rank][0].copy()
            
            if self.rng.random() < self.crossover_rate:
                rank2 = self.rng.geometric(p=0.4) - 1
                rank2 = min(rank2, self.mu - 1)
                p2 = self.population[rank2][0].copy()
                c1, c2 = self._crossover(p1, p2)
                children.append((self._mutate(c1), 0.0))
                if len(children) < self.lamb:
                    children.append((self._mutate(c2), 0.0))
            else:
                children.append((self._mutate(p1), 0.0))
        
        children = children[:self.lamb]
        
        # 2. Оценка детей
        print(f"    Оценка λ={len(children)} детей...", flush=True)
        self._evaluate_list(children)
        
        # 3. (μ+λ) селекция: top μ из parents + children
        combined = self.population + children
        combined.sort(key=lambda x: x[1], reverse=True)
        self.population = combined[:self.mu]
        
        # 4. Adaptive mutation
        self._adapt_mutation()
        
        # Статистика
        fitnesses = [f for _, f in self.population]
        
        # Diversity (L1 между top-5)
        all_w = np.array([w for w, _ in self.population[:10]])
        if len(all_w) >= 2:
            dists = []
            for i in range(min(5, len(all_w))):
                for j in range(i+1, min(5, len(all_w))):
                    dists.append(np.abs(all_w[i] - all_w[j]).mean())
            diversity = np.mean(dists) if dists else 0.0
        else:
            diversity = 0.0
        
        self.history['gen'].append(len(self.history['gen']))
        self.history['best_fitness'].append(fitnesses[0])
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['diversity'].append(diversity)
        self.history['mut_rate'].append(self.mut_rate)
        
        return {
            'best_fitness': fitnesses[0],
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': diversity,
            'mut_rate': self.mut_rate,
            'mut_scale': self.mut_scale,
            'lamb': len(children),
            'best_weights': self.population[0][0].copy(),
        }
    
    def evolve(self, n_generations: int, eval_every: int = 5) -> np.ndarray:
        """Полный цикл эволюции."""
        print(f"\n{'='*60}")
        print(f"  (μ+λ)-EVOLUTION STRATEGY")
        print(f"  μ={self.mu}, λ={self.lamb}, generations={n_generations}")
        print(f"  Mutation: base_rate={self.base_mut_rate}, base_scale={self.base_mut_scale}")
        print(f"  Crossover: rate={self.crossover_rate}")
        print(f"  Adaptive window: {self.adapt_window}")
        print(f"{'='*60}")
        
        for gen in range(n_generations):
            t0 = time.time()
            result = self.step()
            dt = time.time() - t0
            
            if gen % eval_every == 0:
                delta = result['best_fitness'] - self.history['best_fitness'][
                    max(0, gen - eval_every)
                ]
                print(f"  gen {gen:3d}: best={result['best_fitness']:.4f} "
                      f"mean={result['mean_fitness']:.4f} "
                      f"Δ={delta:+.4f} "
                      f"div={result['diversity']:.4f} "
                      f"mut={result['mut_rate']:.2f} "
                      f"λ={result['lamb']} "
                      f"[{dt:.1f}s]", flush=True)
        
        best = self.population[0]
        improvement = best[1] - self.history['best_fitness'][0]
        print(f"\n{'='*60}")
        print(f"  ✅ (μ+λ)-ES ЗАВЕРШЕНА")
        print(f"  Начальный best: {self.history['best_fitness'][0]:.6f}")
        print(f"  Финальный best: {best[1]:.6f}")
        print(f"  Улучшение: {improvement:+.6f}")
        print(f"  Поколений: {len(self.history['gen'])}")
        print(f"  Финальный мутация: rate={self.mut_rate:.3f}, scale={self.mut_scale:.3f}")
        print(f"{'='*60}")
        
        return best[0]


# ═══════════════════════════════════════════════════
#  5. СОХРАНЕНИЕ
# ═══════════════════════════════════════════════════

def save_brain(
    row_ptr, col_indices, values, nt_types, groups,
    history, output_dir: Path, tag: str = "cookie_evolved",
):
    """Сохранить мозг как .taas + историю."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # .taas
    taas_path = output_dir / f"{tag}.taas"
    if HAS_TCM:
        conn = tcm.ConnectomeData(
            row_ptr.astype(np.int32),
            col_indices.astype(np.int32),
            values.astype(np.float32),
            nt_types.astype(np.float32),
            groups,
        )
        tcm.save_connectome(str(taas_path), conn, metadata=[
            ("source", "cookiecutter_evolution"),
            ("generations", str(len(history['gen']))),
            ("best_fitness", f"{history['best_fitness'][-1]:.6f}" if history['best_fitness'] else "0"),
        ])
        print(f"  ✅ .taas: {taas_path}")
    else:
        npz_path = output_dir / f"{tag}.npz"
        np.savez(npz_path,
                 row_ptr=row_ptr, col_indices=col_indices,
                 values=values, nt_types=nt_types,
                 best_fitness=history['best_fitness'][-1] if history['best_fitness'] else 0,
                 n_generations=len(history['gen']))
        print(f"  ⚠ .npz (taas недоступен): {npz_path}")
    
    # История
    hist_path = output_dir / f"{tag}_history.json"
    with open(hist_path, 'w') as f:
        json.dump({
            'gen': [int(g) for g in history['gen']],
            'best_fitness': [float(f) for f in history['best_fitness']],
            'mean_fitness': [float(f) for f in history['mean_fitness']],
            'diversity': [float(d) for d in history['diversity']],
            'mut_rate': [float(m) for m in history.get('mut_rate', [])],
        }, f, indent=2)
    print(f"  ✅ История: {hist_path}")
    
    w_path = output_dir / f"{tag}_weights.npy"
    np.save(w_path, values)
    print(f"  ✅ Веса: {w_path}")


# ═══════════════════════════════════════════════════
#  6. MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  COOKIECUTTER BRAIN — Эволюция весов (v3)")
    print("  Pure Python LIF simulator")
    print("=" * 60)
    
    out_dir = BASE / 'models' / 'cookie_evolved'
    
    # [1] Граф
    print("\n[1/4] Загрузка графа...")
    graph = load_cookie_graph()
    n = graph['n_neurons']
    row_ptr = graph['row_ptr']
    col_indices = graph['col_indices']
    
    # [2] E/I + веса + группы
    print("\n[2/4] Назначение E/I, весов, групп...")
    nt_types, base_weights = assign_ei_and_weights(
        row_ptr, col_indices, n, target_mean_abs=0.5
    )
    groups = identify_groups(row_ptr, col_indices, n)
    
    # Сохраняем оригинал
    save_brain(row_ptr, col_indices, base_weights, nt_types, groups,
               {'gen': [], 'best_fitness': [], 'mean_fitness': [], 'diversity': []},
               out_dir, "cookie_original")
    
    n_sensory = 200
    n_motor = 300
    obs_dim = 18
    act_dim = 8
    
    # Flattened параметры: [connectome_weights, w_in, w_out]
    n_edges = len(col_indices)
    n_w_in = obs_dim * n_sensory
    n_w_out = n_motor * act_dim
    
    # Заполняем base_params
    w_in_init = np.random.default_rng(42).normal(0, 0.1, n_w_in).astype(np.float32) * 0.3
    w_out_init = np.random.default_rng(43).normal(0, 0.1, n_w_out).astype(np.float32) * 0.3
    base_params = np.concatenate([base_weights, w_in_init, w_out_init])
    
    # [3] AntChaseFitness
    print(f"\n[3/4] Инициализация AntChaseFitness...")
    evaluator = AntChaseFitness(
        row_ptr, col_indices, nt_types,
        n_sensory=n_sensory, n_motor=n_motor,
        obs_dim=obs_dim, act_dim=act_dim,
        episode_len=200, dt_ms=1.0,
        n_internal_steps=5,
    )
    
    print(f"  Всего параметров: {len(base_params)} "
          f"(conn={n_edges}, W_in={n_w_in}, W_out={n_w_out})")
    baseline_fit = evaluator.evaluate(base_params, n_seeds=2)
    print(f"  Baseline fitness (Ant chase, avg 2 seeds): {baseline_fit:.4f}")
    
    # [4] Эволюция
    print(f"\n{'='*60}")
    print(f"  [4/4] EVOLVING ANT CHASE BRAIN (GPU+MuJoCo)")
    print(f"{'='*60}")
    
    evo = GeneticEvolution(
        base_weights=base_params,
        evaluator=evaluator,
        mu=15,
        lambda_ratio=1.5,
        mut_rate=0.2,
        mut_scale=0.4,
        crossover_rate=0.7,
        adapt_window=4,
    )
    
    best_params = evo.evolve(n_generations=30, eval_every=5)
    
    # Разобрать best_params обратно
    best_conn_w = best_params[:n_edges]
    best_weights = best_conn_w  # alias for analysis below
    
    # [5] Сохранение
    print(f"\n{'='*60}")
    print(f"  СОХРАНЕНИЕ РЕЗУЛЬТАТА")
    print(f"{'='*60}")
    
    save_brain(row_ptr, col_indices, best_conn_w, nt_types, groups,
               evo.history, out_dir, "cookie_ant_evolved")
    
    # Анализ изменений
    delta = best_weights - base_weights
    changed = np.abs(delta) > 0.01
    print(f"\n  Анализ изменений:")
    print(f"  Изменено: {changed.sum()} / {len(best_weights)} "
          f"({100*changed.sum()/len(best_weights):.1f}%)")
    print(f"  Δ mean: {delta.mean():+.6f}")
    print(f"  |Δ| max: {np.abs(delta).max():.4f}")
    print(f"  |Δ| mean: {np.abs(delta).mean():.4f}")
    
    # Топ изменения
    print(f"\n  Топ-5 синапсов по |Δ|:")
    top_idx = np.argsort(np.abs(delta))[-5:][::-1]
    for idx in top_idx:
        print(f"    syn[{idx:5d}]: {base_weights[idx]:+.4f} → "
              f"{best_weights[idx]:+.4f} (Δ={delta[idx]:+.4f})")
    
    fit_improvement = (
        evo.history['best_fitness'][-1] - evo.history['best_fitness'][0]
        if len(evo.history['best_fitness']) > 1 else 0
    )
    
    print(f"\n{'='*60}")
    print(f"  ✅ ГОТОВО!")
    print(f"  Мозг:  {out_dir / 'cookie_ant_evolved.taas'}")
    print(f"  Нейроны: {n}, Синапсы: {graph['n_edges']}")
    print(f"  Поколений: {len(evo.history['gen'])}")
    print(f"  Fitness: {evo.history['best_fitness'][0]:.4f} → "
          f"{evo.history['best_fitness'][-1]:.4f} "
          f"({fit_improvement:+.4f})")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
