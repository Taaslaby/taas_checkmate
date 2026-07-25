#!/usr/bin/env python3
"""
🧬 DepthWave — depth-modulated ψ-wave evolution на VNC коннектоме.

Перенос из t3-wavelet (crates/tuul/src/neuron/depth_wave.rs).

Ключевая идея — topological depth:
  depth[i] ∈ [0, 1]  (sensory=1, motor=0)

  speed[i] = 1 + depth[i] * mod_range
    → sensory нейроны эволюционируют БЫСТРЕЕ
    → motor нейроны — МЕДЛЕННЕЕ (интегрируют дольше)

  sensory_weight = 1.0 - depth[i]
    → motor (depth≈0) получает БОЛЬШЕ сенсорного входа

  Выход: joint_acts напрямую из ψ-амплитуды моторных нейронов
         → position servo (target angles, не torque)
"""

import os, sys, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import mujoco as mu
from scipy.sparse import csr_matrix

# ═══════════════════════════════════════════════════════════════════
#  DepthWaveVNC — ψ-wave на VNC коннектоме с топологическим depth
# ═══════════════════════════════════════════════════════════════════

class DepthWaveVNC:
    """DepthWave: depth-modulated ψ-wave на VNC CSR коннектоме.

    Архитектура:
      ψ = v + i·u  (continuous wavefunction)
      dψ/dt = -i·H·ψ  (Schrödinger evolution)
      speed[i] = 1 + depth[i]·mod_range  (depth-modulated)
      joint_acts[i] = |ψ_motor[i]|  (amplitude → position servo)

    Параметры
    ----------
    row_ptr, col_idx, vals : np.ndarray
        CSR коннектома VNC (из RealVNCConnectome)
    nt_types : np.ndarray
        Типы нейротрансмиттеров: 1=возб, -1=торм
    n_neurons : int
        Количество нейронов
    depth : np.ndarray или None
        Топологическая глубина [0, 1]. Если None — вычисляется из NT типов
    mod_range : float
        Сила depth-модуляции (0 = plain ψ-wave)
    dt : float
        Временной шаг ψ-эволюции
    sensory_dim : int
        Размерность сенсорного входа

    Атрибуты
    ---------
    v, u : np.ndarray (n,)
        Действительная и мнимая части ψ
    depth : np.ndarray (n,)
        Топологическая глубина каждого нейрона
    speed : np.ndarray (n,)
        Скорость эволюции: 1 + depth * mod_range
    """

    def __init__(self, row_ptr, col_idx, vals, nt_types, n_neurons,
                 depth=None, mod_range=2.0, dt=0.5, seed=42):
        self.n = n_neurons
        self.dt = dt
        self.mod_range = mod_range

        # ── CSR матрица H (симметричная, взвешенная NT) ─────────
        # H_ij = vals[i,j] * nt_types[j] (sign от постсинаптического)
        # Транспонируем: H_ij = sign_j * w_ij
        # Для H: вклад в i-й нейрон от j-го через вес w_ij
        # В ψ-эволюции: (H @ u)_i = Σ_j H_ij * u_j
        # где H_ij = sign_i * W_ji (влияние j на i)
        # 
        # В CSR: row_ptr[i] → col_idx для пре-синаптических нейронов
        # W_ij = вес от j к i (col_idx — пре, row — пост)
        nt_signs = np.asarray(nt_types, dtype=np.float32)
        
        # H = W * sign (постсинаптический знак)
        h_vals = vals.copy()
        for i in range(n_neurons):
            s, e = row_ptr[i], row_ptr[i+1]
            h_vals[s:e] *= nt_signs[i]
        
        self.H = csr_matrix((h_vals, col_idx, row_ptr), shape=(n_neurons, n_neurons)).T.tocsr()
        # ── Топологическая глубина ────────────────────────────────
        if depth is not None:
            self.depth = np.asarray(depth, dtype=np.float32)
        else:
            # Авто-глубина: GABA→1.0 (сенсорика), ACh→0.5, GLUT→0.0 (моторика)
            self.depth = np.ones(n_neurons, dtype=np.float32) * 0.5
            gaba = nt_signs < 0
            glut = nt_signs > 0.5
            self.depth[gaba] = 0.8   # тормозные → глубже
            self.depth[glut] = 0.2   # возбуждающие → мельче
            self.depth[0] = 1.0      # сенсорный полюс
            self.depth[-1] = 0.0     # моторный полюс

        np.clip(self.depth, 0.0, 1.0, out=self.depth)

        # ── Скорость эволюции ────────────────────────────────────
        # speed[i] = 1 + depth[i] * mod_range
        #   depth=1 (sensory) → speed = 1 + mod_range (быстро)
        #   depth=0 (motor) → speed = 1 (медленно)
        self.speed = np.ones(n_neurons, dtype=np.float32) + self.depth * mod_range

        # ── ψ-состояние ───────────────────────────────────────────
        rng = np.random.RandomState(seed)
        phase = rng.rand(n_neurons) * 2 * np.pi
        self.v = np.cos(phase).astype(np.float32)  # Re(ψ)
        self.u = np.sin(phase).astype(np.float32)  # Im(ψ)

        # ── Сенсорный вес: сколько внешнего сигнала получает нейрон ─
        # sensory_weight = 1.0 - depth[i]
        #   depth=0 (motor) → вес = 1.0 (макс входа)
        #   depth=1 (sensory) → вес = 0.0 (мин входа)
        self.sensory_weight = 1.0 - self.depth

        # ── Моторные нейроны (для joint_acts) ────────────────────
        # Нейроны с depth < 0.3 считаются "моторными"
        self.motor_mask = self.depth < 0.3
        self.n_motor = int(self.motor_mask.sum())

        print(f"   🧬 DepthWave: {self.n} neurons, {self.n_motor} motor")
        print(f"      depth range: [{self.depth.min():.2f}, {self.depth.max():.2f}]")
        print(f"      speed range: [{self.speed.min():.2f}, {self.speed.max():.2f}]")
        print(f"      H nnz: {self.H.nnz:,}")

    def compute_Hpsi(self, v, u):
        """Hψ = H @ u (real part), -(H @ v) (imag part).

        H[i,j] = w_{j→i} (влияние j на i).
        (H @ u)_i = Σ_j H_ij * u_j = Σ_j w_{j→i} * u_j.

        Returns h_real, h_imag.
        """
        # (H @ u)_i = Σ_j w_{j→i} * u_j — сумма incoming u
        # (H @ v)_i = Σ_j w_{j→i} * v_j — сумма incoming v
        Hu = self.H.dot(u)  # incoming coupling * u
        Hv = self.H.dot(v)  # incoming coupling * v
        return Hu, -Hv

    def step(self, sensory_input=None, n_steps=1):
        """Один шаг ψ-wave эволюции с depth-модулированной скоростью.

        v += dt · speed[i] · (H @ u)ᵢ
        u -= dt · speed[i] · (H @ v)ᵢ

        Args:
            sensory_input: (n,) внешний сенсорный сигнал или None
            n_steps: количество шагов эволюции

        Returns:
            coherence: float — глобальная когерентность ψ
            joint_acts: np.ndarray (n_motor,) — амплитуды моторных нейронов
        """
        for _ in range(n_steps):
            # Hψ
            h_real, h_imag = self.compute_Hpsi(self.v, self.u)

            # Depth-modulated Euler step
            # v += dt * speed[i] * h_real[i]
            # u += dt * speed[i] * h_imag[i]
            self.v += self.dt * self.speed * h_real
            self.u += self.dt * self.speed * h_imag

            # Inject sensory input (weighted by depth)
            if sensory_input is not None:
                inp = np.asarray(sensory_input, dtype=np.float32)
                n_in = min(len(inp), self.n)
                # sensory_weight = 1 - depth: shallow (motor) gets more
                self.v[:n_in] += self.dt * self.sensory_weight[:n_in] * inp[:n_in]

            # Global L2 normalize (Σ|ψ|² = 1)
            # ⚡ Важно: глобальная норма, не per-neuron!
            norm = np.sqrt(np.sum(self.v**2 + self.u**2) + 1e-9)
            self.v /= norm
            self.u /= norm

        # Coherence = |mean(ψ)|
        coherence = float(np.abs(np.mean(self.v + 1j * self.u)))

        # Joint acts from motor neuron amplitude
        joint_acts = np.sqrt(
            self.v[self.motor_mask]**2 + self.u[self.motor_mask]**2
        ).astype(np.float32)

        return coherence, joint_acts

    def get_psi2(self):
        """Вернуть |ψ|² (вероятностная плотность)."""
        return self.v**2 + self.u**2

    def get_motor_phases(self):
        """Вернуть аргументы ψ для моторных нейронов."""
        return np.arctan2(self.u[self.motor_mask], self.v[self.motor_mask])

    def reset(self, seed=42):
        rng = np.random.RandomState(seed)
        phase = rng.rand(self.n) * 2 * np.pi
        self.v = np.cos(phase).astype(np.float32)
        self.u = np.sin(phase).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  DepthWalk — DepthWave + Position Servo → MuJoCo
# ═══════════════════════════════════════════════════════════════════

class DepthWalkFly:
    """DepthWave VNC → position servo → MuJoCo fly walk.

    Берёт joint_acts из DepthWave и преобразует в target joint angles
    для position servo суставов мухи.

    Анатомия ноги (7 DOF):
      Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1
    """

    # Анатомические диапазоны (Lobato-Rios et al. 2022)
    LEG_DOF_RANGES = {
        'Coxa':        (-0.5,  0.5),
        'Coxa_roll':   (-0.3,  0.3),
        'Coxa_yaw':    (-0.2,  0.2),
        'Femur':       (-1.0,  1.0),
        'Femur_roll':  (-0.3,  0.3),
        'Tibia':       (-1.5,  0.5),
        'Tarsus1':     (-0.5,  0.5),
    }
    DOF_NAMES = list(LEG_DOF_RANGES.keys())
    LEG_NAMES = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    # Амплитуды на DOF (гармонические веса)
    AMP_PER_DOF = [0.02, 0.0, 0.15, 0.02, 0.25, 0.15, 0.0]
    # Фазовые сдвиги на DOF
    PHASE_PER_DOF = [0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.0]

    # Tripod: +1 = group A (T1L, T2R, T3L), -1 = group B (T1R, T2L, T3R)
    TRIPOD = np.array([1, -1, -1, 1, 1, -1])

    def __init__(self, depth_wave: DepthWaveVNC, fly_sim, base_omega=6.0):
        self.dw = depth_wave
        self.fly = fly_sim
        self.base_omega = base_omega
        self.phase = 0.0

        # Сопоставить моторные нейроны DepthWave → 6 ног
        n_motor = depth_wave.n_motor
        self.motor_to_leg = (np.arange(n_motor) * 6 // max(n_motor, 1)).astype(np.int32)
        self.motor_to_dof = np.random.randint(0, len(self.DOF_NAMES), size=n_motor).astype(np.int32)

    def step(self, sensory_input=None, dt_body=5e-4):
        """Один шаг: DepthWave → joint angles → MuJoCo.

        Args:
            sensory_input: (n,) сенсорный сигнал или None
            dt_body: шаг MuJoCo

        Returns:
            dict: pos, joints, speed, coherence
        """
        # 1. DepthWave эволюция
        coherence, joint_acts = self.dw.step(sensory_input, n_steps=1)

        # 2. Phase update (CPG)
        speed_mod = np.clip(coherence * 2.0 + 0.2, 0.2, 1.0)
        self.phase += self.base_omega * speed_mod * dt_body

        # 3. Per-leg modulation from ψ amplitude
        leg_amp = np.zeros(6, dtype=np.float32)
        for i in range(min(len(joint_acts), self.n_motor)):
            leg = self.motor_to_leg[i]
            leg_amp[leg] += joint_acts[i]
        leg_amp /= max(leg_amp.max(), 1e-6)

        # 4. Build joint angle targets (без flygym — через DOF индексы)
        #    Каждая нога имеет 7 DOF. target = 42 элемента
        target_qpos = self.fly.data.qpos.copy()

        for leg_i, leg_name in enumerate(self.LEG_NAMES):
            leg_phase = self.phase * self.TRIPOD[leg_i]
            amp_scale = np.clip(0.4 + 0.6 * leg_amp[leg_i], 0.2, 1.5)

            # DOF для этой ноги
            LEG_MAP = {'LF': 'T1_left', 'LM': 'T2_left', 'LH': 'T3_left',
                       'RF': 'T1_right', 'RM': 'T2_right', 'RH': 'T3_right'}
            leg_label = LEG_MAP.get(leg_name, 'T1_left')
            leg_dofs = self.fly.leg_dofs.get(leg_label, [])
            for j, dof in enumerate(self.DOF_NAMES[:7]):
                if j >= len(leg_dofs):
                    break
                dof_idx = leg_dofs[j]
                lo, hi = self.LEG_DOF_RANGES[dof]
                center = (lo + hi) / 2.0
                rng = (hi - lo) / 2.0
                dof_phase = leg_phase + j * np.pi / 7
                angle = center + self.AMP_PER_DOF[j] * amp_scale * rng * np.sin(dof_phase)
                target_qpos[dof_idx] = np.clip(angle, lo - 0.02, hi + 0.02)

        # 5. Apply to MuJoCo (position control)
        self.fly.data.qpos[:] = target_qpos
        mu.mj_forward(self.fly.model, self.fly.data)

        return {
            'pos': self.fly.data.qpos[:3].copy(),
            'joints': target.copy(),
            'speed_mod': speed_mod,
            'coherence': coherence,
            'leg_amp': leg_amp.copy(),
        }


# ═══════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("🧬 DepthWave — self-test")
    print("=" * 60)

    # ── Load VNC ──────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from vnc_real_atlasov_embodied import RealVNCConnectome

    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons() or not conn.load_connections():
        print("❌ Failed to load VNC")
        sys.exit(1)

    N = conn.n_neurons
    print(f"   VNC: {N:,} neurons, {conn.n_synapses:,} synapses")

    # ── Assign depth from neuron metadata ─────────────────────────
    depth = np.ones(N, dtype=np.float32) * 0.5
    for idx in range(N):
        rid = conn.idx_to_root.get(idx)
        if rid is None:
            continue
        meta = conn.neurons.get(rid, {})
        sc = meta.get('super_class', '')
        if sc == 'sensory':
            depth[idx] = 0.9
        elif sc == 'motor':
            depth[idx] = 0.1
        elif sc == 'descending':
            depth[idx] = 0.7
        elif sc == 'ventral_nerve_cord_intrinsic':
            depth[idx] = 0.5
        elif sc == 'ascending':
            depth[idx] = 0.6

    # ── DepthWave ─────────────────────────────────────────────────
    dw = DepthWaveVNC(
        conn.row_ptr, conn.col_indices, conn.values,
        conn.nt_types, N,
        depth=depth, mod_range=2.0, dt=0.5,
    )

    # ── Test evolution ────────────────────────────────────────────
    print("\n   Evolving ψ-wave (100 steps)...")
    for t in range(100):
        # Simple oscillatory sensory input
        sensory = np.sin(2 * np.pi * 5 * t * 0.0005) * np.ones(N, dtype=np.float32) * 0.1
        coherence, joint_acts = dw.step(sensory, n_steps=1)

        if t % 20 == 0:
            psi2 = dw.get_psi2()
            print(f"   t={t:3d}: coh={coherence:.4f} "
                  f"|ψ|²=[{psi2.min():.6f}, {psi2.max():.6f}] "
                  f"motor_acts=[{joint_acts.min():.4f}, {joint_acts.max():.4f}]")

    print("\n✅ DepthWave self-test complete!")
