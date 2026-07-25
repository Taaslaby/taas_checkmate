#!/usr/bin/env python3
"""
🧬 C. elegans DepthWave — ψ-wave на коннектоме червя (279 нейронов)

DepthWave с топологическим depth-градиентом:
  depth=1.0 (сенсорика) → быстрая эволюция ψ
  depth=0.5 (интернейроны) → средняя скорость
  depth=0.0 (моторика) → медленная, выход на мышцы

Основано на DepthWave из t3-wavelet (Rust) + VNC DepthWave.
"""

import sys, os, time, math
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

# ─── Константы нейротрансмиттеров ──────────────────────────
NT_SIGN: Dict[str, float] = {
    'Ach': 1.0,       # ацетилхолин — возбуждающий
    'GLU': 1.0,       # глутамат — возбуждающий
    'GABA': -1.0,     # GABA — тормозный
    'DA': -1.0,       # дофамин — модуляторный (тормозный)
    '5-HT': 1.0,      # серотонин — модуляторный
    'NA': 0.0,        # неизвестно — нейтрально
    'Ach,5-HT': 1.0,  # смешанный
}

# Роли нейронов: глубина (1=sensory, 0.5=inter, 0=motor)
ROLE_DEPTH: Dict[str, float] = {
    'S': 1.0,    # sensory — максимальная глубина (быстрая эволюция)
    'NA': 0.5,   # interneuron — средняя
    'M': 0.0,    # motor — минимальная глубина (выход на мышцы)
}

# Моторные классы → группа мышц
MOTOR_MAP = {
    'DA': 'dorsal',   # DA → дорсальные мышцы
    'DB': 'dorsal',   # DB → дорсальные
    'DD': 'dorsal_inh', # DD → тормозные дорсальные
    'VA': 'ventral',  # VA → вентральные
    'VB': 'ventral',  # VB → вентральные
    'VD': 'ventral_inh', # VD → тормозные вентральные
    'VC': 'ventral',  # VC → вентральные
}

# ─── CelegansDepthWave ────────────────────────────────────

class CelegansDepthWave:
    """DepthWave для C. elegans — ψ-wave на коннектоме (279 нейронов)."""

    def __init__(self,
                 graphml_path: str = None,
                 n_segments: int = 8,
                 dt: float = 0.001,
                 mod_range: float = 2.0,
                 hbar: float = 1.0):
        if graphml_path is None:
            graphml_path = os.path.join(
                os.path.dirname(__file__),
                '../../../../sardaanaCM/experiments/datasets',
                'c.elegans.herm_pharynx_1.graphml'
            )

        self.dt = dt
        self.mod_range = mod_range
        self.hbar = hbar
        self.n_segments = n_segments

        print(f"📂 Загрузка коннектома C. elegans из {graphml_path}...")
        t0 = time.time()
        self._load_connectome(graphml_path)
        print(f"   ✅ {self.n_neurons} нейронов, {self.n_synapses} синапсов [{time.time()-t0:.1f}s]")

        t0 = time.time()
        self._build_hamiltonian()
        print(f"   ✅ H-матрица построена [{time.time()-t0:.1f}s]")

        self._init_wave_function()
        self._init_motor_output()

        print(f"   🧮 ψ-волна: {self.n_neurons} нейронов, dt={dt}s, mod_range={mod_range}")
        self.print_summary()

    def _load_connectome(self, path: str):
        g = nx.read_graphml(path)
        all_nodes = list(g.nodes(data=True))
        self.n_neurons = len(all_nodes)

        self.node_ids: List[str] = []
        self.node_idx: Dict[str, int] = {}
        self.cell_class: List[str] = []
        self.role: List[str] = []
        self.nt_type: List[str] = []
        self.neuron_name: List[str] = []

        for nid, data in all_nodes:
            idx = len(self.node_ids)
            self.node_ids.append(nid)
            self.node_idx[nid] = idx
            self.cell_class.append(str(data.get('cell_class', 'NA')))
            self.role.append(str(data.get('role', 'NA')))
            nt = str(data.get('neurotransmitters', 'NA'))
            self.nt_type.append(nt)
            self.neuron_name.append(str(data.get('cell_name', nid)))

        self.synapses: List[Tuple[int, int, float, str]] = []
        for pre, post, data in g.edges(data=True):
            if pre not in self.node_idx or post not in self.node_idx:
                continue
            pi = self.node_idx[pre]
            po = self.node_idx[post]
            stype = str(data.get('synapse_type', 'C'))
            weight = float(data.get('weight', 1.0))
            if weight <= 0:
                continue
            self.synapses.append((pi, po, weight, stype))

        self.n_synapses = len(self.synapses)

        self.nt_sign = np.zeros(self.n_neurons, dtype=np.float32)
        for i, nt in enumerate(self.nt_type):
            self.nt_sign[i] = NT_SIGN.get(nt, 0.0)

        self.depth = np.zeros(self.n_neurons, dtype=np.float32)
        for i, r in enumerate(self.role):
            self.depth[i] = ROLE_DEPTH.get(r, 0.5)

        self.speed = 1.0 + self.depth * self.mod_range

    def _build_hamiltonian(self):
        n = self.n_neurons
        self.synapses.sort(key=lambda x: x[0])

        row_ptr = np.zeros(n + 1, dtype=np.int32)
        col_idx = np.zeros(len(self.synapses), dtype=np.int32)
        h_values = np.zeros(len(self.synapses), dtype=np.float32)

        pre_count = np.zeros(n, dtype=np.int32)
        for pre, post, weight, stype in self.synapses:
            pre_count[pre] += 1

        count = 0
        for i in range(n):
            row_ptr[i] = count
            count += pre_count[i]
        row_ptr[n] = count

        cursor = np.zeros(n, dtype=np.int32)
        for pre, post, weight, stype in self.synapses:
            pos = row_ptr[pre] + cursor[pre]
            col_idx[pos] = np.int32(post)
            h_values[pos] = weight * self.nt_sign[pre]
            cursor[pre] += 1

        # CSR → H[pre, post] = weight * sign[pre]; then transpose to H[post, pre]
        self.H = csr_matrix((h_values, col_idx, row_ptr), shape=(n, n)).T.tocsr()
        self.nnz = self.H.nnz
        self.n_chemical = sum(1 for _, _, _, s in self.synapses if s == 'C')
        self.n_electrical = sum(1 for _, _, _, s in self.synapses if s == 'E')

    def _init_wave_function(self):
        n = self.n_neurons
        rng = np.random.RandomState(42)
        phases = rng.uniform(0, 2 * math.pi, n)
        self.v = np.cos(phases).astype(np.float32)
        self.u = np.sin(phases).astype(np.float32)
        norm = np.sqrt(np.sum(self.v**2 + self.u**2) + 1e-9)
        self.v /= norm
        self.u /= norm

    def _init_motor_output(self):
        self.motor_indices: List[int] = []
        self.motor_class: List[str] = []

        for i, cls in enumerate(self.cell_class):
            if cls in MOTOR_MAP:
                self.motor_indices.append(i)
                self.motor_class.append(MOTOR_MAP[cls])

        n_motor = len(self.motor_indices)
        print(f"   🦾 Моторных нейронов: {n_motor}")

        if n_motor == 0:
            print("   ⚠️ Нет моторных нейронов! Использую role='M'")
            for i, r in enumerate(self.role):
                if r == 'M':
                    self.motor_indices.append(i)
                    self.motor_class.append('dorsal')
            n_motor = len(self.motor_indices)

        self.motor_to_segment = np.zeros(n_motor, dtype=np.int32)
        for mi in range(n_motor):
            self.motor_to_segment[mi] = (mi * self.n_segments) // max(n_motor, 1)

        self.dorsal_motors = [mi for mi, mc in enumerate(self.motor_class)
                              if mc in ('dorsal', 'dorsal_inh')]
        self.ventral_motors = [mi for mi, mc in enumerate(self.motor_class)
                               if mc in ('ventral', 'ventral_inh')]
        self.inhibitory = set(i for i, mc in enumerate(self.motor_class)
                              if mc.endswith('_inh'))

        self.n_motor = n_motor
        self.n_dorsal = len(self.dorsal_motors)
        self.n_ventral = len(self.ventral_motors)
        print(f"   🐛 Дорсальных: {self.n_dorsal}, Вентральных: {self.n_ventral}")

    def step(self, sensory_input: Optional[np.ndarray] = None, dt: Optional[float] = None) -> Dict:
        if dt is None:
            dt = self.dt

        # ψ-wave: dv/dt = speed * (H @ u), du/dt = -speed * (H @ v)
        Hu = self.H.dot(self.u)
        Hv = self.H.dot(self.v)

        self.v += dt * self.speed * Hu
        self.u -= dt * self.speed * Hv

        if sensory_input is not None:
            n_in = min(len(sensory_input), self.n_neurons)
            sw = 1.0 - self.depth
            self.v[:n_in] += dt * sw[:n_in] * sensory_input[:n_in]

        # Global L2 normalize
        norm = np.sqrt(np.sum(self.v**2 + self.u**2) + 1e-9)
        self.v /= norm
        self.u /= norm

        psi2 = self.v**2 + self.u**2
        motor_psi = psi2[self.motor_indices]

        segment_dorsal = np.zeros(self.n_segments, dtype=np.float32)
        segment_ventral = np.zeros(self.n_segments, dtype=np.float32)

        for mi, seg in enumerate(self.motor_to_segment):
            amp = motor_psi[mi]
            if mi in self.dorsal_motors:
                if mi in self.inhibitory:
                    segment_dorsal[seg] -= amp
                else:
                    segment_dorsal[seg] += amp
            elif mi in self.ventral_motors:
                if mi in self.inhibitory:
                    segment_ventral[seg] -= amp
                else:
                    segment_ventral[seg] += amp

        max_d = max(np.abs(segment_dorsal).max(), np.abs(segment_ventral).max(), 1e-9)
        segment_dorsal = segment_dorsal / max_d
        segment_ventral = segment_ventral / max_d

        coherence = self._compute_coherence()

        return {
            'psi2': psi2,
            'motor_psi': motor_psi,
            'segment_dorsal': segment_dorsal,
            'segment_ventral': segment_ventral,
            'v': self.v.copy(),
            'u': self.u.copy(),
            'coherence': coherence,
        }

    def _compute_coherence(self) -> float:
        psi = self.v + 1j * self.u
        mean_psi = np.mean(psi)
        mean_psi2 = np.mean(np.abs(psi)**2)
        return float(np.abs(mean_psi)**2 / (mean_psi2 + 1e-9))

    def print_summary(self):
        roles = defaultdict(int)
        nts = defaultdict(int)
        motor_by_class = defaultdict(int)

        for i in range(self.n_neurons):
            roles[self.role[i]] += 1
            nts[self.nt_type[i]] += 1
            cls = self.cell_class[i]
            if cls in MOTOR_MAP:
                motor_by_class[cls] += 1

        print("   📊 Статистика нейронов:")
        print(f"      Роли: {dict(roles)}")
        print(f"      Нейротрансмиттеры: {dict(nts)}")
        print(f"      Моторные классы: {dict(motor_by_class)}")
        print(f"      Синапсы: {self.n_chemical} химических + {self.n_electrical} электрических")


# ─── CelegansBody2D ─────────────────────────────────────────

class CelegansBody2D:
    """
    Минимальное 2D тело C. elegans — N сегментов, spring-damper, мышцы.

    Режим low Reynolds number (как у настоящего червя):
    - Overdamped: vel ∝ F (инерция не важна)
    - Анизотропный drag: нормальный >> тангенциальный
    - Мышечные моменты между сегментами → bending wave
    """

    def __init__(self,
                 n_segments: int = 8,
                 dt_body: float = 0.005,
                 segment_length: float = 0.01,
                 stiffness: float = 100000.0,  # very stiff
                 drag_linear: float = 5000.0,  # strong overdamping
                 drag_angular: float = 0.5,
                 muscle_scale: float = 0.02,  # smaller to prevent explosions
                 fluid: bool = True):
        self.N = n_segments
        self.dt = dt_body
        self.seg_len = segment_length
        self.stiffness = stiffness
        self.drag_lin = drag_linear
        self.drag_ang = drag_angular
        self.muscle_scale = muscle_scale
        self.fluid = fluid
        self.n_joints = n_segments - 1

        self.pos = np.zeros((self.N, 2), dtype=np.float64)
        self.vel = np.zeros_like(self.pos)
        self.theta = np.zeros(self.N, dtype=np.float64)
        self.omega = np.zeros(self.N, dtype=np.float64)
        self.reset()

        self.bend_stiff = 0.1  # passive bending stiffness

    @property
    def head_pos(self) -> Tuple[float, float]:
        return float(self.pos[0, 0]), float(self.pos[0, 1])

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        return float(np.mean(self.pos[:, 0])), float(np.mean(self.pos[:, 1]))

    def step(self, dorsal_act: np.ndarray, ventral_act: np.ndarray) -> Dict:
        """
        Proper overdamped Stokes dynamics:
          Joints: stiff elastic + active bending torques
          drag * vel = F_spring + F_muscle
        """
        n_j = self.n_joints
        da = np.resize(dorsal_act, n_j).astype(np.float64)
        va = np.resize(ventral_act, n_j).astype(np.float64)

        # ─── Joint forces: stiff springs maintain segment distances ───
        diff = self.pos[1:] - self.pos[:-1]
        dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-12)
        unit = diff / dist[:, np.newaxis]

        # Very stiff spring → segments stay at rest length
        spring_force = self.stiffness * (dist - self.seg_len)
        F_spring = -spring_force[:, np.newaxis] * unit  # along joint

        # ─── Active bending torques → perpendicular forces ───
        # dorsal=+ torque (bend CCW), ventral=- torque (bend CW)
        tau_bend = (da - va) * self.muscle_scale
        # Passive straightening
        rel_th = self.theta[1:] - self.theta[:-1]
        tau_bend -= self.bend_stiff * rel_th

        # Perpendicular direction to segment
        perp = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
        # Torque → perpendicular force on each adjacent segment
        F_bend = tau_bend[:, np.newaxis] * perp

        # ─── Net forces on segments ───
        # F_spring = -k*(dist-rest)*unit = inward-pointing force
        # left segment: -F_spring (outward) → pulls RIGHT (toward center)
        # right segment: +F_spring (inward) → pulls LEFT (toward center)
        F_total = np.zeros_like(self.pos)
        F_total[:-1] -= F_spring  # pull left segment RIGHT
        F_total[1:] += F_spring   # pull right segment LEFT
        F_total[:-1] += F_bend    # perpendicular force on left
        F_total[1:] -= F_bend     # opposite on right

        # ─── Overdamped: vel = F / drag ───
        self.vel = F_total / self.drag_lin

        # ─── Angular: net torque on each segment ───
        # segment i gets torque from joint i-1 (right) and joint i (left)
        tau_net = np.zeros(self.N)
        tau_net[:-1] += tau_bend   # joint i bends segment i CCW
        tau_net[1:] -= tau_bend    # joint i bends segment i+1 CW
        self.omega = tau_net / self.drag_ang

        # ─── Integrate ───
        self.pos += self.vel * self.dt
        self.theta += self.omega * self.dt

        # Ground
        below = self.pos[:, 1] < 0.0
        self.pos[below, 1] = 0.0

        speed = float(np.sqrt(np.mean(self.vel[:, 0])**2 + np.mean(self.vel[:, 1])**2))

        return {
            'x': self.pos[:, 0].copy(),
            'y': self.pos[:, 1].copy(),
            'theta': self.theta.copy(),
            'head_x': float(self.pos[0, 0]),
            'head_y': float(self.pos[0, 1]),
            'speed': speed,
            'center_x': float(np.mean(self.pos[:, 0])),
            'center_y': float(np.mean(self.pos[:, 1])),
        }  # ← end step()

    def reset(self):
        self.pos = np.zeros((self.N, 2), dtype=np.float64)
        self.pos[:, 0] = np.arange(self.N) * self.seg_len
        self.vel = np.zeros_like(self.pos)
        self.theta = np.zeros(self.N, dtype=np.float64)
        self.omega = np.zeros(self.N)


# ─── Self-test ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("🧬 C. elegans DepthWave — Self-test")
    print("=" * 60)

    dw = CelegansDepthWave(n_segments=8)
    body = CelegansBody2D(n_segments=8, dt_body=0.005)

    # ψ-wave evolution
    print("\n⏱ Эволюция ψ-волны (2000 шагов)...")
    t0 = time.time()
    for step in range(2000):
        out = dw.step()
        if step % 500 == 0:
            print(f"   Step {step:4d}: |ψ|²=[{out['psi2'].min():.4f}, {out['psi2'].max():.4f}] "
                  f"coh={out['coherence']:.4f} "
                  f"dorsal={out['segment_dorsal'][0]:.3f} ventral={out['segment_ventral'][0]:.3f}")

    dt_sim = time.time() - t0
    print(f"   ✅ {dt_sim:.2f}s = {2000/dt_sim:.0f} steps/s")

    # Check all segments
    print("\n   Активация сегментов (после 2000 ψ-шагов):")
    for seg in range(8):
        print(f"      seg {seg}: dorsal={out['segment_dorsal'][seg]:.4f} "
              f"ventral={out['segment_ventral'][seg]:.4f}")
    print(f"   Моторные |ψ|²: min={out['motor_psi'].min():.6f} "
          f"max={out['motor_psi'].max():.6f} mean={out['motor_psi'].mean():.6f}")

    # Body simulation
    print("\n🏊 Тест плавания (200 body steps)...")
    body.reset()
    t0 = time.time()

    # Run ψ-wave 5x per body step (different timescales)
    psi_steps_per_body = 5
    for step in range(200):
        for _ in range(psi_steps_per_body):
            out = dw.step()
        body_out = body.step(out['segment_dorsal'], out['segment_ventral'])

        if step % 50 == 0:
            print(f"   Body step {step:3d}: head=({body_out['head_x']:.3f}, {body_out['head_y']:.3f}) "
                  f"speed={body_out['speed']:.4f} m/s")

    distance = body.head_pos[0]
    sim_time = time.time() - t0
    print(f"\n🏁 Дистанция: {distance*1000:.1f} мм за {sim_time:.1f}s симуляции")
    print(f"   Скорость: {distance*1000:.1f} mm/s")
    print("✅ Self-test complete!")
