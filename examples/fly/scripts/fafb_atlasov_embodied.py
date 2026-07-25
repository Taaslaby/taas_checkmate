#!/usr/bin/env python3
"""
🧠🦗 FAFB BRAIN → ATLASOV MUSCLES → MUJOCO FLY
=================================================
Первый в мире пайплайн: реальный коннектом мозга дрозофилы (75,919 нейронов)
через сомато-дендритные нейроны с активными HH-каналами → Atlasov Ca²⁺ мышцы
→ Tug-of-War joint impedance → MuJoCo 3D физика.

   [FAFB Connectome] → [SomatoDendritic Neurons] → [Spike Router]
        ↓                                                      ↓
   [Atlasov Ca²⁺ Sarcoplasmic Engine] ← [Motor Neuron Spikes]
        ↓
   [Agonist-Antagonist Tug-of-War: ΔF, K_joint, B_joint]
        ↓
   [MuJoCo 3D Physics: qfrc_applied + impedance modulation]
        ↓
   🦗 FLY WALKS!

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import time
import gzip
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

DLOAD = Path.home() / "Загрузки"
CONN_FILE = DLOAD / "connections_princeton.csv.gz"
SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

import mujoco as mu
import taas_checkmate as tc


# ─── 1. Load FAFB Connectome ─────────────────────────────────────────

def load_fafb_connectome(max_synapses=200000):
    """Load FAFB Princeton connectome, return edges + N."""
    print("📁 Загрузка FAFB коннектома...")
    if not CONN_FILE.exists():
        print(f"❌ Файл не найден: {CONN_FILE}")
        sys.exit(1)

    raw_edges = []
    t0 = time.time()
    with gzip.open(CONN_FILE, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_edges.append((
                int(row["pre_root_id"]),
                int(row["post_root_id"]),
                int(row.get("syn_count", 1))
            ))
            if len(raw_edges) >= max_synapses:
                break

    unique = sorted(set([e[0] for e in raw_edges] + [e[1] for e in raw_edges]))
    node_to_idx = {n: i for i, n in enumerate(unique)}
    edges = [(node_to_idx[p], node_to_idx[o], s) for p, o, s in raw_edges]
    N = len(unique)

    # Build edge map for fast lookups
    edge_map = defaultdict(list)
    for pre, post, syn in edges:
        edge_map[pre].append((post, syn))

    print(f"✅ {len(edges):,} синапсов, {N:,} нейронов за {time.time()-t0:.2f}с")
    return edges, edge_map, N


# ─── 2. Atlasov → MuJoCo Embodied Sim ────────────────────────────────

class AtlasovFlyMuJoCo:
    """Atlasov Neuromotor Engine driving MuJoCo FruitFly."""

    def __init__(self, xml_path=SCENE_XML):
        print(f"🦗 Загрузка MuJoCo мухи: {xml_path}")
        self.model = mu.MjModel.from_xml_path(str(xml_path))
        self.data = mu.MjData(self.model)

        # Joint DOFs (legs)
        self.dof_indices = []
        for i in range(self.model.njnt):
            jname = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_JOINT, i)
            if jname and any(k in jname for k in
                             ['leg', 'coxa', 'femur', 'tibia', 'tarsus',
                              'L1', 'L2', 'L3', 'R1', 'R2', 'R3']):
                self.dof_indices.append(self.model.jnt_dofadr[i])
        if not self.dof_indices:
            self.dof_indices = list(range(6, min(18, self.model.njnt)))

        self.n_joints = len(self.dof_indices)
        self.n_muscles = math.ceil((self.n_joints * 2) / 8) * 8  # SIMD pad
        print(f"   DOFs: {self.n_joints} joints → {self.n_muscles} muscles")

        # Sarcoplasmic muscle state
        self.ca = np.zeros(self.n_muscles, dtype=np.float32)
        self.tension = np.zeros(self.n_muscles, dtype=np.float32)
        self.spikes = np.zeros(self.n_muscles, dtype=np.int32)
        self.resets = np.zeros(self.n_muscles, dtype=np.int32)

        self.d_ca = np.full(self.n_muscles, 0.5, dtype=np.float32)
        self.tau_pump = np.full(self.n_muscles, 20.0, dtype=np.float32)
        self.eta_reset = np.full(self.n_muscles, 0.8, dtype=np.float32)
        self.t_max = np.full(self.n_muscles, 0.01, dtype=np.float32)
        self.kd = np.full(self.n_muscles, 0.3, dtype=np.float32)

        self.joint_df = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_stf = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_dmp = np.zeros(self.n_muscles // 2, dtype=np.float32)

        self.k0 = np.full(self.n_muscles // 2, 0.001, dtype=np.float32)
        self.beta = np.full(self.n_muscles // 2, 0.005, dtype=np.float32)
        self.b0 = np.full(self.n_muscles // 2, 0.0001, dtype=np.float32)
        self.alpha = np.full(self.n_muscles // 2, 0.0005, dtype=np.float32)

        self.dt = self.model.opt.timestep
        self.step_fn = tc.neurons.step_atlasov_neuromotor

    def step(self, motoneuron_spikes):
        """One MuJoCo physics step with Atlasov muscle drive."""
        n_in = min(len(motoneuron_spikes), self.n_muscles)
        self.spikes[:n_in] = motoneuron_spikes[:n_in]
        self.resets[:n_in] = 0

        # Atlasov Ca²⁺ step (Rust SIMD)
        self.step_fn(
            self.ca, self.tension, self.spikes, self.resets,
            float(self.dt * 1000.0),
            self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
            self.joint_df, self.joint_stf, self.joint_dmp,
            self.k0, self.beta, self.b0, self.alpha
        )

        # Apply joint impedance → MuJoCo
        self.data.qfrc_applied.fill(0.0)
        for j in range(self.n_joints):
            dof = self.dof_indices[j]
            q = self.data.qpos[dof]
            qd = self.data.qvel[dof]
            tau = self.joint_df[j] - self.joint_stf[j] * q - self.joint_dmp[j] * qd
            self.data.qfrc_applied[dof] = np.clip(tau, -0.01, 0.01)

        mu.mj_step(self.model, self.data)
        return {
            't': self.data.time,
            'pos': self.data.qpos[:3].copy(),
            'df': self.joint_df[:self.n_joints].copy(),
            'k': self.joint_stf[:self.n_joints].copy(),
        }


# ─── 3. Brain → Motor Spike Router ──────────────────────────────────

class BrainToMotorRouter:
    """Maps FAFB brain spikes to 12 motor channels (6 joints × 2 muscles)."""

    def __init__(self, N_brain, n_motor=12):
        # For each motor channel, pick a random group of brain neurons
        rng = np.random.RandomState(42)
        self.groups = []
        for m in range(n_motor):
            # Each muscle driven by ~2% of brain neurons
            n_per_group = max(1, int(N_brain * 0.02))
            self.groups.append(rng.choice(N_brain, n_per_group, replace=False))
        self.n_motor = n_motor
        self.spike_history = []
        self.step = 0

    def route(self, brain_spikes):
        """Convert per-neuron spikes [N] to motor drive [n_motor]."""
        motor = np.zeros(self.n_motor, dtype=np.int32)
        for m in range(self.n_motor):
            # Population rate code: fire if any neuron in group spiked
            motor[m] = 1 if np.any(brain_spikes[self.groups[m]] > 0) else 0
        if self.step < 100000:
            self.spike_history.append(motor.copy())
        self.step += 1
        return motor


# ═════════════════════════════════════════════════════════════════════
#  MAIN — THE MONSTER PIPELINE
# ═════════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("🧠🦗 FAFB BRAIN → ATLASOV MUSCLES → MUJOCO FLY")
    print("   First ever: Real fly brain connectome driving a 3D fly!")
    print("═" * 80)

    # ── 1. FAFB Connectome ─────────────────────────────────────────
    edges, edge_map, N = load_fafb_connectome(max_synapses=200000)

    # ── 2. Somato-Dendritic Brain ──────────────────────────────────
    print(f"🧠 Создание {N:,} сомато-дендритных нейронов...")
    brain = tc.neurons.SomatoDendriticNeuron(
        n_neurons=N, n_compartments=4,
        r_a=180.0, r_m=120.0, c_m=0.1, dt=0.5,
        tau_m=20.0, v_rest=-65.0, v_thresh=-55.0,
        v_reset=-65.0, tau_ref=2.5
    )
    print("⚡ Активация HH-каналов (Na⁺ m³h, K⁺ n⁴, Ca²⁺ m_ca²)...")
    brain.enable_active_channels(g_na=12.0, g_kv=6.0, g_ca=3.0)

    # SomatoDendriticNeuron возвращает v с паддингом под SIMD
    N_padded = len(brain.v)
    print(f"   N_padded = {N_padded} (SIMD alignment)")

    # ── 3. Atlasov MuJoCo ──────────────────────────────────────────
    fly = AtlasovFlyMuJoCo()

    # ── 4. Brain → Motor Router ────────────────────────────────────
    router = BrainToMotorRouter(N, n_motor=fly.n_muscles)

    # ── 5. Parameters ──────────────────────────────────────────────
    n_warmup = 1000       # brain warmup (500ms)
    n_walk = 10000         # walking steps (5sec)
    n_total = n_warmup + n_walk
    dt_ms = 0.5

    np.random.seed(42)

    print(f"\n{'─' * 80}")
    print(f"🎬 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА:")
    print(f"   Фаза 1 — Разогрев мозга: {n_warmup} шагов ({n_warmup*dt_ms:.0f}ms)")
    print(f"   Фаза 2 — Ходьба:         {n_walk} шагов ({n_walk*dt_ms:.0f}ms)")
    print(f"{'─' * 80}\n")

    # Tracking
    start_pos = None
    displacement = 0.0
    total_brain_spikes = 0
    total_motor_firings = 0
    walk_log = []

    t_start = time.time()

    for step in range(n_total):
        # ── A. Brain simulation ────────────────────────────────────
        # Background noise (pad to N_padded for SIMD)
        I_bg = np.random.randn(N_padded).astype(np.float32) * 4.0 + 8.5

        # Sensory pulse every 30 steps (creates natural rhythm)
        if step % 30 == 0:
            stim = np.random.choice(N, size=int(N * 0.05), replace=False)
            I_bg[stim] += 15.0

        # Synaptic transmission via edge_map
        spikes_prev = brain.spikes
        if step > 0 and spikes_prev.sum() > 0:
            for pre in np.where(spikes_prev == 1)[0]:
                if pre in edge_map:
                    for post, syn in edge_map[pre]:
                        comp = (pre + post) % 4
                        if (pre * 7 + post) % 5 == 0:
                            brain.jump_gaba_a(post, comp, 0.04 * syn)
                        else:
                            brain.jump_ampa(post, comp, 0.05 * syn)
                            brain.jump_nmda(post, comp, 0.02 * syn)

        # Brain step
        brain.step(I_bg, dt_ms)

        current_spikes = int(brain.spikes.sum())
        total_brain_spikes += current_spikes

        # ── B. Route spikes to motors (only during walk phase) ─────
        if step >= n_warmup:
            motor_drive = router.route(brain.spikes)
            total_motor_firings += motor_drive.sum()

            # ── C. Atlasov → MuJoCo step ───────────────────────────
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

        # ── D. Progress ────────────────────────────────────────────
        if step % 200 == 0 or step == 0 or step == n_total - 1:
            elapsed = time.time() - t_start
            fps = (step + 1) / (elapsed + 1e-6)
            phase = "🔥 РАЗОГРЕВ" if step < n_warmup else "🚶 ХОДЬБА"
            bar = "█" * min(20, int(current_spikes / max(1, N * 0.05) * 20)) + "░" * (20 - min(20, int(current_spikes / max(1, N * 0.05) * 20)))
            disp_str = f" | Disp: {displacement:.2f}mm" if step >= n_warmup else ""
            print(f"⏱ Step {step:5d}/{n_total} | {phase} | V: {brain.v[:10].mean():.2f}mV | "
                  f"Spikes: {current_spikes:4d} [{bar}]{disp_str} | {fps:.0f} s/s")

    total_time = time.time() - t_start

    # ═══════════════════════════════════════════════════════════════
    #  GAIT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    walk_log_arr = np.array([(l['t'], l['disp'], l['mean_k'], l['mean_df'])
                             for l in walk_log],
                            dtype=[('t', 'f4'), ('disp', 'f4'), ('k', 'f4'), ('df', 'f4')])

    total_walk_time = n_walk * dt_ms / 1000  # seconds
    speed_mm_s = displacement / total_walk_time if total_walk_time > 0 else 0.0

    # Speed per segment
    seg_size = 2000  # steps per segment
    n_segs = n_walk // seg_size
    seg_speeds = []
    print("\n📊 ПОСЕГМЕНТНЫЙ АНАЛИЗ СКОРОСТИ (каждые 1 сек):")
    print(f"{'Сектор':<10} | {'Время (ms)':<12} | {'Disp (mm)':<12} | {'Скорость (mm/s)':<18} | {'Ср. K (mN/rad)'}")
    print("-" * 70)
    for s in range(n_segs):
        start_idx = s * seg_size
        end_idx = min((s + 1) * seg_size, n_walk)
        if end_idx > start_idx:
            d_start = walk_log_arr['disp'][start_idx]
            d_end = walk_log_arr['disp'][end_idx - 1]
            d_delta = d_end - d_start
            t_seg = (end_idx - start_idx) * dt_ms / 1000
            seg_speed = d_delta / t_seg if t_seg > 0 else 0.0
            avg_k = np.mean(walk_log_arr['k'][start_idx:end_idx]) * 1e3
            seg_speeds.append(seg_speed)
            t_ms = end_idx * dt_ms
            print(f"  {s+1:<8} | {t_ms:<10.0f}  | {d_end:<10.2f}  | {seg_speed:<16.2f}  | {avg_k:<.2f}")

    # Gait classification
    avg_speed = np.mean(seg_speeds) if seg_speeds else 0.0
    speed_trend = seg_speeds[-1] - seg_speeds[0] if len(seg_speeds) > 1 else 0.0
    speed_cv = np.std(seg_speeds) / (np.mean(seg_speeds) + 1e-10) if seg_speeds else 0.0

    if avg_speed < 1.0:
        gait_class = "🟤 ПОЛЗЁТ (crawling)"
    elif avg_speed < 5.0:
        gait_class = "🟡 ИДЁТ (walking)"
    elif avg_speed < 15.0:
        gait_class = "🟢 БЕЖИТ (running)"
    else:
        gait_class = "🟣 ЛЕТИТ (flying / ballistic)"

    if speed_trend > 2.0:
        trend_str = "🚀 УСКОРЯЕТСЯ"
    elif speed_trend < -2.0:
        trend_str = "🐌 ЗАМЕДЛЯЕТСЯ"
    else:
        trend_str = "⚖️ СТАБИЛЬНАЯ СКОРОСТЬ"

    # Gait stability
    if speed_cv < 0.3:
        stability = "🟢 СТАБИЛЬНЫЙ gait — ровная скорость"
    elif speed_cv < 0.8:
        stability = "🟡 НЕСТАБИЛЬНЫЙ — скорость колеблется"
    else:
        stability = "🔴 РЫВКАМИ — stop-and-go"

    print("\n" + "═" * 80)
    print("🏆 РЕЗУЛЬТАТЫ ПОЛНОГО ПАЙПЛАЙНА!")
    print("═" * 80)
    print(f"  🧠 Мозг:")
    print(f"     Нейронов:           {N:,}")
    print(f"     Синапсов:          {len(edges):,}")
    print(f"     Всего спайков:     {total_brain_spikes:,}")
    print(f"     Средняя частота:   {total_brain_spikes/N/(n_total*dt_ms/1000):.2f} Hz")
    print(f"  🦗 Муха — Gait Analysis:")
    print(f"     Суставов:          {fly.n_joints}")
    print(f"     Мышц:              {fly.n_muscles}")
    print(f"     Мото-спайков:      {total_motor_firings:,}")
    print(f"     Дистанция:         {displacement:.2f} mm")
    print(f"     Время ходьбы:      {total_walk_time:.2f} sec")
    print(f"     Средняя скорость:  {speed_mm_s:.2f} mm/s")
    print(f"     Классификация:     {gait_class}")
    print(f"     Тренд:             {trend_str}")
    print(f"     Стабильность:      {stability}")
    print(f"     Средняя K:         {np.mean([l['mean_k'] for l in walk_log])*1e3:.2f} mN/rad")
    print(f"  ⏱ Производительность:")
    print(f"     Физическое время:  {n_total*dt_ms:.0f} ms ({n_total*dt_ms/1000:.2f} s)")
    print(f"     Реальное время:    {total_time:.2f} s")
    print(f"     Скорость:          {n_total/total_time:.0f} step/s")
    print(f"     Симуляция/реальность: {n_total*dt_ms/1000/total_time:.2f}×")
    print("═" * 80)

    # Save telemetry
    out_dir = Path(__file__).parent.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "fafb_walk_telemetry.npz",
             walk_time=np.array([l['t'] for l in walk_log]),
             walk_pos=np.array([l['pos'] for l in walk_log]),
             walk_disp=np.array([l['disp'] for l in walk_log]),
             walk_k=np.array([l['mean_k'] for l in walk_log]),
             walk_df=np.array([l['mean_df'] for l in walk_log]))
    print(f"💾 Телеметрия сохранена: {out_dir / 'fafb_walk_telemetry.npz'}")


if __name__ == "__main__":
    main()
