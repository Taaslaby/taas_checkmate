#!/usr/bin/env python3
"""
🧠🦗 FULL VNC + GAMMA RING + ASTROCYTES → HH → ATLASOV → MUJOCO
v3 — ANALOG DRIVE: глубина сигнала, не бинарные спайки

  Спайки (0/1)          →       Firing Rate [0, 1]       →     Muscle
  дискретный сигнал             непрерывная глубина             плавное движение

Архитектура:
  VNC HH → V_m / spikes → RateBuffer(20ms) → continuous drive [0,1] → Atlasov → MuJoCo

Идея: нейрон спайкует НЕ тогда когда надо дёрнуть мышцей,
а когда надо ИЗМЕНИТЬ глубину сигнала. Частота спайков = глубина = сила.
"""

import os, sys, time, math
from pathlib import Path
from collections import defaultdict, Counter, deque
import numpy as np

os.environ["MUJOCO_GL"] = "egl"
from PIL import Image

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import mujoco as mu
import taas_checkmate as tc

from vnc_real_atlasov_embodied import (
    RealVNCConnectome, AtlasovFlyMuJoCo, SCENE_XML,
)

from vnc_astrocyte_layer import (
    AstrocyteNetwork, GapJunctionConnectome,
)


# ═══════════════════════════════════════════════════════════════════
#  FIX 1: Analog Drive Mapper — глубина сигнала вместо 0/1
# ═══════════════════════════════════════════════════════════════════

class AnalogRateDriveMapper:
    """Карта MN→мышцы с analog drive через firing rate.

    Принцип:
      - Каждый MN имеет буфер последних N спайков
      - Firing rate = количество спайков / время окна
      - drive = clamp(rate / target_rate, 0, 1) * max_force
      - flex/ext XOR: каждый MN — только flex ИЛИ только ext

    Это превращает бинарные спайки в НЕПРЕРЫВНУЮ ГЛУБИНУ сигнала.
    """

    def __init__(self, connectome: RealVNCConnectome, fly_sim, 
                 rate_window: int = 40, target_rate: float = 0.3):
        """
        Args:
            rate_window: шагов для подсчёта firing rate (40 × 0.5ms = 20ms)
            target_rate: firing rate при котором drive = 1.0 (30% времени = ~600Hz)
        """
        self.connectome = connectome
        self.fly = fly_sim
        self.n_muscles = fly_sim.n_muscles
        self.n_joints = fly_sim.n_joints
        self.rate_window = rate_window
        self.target_rate = target_rate

        # leg_dof_list
        self.leg_dof_list = {}
        for leg_label, dofs in fly_sim.leg_dofs.items():
            positions = []
            for dof in dofs:
                if dof in fly_sim.all_dofs:
                    positions.append(list(fly_sim.all_dofs).index(dof))
            self.leg_dof_list[leg_label] = positions

        # MN → (flex_m, ext_m, is_flex) — is_flex=True значит этот MN flex, False = ext
        self.mn_map = {}  # motor_idx → (flex_muscle, ext_muscle, is_flex)
        self._build_mapping()

        # ⚡ РАТЕ BUFFER: для каждого MN храним историю спайков
        # rate_buffer[mn_idx] = deque(maxlen=rate_window) of 0/1
        n_mn = len(connectome.leg_motor)
        self.rate_buffer = [
            deque([0] * rate_window, maxlen=rate_window) 
            for _ in range(n_mn)
        ]

    def _get_leg_from_nerve(self, nerve, body_part):
        """Определить leg_label по nerve + body_part."""
        from vnc_real_atlasov_embodied import NERVE_TO_LEG, BP_TO_NEUROMERE
        leg_label = NERVE_TO_LEG.get(nerve)
        if leg_label:
            return leg_label
        bp_prefix = BP_TO_NEUROMERE.get(body_part)
        if bp_prefix is None:
            return None
        if 'left' in nerve or '_L' in nerve:
            return f'{bp_prefix}_left'
        elif 'right' in nerve or '_R' in nerve:
            return f'{bp_prefix}_right'
        return None

    def _build_mapping(self):
        """Build MN→DOF mapping: каждый MN flex ИЛИ ext, не оба."""
        leg_mn_meta = []
        for idx in self.connectome.leg_motor:
            rid = self.connectome.idx_to_root[idx]
            meta = self.connectome.neurons.get(rid, {})
            nt = meta.get('nt', '')
            if nt == 'GABA':
                continue
            leg_mn_meta.append({
                'idx': idx,
                'body_part': meta.get('body_part', ''),
                'nerve': meta.get('nerve', ''),
                'nt': nt,
            })

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

            for i, mn in enumerate(mns):
                dof_idx = dof_positions[i % n_dofs]
                flex_m = 2 * dof_idx
                ext_m = 2 * dof_idx + 1
                if flex_m >= self.n_muscles:
                    continue
                # is_flex = True если чётный в группе, иначе ext
                self.mn_map[mn['idx']] = (flex_m, ext_m, i % 2 == 0)
                mapped += 1

        # Fallback
        fallback = [mn for mn in leg_mn_meta if mn['idx'] not in self.mn_map]
        all_dofs = []
        for leg in ['T1_left','T1_right','T2_left','T2_right','T3_left','T3_right']:
            all_dofs.extend(self.leg_dof_list.get(leg, []))
        for i, mn in enumerate(fallback):
            if not all_dofs:
                break
            dof_idx = all_dofs[i % len(all_dofs)]
            flex_m = 2 * dof_idx
            ext_m = 2 * dof_idx + 1
            if flex_m < self.n_muscles:
                self.mn_map[mn['idx']] = (flex_m, ext_m, i % 2 == 0)
                mapped += 1

        print(f"  🧬 AnalogRateDrive: {len(self.mn_map)} mappings ({mapped} mapped)")

    def route(self, motor_spikes):
        """Convert motor spikes → analog muscle drive via firing rate depth.

        Args:
            motor_spikes: [n_mn] array of 0/1 spike indicators

        Returns:
            motor_drive: [n_muscles] float32 — continuous drive [0, max_drive]
        """
        n_mn = len(motor_spikes)
        max_drive = 2.0  # максимальная аналоговая глубина

        # 1. Update rate buffers
        for i in range(n_mn):
            self.rate_buffer[i].append(1 if motor_spikes[i] > 0 else 0)

        # 2. Compute firing rate depth for each MN
        #    depth = clamp(rate / target_rate, 0, 1)
        motor_drive = np.zeros(self.n_muscles, dtype=np.float32)

        for mn_idx, (flex_m, ext_m, is_flex) in self.mn_map.items():
            if mn_idx >= n_mn:
                continue

            # ⚡ Firing rate = глубина сигнала
            rate = sum(self.rate_buffer[mn_idx]) / max(self.rate_window, 1)
            depth = min(1.0, rate / max(self.target_rate, 1e-6))

            # Глубина → drive (continuous, не бинарный)
            if is_flex:
                if is_flex:
                motor_drive[flex_m] = max(motor_drive[flex_m], depth * max_drive)
            else:
                motor_drive[ext_m] = max(motor_drive[ext_m], depth * max_drive)

        return motor_drive


# ═══════════════════════════════════════════════════════════════════
#  FIX 2: Smooth Atlasov — повышенная жёсткость + демпфирование
# ═══════════════════════════════════════════════════════════════════

class SmoothFlyMuJoCo(AtlasovFlyMuJoCo):
    """Atlasov с повышенной жёсткостью/демпфированием для плавной ходьбы.
    
    ⚡ Принимает continuous float32 drive (не int) — аналоговый сигнал.
    """
    CLIP_VALUE = 0.3

    def __init__(self, xml_path=SCENE_XML):
        super().__init__(xml_path)
        self.k0 = np.full(self.n_muscles // 2, 0.15, dtype=np.float32)
        self.beta = np.full(self.n_muscles // 2, 0.15, dtype=np.float32)
        self.b0 = np.full(self.n_muscles // 2, 0.30, dtype=np.float32)
        self.alpha = np.full(self.n_muscles // 2, 0.08, dtype=np.float32)
        self.t_max = np.full(self.n_muscles, 1.5, dtype=np.float32)
        self.d_ca = np.full(self.n_muscles, 0.4, dtype=np.float32)
        self.kd = np.full(self.n_muscles, 0.20, dtype=np.float32)

    def step(self, motoneuron_drive):
        """
        Args:
            motoneuron_drive: [n_muscles] float32 — непрерывная глубина [0, 2]
        """
        n_in = min(len(motoneuron_drive), self.n_muscles)
        # ⚡ Аналоговый сигнал: float → int с сохранением глубины
        self.spikes[:n_in] = (motoneuron_drive[:n_in] * 1.5).astype(np.int32)
        self.resets[:n_in] = 0

        self.step_fn(
            self.ca, self.tension, self.spikes, self.resets,
            float(self.dt * 1000.0),
            self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
            self.joint_df, self.joint_stf, self.joint_dmp,
            self.k0, self.beta, self.b0, self.alpha
        )

        # Soft clamp
        self.data.qfrc_applied.fill(0.0)
        for j in range(self.n_joints):
            dof = self.all_dofs[j]
            q = self.data.qpos[dof]
            qd = self.data.qvel[dof]
            tau = self.joint_df[j] - self.joint_stf[j] * q - self.joint_dmp[j] * qd
            tau_clamped = self.CLIP_VALUE * np.tanh(tau / max(self.CLIP_VALUE, 1e-6))
            self.data.qfrc_applied[dof] = tau_clamped

        mu.mj_step(self.model, self.data)
        return {
            't': self.data.time,
            'pos': self.data.qpos[:3].copy(),
            'df': self.joint_df[:self.n_joints].copy(),
            'k': self.joint_stf[:self.n_joints].copy(),
        }


# ═══════════════════════════════════════════════════════════════════
#  BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_fast_gap_junctions(N, p=0.005, g_gap=0.1, seed=42):
    gj = GapJunctionConnectome()
    expected = int(N * (N - 1) / 2 * p)
    if expected == 0:
        return gj
    n_samples = min(expected * 2, N * N // 2)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N, size=n_samples)
    b = rng.integers(0, N, size=n_samples)
    mask = a < b
    a_ok, b_ok = a[mask], b[mask]
    for i in range(len(a_ok)):
        gj.add(a_ok[i], b_ok[i], g_gap)
    return gj


def run_chunked_hh(sim, astro, conn_data, I_bg, n_total, N,
                    dt_ms=0.5, chunk_size=100):
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    all_spikes = []
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_n = end - start
        I_ext = I_bg.copy()
        I_ext *= astro.volume_gain
        chunk_spikes = sim.step_batch(conn_data, n_steps=chunk_n, dt_ms=dt_ms,
                                       external_I=I_ext, scale=1.0).reshape(chunk_n, N)
        for t in range(chunk_n):
            astro.step(dt_ms, chunk_spikes[t])
        all_spikes.append(chunk_spikes)
        if (ci + 1) % 10 == 0:
            print(f"   Chunk {ci+1}/{n_chunks} — {end}/{n_total} "
                  f"({end*dt_ms/1000:.1f}s), gain={astro.volume_gain.mean():.3f}")
    return np.concatenate(all_spikes, axis=0)


# ═══════════════════════════════════════════════════════════════════
#  MAIN — v3 Analog Drive
# ═══════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("🧠🦗 INTEGRATED FLY — v3 ANALOG DRIVE (firing rate depth)")
    print("═" * 80)

    # ─── 1. Load VNC ───────────────────────────────────────────────
    print("\n📥 STEP 1: LOADING VNC CONNECTOME")
    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons() or not conn.load_connections():
        return
    conn_data = conn.build_connectome_data()
    N = conn.n_neurons
    print(f"   ✅ {N:,} neurons, {conn.n_synapses:,} synapses")

    # ─── 2. Astrocytes ─────────────────────────────────────────────
    print("\n🧫 STEP 2: ASTROCYTE LAYER")
    print("─" * 60)
    t0 = time.time()
    gj = build_fast_gap_junctions(N, p=0.005, g_gap=0.1)
    gj.build(N)
    print(f"   ✅ {gj.n_pairs:,} gap pairs in {time.time()-t0:.1f}s")

    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    astro = AstrocyteNetwork(
        simulator=sim, n_astro=N, gap_conn=gj,
        tau_astro_ms=2000.0, g_gap=0.1, mu=0.8,
    )
    print(f"   ✅ Astrocyte μ=0.8, τ=2s")

    # ─── 3. Inputs (LOWER tonic для плавности) ─────────────────────
    print("\n⚡ STEP 3: SETUP INPUTS")
    print("─" * 60)
    rng_state = np.random.RandomState(42)
    I_bg = rng_state.randn(N).astype(np.float32) * 2.0

    if len(conn.leg_motor) > 0:
        I_bg[np.array(conn.leg_motor, dtype=np.intp) % N] += 12.0  # ещё ниже
        print(f"   🦵 Motor tonic: 12 pA × {len(conn.leg_motor)}")
    if len(conn.descending) > 0:
        I_bg[np.array(conn.descending, dtype=np.intp) % N] += 20.0
        print(f"   ⬇️  Descending:  20 pA × {len(conn.descending)}")

    # ─── 4. Smooth Fly + Analog Drive ─────────────────────────────
    print("\n🦗 STEP 4: SMOOTH FLY + ANALOG DRIVE")
    print("─" * 60)
    fly = SmoothFlyMuJoCo()

    # ⚡ ВАЖНО: AnalogRateDriveMapper вместо FrankenMotorMapper!
    mapper = AnalogRateDriveMapper(conn, fly, rate_window=40, target_rate=0.3)
    print(f"   ✅ {fly.n_joints} joints, {fly.n_muscles} muscles")
    print(f"   ✅ Analog drive: rate_window=40steps(20ms), target_rate=0.3")

    # ─── 5. Simulation ─────────────────────────────────────────────
    n_warmup = 500
    n_walk = 2000
    n_total = n_warmup + n_walk
    dt_ms = 0.5
    chunk_size = 100

    print(f"\n🎬 STEP 5: HH SIM ({n_total} steps, {n_total*dt_ms/1000:.1f}s)")
    print("─" * 60)
    print(f"   Warmup: {n_warmup} | Walk: {n_walk} | dt={dt_ms}ms")
    print(f"   Drive: ANALOG (firing rate → continuous [0, {2.0}])")
    print(f"   Clip: ±{fly.CLIP_VALUE} soft tanh | Astro: μ=0.8")

    astro.reset()
    t0 = time.time()
    spike_buf = run_chunked_hh(sim, astro, conn_data, I_bg, n_total, N,
                                dt_ms=dt_ms, chunk_size=chunk_size)
    t_sim = time.time() - t0
    print(f"\n   ✅ HH sim: {t_sim:.1f}s ({n_total/t_sim:.0f} steps/s)")

    # ─── 6. Walk + Video ───────────────────────────────────────────
    print(f"\n🦗 STEP 6: WALK + VIDEO ({n_walk} steps)")
    print("─" * 60)

    renderer = mu.Renderer(fly.model, height=480, width=640)
    cam = mu.MjvCamera()
    cam.type = mu.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.5
    cam.azimuth = 60.0
    cam.elevation = -30.0
    thorax_bid = mu.mj_name2id(fly.model, mu.mjtObj.mjOBJ_BODY, "thorax")

    frames = []
    trajectory = []
    start_pos = None
    total_motor_depth = 0.0

    for step in range(n_total):
        step_spikes = spike_buf[step]

        if step >= n_warmup:
            motor_spikes = np.array([
                float(step_spikes[idx]) for idx in conn.leg_motor if idx < N
            ], dtype=np.float32)

            # ⚡ Analog drive: firing rate → continuous depth
            motor_drive = mapper.route(motor_spikes)
            total_motor_depth += float(motor_drive.sum())

            out = fly.step(motor_drive)

            if start_pos is None:
                start_pos = out['pos'].copy()
            trajectory.append(out['pos'].copy())

            if (step - n_warmup) % 5 == 0:
                if thorax_bid >= 0:
                    cam.lookat = fly.data.xpos[thorax_bid].copy()
                renderer.update_scene(fly.data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))

        if step % 500 == 0:
            phase = "🔥 WARMUP" if step < n_warmup else "🚶 WALK"
            print(f"   Step {step:5d}/{n_total} | {phase} | "
                  f"spikes={step_spikes.sum():.0f}")

    # ─── 7. Results ────────────────────────────────────────────────
    displacement = np.linalg.norm(trajectory[-1] - start_pos) * 1000
    walk_time = n_walk * dt_ms / 1000
    speed = displacement / walk_time if walk_time > 0 else 0.0

    def classify(s):
        if s < 1: return "STANDING ❌"
        if s < 5: return "CRAWLING 🐛"
        if s < 10: return "WALKING 🦗"
        if s < 25: return "FAST WALK 🦗✅"
        if s < 50: return "RUNNING 🏃"
        return "FLYING 🚀"

    gait = classify(speed)

    print(f"\n{'='*80}")
    print("🏆 v3 ANALOG DRIVE — RESULTS")
    print(f"{'='*80}")
    print(f"   🧠 VNC neurons:       {N:,}")
    print(f"   🧫 Gap junctions:     {gj.n_pairs:,} (μ={astro.mu})")
    print(f"   🦗 Joints:            {fly.n_joints}")
    print(f"   🦵 MN→DOF:           {len(mapper.mn_map)}")
    print(f"   ⚙️ Drive type:        ANALOG (firing rate → continuous)")
    print(f"   ⚙️ Rate window:       {mapper.rate_window} steps ({mapper.rate_window*dt_ms:.0f}ms)")
    print(f"   ⚙️ Clip:              ±{fly.CLIP_VALUE} soft tanh")
    print(f"")
    print(f"   🚶 Displacement:      {displacement:.2f} mm")
    print(f"   ⏱ Walk time:         {walk_time:.2f} s")
    print(f"   📏 Speed:             {speed:.2f} mm/s ({speed/3:.1f} BL/s)")
    print(f"   🏃 Gait:              {gait}")
    print(f"   🎬 Frames:            {len(frames)}")

    # ─── 8. Save ───────────────────────────────────────────────────
    webp_path = Path(__file__).parent.parent / "vnc_gamma_astro_walk_v3.webp"
    frames[0].save(webp_path, save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    print(f"\n   💾 Video: {webp_path}")

    # Save data for comparison
    data_path = Path(__file__).parent.parent.parent / "data" / "v3_analog_drive.npz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(data_path,
             speed=speed, displacement=displacement,
             total_motor_depth=total_motor_depth,
             gait=gait)
    print(f"   💾 Data: {data_path}")

    print(f"\n{'='*80}")
    print("✅ v3 COMPLETE! Analog drive: spikes → continuous depth → smooth walk")
    print(f"   Бинарные спайки → firing rate → analog muscle drive")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
