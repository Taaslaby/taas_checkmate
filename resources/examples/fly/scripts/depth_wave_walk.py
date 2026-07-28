#!/usr/bin/env python3
"""
🦗🧬 DepthWave Walk — ψ-wave VNC → position servo → MuJoCo fly

  DepthWaveVNC (25K ψ, depth-modulated)
       ↓ motor_acts
  Position Servo (target joint angles)
       ↓
  MuJoCo Fly (Atlasov body)

Сравнение с HH+VNC v3:
  - v3: HH spikes → analog rate drive → torque → MuJoCo
  - DW: ψ-wave continuous → amplitude → position servo → MuJoCo
"""

import os, sys, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np

os.environ["MUJOCO_GL"] = "egl"
from PIL import Image
import mujoco as mu

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from depth_wave_vnc import DepthWaveVNC
from vnc_real_atlasov_embodied import (
    RealVNCConnectome, AtlasovFlyMuJoCo, SCENE_XML,
)


# ═══════════════════════════════════════════════════════════════════
#  DepthPositionFly — position servo вместо torque
# ═══════════════════════════════════════════════════════════════════

class DepthPositionFly:
    """Atlasov fly с position servo управлением от DepthWave.

    Вместо torque control (Atlasov Ca²⁺) использует position servo:
    задаём target joint angles напрямую в qpos.

    Анатомия (Atlasov, 8 DOF/leg):
      coxa_abduct, coxa_twist, coxa,
      femur_twist, femur, tibia,
      tarsus, tarsus2
    """

    # 8 DOFs per leg in Atlasov MuJoCo model
    N_DOFS_PER_LEG = 8

    # Анатомические диапазоны (Atlasov-совместимые)
    DOF_RANGES = [
        (-0.5,  0.5),   # coxa_abduct
        (-0.3,  0.3),   # coxa_twist
        (-0.5,  0.5),   # coxa
        (-0.3,  0.3),   # femur_twist
        (-1.0,  1.0),   # femur
        (-1.5,  0.5),   # tibia
        (-0.5,  0.5),   # tarsus
        (-0.2,  0.2),   # tarsus2
    ]

    # Амплитуды на DOF
    AMP_PER_DOF = [0.02, 0.0, 0.15, 0.02, 0.25, 0.15, 0.0, 0.0]
    # Фазовые сдвиги
    PHASE_PER_DOF = [0.0, 0.0, 0.0, 0.2, 0.3, 0.5, 0.0, 0.0]

    # Tripod: +1 = group A (T1L, T2R, T3L), -1 = group B (T1R, T2L, T3R)
    TRIPOD = np.array([1, -1, -1, 1, 1, -1])

    # Leg name → Atlasov leg label
    LEG_MAP = {
        'LF': 'T1_left', 'LM': 'T2_left', 'LH': 'T3_left',
        'RF': 'T1_right', 'RM': 'T2_right', 'RH': 'T3_right',
    }

    def __init__(self, dw: DepthWaveVNC, fly: AtlasovFlyMuJoCo, base_omega=6.0):
        self.dw = dw
        self.fly = fly
        self.base_omega = base_omega
        self.phase = 0.0
        self.n_motor = dw.n_motor
        self.n_joints = fly.n_joints

        # MN → leg mapping
        self.motor_to_leg = (np.arange(self.n_motor) * 6 // max(self.n_motor, 1)).astype(np.int32)

        # Warmup: stand pose (reset to stretch)
        self.fly.data.qpos[2] = 0.3  # body height
        self._stand_pose()
        mu.mj_forward(self.fly.model, self.fly.data)

    def _stand_pose(self):
        """Stand pose — все суставы в центре диапазона."""
        for leg_label in ['T1_left', 'T1_right', 'T2_left', 'T2_right', 'T3_left', 'T3_right']:
            dofs = self.fly.leg_dofs.get(leg_label, [])
            for j, (lo, hi) in enumerate(self.DOF_RANGES):
                if j < len(dofs):
                    self.fly.data.qpos[dofs[j]] = (lo + hi) / 2.0

    def step(self, sensory_input=None, dt_body=5e-4):
        """One step: DepthWave → position servo → MuJoCo.

        Returns dict with pos, coherence, leg_amp, displacement.
        """
        # 1. DepthWave evolution
        coherence, joint_acts = self.dw.step(sensory_input, n_steps=1)

        # 2. Phase update (walking rhythm)
        speed_mod = np.clip(coherence * 3.0 + 0.2, 0.2, 1.5)
        self.phase += self.base_omega * speed_mod * dt_body

        # 3. Per-leg amplitude from ψ motor acts
        leg_amp = np.zeros(6, dtype=np.float32)
        for i in range(min(len(joint_acts), self.n_motor)):
            leg = self.motor_to_leg[i]
            leg_amp[leg] += np.clip(joint_acts[i], 0.0, 0.2)
        leg_amp = leg_amp / max(leg_amp.max(), 1e-6)

        # 4. Build position targets for each leg
        for leg_i, (leg_name, leg_label) in enumerate(self.LEG_MAP.items()):
            leg_phase = self.phase * self.TRIPOD[leg_i]
            amp_scale = np.clip(0.3 + 0.7 * leg_amp[leg_i], 0.2, 1.5)

            dofs = self.fly.leg_dofs.get(leg_label, [])
            for j in range(min(self.N_DOFS_PER_LEG, len(dofs))):
                lo, hi = self.DOF_RANGES[j]
                center = (lo + hi) / 2.0
                rng = (hi - lo) / 2.0
                dof_phase = leg_phase + j * np.pi / self.N_DOFS_PER_LEG
                angle = center + self.AMP_PER_DOF[j] * amp_scale * rng * np.sin(dof_phase)
                self.fly.data.qpos[dofs[j]] = np.clip(angle, lo * 1.1, hi * 1.1)

        # 5. Advance physics
        mu.mj_step(self.fly.model, self.fly.data)

        return {
            'pos': self.fly.data.qpos[:3].copy(),
            'coherence': coherence,
            'speed_mod': speed_mod,
            'leg_amp': leg_amp.copy(),
        }


# ═══════════════════════════════════════════════════════════════════
#  MAIN — запись видео ходьбы
# ═══════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("🦗🧬 DEPTHWAVE WALK — ψ-wave VNC → position servo → MuJoCo")
    print("═" * 80)

    # ─── 1. Load VNC connectome ────────────────────────────────────
    print("\n📥 Loading VNC connectome...")
    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons() or not conn.load_connections():
        return
    N = conn.n_neurons
    print(f"   ✅ {N:,} neurons, {conn.n_synapses:,} synapses")

    # ─── 2. Assign topological depth ───────────────────────────────
    print("\n📏 Assigning topological depth...")
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
    print(f"   depth: [{depth.min():.2f}, {depth.max():.2f}]")

    # ─── 3. DepthWave VNC ──────────────────────────────────────────
    print("\n🧬 Initializing DepthWave VNC...")
    dw = DepthWaveVNC(
        conn.row_ptr, conn.col_indices, conn.values,
        conn.nt_types, N,
        depth=depth, mod_range=2.0, dt=0.5,
    )

    # ─── 4. MuJoCo fly ─────────────────────────────────────────────
    print("\n🦗 Initializing MuJoCo fly...")
    fly = AtlasovFlyMuJoCo(SCENE_XML)

    # ⚡ Override Atlasov step for position control
    walker = DepthPositionFly(dw, fly, base_omega=6.0)
    print(f"   ✅ {fly.n_joints} joints, {walker.n_motor} motor neurons")

    # ─── 5. Walk + Record ──────────────────────────────────────────
    n_steps = 2000  # 1s at dt=0.5ms
    dt_body = 5e-4

    print(f"\n🎬 Walking {n_steps} steps ({n_steps*dt_body:.2f}s)...")
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
    coherence_history = []

    t0 = time.time()
    for step in range(n_steps):
        # Simple oscillatory sensory input (body feedback)
        sensory = np.sin(2 * np.pi * 5 * step * dt_body) * np.ones(N, dtype=np.float32) * 0.05

        out = walker.step(sensory, dt_body)
        try:
            coherence_history.append(out['coherence'])
        except Exception:
            pass

        if start_pos is None:
            start_pos = out['pos'].copy()
        trajectory.append(out['pos'].copy())

        # Record frame every 5 steps
        if step % 5 == 0:
            if thorax_bid >= 0:
                cam.lookat = fly.data.xpos[thorax_bid].copy()
            renderer.update_scene(fly.data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))

        if step % 500 == 0:
            print(f"   Step {step:5d}/{n_steps} | "
                  f"coh={out['coherence']:.4f} | "
                  f"leg_amp=[{' '.join(f'{a:.2f}' for a in out['leg_amp'][:3])}]")

    elapsed = time.time() - t0

    # ─── 6. Results ────────────────────────────────────────────────
    displacement = np.linalg.norm(trajectory[-1] - start_pos) * 1000
    walk_time = n_steps * dt_body
    speed = displacement / walk_time if walk_time > 0 else 0.0
    mean_coh = np.mean(coherence_history) if coherence_history else 0.0

    def gait(s):
        if s < 1: return "STANDING ❌"
        if s < 5: return "CRAWLING 🐛"
        if s < 10: return "WALKING 🦗"
        if s < 25: return "FAST WALK 🦗✅"
        if s < 50: return "RUNNING 🏃"
        return "FLYING 🚀"

    print(f"\n{'='*80}")
    print("🏆 DEPTHWAVE WALK — RESULTS")
    print(f"{'='*80}")
    print(f"   🧬 DepthWave:      {N:,} ψ-neurons (VNC)")
    print(f"   🦗 Joints:         {fly.n_joints}")
    print(f"   🦵 Motor neurons:  {walker.n_motor}")
    print(f"   ⚙️ Base omega:     {walker.base_omega:.1f} rad/s")
    print(f"")
    print(f"   🚶 Displacement:   {displacement:.2f} mm")
    print(f"   ⏱ Walk time:      {walk_time:.2f} s")
    print(f"   📏 Speed:          {speed:.2f} mm/s ({speed/3:.1f} BL/s)")
    print(f"   🧠 Coherence:      {mean_coh:.4f}")
    print(f"   🏃 Gait:           {gait(speed)}")
    print(f"   🎬 Frames:         {len(frames)}")
    print(f"   ⏱ Compute:        {elapsed:.1f}s ({n_steps/elapsed:.0f} step/s)")

    # ─── 7. Save video ─────────────────────────────────────────────
    webp_path = Path(__file__).parent.parent / "depthwave_walk.webp"
    frames[0].save(webp_path, save_all=True, append_images=frames[1:],
                   duration=40, loop=0)
    print(f"\n   💾 Video: {webp_path}")

    # Comparison table
    print(f"\n{'='*80}")
    print("📊 COMPARISON: DepthWave vs HH+VNC v3")
    print(f"{'='*80}")
    print(f"   {'Metric':<25} {'DepthWave':<15} {'HH+VNC v3':<15}")
    print(f"   {'-'*25} {'-'*15} {'-'*15}")
    print(f"   {'Neurons':<25} {N:<15} {'25,514':<15}")
    print(f"   {'Drive type':<25} {'ψ-wave continuous':<15} {'Analog firing rate':<15}")
    print(f"   {'Speed (mm/s)':<25} {f'{speed:.0f}':<15} {'698':<15}")
    print(f"   {'Gait':<25} {gait(speed):<15} {'FLYING 🚀':<15}")
    print(f"   {'Smoothness':<25} {'Continuous ψ':<15} {'Filtered spikes':<15}")
    print(f"   {'Motor control':<25} {'Position servo':<15} {'Torque (Atlasov)':<15}")
    print(f"   {'Frames':<25} {len(frames):<15} {'400':<15}")
    print(f"{'='*80}")

    print(f"\n✅ DONE! Video: {webp_path}")


if __name__ == "__main__":
    main()
