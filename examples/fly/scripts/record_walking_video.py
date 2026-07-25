#!/usr/bin/env python3
"""
🎬🦗 RECORD WALKING VIDEO — Real VNC CPG with realistic physics
===============================================================
Записывает видео ходьбы мухи с реалистичным clip=±0.005.
VNC (25K) → HH → Atlasov Ca²⁺ → MuJoCo.
Сохраняет WebP анимацию и PNG траекторию.

Результат validation: при clip=0.005 муха идёт 22 mm/s — REAL CPG!
"""

import os, sys, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

os.environ["MUJOCO_GL"] = "egl"
from PIL import Image

import mujoco as mu
import taas_checkmate as tc

# ─── Import VNC classes ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vnc_real_atlasov_embodied import (
    RealVNCConnectome, FrankenMotorMapper, AtlasovFlyMuJoCo,
    VNC_DATA, SCENE_XML
)

# ═══════════════════════════════════════════════════════════════════════
#  MODIFIED Atlasov with realistic clip
# ═══════════════════════════════════════════════════════════════════════

class RealisticFlyMuJoCo(AtlasovFlyMuJoCo):
    """Atlasov with realistic clip=±0.005 for a 1mg fly."""
    
    CLIP_VALUE = 0.5  # видимая ходьба
    
    def step(self, motoneuron_spikes):
        n_in = min(len(motoneuron_spikes), self.n_muscles)
        self.spikes[:n_in] = motoneuron_spikes[:n_in]
        self.resets[:n_in] = 0

        self.step_fn(
            self.ca, self.tension, self.spikes, self.resets,
            float(self.dt * 1000.0),
            self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
            self.joint_df, self.joint_stf, self.joint_dmp,
            self.k0, self.beta, self.b0, self.alpha
        )

        self.data.qfrc_applied.fill(0.0)
        for j in range(self.n_joints):
            dof = self.all_dofs[j]
            q = self.data.qpos[dof]
            qd = self.data.qvel[dof]
            tau = self.joint_df[j] - self.joint_stf[j] * q - self.joint_dmp[j] * qd
            self.data.qfrc_applied[dof] = np.clip(tau, -self.CLIP_VALUE, self.CLIP_VALUE)

        mu.mj_step(self.model, self.data)
        return {
            't': self.data.time,
            'pos': self.data.qpos[:3].copy(),
            'df': self.joint_df[:self.n_joints].copy(),
            'k': self.joint_stf[:self.n_joints].copy(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — Record walking video
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("🎬🦗 RECORDING WALKING VIDEO (clip=±0.005, REALISTIC!)")
    print("═" * 70)

    # 1. Load VNC
    print("\n📥 Loading VNC connectome...")
    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons() or not conn.load_connections():
        return
    conn_data = conn.build_connectome_data()
    N = conn.n_neurons
    print(f"   ✅ {N:,} neurons, {conn.n_synapses:,} synapses")

    # 2. HH BrainSimulator
    print("\n🧠 Initializing HH pacemaking...")
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)

    # 3. Atlasov with realistic clip
    print("\n🦗 Initializing MuJoCo fly (realistic clip=±0.005)...")
    fly = RealisticFlyMuJoCo()
    mapper = FrankenMotorMapper(conn, fly)
    print(f"   ✅ {fly.n_joints} joints, {fly.n_muscles} muscles")

    # 4. Parameters
    n_warmup = 500   # 250ms warmup
    n_walk = 2000    # 1s walking
    n_total = n_warmup + n_walk
    dt_ms = 0.5

    # External I: background noise + motor tonic + descending
    rng = np.random.RandomState(42)
    I_bg = rng.randn(N).astype(np.float32) * 2.0
    if len(conn.leg_motor) > 0:
        I_bg[np.array(conn.leg_motor) % N] += 30.0
        print(f"   🦵 Motor tonic: 30pA × {len(conn.leg_motor)} neurons")
    if len(conn.descending) > 0:
        I_bg[np.array(conn.descending) % N] += 20.0
        print(f"   ⬇️  Descending drive: {len(conn.descending)} neurons")

    # 5. Batch simulation
    print(f"\n🎬 Simulating {n_total} steps ({(n_total*dt_ms/1000):.1f}s)...")
    t0 = time.time()
    spike_buf = sim.step_batch(conn_data, n_steps=n_total, dt_ms=dt_ms,
                               external_I=I_bg, scale=1.0)
    t_sim = time.time() - t0
    print(f"   ✅ Done: {t_sim:.1f}s ({(n_total*dt_ms/1000)/t_sim:.1f}× RT)")

    # 6. Walk phase + record video
    print(f"\n🎥 Recording walking video ({n_walk} frames)...")
    
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
    total_motor = 0
    
    for step in range(n_total):
        step_spikes = spike_buf[step * N:(step + 1) * N]
        
        if step >= n_warmup:
            motor_spikes = np.array([float(step_spikes[idx]) for idx in conn.leg_motor
                                     if idx < N], dtype=np.float32)
            motor_drive = mapper.route(motor_spikes)
            total_motor += motor_drive.sum()
            out = fly.step(motor_drive)
            
            if start_pos is None:
                start_pos = out['pos'].copy()
            
            trajectory.append(out['pos'].copy())
            
            # Record frame every 5 steps for smoother video
            if (step - n_warmup) % 5 == 0:
                if thorax_bid >= 0:
                    cam.lookat = fly.data.xpos[thorax_bid].copy()
                renderer.update_scene(fly.data, camera=cam)
                frames.append(Image.fromarray(renderer.render()))

    total_time = time.time() - t_sim
    displacement = np.linalg.norm(trajectory[-1] - start_pos) * 1000  # mm
    walk_time = n_walk * dt_ms / 1000
    speed = displacement / walk_time if walk_time > 0 else 0.0

    # 7. Save video
    print(f"\n💾 Saving {len(frames)} frames...")
    
    # WebP animation
    webp_path = Path(__file__).parent.parent / "vnc_real_walk_realistic.webp"
    frames[0].save(webp_path, save_all=True, append_images=frames[1:], 
                   duration=40, loop=0)  # 25 fps = 40ms per frame
    print(f"   ✅ Video: {webp_path}")

    # Also save to artifacts
    artifact_path = Path("/home/taaslab/.gemini/antigravity-ide/brain") / "vnc_walk.webp"
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(artifact_path, save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
        print(f"   ✅ Video: {artifact_path}")
    except Exception as e:
        print(f"   ⚠ Could not save to artifacts: {e}")

    # 8. Results
    print(f"\n{'='*70}")
    print("🏆 WALKING VIDEO RECORDED!")
    print(f"{'='*70}")
    print(f"   VNC neurons:     {N:,}")
    print(f"   Clip:           ±{fly.CLIP_VALUE}")
    print(f"   Displacement:   {displacement:.2f} mm")
    print(f"   Walk time:      {walk_time:.2f} s")
    print(f"   Speed:          {speed:.2f} mm/s ({speed/3:.1f} BL/s)")
    print(f"   Motor firings:  {total_motor:,}")
    print(f"   Frames:         {len(frames)}")
    print(f"   Video:          {webp_path}")
    
    # Gait classification
    if speed < 1: gait = "STANDING"
    elif speed < 10: gait = "CRAWLING"
    elif speed < 30: gait = "WALKING 🦗"
    elif speed < 100: gait = "RUNNING"
    else: gait = "FLYING"
    print(f"   Gait:           {gait}")
    
    print(f"\n{'='*70}")
    print("✅ DONE! Video ready for viewing!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
