#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — 3D Drosophila Walking Animation Renderer.

Renders 3D offscreen frames of the Drosophila walking under Atlasov Muscle Engine Tripod Gait
and outputs a high-resolution animated WebP/GIF video.

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import math
from pathlib import Path
import numpy as np
from PIL import Image

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import mujoco as mu

SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"
OUTPUT_WEBP = Path(__file__).parent.parent / "fly_walk_demo.webp"

def render_walking_animation(n_frames=120, fps=30):
    print("=" * 80)
    print("  Atlasov Neuromotor Engine — 3D Offscreen Animation Renderer")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 80)

    print(f"\n[1/3] Loading 3D Physical MuJoCo Body from: {SCENE_XML}")
    model = mu.MjModel.from_xml_path(str(SCENE_XML))
    data = mu.MjData(model)

    # Leg actuators
    leg_joint_names = [
        'femur_T1_left', 'tibia_T1_left',
        'femur_T1_right', 'tibia_T1_right',
        'femur_T2_left', 'tibia_T2_left',
        'femur_T2_right', 'tibia_T2_right',
        'femur_T3_left', 'tibia_T3_left',
        'femur_T3_right', 'tibia_T3_right',
    ]

    dof_indices = []
    for jname in leg_joint_names:
        jid = mu.mj_name2id(model, mu.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof_indices.append(model.jnt_dofadr[jid])

    n_joints = len(dof_indices)
    n_muscles = math.ceil((n_joints * 2) / 8) * 8

    # Muscle arrays
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    spikes = np.zeros(n_muscles, dtype=np.int32)
    resets = np.zeros(n_muscles, dtype=np.int32)

    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    tau_pump = np.full(n_muscles, 15.0, dtype=np.float32)
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    t_max = np.full(n_muscles, 0.5, dtype=np.float32)
    kd = np.full(n_muscles, 0.3, dtype=np.float32)

    joint_df = np.zeros(n_muscles // 2, dtype=np.float32)
    joint_stf = np.zeros(n_muscles // 2, dtype=np.float32)
    joint_dmp = np.zeros(n_muscles // 2, dtype=np.float32)

    k0 = np.full(n_muscles // 2, 0.05, dtype=np.float32)
    beta = np.full(n_muscles // 2, 0.2, dtype=np.float32)
    b0 = np.full(n_muscles // 2, 0.005, dtype=np.float32)
    alpha = np.full(n_muscles // 2, 0.02, dtype=np.float32)

    dt = model.opt.timestep

    try:
        import taas_checkmate
        step_atlasov_fn = getattr(taas_checkmate, "step_atlasov_neuromotor", None) or getattr(getattr(taas_checkmate, "neurons", None), "step_atlasov_neuromotor", None)
    except ImportError:
        step_atlasov_fn = None

    renderer = mu.Renderer(model, height=480, width=640)
    frames = []

    print(f"[2/3] Simulating and Rendering {n_frames} Frames of 3D Locomotion...")

    substeps_per_frame = int(1.0 / (fps * dt))

    for frame_idx in range(n_frames):
        for _ in range(substeps_per_frame):
            step = int(data.time / dt)
            # Smooth biological CPG sinusoidal wave (15 Hz stride rate)
            cpg_wave = math.sin(2.0 * math.pi * 15.0 * data.time)

            spikes.fill(0)
            if cpg_wave > 0.0:
                # Tripod A Flexors active, Tripod B Extensors active
                spikes[0:4:2] = 1; spikes[13:17:2] = 1; spikes[16:20:2] = 1
                spikes[5:9:2] = 1; spikes[9:13:2] = 1; spikes[21:25:2] = 1
            else:
                # Tripod B Flexors active, Tripod A Extensors active
                spikes[4:8:2] = 1; spikes[8:12:2] = 1; spikes[20:24:2] = 1
                spikes[1:5:2] = 1; spikes[12:16:2] = 1; spikes[17:21:2] = 1

            if step_atlasov_fn is not None:
                step_atlasov_fn(
                    ca, tension, spikes, resets, float(dt * 1000.0),
                    d_ca, tau_pump, eta_reset, t_max, kd,
                    joint_df, joint_stf, joint_dmp,
                    k0, beta, b0, alpha
                )
            else:
                for m in range(n_muscles):
                    if spikes[m] > 0: ca[m] += d_ca[m]
                    ca[m] *= math.exp(-dt * 1000.0 / tau_pump[m])
                    ca2 = ca[m]**2; kd2 = kd[m]**2
                    tension[m] = t_max[m] * (ca2 / (ca2 + kd2 + 1e-8))

                for j in range(n_joints):
                    flx, ext = 2 * j, 2 * j + 1
                    joint_df[j] = tension[ext] - tension[flx]
                    ca_sum = ca[flx] + ca[ext]
                    joint_stf[j] = k0[j] + beta[j] * ca_sum
                    joint_dmp[j] = b0[j] + alpha[j] * ca_sum

            data.qfrc_applied.fill(0.0)
            for j in range(min(n_joints, len(dof_indices))):
                dof = dof_indices[j]
                q = data.qpos[dof]
                q_dot = data.qvel[dof]

                df = joint_df[j]
                k_stf = joint_stf[j]
                b_dmp = joint_dmp[j]
                data.qfrc_applied[dof] = np.clip(df - k_stf * q - b_dmp * q_dot, -1.0, 1.0)

            mu.mj_step(model, data)

        renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(Image.fromarray(rgb))

        if (frame_idx + 1) % 30 == 0:
            print(f"      Rendered {frame_idx + 1}/{n_frames} frames (sim_t = {data.time*1000:.1f} ms)")

    print(f"[3/3] Saving Animated WebP to: {OUTPUT_WEBP}")
    frames[0].save(str(OUTPUT_WEBP), save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"\n✓ Animated WebP Created Successfully! ({OUTPUT_WEBP.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    render_walking_animation(n_frames=120, fps=30)
