#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — Real-time 3D GUI Leg Locomotion & Tripod Gait Visualizer.

Drives the 6 main leg kinematic chains of Drosophila melanogaster (T1, T2, T3 Coxa/Femur/Tibia pairs)
using alternating Tripod Gait Groups (Tripod A vs Tripod B) and Atlasov Impedance Control.

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import time
import math
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import mujoco as mu
import mujoco.viewer as viewer

SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

def run_gui_viewer():
    print("=" * 80)
    print("  Atlasov Neuromotor Engine — 3D Drosophila Tripod Gait Visualizer")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 80)

    print(f"\n[1/3] Loading 3D Physical MuJoCo Body from: {SCENE_XML}")
    model = mu.MjModel.from_xml_path(str(SCENE_XML))
    data = mu.MjData(model)

    # Key Leg Joints for Locomotion (Femur & Tibia extension/flexion for each leg)
    leg_joint_names = [
        # Front Legs (T1)
        'femur_T1_left', 'tibia_T1_left',
        'femur_T1_right', 'tibia_T1_right',
        # Mid Legs (T2)
        'femur_T2_left', 'tibia_T2_left',
        'femur_T2_right', 'tibia_T2_right',
        # Hind Legs (T3)
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

    print(f"[2/3] Mapping {n_joints} Main Leg Actuators to {n_muscles} Atlasov Muscle Channels...")

    # Muscle State Arrays
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    spikes = np.zeros(n_muscles, dtype=np.int32)
    resets = np.zeros(n_muscles, dtype=np.int32)

    # Muscle Kinetics Parameters (Scaled for ~1mg Drosophila leg joints)
    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    tau_pump = np.full(n_muscles, 15.0, dtype=np.float32)
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    t_max = np.full(n_muscles, 0.5, dtype=np.float32) # Torque scale (mN·mm)
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

    print(f"[3/3] Launching Native 3D OpenGL GUI Window...")
    print("\n   Controls:")
    print("   • Left Click + Drag : Rotate Camera")
    print("   • Right Click + Drag: Translate Camera")
    print("   • Scroll             : Zoom In/Out")
    print("   • Spacebar           : Pause / Resume Physics\n")

    step = 0

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = 12.0
        v.cam.elevation = -15.0
        v.cam.azimuth = 60.0

        while v.is_running():
            step_start = time.time()

            # 1. Biological Tripod Gait Pattern Generation (Smooth 15 Hz CPG Sinusoidal Drive)
            cpg_wave = math.sin(2.0 * math.pi * 15.0 * data.time)
            spikes.fill(0)

            if cpg_wave > 0.0:
                # Tripod A Flexors active, Tripod B Extensors active
                spikes[0:4:2] = 1   # T1_left flexors
                spikes[13:17:2] = 1 # T2_right flexors
                spikes[16:20:2] = 1 # T3_left flexors

                spikes[5:9:2] = 1   # T1_right extensors
                spikes[9:13:2] = 1  # T2_left extensors
                spikes[21:25:2] = 1 # T3_right extensors
            else:
                # Tripod B Flexors active, Tripod A Extensors active
                spikes[4:8:2] = 1   # T1_right flexors
                spikes[8:12:2] = 1  # T2_left flexors
                spikes[20:24:2] = 1 # T3_right flexors

                spikes[1:5:2] = 1   # T1_left extensors
                spikes[12:16:2] = 1 # T2_right extensors
                spikes[17:21:2] = 1 # T3_left extensors

            # 2. Atlasov Sarcoplasmic Neuromotor Step
            if step_atlasov_fn is not None:
                step_atlasov_fn(
                    ca, tension, spikes, resets, float(dt * 1000.0),
                    d_ca, tau_pump, eta_reset, t_max, kd,
                    joint_df, joint_stf, joint_dmp,
                    k0, beta, b0, alpha
                )
            else:
                for m in range(n_muscles):
                    if spikes[m] > 0:
                        ca[m] += d_ca[m]
                    ca[m] *= math.exp(-dt * 1000.0 / tau_pump[m])
                    ca2 = ca[m]**2
                    kd2 = kd[m]**2
                    tension[m] = t_max[m] * (ca2 / (ca2 + kd2 + 1e-8))

                for j in range(n_joints):
                    flx, ext = 2 * j, 2 * j + 1
                    joint_df[j] = tension[ext] - tension[flx]
                    ca_sum = ca[flx] + ca[ext]
                    joint_stf[j] = k0[j] + beta[j] * ca_sum
                    joint_dmp[j] = b0[j] + alpha[j] * ca_sum

            # 3. Apply Leg Joint Impedance Forces into MuJoCo
            data.qfrc_applied.fill(0.0)
            for j in range(min(n_joints, len(dof_indices))):
                dof = dof_indices[j]
                q = data.qpos[dof]
                q_dot = data.qvel[dof]

                df = joint_df[j]
                k_stf = joint_stf[j]
                b_dmp = joint_dmp[j]

                tau_applied = np.clip(df - k_stf * q - b_dmp * q_dot, -1.0, 1.0)
                data.qfrc_applied[dof] = tau_applied

            # 4. Step MuJoCo 3D Rigid Physics
            mu.mj_step(model, data)
            v.sync()

            step += 1

            # Sleep to sync with ~60 FPS real-time speed
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

if __name__ == "__main__":
    run_gui_viewer()
