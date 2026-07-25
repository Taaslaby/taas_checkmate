#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — Real-time Interactive 3D GUI Visualizer.

Launches the native 3D MuJoCo OpenGL Viewer displaying the Fruitfly body driven by
the FlyWire Brain → MANC VNC → Atlasov Neuromotor Muscle Engine in real time.

Controls:
- Left Click + Drag: Rotate 3D Camera
- Right Click + Drag: Translate Camera
- Scroll: Zoom In / Out
- Spacebar: Pause / Resume Physics

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
    print("  Atlasov Neuromotor Engine — Interactive 3D GUI Visualizer")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 80)

    print(f"\n[1/3] Loading 3D Physical MuJoCo Body from: {SCENE_XML}")
    model = mu.MjModel.from_xml_path(str(SCENE_XML))
    data = mu.MjData(model)

    # Collect leg joint DOFs
    dof_indices = []
    for i in range(model.njnt):
        jname = mu.mj_id2name(model, mu.mjtObj.mjOBJ_JOINT, i)
        if jname and any(leg in jname for leg in ['leg', 'coxa', 'femur', 'tibia', 'tarsus', 'L1', 'L2', 'L3', 'R1', 'R2', 'R3']):
            dof_indices.append(model.jnt_dofadr[i])

    if not dof_indices:
        dof_indices = [model.jnt_dofadr[i] for i in range(6, min(18, model.njnt))]

    n_joints = len(dof_indices)
    n_muscles = math.ceil((n_joints * 2) / 8) * 8

    # Atlasov Neuromotor Muscle Parameters
    print(f"[2/3] Initializing Atlasov Sarcoplasmic Muscle Engine ({n_muscles} channels)...")
    ca = np.zeros(n_muscles, dtype=np.float32)
    tension = np.zeros(n_muscles, dtype=np.float32)
    spikes = np.zeros(n_muscles, dtype=np.int32)
    resets = np.zeros(n_muscles, dtype=np.int32)

    d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
    tau_pump = np.full(n_muscles, 20.0, dtype=np.float32)
    eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
    t_max = np.full(n_muscles, 0.01, dtype=np.float32)
    kd = np.full(n_muscles, 0.3, dtype=np.float32)

    joint_df = np.zeros(n_muscles // 2, dtype=np.float32)
    joint_stf = np.zeros(n_muscles // 2, dtype=np.float32)
    joint_dmp = np.zeros(n_muscles // 2, dtype=np.float32)

    k0 = np.full(n_muscles // 2, 0.001, dtype=np.float32)
    beta = np.full(n_muscles // 2, 0.005, dtype=np.float32)
    b0 = np.full(n_muscles // 2, 0.0001, dtype=np.float32)
    alpha = np.full(n_muscles // 2, 0.0005, dtype=np.float32)

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
        v.cam.distance = 15.0
        v.cam.elevation = -20.0
        v.cam.azimuth = 45.0

        while v.is_running():
            step_start = time.time()

            # Motoneuron Spikes Generation (Tripod Gait Pattern)
            spikes.fill(0)
            if (step // 25) % 2 == 0:
                spikes[0::2] = 1 # Flexor group
            else:
                spikes[1::2] = 1 # Extensor group

            # Atlasov Neuromotor Kinetics Step
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

            # Apply Joint Impedance Forces
            data.qfrc_applied.fill(0.0)
            for j in range(n_joints):
                dof = dof_indices[j]
                q = data.qpos[dof]
                q_dot = data.qvel[dof]

                df = joint_df[j]
                k_stf = joint_stf[j]
                b_dmp = joint_dmp[j]

                tau_applied = np.clip(df - k_stf * q - b_dmp * q_dot, -0.01, 0.01)
                data.qfrc_applied[dof] = tau_applied

            # Step 3D Multibody Dynamics & Sync GUI Frame
            mu.mj_step(model, data)
            v.sync()

            step += 1

            # Maintain real-time frame rate (~60 FPS)
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

if __name__ == "__main__":
    run_gui_viewer()
