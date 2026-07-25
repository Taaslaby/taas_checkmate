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
    actuator_indices = []
    for jname in leg_joint_names:
        jid = mu.mj_name2id(model, mu.mjtObj.mjOBJ_JOINT, jname)
        aid = mu.mj_name2id(model, mu.mjtObj.mjOBJ_ACTUATOR, jname)
        if jid >= 0:
            dof_indices.append(model.jnt_dofadr[jid])
        if aid >= 0:
            actuator_indices.append(aid)

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
    smooth_ctrl = np.zeros(min(n_joints, len(actuator_indices)), dtype=np.float32)

    cpg_phase = 0.0
    q_rest = np.array([model.qpos0[dof] for dof in dof_indices], dtype=np.float32)

    # --- SNN Initialization ---
    snn_sim = None
    snn_conn = None
    ext_I = np.zeros(128, dtype=np.float32)
    
    # Try loading the Biological MANC Connectome
    cpg_path = "/home/taaslab/Проекты/asm_engines/sardaanaCM/data/generated/manc_vnc.taas"
    
    # Store mapped arrays for fast access in physics loop
    muscle_groups = []
    sensory_group = []
    command_dn_group = []
    
    try:
        import sys
        sys.path.append("/home/taaslab/Проекты/asm_engines/sardaanaCM/src")
        from connectome_builder.dataset import load_connectome
        import taas_checkmate as tcm_brain
        
        # 1. Load from .taas
        sample = load_connectome(cpg_path)
        
        # 2. Build taas_checkmate ConnectomeData
        # We must scale the biological synapse counts so they cross the LIF threshold (1.0).
        # A biological weight of 10 synapses is log1p(10)=2.39. We scale it up so it strongly drives the network.
        snn_conn = tcm_brain.ConnectomeData(
            sample.row_ptr, 
            sample.col_indices, 
            sample.values * 50.0, # Increased synaptic efficacy
            sample.nt_types,
            [(k, v) for k, v in sample.groups.items()]
        )
        
        # 3. Initialize Hodgkin-Huxley (HH) Simulator on GPU
        snn_sim = tcm_brain.BrainSimulator(snn_conn.n_neurons, "hh", True)
        ext_I = np.zeros(snn_conn.n_neurons, dtype=np.float32)
        print(f"\n[INFO] Loaded MANC Connectome! {snn_conn.n_neurons} neurons, {snn_conn.n_synapses} synapses.")
        
        # 4. Map Muscle Groups to Atlasov Channels (0-23)
        ordered_muscle_groups = [
            'T1_L_femur_flx', 'T1_L_femur_ext',
            'T1_L_tibia_flx', 'T1_L_tibia_ext',
            'T1_R_femur_flx', 'T1_R_femur_ext',
            'T1_R_tibia_flx', 'T1_R_tibia_ext',
            'T2_L_femur_flx', 'T2_L_femur_ext',
            'T2_L_tibia_flx', 'T2_L_tibia_ext',
            'T2_R_femur_flx', 'T2_R_femur_ext',
            'T2_R_tibia_flx', 'T2_R_tibia_ext',
            'T3_L_femur_flx', 'T3_L_femur_ext',
            'T3_L_tibia_flx', 'T3_L_tibia_ext',
            'T3_R_femur_flx', 'T3_R_femur_ext',
            'T3_R_tibia_flx', 'T3_R_tibia_ext'
        ]
        
        for g_name in ordered_muscle_groups:
            idx_list = snn_conn.group(g_name)
            muscle_groups.append(idx_list if idx_list else [])
            
        command_dn_group = snn_conn.group("command_dns") or []
        sensory_group = snn_conn.group("sensory") or []
        
    except Exception as e:
        print(f"\n[WARN] Failed to load SNN: {e}. Falling back to math.sin CPG.")

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = 12.0
        v.cam.elevation = -15.0
        v.cam.azimuth = 60.0

        frames = 0
        while v.is_running():
            step_start = time.time()
            cpg_wave = 0.0

            # --- SENSORY PROPRIOCEPTIVE FEEDBACK ---
            avg_err = 0.0
            for j in range(min(n_joints, len(actuator_indices))):
                dof_idx = dof_indices[j]
                jname = leg_joint_names[j]
                bias = 0.25 if 'femur' in jname else (-0.15 if 'tibia' in jname else 0.1)
                ref_q = q_rest[j] + bias
                avg_err += abs(data.qpos[dof_idx] - ref_q)
            avg_err /= max(1, len(actuator_indices))

            # 1. Biological SNN Pattern Generation (Embodied Brain)
            # We step the SNN every 50 physics frames (50 * 0.2ms = 10ms of physics time).
            # We then step the SNN for 10ms (20 steps at dt=0.5).
            if frames % 50 == 0:
                ext_I.fill(0.0)
                
                # 1a. Command Neurons (DNs) get constant drive to initiate walking
                for idx in command_dn_group:
                    ext_I[idx] = 50.0
                
                # 1b. Sensory Feedback: Proprioceptors get current proportional to joint error
                # We map 12 joints evenly into the sensory group (if available)
                if sensory_group:
                    chunk = max(1, len(sensory_group) // 12)
                    for j in range(min(n_joints, 12)):
                        err = abs(data.qpos[dof_indices[j]] - q_rest[j]) * 10.0
                        for s_idx in sensory_group[j*chunk : (j+1)*chunk]:
                            ext_I[s_idx] = err
                    
                # 1c. Step SNN
                if snn_sim is not None and snn_conn is not None:
                    n_snn_steps = 20
                    spike_buf_flat = snn_sim.step_batch(snn_conn, n_steps=n_snn_steps, dt_ms=0.5, external_I=ext_I)
                    spike_buf = spike_buf_flat.reshape((n_snn_steps, snn_conn.n_neurons))
                    
                    # 1d. Read Motor Neuron Spikes
                    # For each of the 24 muscle channels, we do an OR over all biological MNs in that group.
                    mn_spikes_sum = np.sum(spike_buf, axis=0)
                    spikes.fill(0)
                    for m in range(min(n_muscles, len(muscle_groups))):
                        biological_mns = muscle_groups[m]
                        for mn_idx in biological_mns:
                            if mn_spikes_sum[mn_idx] > 0:
                                spikes[m] = 1
                                break # One biological spike is enough to trigger the muscle channel
                else:
                    # Fallback CPG if SNN fails to load
                    nominal_omega = 2.0 * math.pi * 3.0
                    omega_actual = nominal_omega * (1.0 / (1.0 + 3.0 * avg_err))
                    cpg_phase += omega_actual * dt * 50.0  # approximate update since we only enter here every 50 steps
                    
                    cpg_wave = math.sin(cpg_phase)
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
            
            frames += 1

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

            # 3. Apply Leg Joint Impedance & Drive into MuJoCo Actuators with Temporal Low-Pass Filtering
            for j in range(min(n_joints, len(actuator_indices))):
                act_idx = actuator_indices[j]
                dof_idx = dof_indices[j]
                q_dot = data.qvel[dof_idx]
                q = data.qpos[dof_idx]

                df = joint_df[j]
                b_dmp = joint_dmp[j]

                jname = leg_joint_names[j]
                bias = 0.25 if 'femur' in jname else (-0.15 if 'tibia' in jname else 0.1)
                ref_q = q_rest[j] + bias
                
                # Length-Tension Feedback
                gamma = 2.0
                length_multiplier = 1.0 / (1.0 + gamma * (q - ref_q)**2)

                raw_target = np.clip(bias + df * 0.15 * length_multiplier, -0.4, 0.5)
                smooth_ctrl[j] += (dt / 0.04) * (raw_target - smooth_ctrl[j])

                data.ctrl[act_idx] = smooth_ctrl[j]
                # EXPLICIT DAMPING REMOVED: Applying -b_dmp * q_dot explicitly at fly-scale masses 
                # (1e-9 kg) violates the Euler stability limit (b < 2M/dt) and causes QACC explosions!
                # We rely on MuJoCo's built-in implicit joint damping instead.

            # Enable claw adhesion strictly during stance phase for grounded tripod legs
            claw_map_a = [72, 75, 76] # T1_left, T2_right, T3_left
            claw_map_b = [73, 74, 77] # T1_right, T2_left, T3_right

            if cpg_wave > 0.0:
                for c in claw_map_a: data.ctrl[c] = 0.5
                for c in claw_map_b: data.ctrl[c] = 0.0
            else:
                for c in claw_map_b: data.ctrl[c] = 0.5
                for c in claw_map_a: data.ctrl[c] = 0.0

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
