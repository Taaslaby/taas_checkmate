import sys, os, time, math
import mujoco
import numpy as np
from pathlib import Path

# Add SardaanaCM to path for connectome_builder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "sardaanaCM", "src"))
from connectome_builder.dataset import load_connectome
import taas_checkmate as tcm_brain

print("Loading MuJoCo Model...")
xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
model.opt.timestep = 0.0002
data = mujoco.MjData(model)

print("Loading Atlasov Engine...")
import taas_checkmate
step_atlasov_fn = getattr(taas_checkmate, "step_atlasov_neuromotor", None) or getattr(getattr(taas_checkmate, "neurons", None), "step_atlasov_neuromotor", None)

leg_joint_names = [
    'femur_T1_L', 'tibia_T1_L', 'femur_T1_R', 'tibia_T1_R',
    'femur_T2_L', 'tibia_T2_L', 'femur_T2_R', 'tibia_T2_R',
    'femur_T3_L', 'tibia_T3_L', 'femur_T3_R', 'tibia_T3_R'
]
actuator_indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in leg_joint_names]
dof_indices = [model.actuator_trnid[act_idx, 0] for act_idx in actuator_indices]

print("Loading MANC Connectome...")
cpg_path = "/home/taaslab/Проекты/asm_engines/sardaanaCM/data/generated/manc_vnc.taas"
sample = load_connectome(cpg_path)

command_dn_group = sample.groups.get("command_dns", [])
sensory_group = sample.groups.get("sensory", [])

ordered_muscle_groups = [
    'T1_L_femur_flx', 'T1_L_femur_ext', 'T1_L_tibia_flx', 'T1_L_tibia_ext',
    'T1_R_femur_flx', 'T1_R_femur_ext', 'T1_R_tibia_flx', 'T1_R_tibia_ext',
    'T2_L_femur_flx', 'T2_L_femur_ext', 'T2_L_tibia_flx', 'T2_L_tibia_ext',
    'T2_R_femur_flx', 'T2_R_femur_ext', 'T2_R_tibia_flx', 'T2_R_tibia_ext',
    'T3_L_femur_flx', 'T3_L_femur_ext', 'T3_L_tibia_flx', 'T3_L_tibia_ext',
    'T3_R_femur_flx', 'T3_R_femur_ext', 'T3_R_tibia_flx', 'T3_R_tibia_ext'
]
muscle_groups = [sample.groups.get(g, []) for g in ordered_muscle_groups]

def evaluate_gait(params):
    E_scale, I_scale, DN_drive = params
    print(f"Evaluating: E_scale={E_scale:.3f}, I_scale={I_scale:.3f}, DN_drive={DN_drive:.3f}")
    
    # Scale values: excitatory vs inhibitory based on PRE-SYNAPTIC neuron
    new_values = sample.values.copy()
    n_neurons = len(sample.row_ptr) - 1
    for row in range(n_neurons):
        is_inhib = (sample.nt_types[row] < 0)
        start = sample.row_ptr[row]
        end = sample.row_ptr[row+1]
        for i in range(start, end):
            if is_inhib:
                new_values[i] *= I_scale
            else:
                new_values[i] *= E_scale
            
    snn_conn = tcm_brain.ConnectomeData(
        sample.row_ptr, 
        sample.col_indices, 
        new_values, 
        sample.nt_types,
        [(k, v) for k, v in sample.groups.items()]
    )
    snn_sim = tcm_brain.BrainSimulator(snn_conn.n_neurons, "hh", True)
    ext_I = np.zeros(snn_conn.n_neurons, dtype=np.float32)
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_step(model, data)
    q_rest = [data.qpos[dof] for dof in dof_indices]
    
    joint_df = np.zeros(24, dtype=np.float32)
    joint_dmp = np.zeros(24, dtype=np.float32)
    spikes = np.zeros(24, dtype=np.int32)
    smooth_ctrl = np.zeros(len(actuator_indices), dtype=np.float64)
    
    n_physics_steps = int(2.0 / 0.0002) # 2 seconds of simulation
    
    total_spikes = 0
    start_x = data.qpos[0]
    
    for frames in range(n_physics_steps):
        if frames % 50 == 0:
            ext_I.fill(0.0)
            for idx in command_dn_group:
                ext_I[idx] = DN_drive
                
            if sensory_group:
                chunk = max(1, len(sensory_group) // 12)
                for j in range(min(12, len(dof_indices))):
                    err = abs(data.qpos[dof_indices[j]] - q_rest[j]) * 10.0
                    for s_idx in sensory_group[j*chunk : (j+1)*chunk]:
                        ext_I[s_idx] = err
                        
            n_snn_steps = 20
            spike_buf_flat = snn_sim.step_batch(snn_conn, n_steps=n_snn_steps, dt_ms=0.5, external_I=ext_I)
            spike_buf = spike_buf_flat.reshape((n_snn_steps, snn_conn.n_neurons))
            
            mn_spikes_sum = np.sum(spike_buf, axis=0)
            total_spikes += np.sum(mn_spikes_sum)
            
            spikes.fill(0)
            for m in range(24):
                for mn_idx in muscle_groups[m]:
                    if mn_spikes_sum[mn_idx] > 0:
                        spikes[m] = 1
                        break
        
        if step_atlasov_fn is not None:
            step_atlasov_fn(spikes, joint_df, joint_dmp, 24)
            
        for j in range(len(actuator_indices)):
            act_idx = actuator_indices[j]
            dof_idx = dof_indices[j]
            q = data.qpos[dof_idx]
            
            df = joint_df[j]
            jname = leg_joint_names[j]
            bias = 0.25 if 'femur' in jname else (-0.15 if 'tibia' in jname else 0.1)
            ref_q = q_rest[j] + bias
            
            gamma = 2.0
            length_multiplier = 1.0 / (1.0 + gamma * (q - ref_q)**2)
            raw_target = np.clip(bias + df * 0.15 * length_multiplier, -0.4, 0.5)
            smooth_ctrl[j] += (0.0002 / 0.04) * (raw_target - smooth_ctrl[j])
            data.ctrl[act_idx] = smooth_ctrl[j]
                
        mujoco.mj_step(model, data)
        
    end_x = data.qpos[0]
    dist_x = end_x - start_x
    
    seizure_penalty = max(0, total_spikes - 4_000_000) * 1e-6
    loss = -dist_x + seizure_penalty * 10.0
    print(f"  -> Dist X: {dist_x:.3f}, Spikes: {total_spikes}, Loss: {loss:.3f}")
    return loss

best_loss = float('inf')
best_params = None

grid = [
    (10.0, 50.0, 20.0),
    (5.0, 50.0, 20.0),
    (5.0, 25.0, 20.0),
    (2.0, 20.0, 10.0),
    (1.0, 10.0, 10.0),
    (0.5, 10.0, 10.0),
    (0.1, 5.0, 5.0)
]

for params in grid:
    loss = evaluate_gait(params)
    if loss < best_loss:
        best_loss = loss
        best_params = params

print(f"\nBest Params: E_scale={best_params[0]}, I_scale={best_params[1]}, DN_drive={best_params[2]}")
print(f"Best Loss: {best_loss}")
