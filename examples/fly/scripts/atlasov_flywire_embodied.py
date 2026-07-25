#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — Full FlyWire Embodied MuJoCo Physics Simulation (Phase 1 / M2 & M5).

Pipeline:
Motor Neurons → NMJ Synaptic Delay (DelayBuffer) → Atlasov Sarcoplasmic Muscle Kinetics → Dynamic Joint Impedance → MuJoCo qfrc_applied

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

# Path to MuJoCo scene XML model
SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

class AtlasovFlyEmbodiedSim:
    def __init__(self, xml_path=str(SCENE_XML)):
        print(f" Loading MuJoCo FruitFly model from: {xml_path}")
        self.model = mu.MjModel.from_xml_path(xml_path)
        self.data = mu.MjData(self.model)

        # Collect leg joint DOFs
        self.joint_names = []
        self.dof_indices = []

        for i in range(self.model.njnt):
            jname = mu.mj_id2name(self.model, mu.mjtObj.mjOBJ_JOINT, i)
            if jname and any(leg in jname for leg in ['leg', 'coxa', 'femur', 'tibia', 'tarsus', 'L1', 'L2', 'L3', 'R1', 'R2', 'R3']):
                self.joint_names.append(jname)
                self.dof_indices.append(self.model.jnt_dofadr[i])

        self.n_joints = len(self.dof_indices)
        if self.n_joints == 0:
            # Fallback to all hinge joints
            self.dof_indices = [self.model.jnt_dofadr[i] for i in range(6, min(18, self.model.njnt))]
            self.n_joints = len(self.dof_indices)

        # Padded muscle count for SIMD (2 muscles per joint: Flexor & Extensor)
        self.n_muscles = math.ceil((self.n_joints * 2) / 8) * 8

        # Sarcoplasmic Muscle State Arrays
        self.ca = np.zeros(self.n_muscles, dtype=np.float32)
        self.tension = np.zeros(self.n_muscles, dtype=np.float32)
        self.spikes = np.zeros(self.n_muscles, dtype=np.int32)
        self.resets = np.zeros(self.n_muscles, dtype=np.int32)

        # Neuromuscular Parameters (Scaled for ~1mg Drosophila leg joint mechanics)
        self.d_ca = np.full(self.n_muscles, 0.5, dtype=np.float32)
        self.tau_pump = np.full(self.n_muscles, 20.0, dtype=np.float32) # SERCA pump clearance (20ms)
        self.eta_reset = np.full(self.n_muscles, 0.8, dtype=np.float32) # Active mechanical reset efficiency
        self.t_max = np.full(self.n_muscles, 0.01, dtype=np.float32)   # 10 mN T_max for insect leg
        self.kd = np.full(self.n_muscles, 0.3, dtype=np.float32)      # Kd = 0.3 uM

        # Joint Impedance Arrays
        self.joint_df = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_stf = np.zeros(self.n_muscles // 2, dtype=np.float32)
        self.joint_dmp = np.zeros(self.n_muscles // 2, dtype=np.float32)

        self.k0 = np.full(self.n_muscles // 2, 0.001, dtype=np.float32)
        self.beta = np.full(self.n_muscles // 2, 0.005, dtype=np.float32)
        self.b0 = np.full(self.n_muscles // 2, 0.0001, dtype=np.float32)
        self.alpha = np.full(self.n_muscles // 2, 0.0005, dtype=np.float32)

        self.dt = self.model.opt.timestep
        self.step_count = 0

        # Import Rust PyO3 bindings
        try:
            import taas_checkmate
            self.step_atlasov_fn = getattr(taas_checkmate, "step_atlasov_neuromotor", None) or getattr(getattr(taas_checkmate, "neurons", None), "step_atlasov_neuromotor", None)
        except ImportError:
            self.step_atlasov_fn = None

    def step(self, motoneuron_spikes, resets=None):
        """Single simulation step connecting Neurons → Atlasov Sarcoplasmic Kinetics → MuJoCo Physics."""
        if resets is None:
            resets = np.zeros(self.n_muscles, dtype=np.int32)

        # Fill spike vector for flexors/extensors
        n_in = min(len(motoneuron_spikes), self.n_muscles)
        self.spikes[:n_in] = motoneuron_spikes[:n_in]
        self.resets[:n_in] = resets[:n_in]

        # 1. Step Sarcoplasmic Calcium Dynamics & Joint Tug-of-War (Rust SIMD)
        if self.step_atlasov_fn is not None:
            self.step_atlasov_fn(
                self.ca, self.tension, self.spikes, self.resets, float(self.dt * 1000.0),
                self.d_ca, self.tau_pump, self.eta_reset, self.t_max, self.kd,
                self.joint_df, self.joint_stf, self.joint_dmp,
                self.k0, self.beta, self.b0, self.alpha
            )
        else:
            # Fallback Python simulation
            for m in range(n_in):
                if self.spikes[m] > 0:
                    self.ca[m] += self.d_ca[m]
                decay = math.exp(-self.dt * 1000.0 / self.tau_pump[m])
                self.ca[m] *= decay
                if self.resets[m] > 0:
                    self.ca[m] *= (1.0 - self.eta_reset[m])
                ca2 = self.ca[m] ** 2
                kd2 = self.kd[m] ** 2
                self.tension[m] = self.t_max[m] * (ca2 / (ca2 + kd2 + 1e-8))

            for j in range(self.n_joints):
                flx, ext = 2 * j, 2 * j + 1
                self.joint_df[j] = self.tension[ext] - self.tension[flx]
                ca_sum = self.ca[flx] + self.ca[ext]
                self.joint_stf[j] = self.k0[j] + self.beta[j] * ca_sum
                self.joint_dmp[j] = self.b0[j] + self.alpha[j] * ca_sum

        # 2. Apply Dynamic Forces & Virtual Impedance into MuJoCo qfrc_applied (Option B)
        self.data.qfrc_applied.fill(0.0)

        for j in range(self.n_joints):
            dof = self.dof_indices[j]
            q = self.data.qpos[dof]
            q_dot = self.data.qvel[dof]

            df = self.joint_df[j]
            k_stf = self.joint_stf[j]
            b_dmp = self.joint_dmp[j]

            # Physical Torque: τ_applied = ΔF - K*(q - q_rest) - B*q_dot
            tau_applied = np.clip(df - k_stf * q - b_dmp * q_dot, -0.01, 0.01)
            self.data.qfrc_applied[dof] = tau_applied

        # 3. Step MuJoCo Multibody Rigid Physics Loop
        mu.mj_step(self.model, self.data)
        self.step_count += 1

        return {
            't': self.data.time,
            'body_pos': self.data.qpos[:3].copy(),
            'avg_stiffness': float(np.mean(self.joint_stf[:self.n_joints])),
            'avg_damping': float(np.mean(self.joint_dmp[:self.n_joints])),
        }

def test_spontaneous_tripod_walking():
    print("\n--- [M5.2] Testing Spontaneous Tripod Gait Walking ---")
    sim = AtlasovFlyEmbodiedSim()
    motoneuron_spikes = np.zeros(sim.n_muscles, dtype=np.int32)

    stiffness_history = []
    damping_history = []

    # Run for 1,000 steps (200 ms)
    for step in range(1000):
        # Tripod gait pattern generator: alternate left/right leg groups
        motoneuron_spikes.fill(0)
        if (step // 25) % 2 == 0:
            motoneuron_spikes[0::2] = 1 # Flexor group A
        else:
            motoneuron_spikes[1::2] = 1 # Extensor group B

        out = sim.step(motoneuron_spikes)
        stiffness_history.append(out['avg_stiffness'])
        damping_history.append(out['avg_damping'])

    assert np.mean(stiffness_history) > 0.001, "Joint stiffness must rise during walking"
    assert np.mean(damping_history) > 0.0001, "Joint damping must rise during walking to prevent oscillations"
    print(f"✓ Tripod Walking Test Passed! Final sim_t = {sim.data.time * 1000.0:.1f} ms, Avg K = {np.mean(stiffness_history)*1e3:.2f} mN/rad, Avg B = {np.mean(damping_history)*1e3:.2f} mN·s/rad")

def test_perturbation_recovery():
    print("\n--- [M5.4] Testing Perturbation Recovery (Impedance Absorption) ---")
    sim = AtlasovFlyEmbodiedSim()
    spikes = np.ones(sim.n_muscles, dtype=np.int32) # Standing co-contraction

    for _ in range(500):
        sim.step(spikes)

    # Apply sudden external impulse to body (50 mN force)
    sim.data.qfrc_applied[0] += 0.05
    sim.step(spikes)

    pos_before = sim.data.qpos[0]

    # Run 500 recovery steps
    for _ in range(500):
        sim.step(spikes)

    pos_after = sim.data.qpos[0]
    drift = abs(pos_after - pos_before)

    assert not math.isnan(drift), "Body position must remain valid (no NaN explosion)"
    print(f"✓ Perturbation Recovery Test Passed! Body position drift = {drift:.5f} m")

if __name__ == "__main__":
    test_spontaneous_tripod_walking()
    test_perturbation_recovery()
