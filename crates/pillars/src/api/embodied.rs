//! FlyWire Embodied Neuromotor Engine — Phase 1 Pipeline
//!
//! Connects Motor Neuron Spikes → DelayBuffer (NMJ delay) → Neuromotor Engine → Physical Joint Drive (ΔF, K_joint, B_joint).
//!
//! Author / Concept: Anton Atlasov (Атласов Антон)
//!
//! Перенесено из `core::neuron::embodied`— композиция
//! `api::neurons::{DelayBuffer, neuromotor::NeuromotorState}`.

use serde::{Deserialize, Serialize};

use crate::api::neurons::neuromotor::NeuromotorState;
use crate::api::neurons::DelayBuffer;

/// Joint mapping connecting a physical DOF to an Neuromotor agonist-antagonist muscle pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointMapping {
    pub dof_idx: usize,
    pub flexor_muscle_idx: usize,
    pub extensor_muscle_idx: usize,
    pub q_rest: f32,
    pub label: String,
}

/// Kinematic pose state of the embodied fly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlyPose {
    pub body_height: f32,
    pub body_pitch: f32,
    pub body_roll: f32,
    pub joint_angles: Vec<f32>,
}

/// Telemetry output of a single embodied simulation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimStepOutput {
    pub t: f32,
    pub pose: FlyPose,
    pub active_muscles: usize,
    pub avg_joint_stiffness: f32,
    pub avg_joint_damping: f32,
}

/// Main Embodied State engine for FlyWire locomotory simulation.
#[derive(Debug)]
pub struct FlyEmbodiedState {
    pub n_joints: usize,
    pub n_muscles: usize,
    pub neuromotor_state: NeuromotorState,
    pub delay_buffers: Vec<DelayBuffer>,
    pub joint_mappings: Vec<JointMapping>,
    pub step_count: u64,
    pub dt: f32,
    pub simulated_joint_qpos: Vec<f32>,
    pub simulated_joint_qvel: Vec<f32>,
}

impl FlyEmbodiedState {
    /// Initialize a new FlyEmbodiedState with `n_joints` (2 muscles per joint).
    pub fn new(n_joints: usize, dt: f32) -> Self {
        let unpadded_muscles = n_joints * 2;
        let n_muscles = unpadded_muscles.div_ceil(8) * 8;
        let neuromotor_state = NeuromotorState::new(n_muscles);

        let mut delay_buffers = Vec::with_capacity(n_muscles);
        for _ in 0..n_muscles {
            delay_buffers.push(DelayBuffer::new(1));
        }

        let mut joint_mappings = Vec::with_capacity(n_joints);
        for j in 0..n_joints {
            joint_mappings.push(JointMapping {
                dof_idx: j,
                flexor_muscle_idx: 2 * j,
                extensor_muscle_idx: 2 * j + 1,
                q_rest: 0.0,
                label: format!("joint_{j}"),
            });
        }

        Self {
            n_joints,
            n_muscles,
            neuromotor_state,
            delay_buffers,
            joint_mappings,
            step_count: 0,
            dt,
            simulated_joint_qpos: vec![0.0; n_joints],
            simulated_joint_qvel: vec![0.0; n_joints],
        }
    }

    /// Step the embodied pipeline:
    /// 1. Push motoneuron spikes through NMJ DelayBuffers
    /// 2. Update Neuromotor Calcium Muscle Dynamics & Tug-of-War
    /// 3. Compute Net Applied Torque τ = ΔF - K*(q - q_rest) - B*q_dot
    /// 4. Integrate physical joint motion
    pub fn step(&mut self, motoneuron_spikes: &[i32], resets: &[i32]) -> SimStepOutput {
        let n_in = motoneuron_spikes.len().min(self.n_muscles);

        // 1. NMJ Synaptic Delay via DelayBuffer
        let mut delayed_spikes = vec![0i32; self.n_muscles];
        let mut reset_spikes = vec![0i32; self.n_muscles];

        for m in 0..n_in {
            self.delay_buffers[m].push(&[motoneuron_spikes[m] as f32]);
            let delayed = self.delay_buffers[m].read(1);
            delayed_spikes[m] = delayed[0] as i32;
            if m < resets.len() {
                reset_spikes[m] = resets[m];
            }
        }

        // 2. Neuromotor Step (SIMD & Rayon) with Hill length/velocity modulation
        // No lengths/velocities from physics engine yet — use None for isometric mode
        self.neuromotor_state
            .step(&delayed_spikes, &reset_spikes, self.dt, None, None);

        // 3. Compute Generalized Joint Forces & Update Dynamics
        let mut sum_stf = 0.0f32;
        let mut sum_dmp = 0.0f32;

        for j in 0..self.n_joints {
            let map = &self.joint_mappings[j];
            let q = self.simulated_joint_qpos[map.dof_idx];
            let q_dot = self.simulated_joint_qvel[map.dof_idx];

            let df = self.neuromotor_state.joint_delta_f[j];
            let k_stf = self.neuromotor_state.joint_stiffness[j];
            let b_dmp = self.neuromotor_state.joint_damping[j];

            sum_stf += k_stf;
            sum_dmp += b_dmp;

            // Physical Torque Equation: τ = ΔF - K*(q - q_rest) - B*q_dot
            let tau_applied = df - k_stf * (q - map.q_rest) - b_dmp * q_dot;

            // Simple semi-implicit Euler step for joint simulation
            let q_acc = tau_applied; // Unit moment of inertia
            self.simulated_joint_qvel[map.dof_idx] += q_acc * self.dt;
            self.simulated_joint_qpos[map.dof_idx] +=
                self.simulated_joint_qvel[map.dof_idx] * self.dt;
        }

        self.step_count += 1;

        let active_muscles = self
            .neuromotor_state
            .ca
            .iter()
            .filter(|&&ca| ca > 0.01)
            .count();
        let avg_joint_stiffness = sum_stf / self.n_joints as f32;
        let avg_joint_damping = sum_dmp / self.n_joints as f32;

        SimStepOutput {
            t: self.step_count as f32 * self.dt,
            pose: FlyPose {
                body_height: 1.0,
                body_pitch: 0.0,
                body_roll: 0.0,
                joint_angles: self.simulated_joint_qpos.clone(),
            },
            active_muscles,
            avg_joint_stiffness,
            avg_joint_damping,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fly_embodied_pipeline_basic() {
        let n_joints = 6; // 6 leg joints
        let mut fly = FlyEmbodiedState::new(n_joints, 0.001); // 1ms step

        let spikes = vec![1i32; fly.n_muscles];
        let resets = vec![0i32; fly.n_muscles];

        let out = fly.step(&spikes, &resets);

        assert!(out.t > 0.0);
        assert!(
            out.avg_joint_stiffness > 1.0,
            "Joint stiffness should rise with calcium"
        );
        assert!(
            out.avg_joint_damping > 0.1,
            "Joint damping should rise with calcium to stabilize physics"
        );
    }

    #[test]
    fn test_fly_embodied_antagonistic_balance() {
        let n_joints = 2;
        let mut fly = FlyEmbodiedState::new(n_joints, 0.001);

        // Flexor and extensor both spike equally -> Co-contraction!
        let spikes = vec![1i32; fly.n_muscles];
        let resets = vec![0i32; fly.n_muscles];

        for _ in 0..10 {
            fly.step(&spikes, &resets);
        }

        // Torque delta F should be ~0, but stiffness K and damping B should be high!
        let df = fly.neuromotor_state.joint_delta_f[0];
        let k = fly.neuromotor_state.joint_stiffness[0];
        let b = fly.neuromotor_state.joint_damping[0];

        assert!(
            df.abs() < 1e-4,
            "Co-contraction of identical muscles produces near-zero net torque"
        );
        assert!(k > 1.5, "Co-contraction increases joint stiffness");
        assert!(
            b > 0.2,
            "Co-contraction increases joint damping for posture stability"
        );
    }
}
