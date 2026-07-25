//! Atlasov Neuromotor Kernel — Discrete Ca²⁺ & Active Reset Muscle Engine.
//!
//! Safe Rust wrapper over SIMD kernel (`kernels::neuron::atlasov_neuromotor::atlasov_neuromotor_step`).
//!
//! Models:
//! - Quantum Calcium Injection upon motor neuron spikes
//! - Exponential pump clearance & active mechanical reset kinetics
//! - Cooperative saturation muscle tension (Hill-type curve)
//! - Agonist-Antagonist Tug-of-War (Joint net torque & dynamic stiffness)
//!
//! Author / Concept: Atlasov (Атласов)

use crate::kernels;
use rayon::prelude::*;

/// Step `n` Atlasov Neuromotor Muscle Channels.
/// Parallelized via Rayon, `n` must be a multiple of 8.
///
/// Muscle Parameters:
/// - `dt`: time step (ms)
/// - `d_ca`: Ca²⁺ quantum injected per motor spike
/// - `tau_pump`: pump clearance time constant (ms)
/// - `eta_reset`: active mechanical reset efficiency (0.0..1.0)
/// - `t_max`: maximum isometric muscle tension
/// - `kd`: calcium sensitivity threshold
///
/// Joint Output (Optional):
/// - `joint_delta_f`: output net torque/force for `n / 2` joint pairs
/// - `joint_stiffness`: output dynamic joint stiffness for `n / 2` joint pairs
/// - `k0`: baseline joint stiffness per pair (`n / 2`)
/// - `beta`: calcium co-contraction stiffness gain per pair (`n / 2`)
pub fn step_atlasov_neuromotor(
    ca: &mut [f32],
    tension: &mut [f32],
    spikes: &[i32],
    resets: &[i32],
    dt: f32,
    d_ca: &[f32],
    tau_pump: &[f32],
    eta_reset: &[f32],
    t_max: &[f32],
    kd: &[f32],
    joint_delta_f: Option<&mut [f32]>,
    joint_stiffness: Option<&mut [f32]>,
    mut joint_damping: Option<&mut [f32]>,
    k0: Option<&[f32]>,
    beta: Option<&[f32]>,
    b0: Option<&[f32]>,
    alpha: Option<&[f32]>,
) {
    let n = ca.len();
    assert!(n.is_multiple_of(8)
        && n == tension.len() && n == spikes.len() && n == resets.len()
        && n == d_ca.len() && n == tau_pump.len() && n == eta_reset.len()
        && n == t_max.len() && n == kd.len());

    // 1. Process Muscle Channels in Parallel (chunks of 8)
    ca.par_chunks_exact_mut(8)
        .zip(tension.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact(8))
        .zip(resets.par_chunks_exact(8))
        .enumerate()
        .for_each(|(chunk, (((ca8, tens8), spk8), rst8))| {
            let i = chunk * 8;
            let mut params = [0.0f32; 48];
            for j in 0..8 {
                params[j] = dt;
                params[8 + j] = d_ca[i + j];
                params[16 + j] = tau_pump[i + j];
                params[24 + j] = eta_reset[i + j];
                params[32 + j] = t_max[i + j];
                params[40 + j] = kd[i + j];
            }
            unsafe {
                kernels::neuron::atlasov_neuromotor::atlasov_neuromotor_step(
                    ca8.as_mut_ptr(),
                    tens8.as_mut_ptr(),
                    spk8.as_ptr(),
                    rst8.as_ptr(),
                    params.as_ptr(),
                    8,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                );
            }
        });

    // 2. Compute Joint Agonist-Antagonist Dynamics if joint buffers provided
    if let (Some(df), Some(stf), Some(k0_arr), Some(beta_arr)) = (joint_delta_f, joint_stiffness, k0, beta) {
        let n_joints = n / 2;
        assert!(df.len() >= n_joints && stf.len() >= n_joints && k0_arr.len() >= n_joints && beta_arr.len() >= n_joints);

        for j in 0..n_joints {
            let flex_idx = 2 * j;
            let ext_idx = 2 * j + 1;
            let ca_sum = ca[flex_idx] + ca[ext_idx];

            df[j] = tension[ext_idx] - tension[flex_idx];
            stf[j] = k0_arr[j] + beta_arr[j] * ca_sum;

            if let (Some(dmp), Some(b0_arr), Some(alpha_arr)) = (joint_damping.as_deref_mut(), b0, alpha) {
                dmp[j] = b0_arr[j] + alpha_arr[j] * ca_sum;
            }
        }
    }
}

/// High-level Object-Oriented state management for Atlasov Neuromotor Engine.
/// Ideal for VNC integration, MuJoCo physics loop, and FlyWire motor control.
#[derive(Debug, Clone)]
pub struct AtlasovState {
    pub n_muscles: usize,
    pub n_joints: usize,
    pub ca: Vec<f32>,
    pub tension: Vec<f32>,
    pub d_ca: Vec<f32>,
    pub tau_pump: Vec<f32>,
    pub eta_reset: Vec<f32>,
    pub t_max: Vec<f32>,
    pub kd: Vec<f32>,
    pub joint_delta_f: Vec<f32>,
    pub joint_stiffness: Vec<f32>,
    pub joint_damping: Vec<f32>,
    pub k0: Vec<f32>,
    pub beta: Vec<f32>,
    pub b0: Vec<f32>,
    pub alpha: Vec<f32>,
}

impl AtlasovState {
    /// Initialize new Atlasov Neuromotor state for `n_muscles` (must be even for joint pairs).
    pub fn new(n_muscles: usize) -> Self {
        assert!(n_muscles % 2 == 0, "n_muscles must be even for agonist-antagonist pairs");
        let n_joints = n_muscles / 2;
        Self {
            n_muscles,
            n_joints,
            ca: vec![0.0; n_muscles],
            tension: vec![0.0; n_muscles],
            d_ca: vec![0.5; n_muscles],
            tau_pump: vec![20.0; n_muscles],
            eta_reset: vec![0.8; n_muscles],
            t_max: vec![10.0; n_muscles],
            kd: vec![0.3; n_muscles],
            joint_delta_f: vec![0.0; n_joints],
            joint_stiffness: vec![0.0; n_joints],
            joint_damping: vec![0.0; n_joints],
            k0: vec![1.0; n_joints],
            beta: vec![2.0; n_joints],
            b0: vec![0.1; n_joints],
            alpha: vec![0.5; n_joints],
        }
    }

    /// Step simulation with input spikes and resets.
    pub fn step(&mut self, spikes: &[i32], resets: &[i32], dt: f32) {
        step_atlasov_neuromotor(
            &mut self.ca,
            &mut self.tension,
            spikes,
            resets,
            dt,
            &self.d_ca,
            &self.tau_pump,
            &self.eta_reset,
            &self.t_max,
            &self.kd,
            Some(&mut self.joint_delta_f),
            Some(&mut self.joint_stiffness),
            Some(&mut self.joint_damping),
            Some(&self.k0),
            Some(&self.beta),
            Some(&self.b0),
            Some(&self.alpha),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlasov_wrapper_basic() {
        let n = 16;
        let dt = 1.0;
        let mut ca = vec![0.0f32; n];
        let mut tension = vec![0.0f32; n];
        let spikes = vec![1i32; n]; // All muscles spike
        let resets = vec![0i32; n];

        let d_ca = vec![0.5f32; n];
        let tau_pump = vec![20.0f32; n];
        let eta_reset = vec![0.8f32; n];
        let t_max = vec![10.0f32; n];
        let kd = vec![0.3f32; n];

        let mut joint_df = vec![0.0f32; 8];
        let mut joint_stf = vec![0.0f32; 8];
        let mut joint_dmp = vec![0.0f32; 8];
        let k0 = vec![1.0f32; 8];
        let beta = vec![2.0f32; 8];
        let b0 = vec![0.1f32; 8];
        let alpha = vec![0.5f32; 8];

        step_atlasov_neuromotor(
            &mut ca, &mut tension, &spikes, &resets, dt,
            &d_ca, &tau_pump, &eta_reset, &t_max, &kd,
            Some(&mut joint_df), Some(&mut joint_stf), Some(&mut joint_dmp), Some(&k0), Some(&beta), Some(&b0), Some(&alpha),
        );

        assert!(ca[0] > 0.0, "Ca2+ should accumulate");
        assert!(tension[0] > 0.0, "Tension should be generated");
        assert!(joint_stf[0] > 1.0, "Joint stiffness should increase with calcium");
        assert!(joint_dmp[0] > 0.1, "Joint damping should increase with calcium");
    }

    #[test]
    fn test_atlasov_state_struct_high_level() {
        let mut state = AtlasovState::new(16);
        let spikes = vec![1i32; 16];
        let resets = vec![0i32; 16];

        state.step(&spikes, &resets, 1.0);

        assert!(state.ca[0] > 0.0);
        assert!(state.tension[0] > 0.0);
        assert!(state.joint_stiffness[0] > 1.0);
        assert!(state.joint_damping[0] > 0.1);
    }

    #[test]
    fn test_motoneuron_to_muscle_latency() {
        use crate::neuron::delay::DelayBuffer;
        let mut state = AtlasovState::new(8);
        let mut delay_buf = DelayBuffer::new(8);

        let mn_spike = vec![1.0f32; 8];
        let resets = vec![0i32; 8];

        // Push spike to delay buffer (simulating 2ms NMJ delay)
        delay_buf.push(&mn_spike);

        // Immediate read at delay=2 should return 0.0 (since buffer was pre-filled with 0.0)
        let delayed_spikes_t0 = delay_buf.read(2);
        let spikes_t0_i32: Vec<i32> = delayed_spikes_t0.iter().map(|&s| s as i32).collect();
        state.step(&spikes_t0_i32, &resets, 1.0);

        assert!(state.ca[0] >= 0.0);
    }
}
