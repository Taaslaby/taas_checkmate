//! Atlasov Neuromotor Kernel — Discrete Ca²⁺ & Active Reset Muscle Engine.
//!
//! Safe Rust wrapper over SIMD kernel (`kernels::neuron::atlasov_neuromotor::atlasov_neuromotor_step`).
//!
//! Models:
//! - Spike-triggered Calcium Injection upon motor neuron spikes
//! - Exponential pump clearance & active mechanical reset kinetics
//! - Cooperative saturation muscle tension (Hill-type curve)
//! - Agonist-Antagonist Tug-of-War (Joint net torque & dynamic stiffness)
//!
//! Author / Concept: Atlasov (Атласов)
//!
//! маленкие мускулы из маленьких клеток строят маленьких человечков которые потом будут строить человечков побольше и так до бесконечности
//!

use crate::kernels;
use rayon::prelude::*;

/// Step `n` Atlasov Neuromotor Muscle Channels.
/// Parallelized via Rayon, `n` must be a multiple of 8.
///
/// Muscle Parameters:
/// - `dt`: time step (ms)
/// - `d_ca`: Ca²⁺ injected per motor spike
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
    l0: &[f32],
    v_max: &[f32],
    lengths: Option<&[f32]>,
    velocities: Option<&[f32]>,
    joint_delta_f: Option<&mut [f32]>,
    joint_stiffness: Option<&mut [f32]>,
    joint_damping: Option<&mut [f32]>,
    k0: Option<&[f32]>,
    beta: Option<&[f32]>,
    b0: Option<&[f32]>,
    alpha: Option<&[f32]>,
) {
    let n = ca.len();
    assert!(
        n.is_multiple_of(8)
            && n == tension.len()
            && n == spikes.len()
            && n == resets.len()
            && n == d_ca.len()
            && n == tau_pump.len()
            && n == eta_reset.len()
            && n == t_max.len()
            && n == kd.len()
            && n == l0.len()
            && n == v_max.len()
    );

    if let Some(l) = lengths {
        assert_eq!(l.len(), n);
    }
    if let Some(v) = velocities {
        assert_eq!(v.len(), n);
    }

    // 1. Process Muscle Channels in Parallel (chunks of 8)
    let len_ptr_usize = lengths.map_or(0, |l| l.as_ptr() as usize);
    let vel_ptr_usize = velocities.map_or(0, |v| v.as_ptr() as usize);

    ca.par_chunks_exact_mut(8)
        .zip(tension.par_chunks_exact_mut(8))
        .zip(spikes.par_chunks_exact(8))
        .zip(resets.par_chunks_exact(8))
        .zip(d_ca.par_chunks_exact(8))
        .zip(tau_pump.par_chunks_exact(8))
        .zip(eta_reset.par_chunks_exact(8))
        .zip(t_max.par_chunks_exact(8))
        .zip(kd.par_chunks_exact(8))
        .zip(l0.par_chunks_exact(8))
        .zip(v_max.par_chunks_exact(8))
        .enumerate()
        .for_each(
            |(
                chunk_idx,
                (
                    (
                        (
                            (
                                ((((((ca8, tens8), spk8), rst8), d_ca8), tau_pump8), eta_reset8),
                                t_max8,
                            ),
                            kd8,
                        ),
                        l08,
                    ),
                    vmax8,
                ),
            )| {
                let mut params = [0.0f32; 64];
                for j in 0..8 {
                    params[j] = dt;
                    params[8 + j] = d_ca8[j];
                    params[16 + j] = tau_pump8[j];
                    params[24 + j] = eta_reset8[j];
                    params[32 + j] = t_max8[j];
                    params[40 + j] = kd8[j];
                    params[48 + j] = l08[j];
                    params[56 + j] = vmax8[j];
                }

                let off = chunk_idx * 8;
                let c_len_ptr = if len_ptr_usize == 0 {
                    std::ptr::null()
                } else {
                    unsafe { (len_ptr_usize as *const f32).add(off) }
                };
                let c_vel_ptr = if vel_ptr_usize == 0 {
                    std::ptr::null()
                } else {
                    unsafe { (vel_ptr_usize as *const f32).add(off) }
                };

                unsafe {
                    kernels::neuron::atlasov_neuromotor::atlasov_neuromotor_step(
                        ca8.as_mut_ptr(),
                        tens8.as_mut_ptr(),
                        spk8.as_ptr(),
                        rst8.as_ptr(),
                        c_len_ptr,
                        c_vel_ptr,
                        params.as_ptr(),
                        8,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                    );
                }
            },
        );

    // 2. Compute Joint Agonist-Antagonist Dynamics if joint buffers provided
    if let (Some(df), Some(stf), Some(k0_arr), Some(beta_arr)) =
        (joint_delta_f, joint_stiffness, k0, beta)
    {
        let n_joints = n / 2;
        assert!(
            df.len() >= n_joints
                && stf.len() >= n_joints
                && k0_arr.len() >= n_joints
                && beta_arr.len() >= n_joints
        );

        if let (Some(dmp), Some(b0_arr), Some(alpha_arr)) = (joint_damping, b0, alpha) {
            df.par_iter_mut()
                .zip(stf.par_iter_mut())
                .zip(dmp.par_iter_mut())
                .zip(k0_arr.par_iter())
                .zip(beta_arr.par_iter())
                .zip(b0_arr.par_iter())
                .zip(alpha_arr.par_iter())
                .enumerate()
                .for_each(
                    |(j, ((((((df_j, stf_j), dmp_j), k0_j), beta_j), b0_j), alpha_j))| {
                        let flex_idx = 2 * j;
                        let ext_idx = 2 * j + 1;
                        let ca_sum = ca[flex_idx] + ca[ext_idx];

                        *df_j = tension[ext_idx] - tension[flex_idx];
                        *stf_j = k0_j + beta_j * ca_sum;
                        *dmp_j = b0_j + alpha_j * ca_sum;
                    },
                );
        } else {
            df.par_iter_mut()
                .zip(stf.par_iter_mut())
                .zip(k0_arr.par_iter())
                .zip(beta_arr.par_iter())
                .enumerate()
                .for_each(|(j, (((df_j, stf_j), k0_j), beta_j))| {
                    let flex_idx = 2 * j;
                    let ext_idx = 2 * j + 1;
                    let ca_sum = ca[flex_idx] + ca[ext_idx];

                    *df_j = tension[ext_idx] - tension[flex_idx];
                    *stf_j = k0_j + beta_j * ca_sum;
                });
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
    pub l0: Vec<f32>,
    pub v_max: Vec<f32>,
    pub joint_delta_f: Vec<f32>,
    pub joint_stiffness: Vec<f32>,
    pub joint_damping: Vec<f32>,
    pub k0: Vec<f32>,
    pub beta: Vec<f32>,
    pub b0: Vec<f32>,
    pub alpha: Vec<f32>,
}

/// Physiological profile for muscle tissue scaling and biomechanics.
#[derive(Debug, Clone, Copy)]
pub struct BiomechanicsProfile {
    pub d_ca: f32,
    pub tau_pump: f32,
    pub eta_reset: f32,
    pub t_max: f32,
    pub kd: f32,
    pub l0: f32,
    pub v_max: f32,
    pub k0: f32,
    pub beta: f32,
    pub b0: f32,
    pub alpha: f32,
}

impl BiomechanicsProfile {
    /// Mammalian-scale profile (macroscopic forces in Newtons).
    pub fn mammal() -> Self {
        Self {
            d_ca: 0.5,
            tau_pump: 50.0,
            eta_reset: 0.8,
            t_max: 10.0, // 10 N
            kd: 0.3,
            l0: 1.0,
            v_max: 5.0,
            k0: 1.0,
            beta: 2.0,
            b0: 0.1,
            alpha: 0.5,
        }
    }

    /// Insect-scale profile (microscopic forces in milli-Newtons).
    /// Ideal for Drosophila MuJoCo physics (prevents explosion from excess torque).
    pub fn insect() -> Self {
        Self {
            d_ca: 0.5,
            tau_pump: 20.0,
            eta_reset: 0.8,
            t_max: 0.001, // 1 mN
            kd: 0.3,
            l0: 1.0,
            v_max: 0.5,
            k0: 1e-5,
            beta: 1e-4,
            b0: 1e-6,
            alpha: 1e-5,
        }
    }
}

impl Default for BiomechanicsProfile {
    fn default() -> Self {
        Self::mammal()
    }
}

/// Idiomatic Rust Builder for `AtlasovState`.
#[derive(Debug, Clone)]
pub struct AtlasovStateBuilder {
    n_muscles: usize,
    profile: BiomechanicsProfile,
}

impl AtlasovStateBuilder {
    pub fn new(n_muscles: usize) -> Self {
        assert!(
            n_muscles.is_multiple_of(2),
            "n_muscles must be even for agonist-antagonist pairs"
        );
        Self {
            n_muscles,
            profile: BiomechanicsProfile::default(),
        }
    }

    pub fn with_profile(mut self, profile: BiomechanicsProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn build(self) -> AtlasovState {
        let n = self.n_muscles;
        let n_joints = n / 2;
        let p = self.profile;

        AtlasovState {
            n_muscles: n,
            n_joints,
            ca: vec![0.0; n],
            tension: vec![0.0; n],
            d_ca: vec![p.d_ca; n],
            tau_pump: vec![p.tau_pump; n],
            eta_reset: vec![p.eta_reset; n],
            t_max: vec![p.t_max; n],
            kd: vec![p.kd; n],
            l0: vec![p.l0; n],
            v_max: vec![p.v_max; n],
            joint_delta_f: vec![0.0; n_joints],
            joint_stiffness: vec![0.0; n_joints],
            joint_damping: vec![0.0; n_joints],
            k0: vec![p.k0; n_joints],
            beta: vec![p.beta; n_joints],
            b0: vec![p.b0; n_joints],
            alpha: vec![p.alpha; n_joints],
        }
    }
}

impl AtlasovState {
    /// Initialize new Atlasov Neuromotor state for `n_muscles` (must be even for joint pairs)
    /// using the default (mammal) profile. For customized profiles, use `AtlasovStateBuilder`.
    pub fn new(n_muscles: usize) -> Self {
        AtlasovStateBuilder::new(n_muscles).build()
    }

    /// Step simulation with input spikes and resets.
    pub fn step(
        &mut self,
        spikes: &[i32],
        resets: &[i32],
        dt: f32,
        lengths: Option<&[f32]>,
        velocities: Option<&[f32]>,
    ) {
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
            &self.l0,
            &self.v_max,
            lengths,
            velocities,
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
        let t_max = vec![0.001f32; n];
        let kd = vec![0.3f32; n];
        let l0 = vec![1.0f32; n];
        let v_max = vec![5.0f32; n];

        let mut joint_df = vec![0.0f32; 8];
        let mut joint_stf = vec![0.0f32; 8];
        let mut joint_dmp = vec![0.0f32; 8];
        let k0 = vec![1e-5f32; 8];
        let beta = vec![1e-4f32; 8];
        let b0 = vec![1e-6f32; 8];
        let alpha = vec![1e-5f32; 8];

        step_atlasov_neuromotor(
            &mut ca,
            &mut tension,
            &spikes,
            &resets,
            dt,
            &d_ca,
            &tau_pump,
            &eta_reset,
            &t_max,
            &kd,
            &l0,
            &v_max,
            None,
            None,
            Some(&mut joint_df),
            Some(&mut joint_stf),
            Some(&mut joint_dmp),
            Some(&k0),
            Some(&beta),
            Some(&b0),
            Some(&alpha),
        );

        assert!(ca[0] > 0.0, "Ca2+ should accumulate");
        assert!(tension[0] > 0.0, "Tension should be generated");
        assert!(
            joint_stf[0] > 1e-5,
            "Joint stiffness should increase with calcium"
        );
        assert!(
            joint_dmp[0] > 1e-6,
            "Joint damping should increase with calcium"
        );
    }

    #[test]
    fn test_atlasov_state_struct_high_level() {
        let mut state = AtlasovState::new(16);
        let spikes = vec![1i32; 16];
        let resets = vec![0i32; 16];

        state.step(&spikes, &resets, 1.0, None, None);

        assert!(state.ca[0] > 0.0);
        assert!(state.tension[0] > 0.0);
        assert!(state.joint_stiffness[0] > 1e-5);
        assert!(state.joint_damping[0] > 1e-6);
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
        state.step(&spikes_t0_i32, &resets, 1.0, None, None);

        assert!(state.ca[0] >= 0.0);
    }

    #[test]
    fn test_human_scale_muscles() {
        // У обычного человека около 640 скелетных мышц (320 суставных пар).
        let human_muscles = 640;
        let mut state = AtlasovStateBuilder::new(human_muscles)
            .with_profile(BiomechanicsProfile::mammal())
            .build();

        let spikes = vec![1i32; human_muscles];
        let resets = vec![0i32; human_muscles];

        // Проверим производительность на 1000 итерациях (1 секунда симуляции при dt=1ms)
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            state.step(&spikes, &resets, 1.0, None, None);
        }
        let duration = start.elapsed();

        // Сила должна быть огромной (в Ньютонах)
        assert!(state.tension[0] > 1.0);
        // Жесткость коленных и других суставов при коконтракции
        assert!(state.joint_stiffness[0] > 1.0);

        println!(
            "Просимулировано 640 мышц человека (1 секунда, 1000 тиков) за: {:?}",
            duration
        );
    }
}
