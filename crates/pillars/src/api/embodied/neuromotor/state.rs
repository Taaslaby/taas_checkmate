use super::*;

/// High-level Object-Oriented state management for Neuromotor Engine.
/// Ideal for VNC integration, MuJoCo physics loop, and FlyWire motor control.
#[derive(Debug, Clone)]
pub struct NeuromotorState {
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

pub struct NeuromotorStateBuilder {
    n_muscles: usize,
    profile: BiomechanicsProfile,
}

impl NeuromotorStateBuilder {
    pub fn new(n_muscles: usize) -> Self {
        assert!(
            n_muscles.is_multiple_of(8),
            "n_muscles must be multiple of 8 for SIMD alignment"
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

    pub fn build(self) -> NeuromotorState {
        let n = self.n_muscles;
        let n_joints = n / 2;
        let p = self.profile;

        NeuromotorState {
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

impl NeuromotorState {
    /// Initialize new Neuromotor state for `n_muscles` (must be multiple of 8)
    /// using the default (mammal) profile. For customized profiles, use `NeuromotorStateBuilder`.
    pub fn new(n_muscles: usize) -> Self {
        NeuromotorStateBuilder::new(n_muscles).build()
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
        step_neuromotor(
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
