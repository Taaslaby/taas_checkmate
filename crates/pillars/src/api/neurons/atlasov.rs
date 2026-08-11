//! # Atlasov Neuromotor Kernel — Discrete Ca²⁺ & Active Reset Muscle Engine
//!
//! Канон pillars::api: step-функция и stateful-классы перенесены
//! из core. Модель — `run_neuron::<AtlasovNeuromotorModel>` (single source
//! of truth).
//!
//! Models:
//! - Spike-triggered Calcium Injection upon motor neuron spikes
//! - Exponential pump clearance & active mechanical reset kinetics
//! - Cooperative saturation muscle tension (Hill-type curve)
//! - Agonist-Antagonist Tug-of-War (Joint net torque & dynamic stiffness)
//!
//! Author / Concept: Atlasov (Атласов)
//!
//! маленькие мускулы из маленьких клеток строят маленьких человечков,
//! которые потом будут строить человечков побольше — и так до бесконечности

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::core::runner::run_neuron;
use crate::zoo::neurons::AtlasovNeuromotorModel;

/// Обработать чанк из 8 мышечных каналов через pillars-модель.
#[allow(clippy::too_many_arguments)]
#[inline]
fn process_chunk(
    ca: &mut [f32],
    tens: &mut [f32],
    spk: &[i32],
    rst: &[i32],
    d_ca: &[f32],
    tau_pump: &[f32],
    eta_reset: &[f32],
    t_max: &[f32],
    kd: &[f32],
    l0: &[f32],
    v_max: &[f32],
    dt: f32,
    lengths: Option<&[f32]>,
    off: usize,
    velocities: Option<&[f32]>,
) {
    // AtlasovNeuromotorModel 9-col: [dt, d_ca, ca_decay_const, eta_reset,
    //   t_max, kd, l0, inv_vmax, inv_width]; state [ca, tension, reset_flag,
    //   length, velocity].
    // Легаси-семантика сохранена на границе: ca_decay_const = exp(-dt/τ_pump),
    // inv_vmax = 1/v_max, inv_width = 1/(l0·0.5); нет длины/скорости → слоты
    // = l0 / 0 (f_l = f_v = 1).
    let mut params = [0.0f32; 72];
    let mut i_in = [0.0f32; 8];
    let mut st_rst = [0.0f32; 8];
    let mut st_len = [0.0f32; 8];
    let mut st_vel = [0.0f32; 8];
    let mut spk_out = [0i32; 8];
    for j in 0..8 {
        params[j] = dt;
        params[8 + j] = d_ca[j];
        params[16 + j] = (-dt / tau_pump[j].max(1e-6)).exp();
        params[24 + j] = eta_reset[j];
        params[32 + j] = t_max[j];
        params[40 + j] = kd[j];
        params[48 + j] = l0[j];
        params[56 + j] = 1.0 / v_max[j].max(1e-6);
        params[64 + j] = 1.0 / (l0[j] * 0.5).max(1e-6);
        i_in[j] = spk[j] as f32;
        st_rst[j] = rst[j] as f32;
        st_len[j] = lengths.map_or(l0[j], |l| l[off + j]);
        st_vel[j] = velocities.map_or(0.0, |v| v[off + j]);
    }
    unsafe {
        run_neuron::<AtlasovNeuromotorModel>(
            ca.as_mut_ptr(),
            tens.as_mut_ptr(),
            st_rst.as_mut_ptr(),
            st_len.as_mut_ptr(),
            st_vel.as_mut_ptr(),
            std::ptr::null_mut(),
            i_in.as_ptr(),
            spk_out.as_mut_ptr(),
            params.as_ptr(),
            8,
        );
    }
}

/// Обработать сустав `j` (agonista-antagonist tug-of-war).
/// Используется только в serial-ветке (rayon-ветки инлайн-zip).
#[cfg(not(feature = "rayon"))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn process_joint(
    j: usize,
    df: &mut [f32],
    stf: &mut [f32],
    dmp: Option<&mut [f32]>,
    k0: &[f32],
    beta: &[f32],
    b0: Option<&[f32]>,
    alpha: Option<&[f32]>,
    ca: &[f32],
    tension: &[f32],
) {
    let flex_idx = 2 * j;
    let ext_idx = 2 * j + 1;
    let ca_sum = ca[flex_idx] + ca[ext_idx];

    df[j] = tension[ext_idx] - tension[flex_idx];
    stf[j] = k0[j] + beta[j] * ca_sum;
    if let (Some(dmp), Some(b0), Some(alpha)) = (dmp, b0, alpha) {
        dmp[j] = b0[j] + alpha[j] * ca_sum;
    }
}

/// Step `n` Atlasov Neuromotor Muscle Channels.
/// Parallelized via Rayon (или serial без feature), `n` must be a multiple of 8.
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

    // 1. Process Muscle Channels (chunks of 8)
    #[cfg(feature = "rayon")]
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
                process_chunk(
                    ca8,
                    tens8,
                    spk8,
                    rst8,
                    d_ca8,
                    tau_pump8,
                    eta_reset8,
                    t_max8,
                    kd8,
                    l08,
                    vmax8,
                    dt,
                    lengths,
                    chunk_idx * 8,
                    velocities,
                );
            },
        );
    #[cfg(not(feature = "rayon"))]
    for chunk_idx in 0..n / 8 {
        let off = chunk_idx * 8;
        process_chunk(
            &mut ca[off..off + 8],
            &mut tension[off..off + 8],
            &spikes[off..off + 8],
            &resets[off..off + 8],
            &d_ca[off..off + 8],
            &tau_pump[off..off + 8],
            &eta_reset[off..off + 8],
            &t_max[off..off + 8],
            &kd[off..off + 8],
            &l0[off..off + 8],
            &v_max[off..off + 8],
            dt,
            lengths,
            off,
            velocities,
        );
    }

    // 2. Compute Joint Agonist-Antagonist Dynamics if joint buffers provided.
    // rayon-ветки — inline zip-замыкания, serial — process_joint: две копии
    // логики (rayon требует Fn-совместимых замыканий). При правке синхронизировать.
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
            #[cfg(feature = "rayon")]
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
            #[cfg(not(feature = "rayon"))]
            for j in 0..n_joints {
                process_joint(
                    j,
                    df,
                    stf,
                    Some(dmp),
                    k0_arr,
                    beta_arr,
                    Some(b0_arr),
                    Some(alpha_arr),
                    ca,
                    tension,
                );
            }
        } else {
            #[cfg(feature = "rayon")]
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
            #[cfg(not(feature = "rayon"))]
            for j in 0..n_joints {
                process_joint(j, df, stf, None, k0_arr, beta_arr, None, None, ca, tension);
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
    pub const fn mammal() -> Self {
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
    pub const fn insect() -> Self {
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

    /// Drosophila-scale profile (forces in micro-Newtons).
    pub const fn drosophila() -> Self {
        Self {
            d_ca: 0.3,
            tau_pump: 10.0,
            eta_reset: 0.9,
            t_max: 2e-6, // 2 µN — intermediate fiber
            kd: 0.2,
            l0: 0.0003, // 300 µm fiber length
            v_max: 0.5,
            k0: 1e-8,
            beta: 1e-7,
            b0: 1e-9,
            alpha: 1e-8,
        }
    }

    /// Drosophila slow fiber pool — postural tone (~0.1 µN per spike).
    pub const fn drosophila_slow() -> Self {
        Self {
            d_ca: 0.1,
            tau_pump: 20.0,
            eta_reset: 0.9,
            t_max: 1e-7, // 0.1 µN
            kd: 0.15,
            l0: 0.0002, // 200 µm
            v_max: 0.1,
            k0: 1e-8,
            beta: 1e-8,
            b0: 1e-9,
            alpha: 1e-9,
        }
    }

    /// Drosophila fast fiber pool — ballistic (~10 µN per spike).
    pub const fn drosophila_fast() -> Self {
        Self {
            d_ca: 1.0,
            tau_pump: 8.0,
            eta_reset: 0.7,
            t_max: 1e-5, // 10 µN
            kd: 0.3,
            l0: 0.0004, // 400 µm
            v_max: 1.0,
            k0: 1e-8,
            beta: 1e-7,
            b0: 1e-9,
            alpha: 1e-8,
        }
    }

    /// TDT (Tergal Depressor of Trochanter) — catapult muscle for jumping.
    pub const fn drosophila_tdt() -> Self {
        Self {
            d_ca: 2.0,      // massive Ca²⁺ injection
            tau_pump: 50.0, // slow pump
            eta_reset: 0.5, // passive reset
            t_max: 2e-4,    // 200 µN — 100× body weight!
            kd: 0.1,
            l0: 0.0005, // 500 µm — spans thorax
            v_max: 2.0, // 2 m/s — ballistic
            k0: 1e-7,
            beta: 1e-6,
            b0: 1e-8,
            alpha: 1e-7,
        }
    }
}

impl Default for BiomechanicsProfile {
    fn default() -> Self {
        Self::mammal()
    }
}

/// Configuration for a single Drosophila muscle fiber pool.
#[derive(Debug, Clone, Copy)]
pub struct FiberPoolConfig {
    /// Number of fibers in this pool (typically 3-7 per fly muscle).
    pub n_fibers: usize,
    /// Biophysical profile for this pool.
    pub profile: BiomechanicsProfile,
}

impl FiberPoolConfig {
    /// Slow postural fibers: 5-7 fibers, ~0.1 µN each.
    pub const fn slow() -> Self {
        Self {
            n_fibers: 6,
            profile: BiomechanicsProfile::drosophila_slow(),
        }
    }
    /// Intermediate walking fibers: 4-6 fibers, ~2 µN each.
    pub const fn intermediate() -> Self {
        Self {
            n_fibers: 5,
            profile: BiomechanicsProfile::drosophila(),
        }
    }
    /// Fast ballistic fibers: 3-5 fibers, ~10 µN each.
    pub const fn fast() -> Self {
        Self {
            n_fibers: 4,
            profile: BiomechanicsProfile::drosophila_fast(),
        }
    }
    /// TDT catapult: 3 super-thick fibers, ~200 µN each.
    pub const fn tdt() -> Self {
        Self {
            n_fibers: 3,
            profile: BiomechanicsProfile::drosophila_tdt(),
        }
    }
}

/// Full Drosophila leg musculature configuration.
#[derive(Debug, Clone)]
pub struct DrosophilaLegConfig {
    /// Number of legs.
    pub n_legs: usize,
    /// Number of muscles per leg.
    pub muscles_per_leg: usize,
    /// Number of joints per leg.
    pub joints_per_leg: usize,
    /// Total muscles across all legs.
    pub total_muscles: usize,
    /// Total fiber channels (sum of all pool.n_fibers).
    pub total_channels: usize,
    /// Whether to include TDT catapult muscles (middle legs only).
    pub include_tdt: bool,
    /// Fiber pool assignments for each muscle [n_muscles × pools].
    pub pools: Vec<Vec<FiberPoolConfig>>,
}

impl Default for DrosophilaLegConfig {
    fn default() -> Self {
        Self::hexapod()
    }
}

impl DrosophilaLegConfig {
    /// Standard 6-legged fly configuration.
    ///
    /// Each leg: 19 muscles × [slow, intermediate, fast] = 57 channels.
    /// Middle legs (T2, 2 legs): +3 TDT fibers each = +6 channels.
    /// Total: 6 × 57 + 6 = 348 channels → padded to 352 (44×8).
    pub fn hexapod() -> Self {
        let n_legs = 6;
        let muscles_per_leg = 19;
        let joints_per_leg = 5;

        let standard_pools = vec![
            FiberPoolConfig::slow(),
            FiberPoolConfig::intermediate(),
            FiberPoolConfig::fast(),
        ];

        let mut pools = Vec::new();
        for leg in 0..n_legs {
            for muscle in 0..muscles_per_leg {
                let mut muscle_pools = standard_pools.clone();
                // TDT is a SINGLE muscle (muscle 18) in middle legs only
                if (leg == 1 || leg == 2) && muscle == muscles_per_leg - 1 {
                    muscle_pools.push(FiberPoolConfig::tdt());
                }
                pools.push(muscle_pools);
            }
        }

        let total_channels: usize = pools
            .iter()
            .flat_map(|p| p.iter())
            .map(|pool| pool.n_fibers)
            .sum();

        // Pad to multiple of 8 for SIMD
        let n_padded = total_channels.div_ceil(8) * 8;

        Self {
            n_legs,
            muscles_per_leg,
            joints_per_leg,
            total_muscles: muscles_per_leg * n_legs,
            total_channels: n_padded, // store padded count for SIMD alignment
            include_tdt: true,
            pools,
        }
    }

    /// Build an AtlasovState for the full fly with per-fiber parameters.
    pub fn build_state(&self) -> AtlasovState {
        let n_channels = self.total_channels; // already padded to ×8

        let mut state = AtlasovStateBuilder::new(n_channels)
            .with_profile(BiomechanicsProfile::drosophila())
            .build();

        let mut idx = 0;
        for muscle_pools in &self.pools {
            for pool in muscle_pools {
                for _ in 0..pool.n_fibers {
                    if idx < n_channels {
                        state.d_ca[idx] = pool.profile.d_ca;
                        state.tau_pump[idx] = pool.profile.tau_pump;
                        state.eta_reset[idx] = pool.profile.eta_reset;
                        state.t_max[idx] = pool.profile.t_max;
                        state.kd[idx] = pool.profile.kd;
                        state.l0[idx] = pool.profile.l0;
                        state.v_max[idx] = pool.profile.v_max;
                    }
                    idx += 1;
                }
            }
        }

        state
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
    /// Initialize new Atlasov Neuromotor state for `n_muscles` (must be multiple of 8)
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
        let spikes = vec![1i32; n];
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
        assert!(joint_stf[0] > 1e-5, "Joint stiffness should increase");
        assert!(joint_dmp[0] > 1e-6, "Joint damping should increase");
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
    fn test_human_scale_muscles() {
        let human_muscles = 640;
        let mut state = AtlasovStateBuilder::new(human_muscles)
            .with_profile(BiomechanicsProfile::mammal())
            .build();

        let spikes = vec![1i32; human_muscles];
        let resets = vec![0i32; human_muscles];

        for _ in 0..1000 {
            state.step(&spikes, &resets, 1.0, None, None);
        }

        assert!(state.tension[0] > 1.0);
        assert!(state.joint_stiffness[0] > 1.0);
    }

    #[test]
    fn test_drosophila_profile_slow_fiber_pool() {
        let n = 8;
        let mut state = AtlasovStateBuilder::new(n)
            .with_profile(BiomechanicsProfile::drosophila_slow())
            .build();

        let spikes = vec![1i32; n];
        let resets = vec![0i32; n];

        for _ in 0..100 {
            state.step(&spikes, &resets, 0.5, None, None);
        }

        assert!(state.tension[0] < 1e-4, "Slow fiber tension should be tiny");
        assert!(state.tension[0] > 0.0, "Should have some tension");
    }

    #[test]
    fn test_drosophila_fast_fiber_pool() {
        let n = 8;
        let mut state = AtlasovStateBuilder::new(n)
            .with_profile(BiomechanicsProfile::drosophila_fast())
            .build();

        let spikes = vec![1i32; n];
        let resets = vec![0i32; n];

        for _ in 0..100 {
            state.step(&spikes, &resets, 0.5, None, None);
        }

        assert!(state.tension[0] > 0.0, "Fast fiber should have tension");
    }

    #[test]
    fn test_drosophila_tdt_profile() {
        let n = 8;
        let mut state = AtlasovStateBuilder::new(n)
            .with_profile(BiomechanicsProfile::drosophila_tdt())
            .build();

        let spikes = vec![1i32; n];
        let resets = vec![0i32; n];

        for _ in 0..200 {
            state.step(&spikes, &resets, 0.5, None, None);
        }

        assert!(state.tension[0] > state.tension[1] * 0.5);
    }

    #[test]
    fn test_drosophila_leg_config_hexapod() {
        let config = DrosophilaLegConfig::hexapod();
        let channels = config.total_channels;

        assert!(channels > 0 && channels.is_multiple_of(8));
        assert!(config.include_tdt);
        let total_tdt: usize = config
            .pools
            .iter()
            .flat_map(|p| p.iter())
            .filter(|pool| (pool.profile.t_max - 2e-4).abs() < 1e-5)
            .map(|p| p.n_fibers)
            .sum();
        assert_eq!(total_tdt, 3 * 2, "TDT: 3 fibers × 2 middle legs = 6 total");

        // Build and test state
        let state = config.build_state();
        assert_eq!(state.n_muscles, channels);
        assert!(state.t_max.contains(&2e-4), "TDT tension must exist");
    }
}
