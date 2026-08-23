//! -------------------------------------------------------------------------
//! Filename:    Human_hand.rs
//! Class:       valide
//! Purpose:     Human hand/forearm reference profile for Neuromotor validation.
//! Description:
//!
//! This is deliberately a **reference model**, not a clinical patient model.
//! Morphology is taken from the open-access integrated forearm/hand dataset of
//! Kerkhof et al. (2018, DOI: 10.1111/joa.12877), while the 35 N/cm² maximum
//! muscle stress convention follows Goislard de Monsabert et al. (2018,
//! DOI: 10.1007/s10439-017-1936-z).
//!
//! //! The compact profile uses eight representative muscles: wrist flexors and
//! extensors plus the index finger flexors and first dorsal interosseous. PCSA
//! and force are source-backed; the signed moment arms are explicitly marked
//! as representative geometry for this reduced model and must be replaced by
//! joint-angle-dependent tendon paths for subject-specific work.
//!
//! All public physical quantities use SI units:
//! - PCSA: `mm²`
//! - fibre length: `m`
//! - specific tension: `Pa`
//! - force: `N`
//! - moment arm: `m`
//! - joint torque: `N·m`
//!
//! Author/Concept: Anton Atlasov
//! -------------------------------------------------------------------------

use super::neurons::neuromotor::{BiomechanicsProfile, NeuromotorState};

/// Maximum muscle stress used by the cited wrist model: 35 N/cm².
pub const HUMAN_SPECIFIC_TENSION_PA: f32 = 350_000.0;

/// Muscles represented by [`HumanHandProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanMuscle {
    Fcr,
    Fcu,
    Ecrl,
    Ecrb,
    Ecu,
    FdsIndex,
    FdpIndex,
    Fdi,
}

impl HumanMuscle {
    /// Stable channel order used by [`HumanHandState`].
    pub const ALL: [Self; 8] = [
        Self::Fcr,
        Self::Ecrl,
        Self::Fcu,
        Self::Ecu,
        Self::FdsIndex,
        Self::FdpIndex,
        Self::Ecrb,
        Self::Fdi,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Fcr => 0,
            Self::Ecrl => 1,
            Self::Fcu => 2,
            Self::Ecu => 3,
            Self::FdsIndex => 4,
            Self::FdpIndex => 5,
            Self::Ecrb => 6,
            Self::Fdi => 7,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Fcr => "FCR",
            Self::Fcu => "FCU",
            Self::Ecrl => "ECRL",
            Self::Ecrb => "ECRB",
            Self::Ecu => "ECU",
            Self::FdsIndex => "FDS-II",
            Self::FdpIndex => "FDP-II",
            Self::Fdi => "FDI",
        }
    }
}

/// Signed wrist moment arm (mm) as a piecewise-linear function of wrist angle
/// in degrees, where **flexion is positive** and **extension is negative**.
///
/// Control points are fitted to published ranges so the neutral pose matches
/// [`HumanMuscleSpec::wrist_moment_arm_m`]. They are a reduced-model
/// approximation, not a measured tendon-excursion curve.
#[derive(Debug, Clone, Copy)]
pub struct WristMomentArmCurve {
    /// `(angle_deg, signed_mm)` control points, ascending in angle.
    pub control_points: [(f32, f32); 5],
}

impl WristMomentArmCurve {
    /// Piecewise-linear interpolation in mm; clamps outside the end points.
    pub fn arm_at_deg_mm(self, angle_deg: f32) -> f32 {
        let pts = &self.control_points;
        if angle_deg <= pts[0].0 {
            return pts[0].1;
        }
        if angle_deg >= pts[4].0 {
            return pts[4].1;
        }
        for w in pts.windows(2) {
            let (a0, v0) = w[0];
            let (a1, v1) = w[1];
            if angle_deg <= a1 {
                let t = (angle_deg - a0) / (a1 - a0).max(1e-9);
                return v0 + t * (v1 - v0);
            }
        }
        pts[4].1
    }

    /// Same curve converted to metres.
    pub fn arm_at_deg_m(self, angle_deg: f32) -> f32 {
        self.arm_at_deg_mm(angle_deg) * 1.0e-3
    }
}

impl HumanHandProfile {
    /// Wrist moment arm in metres at a given wrist angle.
    ///
    /// Flexors produce positive (flexion) moments, extensors negative
    /// (extension) moments. Non-wrist channels return zero.
    pub fn wrist_moment_arm_m_at(&self, muscle: HumanMuscle, angle_deg: f32) -> f32 {
        let curve = match muscle {
            HumanMuscle::Fcr => WRIST_ARM_FCR,
            HumanMuscle::Ecrl => WRIST_ARM_ECRL,
            HumanMuscle::Fcu => WRIST_ARM_FCU,
            HumanMuscle::Ecu => WRIST_ARM_ECU,
            HumanMuscle::Ecrb => WRIST_ARM_ECRB,
            _ => return 0.0,
        };
        curve.arm_at_deg_m(angle_deg)
    }

    /// Peak wrist moment in N·m at a given wrist angle, assuming all muscles
    /// develop their maximum isometric force.
    pub fn peak_wrist_moment_nm_at_angle(&self, angle_deg: f32) -> f32 {
        self.muscles
            .iter()
            .map(|spec| {
                spec.max_isometric_force_n() * self.wrist_moment_arm_m_at(spec.muscle, angle_deg)
            })
            .sum()
    }
}

/// FCR: grows from ~8 mm (extension) to ~15 mm (flexion); neutral 10 mm.
pub const WRIST_ARM_FCR: WristMomentArmCurve = WristMomentArmCurve {
    control_points: [
        (-60.0, 8.0),
        (-30.0, 9.0),
        (0.0, 10.0),
        (30.0, 12.0),
        (60.0, 15.0),
    ],
};

/// ECRL: extensor, magnitude ~8–14 mm; neutral −10 mm.
pub const WRIST_ARM_ECRL: WristMomentArmCurve = WristMomentArmCurve {
    control_points: [
        (-60.0, -8.0),
        (-30.0, -9.0),
        (0.0, -10.0),
        (30.0, -12.0),
        (60.0, -14.0),
    ],
};

/// FCU: drops toward ~5 mm in extension, rises in flexion; neutral 12 mm.
/// The reported range is ≈5–12 mm; our neutral sits at its upper bound and
/// deep flexion grows the arm to ~14 mm (wrap-around).
pub const WRIST_ARM_FCU: WristMomentArmCurve = WristMomentArmCurve {
    control_points: [
        (-60.0, 5.0),
        (-30.0, 8.0),
        (0.0, 12.0),
        (30.0, 13.0),
        (60.0, 14.0),
    ],
};

/// ECU: extensor, magnitude ~3–12 mm; neutral −4 mm.
pub const WRIST_ARM_ECU: WristMomentArmCurve = WristMomentArmCurve {
    control_points: [
        (-60.0, -3.0),
        (-30.0, -3.5),
        (0.0, -4.0),
        (30.0, -8.0),
        (60.0, -12.0),
    ],
};

/// ECRB: extensor, magnitude ~8–13 mm; neutral −10 mm.
pub const WRIST_ARM_ECRB: WristMomentArmCurve = WristMomentArmCurve {
    control_points: [
        (-60.0, -8.0),
        (-30.0, -9.0),
        (0.0, -10.0),
        (30.0, -12.0),
        (60.0, -13.0),
    ],
};

/// One muscle's morphology and reduced-model geometry.
#[derive(Debug, Clone, Copy)]
pub struct HumanMuscleSpec {
    pub muscle: HumanMuscle,
    /// Physiological cross-sectional area from the cited cadaver dataset.
    pub pcsa_mm2: f32,
    /// Optimal fibre length from the same dataset, converted to metres.
    pub optimal_fibre_length_m: f32,
    /// Maximum active stress in pascals.
    pub specific_tension_pa: f32,
    /// Signed wrist moment arm in metres at the **neutral pose**.
    /// Positive is flexion; negative is extension in this compact convention.
    /// Zero means that this channel is not assigned to the wrist DOF.
    /// Angle-dependent values come from the matching [`WristMomentArmCurve`].
    pub wrist_moment_arm_m: f32,
    /// Signed representative arm for the channel's primary finger/hand DOF.
    /// This is kept per channel; unlike the wrist output, these values are not
    /// summed because the channels act on different finger joints.
    pub primary_joint_moment_arm_m: f32,
}

impl HumanMuscleSpec {
    /// Maximum isometric force implied by PCSA and specific tension.
    #[inline]
    pub const fn max_isometric_force_n(self) -> f32 {
        self.pcsa_mm2 * 1.0e-6 * self.specific_tension_pa
    }

    /// Linear capacity estimate at a normalized activation in `[0, 1]`.
    ///
    /// This is deliberately named `capacity`: it is not Neuromotor's nonlinear
    /// `ca² / (ca² + kd²)` activation law and must not be read as a measured
    /// force response.
    #[inline]
    pub fn capacity_at_activation(self, activation: f32) -> f32 {
        self.max_isometric_force_n() * activation.clamp(0.0, 1.0)
    }

    /// Wrist moment from the linear capacity estimate.
    #[inline]
    pub fn wrist_moment_at_capacity(self, activation: f32) -> f32 {
        self.capacity_at_activation(activation) * self.wrist_moment_arm_m
    }

    /// Per-channel primary-joint moment from the linear capacity estimate.
    #[inline]
    pub fn primary_joint_moment_at_capacity(self, activation: f32) -> f32 {
        self.capacity_at_activation(activation) * self.primary_joint_moment_arm_m
    }
}

/// Eight-channel human hand/forearm reference profile.
#[derive(Debug, Clone, Copy)]
pub struct HumanHandProfile {
    pub muscles: [HumanMuscleSpec; 8],
}

impl HumanHandProfile {
    /// Build the compact profile from the published anatomical values.
    ///
    /// PCSA/fibre lengths:
    /// Kerkhof et al., *The digital human forearm and hand*, Table 1,
    /// <https://pmc.ncbi.nlm.nih.gov/articles/PMC6183001/>.
    ///
    /// Maximum stress:
    /// Goislard de Monsabert et al., Table 6/model methods,
    /// <https://pmc.ncbi.nlm.nih.gov/articles/PMC5754461/>.
    ///
    /// Moment arms are representative fixed values for this reduced model,
    /// not a claim that one value is valid over the full wrist range of motion.
    pub const fn reference() -> Self {
        let sigma = HUMAN_SPECIFIC_TENSION_PA;
        Self {
            muscles: [
                // FCR: 358 mm², 63 mm fibre length; representative +10 mm.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Fcr,
                    pcsa_mm2: 358.0,
                    optimal_fibre_length_m: 0.063,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: 0.010,
                    primary_joint_moment_arm_m: 0.0,
                },
                // ECRL: 483 mm², 160 mm fibre length; representative -10 mm.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Ecrl,
                    pcsa_mm2: 483.0,
                    optimal_fibre_length_m: 0.160,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: -0.010,
                    primary_joint_moment_arm_m: 0.0,
                },
                // FCU: 321 mm², 99 mm fibre length; representative +12 mm.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Fcu,
                    pcsa_mm2: 321.0,
                    optimal_fibre_length_m: 0.099,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: 0.012,
                    primary_joint_moment_arm_m: 0.0,
                },
                // ECU: 620 mm², 45 mm fibre length; representative -4 mm.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Ecu,
                    pcsa_mm2: 620.0,
                    optimal_fibre_length_m: 0.045,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: -0.004,
                    primary_joint_moment_arm_m: 0.0,
                },
                // FDS-II: 287 mm², 89 mm fibre length; finger flexor channel.
                HumanMuscleSpec {
                    muscle: HumanMuscle::FdsIndex,
                    pcsa_mm2: 287.0,
                    optimal_fibre_length_m: 0.089,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: 0.0,
                    primary_joint_moment_arm_m: 0.008,
                },
                // FDP-II: 254 mm², 110 mm fibre length; finger flexor channel.
                HumanMuscleSpec {
                    muscle: HumanMuscle::FdpIndex,
                    pcsa_mm2: 254.0,
                    optimal_fibre_length_m: 0.110,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: 0.0,
                    primary_joint_moment_arm_m: 0.006,
                },
                // ECRB: 483 mm², 160 mm fibre length; representative -10 mm.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Ecrb,
                    pcsa_mm2: 483.0,
                    optimal_fibre_length_m: 0.160,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: -0.010,
                    primary_joint_moment_arm_m: 0.0,
                },
                // FDI: 196 mm², 46 mm fibre length; intrinsic hand muscle.
                HumanMuscleSpec {
                    muscle: HumanMuscle::Fdi,
                    pcsa_mm2: 196.0,
                    optimal_fibre_length_m: 0.046,
                    specific_tension_pa: sigma,
                    wrist_moment_arm_m: 0.0,
                    primary_joint_moment_arm_m: 0.004,
                },
            ],
        }
    }

    pub const fn muscle(self, muscle: HumanMuscle) -> HumanMuscleSpec {
        self.muscles[muscle.index()]
    }

    pub fn peak_force_n(self) -> f32 {
        self.muscles
            .iter()
            .map(|spec| spec.max_isometric_force_n())
            .sum()
    }

    pub fn peak_wrist_moment_nm(self) -> f32 {
        self.peak_wrist_moment_nm_at_angle(0.0)
    }
}

impl Default for HumanHandProfile {
    fn default() -> Self {
        Self::reference()
    }
}
/// Neuromotor-backed runtime for the compact human hand reference.
///
/// `tension` is in newtons because each channel's Neuromotor `t_max` is populated
/// from PCSA × specific tension. The Neuromotor calcium pool remains dimensionless;
/// only the mechanical output is calibrated to SI force.
#[derive(Debug, Clone)]
pub struct HumanHandState {
    pub profile: HumanHandProfile,
    pub neuromotor: NeuromotorState,
    /// Current wrist flexion angle in degrees (positive = flexion).
    pub wrist_angle_deg: f32,
    /// Current muscle forces in newtons, one per profile channel.
    pub force_n: [f32; 8],
    /// Net reduced wrist moment in N·m at the current wrist angle.
    /// Finger/hand channels are excluded.
    pub wrist_moment_nm: f32,
    /// Per-channel primary-joint moments in N·m. These are intentionally not
    /// reduced to one scalar because FDS/FDP/FDI act on different DOFs.
    pub primary_joint_moment_nm: [f32; 8],
}

impl HumanHandState {
    pub fn new() -> Self {
        let profile = HumanHandProfile::reference();
        let base = BiomechanicsProfile {
            d_ca: 0.3,
            tau_pump: 0.05,
            eta_reset: 0.9,
            t_max: 1.0,
            kd: 0.2,
            l0: 0.1,
            v_max: 1.0,
            k0: 0.0,
            beta: 0.0,
            b0: 0.0,
            alpha: 0.0,
        };
        let mut neuromotor = crate::api::neurons::neuromotor::NeuromotorStateBuilder::new(8)
            .with_profile(base)
            .build();
        for (idx, spec) in profile.muscles.iter().enumerate() {
            neuromotor.t_max[idx] = spec.max_isometric_force_n();
            neuromotor.l0[idx] = spec.optimal_fibre_length_m;
            neuromotor.v_max[idx] = 1.0;
            neuromotor.kd[idx] = 0.2;
        }
        Self {
            profile,
            neuromotor,
            wrist_angle_deg: 0.0,
            force_n: [0.0; 8],
            wrist_moment_nm: 0.0,
            primary_joint_moment_nm: [0.0; 8],
        }
    }

    /// Set the wrist flexion angle (degrees, positive = flexion). The wrist
    /// moment arm of each wrist channel is re-evaluated from its angle curve,
    /// so `wrist_moment_nm` stays consistent with the new angle immediately.
    pub fn set_wrist_angle_deg(&mut self, angle_deg: f32) {
        self.wrist_angle_deg = angle_deg;
        self.wrist_moment_nm = self.wrist_moment_nm_at_angle(angle_deg);
    }

    /// Net wrist moment in N·m at an arbitrary angle, using the current forces.
    pub fn wrist_moment_nm_at_angle(&self, angle_deg: f32) -> f32 {
        self.force_n
            .iter()
            .zip(self.profile.muscles.iter())
            .map(|(force, spec)| force * self.profile.wrist_moment_arm_m_at(spec.muscle, angle_deg))
            .sum()
    }

    /// Advance the hand by one step with external motor-neuron spikes.
    /// Lengths are metres, velocities are metres/second, and `dt` is seconds.
    pub fn step(
        &mut self,
        spikes: &[i32],
        resets: &[i32],
        dt: f32,
        lengths_m: Option<&[f32]>,
        velocities_m_s: Option<&[f32]>,
    ) {
        self.neuromotor
            .step(spikes, resets, dt, lengths_m, velocities_m_s);
        self.force_n.copy_from_slice(&self.neuromotor.tension[..8]);
        self.wrist_moment_nm = self.wrist_moment_nm_at_angle(self.wrist_angle_deg);
        for (idx, (force, spec)) in self
            .force_n
            .iter()
            .zip(self.profile.muscles.iter())
            .enumerate()
        {
            self.primary_joint_moment_nm[idx] = force * spec.primary_joint_moment_arm_m;
        }
    }
}

impl Default for HumanHandState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_pcsas_produce_newton_scale_forces() {
        let p = HumanHandProfile::reference();
        assert!((p.muscle(HumanMuscle::Fcr).max_isometric_force_n() - 125.3).abs() < 1e-3);
        assert!((p.muscle(HumanMuscle::Ecu).max_isometric_force_n() - 217.0).abs() < 1e-3);
        assert!((p.muscle(HumanMuscle::Fdi).max_isometric_force_n() - 68.6).abs() < 1e-3);
        assert!((p.peak_force_n() - 1050.7).abs() < 1e-2);
    }

    #[test]
    fn representative_wrist_moment_is_in_newton_metres() {
        let p = HumanHandProfile::reference();
        let moment = p.peak_wrist_moment_nm();
        assert!(moment.is_finite());
        assert!(moment.abs() < 5.0, "compact reference moment: {moment} N·m");
        assert_eq!(p.muscle(HumanMuscle::Fdi).wrist_moment_arm_m, 0.0);
    }

    #[test]
    fn neuromotor_runtime_outputs_force_in_newtons() {
        let mut hand = HumanHandState::new();
        let spikes = [1_i32; 8];
        let resets = [0_i32; 8];
        for _ in 0..200 {
            hand.step(&spikes, &resets, 0.001, None, None);
        }
        for (force, spec) in hand.force_n.iter().zip(hand.profile.muscles.iter()) {
            assert!(force.is_finite() && *force >= 0.0);
            assert!(*force <= spec.max_isometric_force_n() * 1.5 + 1e-4);
        }
        assert!(hand.wrist_moment_nm.is_finite());
        assert!(hand.wrist_moment_nm.abs() < 5.0);
        assert!(hand.primary_joint_moment_nm[7].is_finite());
    }

    #[test]
    fn moment_arm_curves_match_neutral_spec() {
        let p = HumanHandProfile::reference();
        for muscle in [
            HumanMuscle::Fcr,
            HumanMuscle::Ecrl,
            HumanMuscle::Fcu,
            HumanMuscle::Ecu,
            HumanMuscle::Ecrb,
        ] {
            let spec = p.muscle(muscle);
            let arm_neutral = p.wrist_moment_arm_m_at(muscle, 0.0);
            assert!(
                (arm_neutral - spec.wrist_moment_arm_m).abs() < 1e-6,
                "{} neutral arm {arm_neutral} != spec {}",
                muscle.name(),
                spec.wrist_moment_arm_m
            );
        }
        // Non-wrist channels keep zero wrist arm at any angle.
        assert_eq!(p.wrist_moment_arm_m_at(HumanMuscle::Fdi, 45.0), 0.0);
    }

    #[test]
    fn moment_arm_curves_follow_published_ranges_and_physiology() {
        // Flexors grow with flexion; extensors grow (in magnitude) with flexion
        // wrap-around, and remain finite at the range extremes.
        let p = HumanHandProfile::reference();
        let fcr_ext = p.wrist_moment_arm_m_at(HumanMuscle::Fcr, -60.0);
        let fcr_neutral = p.wrist_moment_arm_m_at(HumanMuscle::Fcr, 0.0);
        let fcr_flex = p.wrist_moment_arm_m_at(HumanMuscle::Fcr, 60.0);
        assert!(fcr_ext > 0.0 && fcr_neutral > fcr_ext && fcr_flex > fcr_neutral);
        // Published FCR range ≈ 8–15 mm.
        assert!(fcr_ext >= 7.5e-3 && fcr_flex <= 15.5e-3);

        let ecu_ext = p.wrist_moment_arm_m_at(HumanMuscle::Ecu, -60.0);
        let ecu_flex = p.wrist_moment_arm_m_at(HumanMuscle::Ecu, 60.0);
        assert!(ecu_ext < 0.0 && ecu_flex < 0.0);
        // Published ECU range ≈ 3–12 mm.
        assert!(ecu_ext.abs() >= 2.5e-3 && ecu_flex.abs() <= 12.5e-3);

        for angle in [-60.0, -30.0, 0.0, 30.0, 60.0] {
            let mut total = 0.0f32;
            for muscle in [
                HumanMuscle::Fcr,
                HumanMuscle::Ecrl,
                HumanMuscle::Fcu,
                HumanMuscle::Ecu,
                HumanMuscle::Ecrb,
            ] {
                total += p.wrist_moment_arm_m_at(muscle, angle);
            }
            assert!(total.is_finite());
        }
    }

    #[test]
    fn wrist_moment_depends_on_angle() {
        let mut hand = HumanHandState::new();
        let spikes = [1_i32; 8];
        let resets = [0_i32; 8];
        for _ in 0..200 {
            hand.step(&spikes, &resets, 0.001, None, None);
        }
        let p = hand.profile;
        // Flexor and extensor moment sums, decomposed per muscle group.
        let group_moment = |angle_deg: f32, muscles: &[HumanMuscle]| -> f32 {
            muscles
                .iter()
                .map(|&m| {
                    let force = hand.force_n[m.index()];
                    force * p.wrist_moment_arm_m_at(m, angle_deg)
                })
                .sum()
        };
        let flexors = [HumanMuscle::Fcr, HumanMuscle::Fcu];
        let extensors = [HumanMuscle::Ecrl, HumanMuscle::Ecrb, HumanMuscle::Ecu];

        let moment_neutral = hand.wrist_moment_nm;
        let flex_ext = group_moment(-60.0, &flexors);
        let flex_neutral = group_moment(0.0, &flexors);
        let flex_flex = group_moment(60.0, &flexors);
        let ext_ext = group_moment(-60.0, &extensors);
        let ext_flex = group_moment(60.0, &extensors);

        // The public field must reflect the new angle immediately, without
        // another step(): setting the angle re-evaluates wrist_moment_nm.
        hand.set_wrist_angle_deg(60.0);
        assert_eq!(hand.wrist_moment_nm, hand.wrist_moment_nm_at_angle(60.0));

        // Flexors gain leverage with flexion.
        assert!(flex_ext > 0.0 && flex_flex > flex_neutral && flex_neutral > flex_ext);
        // Extensors are negative and their magnitude grows with flexion
        // wrap-around in this fitted approximation.
        assert!(ext_ext < 0.0 && ext_flex < ext_ext);
        // The angle change moved the cached public field.
        assert_ne!(moment_neutral, hand.wrist_moment_nm);
        // The net balance stays bounded across the range of motion.
        for angle in [-60.0, 0.0, 60.0] {
            let net = hand.wrist_moment_nm_at_angle(angle);
            assert!(net.is_finite());
            assert!(net.abs() < 5.0, "net moment {net} at {angle}°");
        }
    }

    #[test]
    fn force_velocity_stays_bounded_for_human_units() {
        let mut hand = HumanHandState::new();
        let spikes = [1_i32; 8];
        let resets = [0_i32; 8];
        let lengths = [0.063, 0.160, 0.099, 0.045, 0.089, 0.110, 0.160, 0.046];
        let velocities = [-1.0_f32; 8];
        hand.step(
            &spikes,
            &resets,
            0.001,
            Some(&lengths[..]),
            Some(&velocities[..]),
        );
        for (force, spec) in hand.force_n.iter().zip(hand.profile.muscles.iter()) {
            assert!(*force <= spec.max_isometric_force_n() * 1.5 + 1e-4);
        }
    }
}
