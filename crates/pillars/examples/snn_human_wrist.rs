//! SNN → Neuromotor → human wrist showcase.
//!
//! A compact closed-loop vertical slice using only canonical `pillars` APIs:
//!
//! ```text
//! target angle → proprioceptive current → VSN motor spikes
//!             → Neuromotor human-hand muscle forces (N)
//!             → wrist moment (N·m) → inertial wrist dynamics
//!             ↺
//! ```
//!
//! The eight-channel human-hand profile is calibrated from published PCSA
//! values. This example drives its four wrist muscles (FCR/ECRL/FCU/ECU),
//! while the other four hand channels remain present in the calibrated model.
//! It is a physics/control showcase, not a clinical patient simulation.
//!
//! Run:
//! ```text
//! cargo run -p pillars --example snn_human_wrist --release
//! ```

use pillars::api::{HumanHandState, HumanMuscle};
use pillars::prelude::{NeuralPopulation, VSNBuilder, VSNModel};

const BODY_DT_S: f32 = 0.001;
const NEURON_DT_MS: f32 = 1.0;
const STEPS: usize = 4_000;
const INERTIA_KG_M2: f32 = 0.08;
const PASSIVE_DAMPING_NM_S: f32 = 0.65;
const DRIVE_GAIN: f32 = 4.0;
const BASE_CURRENT: f32 = 4.0;
const MAX_TARGET_DEG: f32 = 28.0;

#[derive(Debug, Clone, Copy)]
struct WristTelemetry {
    angle_deg: f32,
    target_deg: f32,
    angular_velocity_rad_s: f32,
    moment_nm: f32,
    flexor_force_n: f32,
    extensor_force_n: f32,
    active_muscles: usize,
    spikes: usize,
}

#[derive(Debug, Clone, Copy)]
struct WristSummary {
    angle_rmse_deg: f32,
    final_error_deg: f32,
    peak_wrist_pair_force_n: f32,
    peak_moment_nm: f32,
    mean_spikes_per_step: f32,
    max_abs_angle_deg: f32,
}

fn target_angle_deg(step: usize) -> f32 {
    let phase = 2.0 * std::f32::consts::PI * step as f32 / 1_600.0;
    MAX_TARGET_DEG * phase.sin()
}

/// Convert a desired signed angle error into four agonist/antagonist drive values.
/// Positive error recruits wrist flexors; negative error recruits extensors.
///
/// These are dimensionless control currents for the VSN model, not calibrated
/// physiological amperes. Neuromotor's downstream force and torque outputs are
/// the SI-calibrated quantities in this showcase.
fn motor_currents(error_deg: f32) -> [f32; 4] {
    let flex = BASE_CURRENT + DRIVE_GAIN * error_deg.max(0.0);
    let extend = BASE_CURRENT + DRIVE_GAIN * (-error_deg).max(0.0);
    [flex, extend, flex, extend]
}

fn make_motor_population() -> NeuralPopulation<VSNModel> {
    VSNBuilder::new(4)
        .dt(NEURON_DT_MS)
        .tau_m(10.0)
        .v_rest(-70.0)
        .v_thresh(-54.0)
        .v_reset(-70.0)
        .tau_ref(2.0)
        .beta(0.9)
        .surrogate_gain(10.0)
        .build()
}

fn run_wrist(steps: usize, print_telemetry: bool) -> WristSummary {
    let mut motor = make_motor_population();
    let mut hand = HumanHandState::new();
    let mut spikes = [0i32; 4];
    let muscle_resets = [0i32; 8];
    let mut angle_rad = 0.0f32;
    let mut velocity_rad_s = 0.0f32;
    let mut squared_error = 0.0f32;
    let mut peak_force = 0.0f32;
    let mut peak_moment = 0.0f32;
    let mut total_spikes = 0usize;
    let mut max_abs_angle = 0.0f32;

    for step in 0..steps {
        let target_deg = target_angle_deg(step);
        let angle_deg = angle_rad.to_degrees();
        let error_deg = target_deg - angle_deg;
        let currents = motor_currents(error_deg);
        motor.step(&currents, &mut spikes);

        // Expand the four VSN outputs into the calibrated 8-channel hand.
        // Channel order: FCR, ECRL, FCU, ECU, FDS-II, FDP-II, ECRB, FDI.
        let muscle_spikes = [spikes[0], spikes[1], spikes[2], spikes[3], 0, 0, 0, 0];
        hand.set_wrist_angle_deg(angle_deg);
        hand.step(&muscle_spikes, &muscle_resets, BODY_DT_S, None, None);

        let flexor_force =
            hand.force_n[HumanMuscle::Fcr.index()] + hand.force_n[HumanMuscle::Fcu.index()];
        let extensor_force =
            hand.force_n[HumanMuscle::Ecrl.index()] + hand.force_n[HumanMuscle::Ecu.index()];
        let moment_nm = hand.wrist_moment_nm;

        // Unit-consistent one-DOF wrist dynamics: τ = I·α + b·ω.
        let acceleration_rad_s2 =
            (moment_nm - PASSIVE_DAMPING_NM_S * velocity_rad_s) / INERTIA_KG_M2;
        velocity_rad_s += acceleration_rad_s2 * BODY_DT_S;
        angle_rad += velocity_rad_s * BODY_DT_S;

        let next_error = target_angle_deg(step + 1) - angle_rad.to_degrees();
        squared_error += next_error * next_error;
        peak_force = peak_force.max(flexor_force.max(extensor_force));
        peak_moment = peak_moment.max(moment_nm.abs());
        max_abs_angle = max_abs_angle.max(angle_rad.to_degrees().abs());
        total_spikes += spikes.iter().map(|&s| s as usize).sum::<usize>();

        if print_telemetry && (step % 400 == 0 || step + 1 == steps) {
            let telemetry = WristTelemetry {
                angle_deg: angle_rad.to_degrees(),
                target_deg,
                angular_velocity_rad_s: velocity_rad_s,
                moment_nm,
                flexor_force_n: flexor_force,
                extensor_force_n: extensor_force,
                active_muscles: hand.force_n.iter().filter(|&&f| f > 0.0).count(),
                spikes: spikes.iter().filter(|&&s| s != 0).count(),
            };
            println!(
                "telemetry step={step:4} target={:.2}° angle={:.2}° ω={:.3} rad/s moment={:.3} N·m flex={:.2}N ext={:.2}N active={} spikes={}",
                telemetry.target_deg,
                telemetry.angle_deg,
                telemetry.angular_velocity_rad_s,
                telemetry.moment_nm,
                telemetry.flexor_force_n,
                telemetry.extensor_force_n,
                telemetry.active_muscles,
                telemetry.spikes,
            );
        }

        assert!(angle_rad.is_finite() && velocity_rad_s.is_finite());
        assert!(moment_nm.is_finite());
        assert!(hand
            .force_n
            .iter()
            .all(|force| force.is_finite() && *force >= 0.0));
    }

    let final_target = target_angle_deg(steps);
    WristSummary {
        angle_rmse_deg: (squared_error / steps as f32).sqrt(),
        final_error_deg: final_target - angle_rad.to_degrees(),
        peak_wrist_pair_force_n: peak_force,
        peak_moment_nm: peak_moment,
        mean_spikes_per_step: total_spikes as f32 / steps as f32,
        max_abs_angle_deg: max_abs_angle,
    }
}

fn main() {
    println!("snn_human_wrist");
    println!("pipeline=VSN[4] -> Neuromotor[8 human channels] -> wrist torque");
    println!("dt_body={BODY_DT_S}s dt_neuron={NEURON_DT_MS}ms inertia={INERTIA_KG_M2}kg·m²");
    println!("muscles=FCR,ECRL,FCU,ECU | force=Newtons | moment=N·m");

    let summary = run_wrist(STEPS, true);
    println!(
        "SUMMARY status=prototype angle_rmse={:.3}° final_error={:.3}° peak_wrist_pair_force={:.2}N peak_moment={:.3}N·m mean_spikes={:.2}/step max_abs_angle={:.2}°",
        summary.angle_rmse_deg,
        summary.final_error_deg,
        summary.peak_wrist_pair_force_n,
        summary.peak_moment_nm,
        summary.mean_spikes_per_step,
        summary.max_abs_angle_deg,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_error_recruits_flexors() {
        let currents = motor_currents(10.0);
        assert!(currents[0] > currents[1]);
        assert_eq!(currents[0], currents[2]);
        assert_eq!(currents[1], currents[3]);
    }

    #[test]
    fn negative_error_recruits_extensors() {
        let currents = motor_currents(-10.0);
        assert!(currents[1] > currents[0]);
    }

    #[test]
    fn closed_loop_wrist_stays_finite_and_produces_si_force() {
        let summary = run_wrist(800, false);
        assert!(summary.angle_rmse_deg.is_finite());
        assert!(
            summary.peak_wrist_pair_force_n > 0.0,
            "SNN must activate a wrist muscle"
        );
        assert!(summary.peak_moment_nm.is_finite());
        assert!(summary.max_abs_angle_deg < 180.0, "wrist runaway");
        assert!(
            summary.mean_spikes_per_step > 0.0,
            "no motor spikes emitted"
        );
    }

    #[test]
    fn calibrated_muscles_have_opposing_wrist_moments() {
        let resets = [0i32; 8];
        let mut flexor_hand = HumanHandState::new();
        let mut extensor_hand = HumanHandState::new();
        let flexor_spikes = [1, 0, 1, 0, 0, 0, 0, 0];
        let extensor_spikes = [0, 1, 0, 1, 0, 0, 0, 0];
        for _ in 0..200 {
            flexor_hand.step(&flexor_spikes, &resets, BODY_DT_S, None, None);
            extensor_hand.step(&extensor_spikes, &resets, BODY_DT_S, None, None);
        }
        assert!(
            flexor_hand.wrist_moment_nm > 0.0,
            "flexors must produce flexion moment"
        );
        assert!(
            extensor_hand.wrist_moment_nm < 0.0,
            "extensors must produce extension moment"
        );
    }
}
