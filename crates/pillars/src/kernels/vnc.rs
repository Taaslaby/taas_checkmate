//! # 🦾 VNC Motor Pipeline — MN Firing → Actuator Torque
//!
//! The final stage of the motor control pipeline: maps motor-neuron (MN)
//! firing rates to actuator drive signals. Was ~55% of Python simulation
//! time for the VNC walker — ported to Rust with flat arrays.
//!
//! ## Pipeline
//!
//! ```text
//! MN rates → scale·clamp(rate·bg + cb) → accumulate → normalize → L/R symmetrize → apply gain
//!                                    ↑                        ↑
//!                              balance_gain  per-leg     mirror_left/right
//!                              contact_boost              actuator pairs
//! ```
//!
//! ## Key design
//!
//! - **Flat arrays** — no dict lookups, no dynamic dispatch
//! - **Balance gain per leg** — `bg[leg]` scales rate before clamp
//! - **Contact boost per leg** — `cb[leg]` added pre-clamp (ground contact reflex)
//! - **L/R symmetrization** — averages mirrored actuator pairs to prevent steering drift
//! - **Output clamp [-1, 1]** — safe PWM range for servo controllers

/// VNC MN→actuator step: computes actuator drive from MN rates.
///
/// # Algorithm
/// 1. **Accumulate**: for each mapping entry, `act_drive[act] += scale·clamp(rate·bg + cb)`.
/// 2. **Normalize**: average when multiple MNs map to the same actuator.
/// 3. **Symmetrize**: average mirrored left/right actuator pairs.
/// 4. **Apply**: `ctrl_out[i] = clamp(act_drive[i]·actuator_gain, -1, 1)`.
///
/// # Args
/// * `mn_rates` — MN firing rates `[n_mn]`
/// * `balance_gain` — per-leg balance gain `[6]`
/// * `contact_boost` — per-leg contact boost `[6]`
/// * `actuator_gain` — overall actuator gain (f64)
/// * `n_actuators` — total number of actuators
/// * `map_mn_idx` / `map_act_idx` / `map_scale` / `map_leg_idx` — mapping entries
/// * `mirror_left` / `mirror_right` — symmetrization index pairs
/// * `ctrl_out` — output actuator control values `[n_actuators]`
#[allow(clippy::too_many_arguments)]
pub fn vnc_step_mapping(
    mn_rates: &[f32],
    balance_gain: &[f64],
    contact_boost: &[f64],
    actuator_gain: f64,
    n_actuators: usize,
    map_mn_idx: &[i32],
    map_act_idx: &[i32],
    map_scale: &[f32],
    map_leg_idx: &[i32],
    mirror_left: &[i32],
    mirror_right: &[i32],
    ctrl_out: &mut [f64],
) {
    let n_mapped = map_mn_idx.len();
    assert_eq!(map_act_idx.len(), n_mapped);
    assert_eq!(map_scale.len(), n_mapped);
    assert_eq!(map_leg_idx.len(), n_mapped);
    assert_eq!(mirror_left.len(), mirror_right.len());
    assert!(ctrl_out.len() >= n_actuators, "ctrl_out too short");

    let bg = |li: usize| -> f64 {
        if li < balance_gain.len() {
            balance_gain[li]
        } else {
            1.0
        }
    };
    let cb = |li: usize| -> f64 {
        if li < contact_boost.len() {
            contact_boost[li]
        } else {
            0.0
        }
    };

    // ─── Phase 1: accumulate act_drive + act_count ───
    let mut act_drive = vec![0.0f64; n_actuators];
    let mut act_count = vec![0i32; n_actuators];

    for i in 0..n_mapped {
        let mn_i = map_mn_idx[i] as usize;
        let rate = if mn_i < mn_rates.len() {
            mn_rates[mn_i] as f64
        } else {
            0.0
        };
        let act_idx = map_act_idx[i] as usize;
        if act_idx >= n_actuators {
            continue;
        }
        let scale = map_scale[i] as f64;
        let leg_i = map_leg_idx[i] as usize;

        let val = rate * bg(leg_i) + cb(leg_i);
        let clamped = val.clamp(-1.0, 1.0);
        act_drive[act_idx] += scale * clamped;
        act_count[act_idx] += 1;
    }

    // Normalize: average if multiple MNs map to the same actuator
    for i in 0..n_actuators {
        if act_count[i] > 1 {
            act_drive[i] /= act_count[i] as f64;
        }
    }

    // ─── Phase 2: symmetrize L/R ───
    for i in 0..mirror_left.len() {
        let l_idx = mirror_left[i] as usize;
        let r_idx = mirror_right[i] as usize;
        if l_idx >= n_actuators || r_idx >= n_actuators {
            continue;
        }
        let avg = (act_drive[l_idx] + act_drive[r_idx]) * 0.5;
        // Symmetrize unconditionally: an antagonistic pair (±x) averages to 0
        // and MUST still be forced together — guarding on `avg != 0.0` leaves
        // them asymmetric.
        act_drive[l_idx] = avg;
        act_drive[r_idx] = avg;
    }

    // ─── Phase 3: apply to ctrl_out ───
    for i in 0..n_actuators {
        ctrl_out[i] = (act_drive[i] * actuator_gain).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vnc_mapping_basic() {
        let mn_rates = vec![10.0f32, 20.0, 5.0];
        let balance_gain = vec![1.0f64; 6];
        let contact_boost = vec![0.0f64; 6];
        let n_actuators = 4;
        let map_mn_idx = vec![0i32, 1, 2];
        let map_act_idx = vec![0i32, 1, 2];
        let map_scale = vec![1.0f32, 1.0, 0.5];
        let map_leg_idx = vec![0i32, 1, 2];
        let mirror_left = vec![0i32];
        let mirror_right = vec![1i32];
        let mut ctrl_out = vec![0.0f64; n_actuators];

        vnc_step_mapping(
            &mn_rates,
            &balance_gain,
            &contact_boost,
            0.5,
            n_actuators,
            &map_mn_idx,
            &map_act_idx,
            &map_scale,
            &map_leg_idx,
            &mirror_left,
            &mirror_right,
            &mut ctrl_out,
        );

        // rate 10·1=10 → clamp 1 → 1·0.5 = 0.5
        assert!(
            (ctrl_out[0] - 0.5).abs() < 1e-9,
            "ctrl[0] should be 0.5, got {}",
            ctrl_out[0]
        );
        // rate 20·1=20 → clamp 1 → 1·0.5 = 0.5
        assert!(
            (ctrl_out[1] - 0.5).abs() < 1e-9,
            "ctrl[1] should be 0.5, got {}",
            ctrl_out[1]
        );
        // rate 5·1=5 → clamp 1 → 1·scale(0.5)=0.5 → 0.5·gain(0.5) = 0.25
        assert!(
            (ctrl_out[2] - 0.25).abs() < 1e-9,
            "ctrl[2] should be 0.25, got {}",
            ctrl_out[2]
        );
        assert_eq!(ctrl_out[3], 0.0, "unmapped actuator stays 0");
    }

    #[test]
    fn vnc_mapping_symmetrizes() {
        let mn_rates = vec![10.0f32, 30.0];
        let balance_gain = vec![1.0f64; 6];
        let contact_boost = vec![0.0f64; 6];
        let map_mn_idx = vec![0i32, 1];
        let map_act_idx = vec![0i32, 1];
        let map_scale = vec![1.0f32, 1.0];
        let map_leg_idx = vec![0i32, 1];
        let mirror_left = vec![0i32];
        let mirror_right = vec![1i32];
        let mut ctrl_out = vec![0.0f64; 2];

        vnc_step_mapping(
            &mn_rates,
            &balance_gain,
            &contact_boost,
            0.1,
            2,
            &map_mn_idx,
            &map_act_idx,
            &map_scale,
            &map_leg_idx,
            &mirror_left,
            &mirror_right,
            &mut ctrl_out,
        );

        assert!(
            (ctrl_out[0] - ctrl_out[1]).abs() < 1e-12,
            "mirrored actuators must be equal: {} vs {}",
            ctrl_out[0],
            ctrl_out[1]
        );
    }
}
