//! VNC MN→actuator mapping — Rust-ускорение главного bottleneck (55% времени).
//!
//! Переносит цикл `for i, rid in enumerate(csr.mn_root_ids): ... for act_idx, scale, motion in ctrl_list:`
//! из Python в Rust, заменяя dict lookups на flat массивы.

/// VNC MN→actuator step: вычисляет actuator drive из MN rates.
///
/// # Arguments
///
/// * `mn_rates` — MN firing rates [n_mn], из `brain.get_mn_rates()`
/// * `balance_gain` — per-leg balance gain [6]
/// * `contact_boost` — per-leg contact boost [6]
/// * `actuator_gain` — overall actuator gain f64
/// * `n_actuators` — total number of actuators in model
/// * `map_mn_idx` — MN index (into mn_rates) for each mapping entry [n_mapped]
/// * `map_act_idx` — actuator index for each mapping entry [n_mapped]
/// * `map_scale` — scale factor for each mapping entry [n_mapped]
/// * `map_leg_idx` — leg index (0..5) for each mapping entry [n_mapped]
/// * `mirror_left` — left actuator indices for symmetrization [n_mirror]
/// * `mirror_right` — right actuator indices for symmetrization [n_mirror]
/// * `ctrl_out` — output actuator control values [n_actuators]
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

    // ─── Phase 1: accumulate act_drive + act_count ───
    let mut act_drive = vec![0.0f64; n_actuators];
    let mut act_count = vec![0i32; n_actuators];

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

    // Normalize: average if multiple MNs map to same actuator
    for i in 0..n_actuators {
        if act_count[i] > 1 {
            act_drive[i] /= act_count[i] as f64;
        }
    }

    // ─── Phase 2: Symmetrize L/R ───
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

    // ─── Phase 3: Apply to ctrl_out ───
    for i in 0..n_actuators {
        let val = act_drive[i] * actuator_gain;
        ctrl_out[i] = val.clamp(-1.0, 1.0);
    }
}
