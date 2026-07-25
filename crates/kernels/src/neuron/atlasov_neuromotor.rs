#![allow(non_snake_case)]
use wide::f32x8;
use super::{load8, load8_i32, store8};

// ═══════════════════════════════════════════════════════
//  Atlasov Neuromotor Kernel (Discrete Ca²⁺ & Active Reset Engine)
// ═══════════════════════════════════════════════════════
//
// Biomechanical & Neurophysiological Engine for Muscle Tissue:
// 1. Quantum Calcium Injection: ca += spike * d_ca
// 2. Pump Clearance & Active Mechanical Reset: ca = ca * decay * (1 - reset * eta_reset)
// 3. Cooperative Saturation Tension: T = T_max * (ca² / (ca² + Kd²))
// 4. Antagonistic Tug-of-War (Flexor vs Extensor Pairs):
//      delta_F = T_extensor - T_flexor
//      K_joint = K_0 + beta * (ca_extensor + ca_flexor)
//
// Author / Concept: Atlasov (Атласов)

/// Step `n` Atlasov Neuromotor Muscle Channels. Uses `f32x8` SIMD.
///
/// Muscle Params layout (column-major, size `6 * n`):
///   - params[0..n]: dt (ms)
///   - params[n..2n]: d_ca (calcium quantum per spike)
///   - params[2n..3n]: tau_pump (calcium pump clearance time constant in ms)
///   - params[3n..4n]: eta_reset (active mechanical reset efficiency, 0.0..1.0)
///   - params[4n..5n]: T_max (maximum isometric tension)
///   - params[5n..6n]: K_d (calcium sensitivity threshold)
///
/// # Safety
/// All pointers must be valid for `n` elements (or joint arrays `n / 2`).
pub unsafe fn atlasov_neuromotor_step(
    ca: *mut f32,
    tension: *mut f32,
    spikes: *const i32,
    resets: *const i32,
    params: *const f32,
    n: usize,
    joint_delta_f: *mut f32,
    joint_stiffness: *mut f32,
    joint_damping: *mut f32,
    joint_params: *const f32,
) {
    let p = std::slice::from_raw_parts(params, n * 6);
    let dt_val = p[0];
    let n8 = n & !7;

    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    for off in (0..n8).step_by(8) {
        let mut ca_v = load8(ca, off);

        // 1. Spike input -> Quantum Ca²⁺ Injection
        let spk_v = load8_i32(spikes, off); // 1.0 if spiked, 0.0 otherwise
        let d_ca_v = load8(params.wrapping_add(n), off);
        ca_v += spk_v * d_ca_v;

        // 2. Exponential Pump Clearance: decay = exp(-dt / tau_pump)
        let tau_pump_v = load8(params.wrapping_add(2 * n), off).max(f32x8::splat(1e-6));
        let decay_v = (-f32x8::splat(dt_val) / tau_pump_v).exp();
        ca_v *= decay_v;

        // 3. Active Mechanical Reset: if reset signal > 0, clear Ca²⁺ by eta_reset factor
        let reset_v = load8_i32(resets, off); // 1.0 if reset active
        let eta_reset_v = load8(params.wrapping_add(3 * n), off);
        let reset_factor = one - (reset_v * eta_reset_v);
        ca_v *= reset_factor;
        ca_v = ca_v.max(zero); // Non-negative Ca²⁺

        // 4. Cooperative Saturation Tension: T = T_max * (ca² / (ca² + Kd²))
        let t_max_v = load8(params.wrapping_add(4 * n), off);
        let kd_v = load8(params.wrapping_add(5 * n), off);
        let ca2 = ca_v * ca_v;
        let kd2 = kd_v * kd_v;
        let denom = (ca2 + kd2).max(f32x8::splat(1e-8));
        let tens_v = t_max_v * (ca2 / denom);

        store8(ca, off, ca_v);
        store8(tension, off, tens_v);
    }

    // Scalar tail for remainder
    if n8 < n {
        let ca_slice = std::slice::from_raw_parts_mut(ca, n);
        let tens_slice = std::slice::from_raw_parts_mut(tension, n);
        let spk_slice = std::slice::from_raw_parts(spikes, n);
        let rst_slice = std::slice::from_raw_parts(resets, n);

        for i in n8..n {
            let d_ca = p[n + i];
            let tau_pump = p[2 * n + i].max(1e-6);
            let eta_reset = p[3 * n + i];
            let t_max = p[4 * n + i];
            let kd = p[5 * n + i];

            // 1. Quantum Injection
            if spk_slice[i] > 0 {
                ca_slice[i] += d_ca;
            }

            // 2. Pump clearance
            let decay = (-dt_val / tau_pump).exp();
            ca_slice[i] *= decay;

            // 3. Active reset
            if rst_slice[i] > 0 {
                ca_slice[i] *= 1.0 - eta_reset;
            }
            if ca_slice[i] < 0.0 {
                ca_slice[i] = 0.0;
            }

            // 4. Tension
            let ca2 = ca_slice[i] * ca_slice[i];
            let kd2 = kd * kd;
            tens_slice[i] = t_max * (ca2 / (ca2 + kd2 + 1e-8));
        }
    }

    // 5. Antagonistic Joint Tug-of-War (Pairs: Even = Flexor, Odd = Extensor)
    if !joint_delta_f.is_null() && !joint_stiffness.is_null() && !joint_params.is_null() {
        let n_joints = n / 2;
        let tens = std::slice::from_raw_parts(tension, n);
        let ca_s = std::slice::from_raw_parts(ca, n);
        let df_out = std::slice::from_raw_parts_mut(joint_delta_f, n_joints);
        let stf_out = std::slice::from_raw_parts_mut(joint_stiffness, n_joints);
        let mut dmp_out = if !joint_damping.is_null() {
            Some(std::slice::from_raw_parts_mut(joint_damping, n_joints))
        } else {
            None
        };
        // Layout: [k0 (n_joints), beta (n_joints), b0 (n_joints), alpha (n_joints)]
        let j_p = std::slice::from_raw_parts(joint_params, n_joints * 4);

        for j in 0..n_joints {
            let flex_idx = 2 * j;
            let ext_idx = 2 * j + 1;

            let k0 = j_p[j];
            let beta = j_p[n_joints + j];
            let b0 = j_p[2 * n_joints + j];
            let alpha = j_p[3 * n_joints + j];

            // Delta Force = Extensor Tension - Flexor Tension
            df_out[j] = tens[ext_idx] - tens[flex_idx];

            // Dynamic Joint Stiffness = K_0 + beta * (Ca_flex + Ca_ext)
            let ca_sum = ca_s[flex_idx] + ca_s[ext_idx];
            stf_out[j] = k0 + beta * ca_sum;

            // Dynamic Joint Damping = B_0 + alpha * (Ca_flex + Ca_ext)
            if let Some(ref mut dmp) = dmp_out {
                dmp[j] = b0 + alpha * ca_sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlasov_twitch_summation() {
        let n = 8;
        let dt = 1.0; // 1 ms
        let mut ca = vec![0.0f32; n];
        let mut tension = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];
        let resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.5;      // d_ca = 0.5
            params[2 * n + i] = 20.0;  // tau_pump = 20ms
            params[3 * n + i] = 0.8;   // eta_reset = 0.8
            params[4 * n + i] = 10.0;  // T_max = 10 N
            params[5 * n + i] = 0.3;   // Kd = 0.3
        }

        // Single spike -> twitch response
        spikes[0] = 1;
        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        assert!(ca[0] > 0.0, "Spike should inject Ca2+");
        assert!(tension[0] > 0.0, "Ca2+ should generate tension");
        let t_single = tension[0];

        // Second spike -> summation (tetanus accumulation)
        spikes[0] = 1;
        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        assert!(tension[0] > t_single, "Repeated spikes should cause twitch summation (T2 > T1)");
    }

    #[test]
    fn test_atlasov_active_reset() {
        let n = 8;
        let dt = 1.0;
        let mut ca = vec![1.0f32; n]; // pre-charged Ca2+
        let mut tension = vec![0.0f32; n];
        let spikes = vec![0i32; n];
        let mut resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.5;
            params[2 * n + i] = 50.0;  // slow pump
            params[3 * n + i] = 0.9;   // 90% reset efficiency
            params[4 * n + i] = 10.0;
            params[5 * n + i] = 0.3;
        }

        let ca_before = ca[0];
        resets[0] = 1; // Trigger active mechanical reset

        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        assert!(ca[0] < ca_before * 0.2, "Active reset should rapidly clear Ca2+");
    }

    #[test]
    fn test_atlasov_antagonistic_tug_of_war() {
        let n = 2; // 1 joint pair (0=flexor, 1=extensor)
        let dt = 1.0;
        let mut ca = vec![0.5f32, 1.0f32]; // Extensor has more Ca2+
        let mut tension = vec![0.0f32; n];
        let spikes = vec![0i32; n];
        let resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.5;
            params[2 * n + i] = 50.0;
            params[3 * n + i] = 0.5;
            params[4 * n + i] = 10.0; // T_max
            params[5 * n + i] = 0.3; // Kd
        }

        let mut joint_df = vec![0.0f32; 1];
        let mut joint_stf = vec![0.0f32; 1];
        let mut joint_dmp = vec![0.0f32; 1];
        let joint_params = vec![1.0f32, 2.0f32, 0.1f32, 0.5f32]; // K0 = 1.0, beta = 2.0, B0 = 0.1, alpha = 0.5

        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                joint_df.as_mut_ptr(), joint_stf.as_mut_ptr(), joint_dmp.as_mut_ptr(), joint_params.as_ptr(),
            );
        }

        // Extensor Ca = 1.0 > Flexor Ca = 0.5 => Delta Force > 0
        assert!(joint_df[0] > 0.0, "Net force should favor extensor when Ca_ext > Ca_flex");
        // Joint stiffness K = K0 + beta * (Ca_flex + Ca_ext) = 1.0 + 2.0 * (ca_flex + ca_ext)
        assert!(joint_stf[0] > 1.0, "Co-contraction of calcium should increase joint stiffness");
        assert!(joint_dmp[0] > 0.1, "Co-contraction of calcium should increase joint damping to prevent oscillations");
    }

    #[test]
    fn test_kd_affects_twitch_kinetics() {
        let n = 8;
        let dt = 1.0;
        let mut ca = vec![0.2f32; n];
        let mut tension = vec![0.0f32; n];
        let spikes = vec![0i32; n];
        let resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.5;
            params[2 * n + i] = 50.0;
            params[3 * n + i] = 0.0;
            params[4 * n + i] = 10.0;
            // Channel 0 (fast fiber): Kd = 0.15 uM
            // Channel 1 (slow fiber): Kd = 0.60 uM
            params[5 * n + i] = if i == 0 { 0.15 } else { 0.60 };
        }

        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        // Fast fiber (smaller Kd) should generate higher tension for identical Ca2+=0.2
        assert!(tension[0] > tension[1], "Fast fiber (smaller Kd) must generate higher tension at subthreshold Ca2+");
    }

    #[test]
    fn test_calcium_saturation_nonlinearity() {
        let n = 8;
        let dt = 1.0;
        let mut ca1 = vec![0.1f32; n]; // low Ca
        let mut ca2 = vec![0.3f32; n]; // Kd Ca
        let mut ca3 = vec![1.5f32; n]; // high Ca (saturating)

        let mut tens1 = vec![0.0f32; n];
        let mut tens2 = vec![0.0f32; n];
        let mut tens3 = vec![0.0f32; n];

        let spikes = vec![0i32; n];
        let resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.0;
            params[2 * n + i] = 1000.0; // no decay
            params[3 * n + i] = 0.0;
            params[4 * n + i] = 10.0;  // T_max = 10
            params[5 * n + i] = 0.3;   // Kd = 0.3
        }

        unsafe {
            atlasov_neuromotor_step(ca1.as_mut_ptr(), tens1.as_mut_ptr(), spikes.as_ptr(), resets.as_ptr(), params.as_ptr(), n, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null());
            atlasov_neuromotor_step(ca2.as_mut_ptr(), tens2.as_mut_ptr(), spikes.as_ptr(), resets.as_ptr(), params.as_ptr(), n, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null());
            atlasov_neuromotor_step(ca3.as_mut_ptr(), tens3.as_mut_ptr(), spikes.as_ptr(), resets.as_ptr(), params.as_ptr(), n, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null());
        }

        // At Ca = Kd = 0.3, T should be 50% of T_max (5.0 N)
        assert!((tens2[0] - 5.0).abs() < 0.2, "Tension at Kd should be ~50% of T_max");
        // Non-linear Hill saturation: tens3 should approach T_max = 10.0 but not exceed it
        assert!(tens3[0] > 9.0 && tens3[0] <= 10.0, "Tension at high Ca2+ must saturate at T_max");
    }

    #[test]
    fn test_reset_decay_interplay() {
        let n = 8;
        let dt = 1.0;
        let mut ca_decay_only = vec![1.0f32; n];
        let mut ca_both = vec![1.0f32; n];
        let mut tension = vec![0.0f32; n];
        let spikes = vec![0i32; n];
        let mut resets_none = vec![0i32; n];
        let mut resets_active = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.5;
            params[2 * n + i] = 10.0;  // tau_pump = 10ms
            params[3 * n + i] = 0.5;   // eta_reset = 0.5
            params[4 * n + i] = 10.0;
            params[5 * n + i] = 0.3;
            resets_active[i] = 1;
        }

        unsafe {
            atlasov_neuromotor_step(
                ca_decay_only.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets_none.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
            atlasov_neuromotor_step(
                ca_both.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets_active.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        // Decay only: ca = 1.0 * exp(-1/10) ~ 0.9048
        // Both: ca = 1.0 * exp(-1/10) * (1 - 0.5) ~ 0.4524
        assert!(ca_both[0] < ca_decay_only[0] * 0.6, "Active reset + pump clearance interplay should compound linearly");
    }

    #[test]
    fn test_simd_scalar_parity_unaligned() {
        let n = 11; // 8 SIMD + 3 scalar tail
        let dt = 1.0;
        let mut ca = vec![0.2f32; n];
        let mut tension = vec![0.0f32; n];
        let spikes = vec![1i32; n];
        let resets = vec![0i32; n];
        let mut params = vec![0.0f32; n * 6];

        for i in 0..n {
            params[i] = dt;
            params[n + i] = 0.4;
            params[2 * n + i] = 15.0;
            params[3 * n + i] = 0.7;
            params[4 * n + i] = 12.0;
            params[5 * n + i] = 0.25;
        }

        unsafe {
            atlasov_neuromotor_step(
                ca.as_mut_ptr(), tension.as_mut_ptr(),
                spikes.as_ptr(), resets.as_ptr(),
                params.as_ptr(), n,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(),
            );
        }

        // SIMD element 0 vs scalar tail element 8 should produce identical math (within f32 precision)
        let diff_ca = (ca[0] - ca[8]).abs();
        let diff_t = (tension[0] - tension[8]).abs();
        assert!(diff_ca < 1e-5, "SIMD vs scalar tail Ca2+ parity error: diff={diff_ca}");
        assert!(diff_t < 1e-5, "SIMD vs scalar tail Tension parity error: diff={diff_t}");
    }
}
