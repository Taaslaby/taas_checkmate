use wide::{f32x8, CmpLt};
use crate::vm::math::utils::fast_exp_simd;

/// IV Bolus / Elimination PK Model
///
/// The simplest PK model: first-order elimination
/// dC/dt = -ke * C
///
/// Analytical solution: C(t+dt) = C(t) * exp(-ke * dt)
///
/// For bolus dosing, set c_reg directly to the initial concentration,
/// then each timestep eliminates exponentially.
///
/// ke = ln(2) / half_life
///
/// Examples:
///   Ketamine:   t½ ≈ 3h  → ke = 0.231/h
///   Diazepam:   t½ ≈ 24h → ke = 0.029/h
///   Lidocaine:  t½ ≈ 1.5h → ke = 0.462/h
#[inline(always)]
pub fn op_pk_elimination(
    regs: &mut [f32x8; 16],
    c_reg: u8,
    ke: f32,
    dt: f32,
) {
    let c = regs[c_reg as usize];
    let decay = fast_exp_simd(f32x8::splat(-ke * dt));
    regs[c_reg as usize] = c * decay;
}

/// Oral Absorption PK Model (1-compartment with absorption)
///
/// dG/dt = -ka * G       (gut compartment: absorption)
/// dB/dt = ka * G - ke * B  (blood compartment: absorption - elimination)
///
/// Where:
/// - G: drug amount in gut (absorbed fraction)
/// - B: drug concentration in blood (what reaches the brain)
/// - ka: absorption rate constant
/// - ke: elimination rate constant
///
/// This model captures the typical oral PK curve:
/// [drug] rises (absorption > elimination), peaks, then falls (elimination dominates)
#[inline(always)]
pub fn op_pk_oral(
    regs: &mut [f32x8; 16],
    gut_reg: u8,
    blood_reg: u8,
    ka: f32,
    ke: f32,
    dt: f32,
) {
    let gut = regs[gut_reg as usize];
    let blood = regs[blood_reg as usize];
    let ka_v = f32x8::splat(ka);
    let ke_v = f32x8::splat(ke);
    let dt_v = f32x8::splat(dt);

    // Gut: dG = -ka * G * dt
    let d_gut = -ka_v * gut * dt_v;
    let gut_new = gut + d_gut;

    // Blood: dB = (ka * G - ke * B) * dt
    let d_blood = (ka_v * gut - ke_v * blood) * dt_v;
    let blood_new = blood + d_blood;

    // Clamp to prevent negative concentrations
    let zero = f32x8::splat(0.0);
    regs[gut_reg as usize] = gut_new.cmp_lt(zero).blend(zero, gut_new);
    regs[blood_reg as usize] = blood_new.cmp_lt(zero).blend(zero, blood_new);
}

/// Blood-Brain Barrier (BBB) Transport
///
/// Models the transfer of drug from blood into brain tissue:
///   dC_brain/dt = k_in * C_blood - k_out * C_brain
///
/// k_in = permeability * surface_area (logP-dependent, typically higher for lipophilic drugs)
/// k_out = elimination from brain compartment
///
/// This allows modeling the delay between plasma concentration and CNS effect.
#[inline(always)]
pub fn op_pb_bbb(
    regs: &mut [f32x8; 16],
    brain_reg: u8,
    blood_reg: u8,
    k_in: f32,
    k_out: f32,
    dt: f32,
) {
    let brain = regs[brain_reg as usize];
    let blood = regs[blood_reg as usize];
    let k_in_v = f32x8::splat(k_in);
    let k_out_v = f32x8::splat(k_out);
    let dt_v = f32x8::splat(dt);

    // dBrain = (k_in * Blood - k_out * Brain) * dt
    let d_brain = (k_in_v * blood - k_out_v * brain) * dt_v;
    let brain_new = brain + d_brain;

    let zero = f32x8::splat(0.0);
    regs[brain_reg as usize] = brain_new.cmp_lt(zero).blend(zero, brain_new);
}
