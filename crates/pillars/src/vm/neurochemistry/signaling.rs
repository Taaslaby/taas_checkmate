use wide::f32x8;

/// ███ Cool Chemistry: NO/cGMP Signaling Pathway █████████████████████████████
///
/// Nitric oxide (NO) is a gaseous retrograde messenger synthesized by nNOS
/// in response to Ca²⁺ influx. NO diffuses to presynaptic terminals and
/// activates guanylyl cyclase → cGMP → modulates transmitter release.
///
///   d[NO]/dt = k_nNOS * Ca - k_deg_NO * [NO]
///   d[cGMP]/dt = k_GC * [NO] - k_PDE * [cGMP]
///   mod_cgmp = 1 + gain * cGMP / (cGMP + Kd)
///
/// Registers:
///   no_reg (in/out): NO concentration
///   cgmp_reg (in/out): cGMP concentration
///   ca_reg (in): Intracellular Ca²⁺ (activates nNOS)
///   mod_factor (out): cGMP-dependent presynaptic modulation factor
///   k_nnos (const): nNOS activation rate by Ca²⁺
///   k_deg_no (const): NO degradation rate
///   k_gc (const): Guanylyl cyclase activation by NO
///   k_pde (const): PDE degradation of cGMP
///   gain (const): Modulation gain
///   kd_half (const): Half-activation for cGMP modulation
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_no_cgmp(
    regs: &mut [f32x8; 16],
    no_reg: u8,
    cgmp_reg: u8,
    ca_reg: u8,
    mod_factor: u8,
    k_nnos: f32,
    k_deg_no: f32,
    k_gc: f32,
    k_pde: f32,
    gain: f32,
    kd_half: f32,
    dt: f32,
) {
    let no = regs[no_reg as usize];
    let cgmp = regs[cgmp_reg as usize];
    let ca = regs[ca_reg as usize];
    let dt_v = f32x8::splat(dt);

    // d[NO]/dt = k_nNOS * Ca - k_deg_NO * [NO]
    let dno = f32x8::splat(k_nnos) * ca - f32x8::splat(k_deg_no) * no;
    let new_no = (no + dno * dt_v).max(f32x8::splat(0.0));
    regs[no_reg as usize] = new_no;

    // d[cGMP]/dt = k_GC * [NO] - k_PDE * [cGMP]
    let dcgmp = f32x8::splat(k_gc) * new_no - f32x8::splat(k_pde) * cgmp;
    let new_cgmp = (cgmp + dcgmp * dt_v).max(f32x8::splat(0.0));
    regs[cgmp_reg as usize] = new_cgmp;

    // mod = 1 + gain * cGMP / (cGMP + Kd)
    let one = f32x8::splat(1.0);
    let g = f32x8::splat(gain);
    let kd = f32x8::splat(kd_half);
    regs[mod_factor as usize] = one + g * new_cgmp / (new_cgmp + kd);
}

/// ███ Cool Chemistry: CaMKII Autophosphorylation ████████████████████████████
///
/// Ca²⁺/calmodulin-dependent kinase II — molecular memory for LTP/LTD.
/// Ca²⁺ activates CaMKII, which then autophosphorylates (becomes
/// Ca²⁺-independent for a period — molecular memory).
///
///   d[CaMKII*]/dt = k_on * Ca * (1 - [CaMKII*]) - k_off * [CaMKII*]
///
/// Registers:
///   camkii_p (in/out): Fraction of phosphorylated (active) CaMKII (0-1)
///   ca_reg (in): Intracellular Ca²⁺
///   k_on (const): Ca²⁺-dependent activation rate
///   k_off (const): Dephosphorylation rate
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_camkii(
    regs: &mut [f32x8; 16],
    camkii_p: u8,
    ca_reg: u8,
    k_on: f32,
    k_off: f32,
    dt: f32,
) {
    let camkii = regs[camkii_p as usize];
    let ca = regs[ca_reg as usize];
    let kon = f32x8::splat(k_on);
    let koff = f32x8::splat(k_off);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // d[CaMKII*]/dt = k_on * Ca * (1 - CaMKII*) - k_off * CaMKII*
    let dcamkii = kon * ca * (one - camkii) - koff * camkii;
    let new_camkii = (camkii + dcamkii * dt_v).max(f32x8::splat(0.0)).min(one);
    regs[camkii_p as usize] = new_camkii;
}

/// ███ Cool Chemistry: Endocannabinoid Retrograde Signal █████████████████████
///
/// 2-AG (endocannabinoid) produced by postsynaptic Ca²⁺ or mGluR activation.
/// Diffuses retrograde to presynaptic CB1 receptors → inhibits release.
///
///   d[2-AG]/dt = k_prod * Ca² - k_deg * [2-AG] + k_mGluR * Gq
///   release_mod = 1 - gain * [2-AG] / ([2-AG] + Kd)
///
/// Registers:
///   ecb_reg (in/out): Endocannabinoid (2-AG) concentration
///   ca_reg (in): Postsynaptic Ca²⁺
///   gq_reg (in): Gq protein activity (from mGluR)
///   mod_factor (out): Release modulation factor (0-1)
///   k_prod (const): Production rate by Ca²⁺
///   k_mglur (const): Production rate by mGluR/Gq
///   k_deg (const): Degradation rate
///   gain (const): Max inhibition fraction
///   kd (const): Half-activation
///   dt (const): Time step
#[inline(always)]
pub fn op_chem_endocb(
    regs: &mut [f32x8; 16],
    ecb_reg: u8,
    ca_reg: u8,
    gq_reg: u8,
    mod_factor: u8,
    k_prod: f32,
    k_mglur: f32,
    k_deg: f32,
    gain: f32,
    kd: f32,
    dt: f32,
) {
    let ecb = regs[ecb_reg as usize];
    let ca = regs[ca_reg as usize];
    let gq = regs[gq_reg as usize];
    let dt_v = f32x8::splat(dt);

    // d[2-AG]/dt = k_prod * Ca² + k_mGluR * Gq - k_deg * [2-AG]
    let decb = f32x8::splat(k_prod) * ca * ca  // Ca² dependence for Ca²⁺-sensitive PLC
        + f32x8::splat(k_mglur) * gq
        - f32x8::splat(k_deg) * ecb;
    let new_ecb = (ecb + decb * dt_v).max(f32x8::splat(0.0));
    regs[ecb_reg as usize] = new_ecb;

    // mod = 1 - gain * [2-AG] / ([2-AG] + Kd), clamped to [1-gain, 1]
    let one = f32x8::splat(1.0);
    let g = f32x8::splat(gain);
    let kd_v = f32x8::splat(kd);
    let mod_val = one - g * new_ecb / (new_ecb + kd_v);
    let mod_clamped = mod_val.max(one - g).min(one);
    regs[mod_factor as usize] = mod_clamped;
}
