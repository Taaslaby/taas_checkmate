use wide::{f32x8, i32x8};

/// Extracellular K⁺ Concentration Dynamics
///
///   d[K⁺]ₒ/dt = k_release * spike_rate - k_uptake * ([K⁺]ₒ - [K⁺]_rest)
///
/// Each spike releases K⁺ into the extracellular space (via delayed
/// rectifier channels). Astrocytes (glia) take up excess K⁺.
///
/// [K⁺]ₒ affects the K⁺ reversal potential:
///   E_K = 26.7 * ln([K⁺]ₒ / [K⁺]ᵢ)
///
/// At rest: [K⁺]ₒ ≈ 3.5 mM, [K⁺]ᵢ ≈ 140 mM → E_K ≈ -96 mV
/// During 300 Hz firing in fly brain: [K⁺]ₒ can reach 10-15 mM
/// After intense activity: [K⁺]ₒ can reach 20 mM → E_K ≈ -50mV!
///
/// The opcode updates TWO registers:
///   k_out_reg: [K⁺]ₒ in mM (updated in place)
///   e_k_reg: E_K in mV (computed from Nernst equation)
///
/// Parameters:
///   k_release: K⁺ released per spike (≈ 0.1-0.5 mM·spike⁻¹)
///   k_uptake: K⁺ clearance rate by glia (≈ 1-10 ms⁻¹)
///   k_rest: resting [K⁺]ₒ (≈ 3.5 mM for extracellular)
///   k_in: intracellular [K⁺] (≈ 140 mM, constant)
///   dt: timestep (ms)
///
/// For glial buffering, this also models the spatial buffer:
/// astrocytes take up K⁺ at active sites and release it elsewhere.
/// The `k_uptake` parameter captures both glial uptake and diffusion.
///
/// Drosophila note: The fly blood-brain barrier and cortical glia
/// regulate [K⁺]ₒ around the brain. Without this, high-frequency
/// firing in lamina (100-300 Hz) would quickly depolarize all neurons.
#[inline(always)]
pub fn op_bio_k_concentration(
    regs: &mut [f32x8; 16],
    k_out_reg: u8,
    e_k_reg: u8,
    spike_reg: u8,
    k_release: f32,
    k_uptake: f32,
    k_rest: f32,
    k_in: f32,
    dt: f32,
) {
    let k_out_old = regs[k_out_reg as usize];
    let spike = regs[spike_reg as usize];
    let kr = f32x8::splat(k_release);
    let ku = f32x8::splat(k_uptake);
    let krst = f32x8::splat(k_rest);
    let dt_v = f32x8::splat(dt);

    // d[K⁺]ₒ/dt = k_release * spike - k_uptake * ([K⁺]ₒ - [K⁺]_rest)
    let release = kr * spike;
    let uptake = ku * (k_out_old - krst);
    let dk = release - uptake;
    let k_out_new = k_out_old + dk * dt_v;

    // Clamp to prevent negative or extreme concentrations
    let zero = f32x8::splat(0.0);
    let k_out = k_out_new.max(zero).min(f32x8::splat(50.0));

    regs[k_out_reg as usize] = k_out;

    // E_K = 26.7 * ln([K⁺]ₒ / [K⁺]ᵢ) mV at 37°C
    // Uses accurate log2 via IEEE 754 exponent extraction:
    //   log2(x) = exponent + log2(1.mantissa) ≈ exp - 127 + mant / 2^23
    let ki = f32x8::splat(k_in);
    let ratio = (k_out / ki).max(f32x8::splat(1e-8)).min(f32x8::splat(1.0));

    // Reinterpret f32 bits as i32 for exponent/mantissa extraction
    let bits: i32x8 = bytemuck::cast(ratio);
    // Type annotation needed for shift result
    let exp_part: i32x8 = (bits >> 23_i32) - i32x8::splat(127);
    let mant_bits = bits & i32x8::splat(0x7FFFFF);

    // Convert i32 mantissa to f32 array: mant_f32 = mant_bits as f32
    let mant_arr: [i32; 8] = mant_bits.to_array();
    let mant_f32_arr: [f32; 8] = [
        mant_arr[0] as f32, mant_arr[1] as f32,
        mant_arr[2] as f32, mant_arr[3] as f32,
        mant_arr[4] as f32, mant_arr[5] as f32,
        mant_arr[6] as f32, mant_arr[7] as f32,
    ];
    let mant_f32 = f32x8::from(mant_f32_arr);
    // log2(1.m) ≈ mant / 2^23 (linear, ~5% max error)
    let log2_mant = mant_f32 * f32x8::splat(1.192_092_9e-7);

    // Convert i32 exponent to f32
    let exp_arr: [i32; 8] = exp_part.to_array();
    let exp_f32_arr: [f32; 8] = [
        exp_arr[0] as f32, exp_arr[1] as f32,
        exp_arr[2] as f32, exp_arr[3] as f32,
        exp_arr[4] as f32, exp_arr[5] as f32,
        exp_arr[6] as f32, exp_arr[7] as f32,
    ];
    let exp_f32 = f32x8::from(exp_f32_arr);

    // log2(ratio) = exponent + log2(1.mantissa)
    let log2_ratio = exp_f32 + log2_mant;

    // ln(ratio) = log2(ratio) * ln(2)
    let ln_ratio = log2_ratio * f32x8::splat(std::f32::consts::LN_2);
    // Nernst: E_K = (RT/zF) * 1000 * ln([K⁺]ₒ/[K⁺]ᵢ) ≈ 26.7 * ln(ratio) at 37°C
    let e_k = f32x8::splat(26.7) * ln_ratio;

    regs[e_k_reg as usize] = e_k;
}
