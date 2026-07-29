use wide::f32x8;
use crate::vm::math::utils::fast_exp_simd;

/// Kir (Inward Rectifier K⁺) Current
///
///   I_Kir = g_Kir * (V - E_K) * Boltzmann_block(V, E_K)
///
///   Boltzmann_block = 1 / (1 + exp((V - E_K + V_offset) / k))
///
/// Inward rectifiers pass K⁺ more EASILY inward (into the cell) than
/// outward. At rest (V ≈ E_K), Kir sets the resting membrane potential.
/// During depolarization, Kir channels close, preventing K⁺ efflux.
///
/// Key properties:
///   - Sets resting V near E_K (important for glia and some neurons)
///   - Blocked by Ba²⁺ (millimolar) and Cs⁺
///   - G-protein gated Kir = GIRK (separate opcode)
///   - ATP-sensitive Kir = K_ATP (metabolic sensor)
///
/// Distribution:
///   - Glial cells (astrocytes — K⁺ buffering)
///   - Cardiac myocytes (I_K1)
///   - Drosophila: Kir expressed in muscle and glia
///
/// Default parameters:
///   V_offset ≈ 10 mV (block begins ~10mV above E_K)
///   k ≈ 5 (steepness of block)
///   g_Kir ≈ 0.5-5 nS
#[inline(always)]
pub fn op_bio_kir_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    g_max: f32,
    e_rev: f32,
    v_offset: f32,
    k_slope: f32,
) {
    let v = regs[v_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let one = f32x8::splat(1.0);

    // Boltzmann block: f(V) = 1 / (1 + exp((V - E_K + V_offset) / k))
    // At V = E_K: exp(V_offset/k) — partial block at rest
    // At V >> E_K: large → block activated → channel CLOSED
    // At V << E_K: negative → block OFF → channel OPEN (inward rectification)
    let exp_arg = (v - e + f32x8::splat(v_offset)) / f32x8::splat(k_slope);
    let exp_val = fast_exp_simd(exp_arg.max(f32x8::splat(-50.0)).min(f32x8::splat(50.0)));
    let block = one / (one + exp_val);

    // I_Kir = g * (V - E_K) * block
    regs[i_reg as usize] = g * (v - e) * block;
}
