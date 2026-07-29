use wide::f32x8;

/// K_Na (Slack / Slo2) — Na⁺-Activated K⁺ Current
///
///   I_KNa = g_KNa * ([Na]ⁱ / ([Na]ⁱ + K_d)) * (V - E_K)
///
/// Activated by INTRACELLULAR Na⁺ accumulation during high-frequency
/// firing. NOT directly voltage-gated (though some isoforms have
/// weak voltage dependence).
///
/// This current generates the slow afterhyperpolarization (sAHP)
/// during repetitive firing and contributes to spike frequency
/// adaptation during sustained activity.
///
/// Key properties:
///   - K_d ≈ 30-80 mM (requires significant Na⁺ influx)
///   - Very slow activation (Na⁺ accumulates over 100ms-1s)
///   - Ubiquitous in CNS neurons
///   - Distinct from SK (Ca²⁺-activated) and BK (Ca²⁺+voltage)
///
/// Distribution:
///   - Cortical pyramidal neurons
///   - Hippocampal CA1 and CA3
///   - Drosophila: expressed in motor neurons (dK_Na)
///
/// [Na]ⁱ is typically maintained at ~10 mM by Na⁺/K⁺ ATPase.
/// Each spike lets in ~3-5 µM Na⁺ per cm², so [Na]ⁱ rises
/// ~0.5-2 mM per 100 spikes.
#[inline(always)]
pub fn op_bio_kna_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    na_reg: u8,
    g_max: f32,
    e_rev: f32,
    kd: f32,
) {
    let v = regs[v_reg as usize];
    let na = regs[na_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);
    let kd_v = f32x8::splat(kd);

    // Hill equation (n=1 approx): [Na] / ([Na] + K_d)
    let open_prob = na / (na + kd_v);

    // I_KNa = g * open_prob * (V - E_K)
    regs[i_reg as usize] = g * open_prob * (v - e);
}
