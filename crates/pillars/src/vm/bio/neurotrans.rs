use wide::f32x8;

/// GIRK (G Protein-Coupled Inward Rectifier K⁺) Current
///
///   I_GIRK = g_GIRK * G_protein * (V - E_K)
///
/// GIRK channels are activated by Gβγ subunits released from Gi/o
/// proteins. They are the final common pathway for:
///   - GABA-B receptors (main! — PIRK in Drosophila)
///   - D2 dopamine receptors
///   - 5-HT1A serotonin receptors
///   - A1 adenosine receptors
///   - CB1 cannabinoid receptors
///   - M2 muscarinic receptors
///
/// This opcode computes I_GIRK from a pre-computed G_protein activation
/// level (0..1, typically from GPCRModulation or differential equation).
///
/// Parameters:
///   g_girk ≈ 0.1-2 nS (GIRK conductance is small but hyperpolarizing)
///   E_K ≈ -85 mV
#[inline(always)]
pub fn op_bio_girk_current(
    regs: &mut [f32x8; 16],
    i_reg: u8,
    v_reg: u8,
    g_protein_reg: u8,
    g_max: f32,
    e_rev: f32,
) {
    let v = regs[v_reg as usize];
    let gp = regs[g_protein_reg as usize];
    let g = f32x8::splat(g_max);
    let e = f32x8::splat(e_rev);

    // I_GIRK = g * G_protein * (V - E_K)
    regs[i_reg as usize] = g * gp * (v - e);
}

/// GABA-B Receptor Activation (Slow Metabotropic Inhibition)
///
///   dG/dt = k_on * [GABA] * (1 - G) - k_off * G
///
/// GABA-B receptors are metabotropic (GPCR) — they activate Gi/o
/// proteins, which in turn:
///   1. Open GIRK channels (postsynaptic — slow IPSP, 50-200ms)
///   2. Close Ca²⁺ channels (presynaptic — reduce NT release)
///
/// This opcode updates G_protein activation level (used by GIRK opcode).
///
/// Key differences from GABA-A (ionotropic):
///   - GABA-A: fast IPSP (1-5ms), Cl⁻ channel, target of benzodiazepines
///   - GABA-B: slow IPSP (50-200ms), K⁺ channel, target of baclofen
///
/// Pharmacology:
///   - Agonist: baclofen (muscle relaxant, antispastic)
///   - Antagonist: CGP 35348, phaclofen
///   - Drosophila: metabotropic GABA-B receptor (mGABA-B) is expressed
///     throughout the brain, modulating olfactory processing
#[inline(always)]
pub fn op_bio_gaba_b_activation(
    regs: &mut [f32x8; 16],
    g_protein_reg: u8,
    gaba_reg: u8,
    kon: f32,
    koff: f32,
    dt: f32,
) {
    let g_old = regs[g_protein_reg as usize];
    let gaba = regs[gaba_reg as usize];
    let kon_v = f32x8::splat(kon);
    let koff_v = f32x8::splat(koff);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // dG/dt = kon * [GABA] * (1 - G) - koff * G
    let dg = kon_v * gaba * (one - g_old) - koff_v * g_old;
    let g_new = g_old + dg * dt_v;

    // Clamp to [0, 1]
    let zero = f32x8::splat(0.0);
    regs[g_protein_reg as usize] = g_new.max(zero).min(one);
}

/// mGluR (Metabotropic Glutamate Receptor) — Group I/II Activation
///
///   dG_q/dt = k_on * [glu] * (1 - G_q) - k_off * G_q
///
/// mGluR activation leads to:
///   Group I (mGluR1/5, Gq): PLC → IP3 → Ca²⁺ release (excitatory)
///   Group II (mGluR2/3, Gi): ↓cAMP (inhibitory, like GABA-B)
///
/// This opcode outputs G_q activation for IP3/Ca²⁺ signaling.
/// Use with SecondMessenger opcode for full cascade.
///
/// Pharmacology:
///   - Agonist: DHPG (Group I), DCG-IV (Group II)
///   - Antagonist: MPEP (Group I), LY341495 (Group II)
///   - Drosophila: DmGluRA is expressed in the antennal lobe
///     and mushroom bodies — olfactory learning!
#[inline(always)]
pub fn op_bio_mglur_activation(
    regs: &mut [f32x8; 16],
    g_q_reg: u8,
    glu_reg: u8,
    kon: f32,
    koff: f32,
    dt: f32,
) {
    let gq_old = regs[g_q_reg as usize];
    let glu = regs[glu_reg as usize];
    let kon_v = f32x8::splat(kon);
    let koff_v = f32x8::splat(koff);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // dGq/dt = kon * [glu] * (1 - Gq) - koff * Gq
    let dgq = kon_v * glu * (one - gq_old) - koff_v * gq_old;
    let gq_new = gq_old + dgq * dt_v;

    let zero = f32x8::splat(0.0);
    regs[g_q_reg as usize] = gq_new.max(zero).min(one);
}

/// CB1 Cannabinoid Receptor — Retrograde Signaling
///
///   dG_i/dt = k_on * [endoCB] * (1 - G_i) - k_off * G_i
///
/// Endocannabinoids (2-AG, anandamide) are produced POSTSYNAPTICALLY
/// and diffuse BACK to the presynaptic terminal, activating CB1 receptors.
/// This produces DEPOLARIZATION-INDUCED SUPPRESSION OF INHIBITION (DSI)
/// or EXCITATION (DSE) — a form of short-term plasticity.
///
/// CB1 is Gi-coupled → opens GIRK → reduces presynaptic Ca²⁺ → ↓NT release.
///
/// The `endocb_reg` holds endocannabinoid concentration (produced by
/// postsynaptic Ca²⁺, typically via a spike train or depolarization).
///
/// Pharmacology:
///   - Agonist: THC (Δ⁹-THC — psychoactive component of cannabis)
///   - Endogenous: 2-AG, anandamide (AEA)
///   - Antagonist: rimonabant (SR141716)
#[inline(always)]
pub fn op_bio_cb1_activation(
    regs: &mut [f32x8; 16],
    g_i_reg: u8,
    endocb_reg: u8,
    kon: f32,
    koff: f32,
    dt: f32,
) {
    let gi_old = regs[g_i_reg as usize];
    let ecb = regs[endocb_reg as usize];
    let kon_v = f32x8::splat(kon);
    let koff_v = f32x8::splat(koff);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // dGi/dt = kon * [ECB] * (1 - Gi) - koff * Gi
    let dgi = kon_v * ecb * (one - gi_old) - koff_v * gi_old;
    let gi_new = gi_old + dgi * dt_v;

    let zero = f32x8::splat(0.0);
    regs[g_i_reg as usize] = gi_new.max(zero).min(one);
}

/// A1 Adenosine Receptor — Tonic Inhibition
///
///   dG_i/dt = k_on * [ado] * (1 - G_i) - k_off * G_i
///
/// Adenosine is a breakdown product of ATP. During high activity,
/// extracellular adenosine accumulates, activating A1 receptors.
/// A1 is the brain's "brake" — it tonically inhibits via Gi/GIRK.
///
/// This mediates:
///   - Sleep pressure (adenosine buildup during wakefulness)
///   - Caffeine's mechanism! Caffeine is an A1 antagonist.
///   - Activity-dependent homeostatic scaling
///
/// The adenosine concentration can be tracked via:
///   d[ado]/dt = k_atp_release * spikes - k_ado_degradation * [ado]
#[inline(always)]
pub fn op_bio_a1_activation(
    regs: &mut [f32x8; 16],
    g_i_reg: u8,
    ado_reg: u8,
    kon: f32,
    koff: f32,
    dt: f32,
) {
    let gi_old = regs[g_i_reg as usize];
    let ado = regs[ado_reg as usize];
    let kon_v = f32x8::splat(kon);
    let koff_v = f32x8::splat(koff);
    let dt_v = f32x8::splat(dt);
    let one = f32x8::splat(1.0);

    // dGi/dt = kon * [ado] * (1 - Gi) - koff * Gi
    let dgi = kon_v * ado * (one - gi_old) - koff_v * gi_old;
    let gi_new = gi_old + dgi * dt_v;

    let zero = f32x8::splat(0.0);
    regs[g_i_reg as usize] = gi_new.max(zero).min(one);
}

/// Adenosine Accumulation from ATP Breakdown
///
///   d[ado]/dt = k_release * spikes - k_deg * [ado]
///
/// Each spike releases ATP as co-transmitter. ATP is rapidly
/// broken down to adenosine by ectonucleotidases.
///
/// Parameters:
///   k_release: adenosine produced per spike (≈ 0.01-0.1 µM)
///   k_deg: degradation rate (≈ 0.01-0.1 ms⁻¹)
#[inline(always)]
pub fn op_bio_adenosine_dynamics(
    regs: &mut [f32x8; 16],
    ado_reg: u8,
    spike_reg: u8,
    k_release: f32,
    k_deg: f32,
    dt: f32,
) {
    let ado_old = regs[ado_reg as usize];
    let spike = regs[spike_reg as usize];
    let kre = f32x8::splat(k_release);
    let kde = f32x8::splat(k_deg);
    let dt_v = f32x8::splat(dt);

    // d[ado]/dt = k_release * spike - k_deg * [ado]
    let dado = kre * spike - kde * ado_old;
    let ado_new = ado_old + dado * dt_v;

    let zero = f32x8::splat(0.0);
    regs[ado_reg as usize] = ado_new.max(zero);
}
