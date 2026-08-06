//! # GPCR — G-Protein Coupled Receptor Cascades
//!
//! Метаботропные рецепторы — медленная химическая модуляция нейронов.
//! В отличие от ионотропных (AMPA, NMDA, GABA-A), GPCR работают через
//! G-белки и вторичные мессенджеры (cAMP, IP₃, DAG), изменяя
//! проводимость каналов на секундных масштабах.
//!
//! ## Пути
//!
//! | Путь | G-белок | Эффектор | Вторичный мессенджер | Мишень |
//! |------|---------|----------|----------------------|--------|
//! | **Gs** | Gαₛ → AC ↑ | Adenylyl cyclase | cAMP ↑ | PKA → KM ↓, HCN ↑ |
//! | **Gq** | Gαq → PLC ↑ | Phospholipase C | IP₃↑, DAG↑, Ca²⁺↑ | PKC → TRP, CaM |
//! | **Gi** | Gαᵢ → AC ↓ | Adenylyl cyclase | cAMP ↓ | PKA ↓ → disinhibition |
//!
//! ## Нейромедиаторы (Drosophila)
//!
//! | Медиатор | Рецептор | G-белок | Эффект на нейрон |
//! |-----------|----------|---------|------------------|
//! | **Dopamine** | D1 (DopR) | Gs → cAMP ↑ | Возбуждение (PKA modulates K⁺) |
//! | **Dopamine** | D2 (DopR2) | Gi → cAMP ↓ | Торможение |
//! | **Octopamine** | OAMB | Gs → cAMP ↑ | Возбуждение (flight-or-fight) |
//! | **Serotonin** | 5-HT₇ | Gs → cAMP ↑ | Модуляция циркадного ритма |
//! | **Serotonin** | 5-HT₂ | Gq → IP₃ ↑ | Ca²⁺-зависимые K⁺ каналы |
//! | **ACh (musc)** | mAChR-A | Gq → IP₃ ↑ | Медленное возбуждение |
//! | **ACh (musc)** | mAChR-B | Gi → cAMP ↓ | Медленное торможение |
//!
//! # Convention
//! Все функции возвращают (`state_new`, ...) — новое значение состояния.
//! Все параметры — `S: SimdLane` (per-neuron SoA), кроме n_hill (константа типа).

use crate::core::lane::SimdLane;

// ═══════════════════════════════════════════════════════════════
//  Gs cascade: Gs → AC → cAMP → PKA
// ═══════════════════════════════════════════════════════════════

/// Gs cascade step: `cAMP' = cAMP + (k_ac · Gs* - k_pde · cAMP) · dt`
///
/// `ligand` — концентрация нейромедиатора (0..1, нормализованная).
/// `k_ac` — скорость AC (аденилатциклазы).
/// `k_pde` — скорость PDE (фосфодиэстеразы).
#[inline(always)]
pub fn gs_cascade<S: SimdLane>(camp: S, ligand: S, k_ac: S, k_pde: S, dt: S) -> S {
    let gs_active = ligand;
    let new = camp + (k_ac * gs_active - k_pde * camp) * dt;
    new.max(S::zero())
}

/// Gs cascade with Hill kinetics for G-protein activation.
///
/// `r_frac` — доля активированных рецепторов [0..1].
/// `g_total` — общая концентрация G-белка.
/// `kd` — EC₅₀ G-белка к рецептору.
/// `k_ac` — удельная активность AC на единицу Gs*.
#[inline(always)]
pub fn gs_cascade_hill<S: SimdLane>(
    camp: S,
    r_frac: S,
    g_total: S,
    kd: S,
    k_ac: S,
    k_pde: S,
    dt: S,
) -> S {
    let gs_act = r_frac * g_total / (r_frac + kd);
    let new = camp + (k_ac * gs_act - k_pde * camp) * dt;
    new.max(S::zero())
}

// ═══════════════════════════════════════════════════════════════
//  Gq cascade: Gq → PLC → IP₃ → Ca²⁺ release
// ═══════════════════════════════════════════════════════════════

/// Gq cascade step: `IP₃' = IP₃ + (k_plc · Gq* - k_deg · IP₃) · dt`
///
/// Returns: (IP₃_new, ca_release_flux)
/// `ca_release_flux` должна быть добавлена в кальциевую динамику: `Ca += ca_release_flux · dt`.
#[inline(always)]
pub fn gq_ip3_step<S: SimdLane>(
    ip3: S,
    ligand: S,
    k_plc: S,
    k_ip3_deg: S,
    k_release: S,
    ca_er: S,
    ca: S,
    dt: S,
) -> (S, S) {
    let gq_active = ligand;
    let new_ip3 = (ip3 + (k_plc * gq_active - k_ip3_deg * ip3) * dt).max(S::zero());
    let ca_gradient = (ca_er - ca).max(S::zero());
    let ca_release_flux = k_release * new_ip3 * ca_gradient;
    (new_ip3, ca_release_flux)
}

/// Gq cascade simplified — без явного IP₃, аппроксимация Ca²⁺ release.
///
/// Использует `tanh` для плавного включения:
/// ```text
/// ca_release = k_q · ligand · (1 + tanh(5·(Ca_ER - Ca))) / 2 · dt
/// ```
/// tanh реализован через `exp`: `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`.
#[inline(always)]
pub fn gq_ca_release_simple<S: SimdLane>(ligand: S, ca_er: S, ca: S, k_q: S, dt: S) -> S {
    let x = (ca_er - ca) * S::splat(5.0);
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    let exp2x = (x * S::splat(2.0)).exp();
    let tanh_x = (exp2x - S::one()) / (exp2x + S::one());
    let drive = ligand * (S::one() + tanh_x) * S::splat(0.5);
    drive * k_q * dt
}

// ═══════════════════════════════════════════════════════════════
//  Gi cascade: Gi → AC ↓ → cAMP ↓
// ═══════════════════════════════════════════════════════════════

/// Gi-mediated inhibition of AC — снижает продукцию cAMP.
///
/// `camp` — текущий cAMP.
/// `ligand` — агонист Gi (dopamine D2, mAChR-B).
/// `k_inhibit` — максимальная фракция ингибирования AC [0..1].
/// `kd_inhibit` — EC₅₀ для Gi.
/// `k_pde` — фоновая деградация cAMP.
/// `camp_basal_prod` — базальная продукция cAMP (без Gs).
#[inline(always)]
pub fn gi_inhibit_camp<S: SimdLane>(
    camp: S,
    ligand: S,
    k_inhibit: S,
    kd_inhibit: S,
    k_pde: S,
    camp_basal_prod: S,
    dt: S,
) -> S {
    let gi_active = ligand / (ligand + kd_inhibit);
    let ac_eff = camp_basal_prod * (S::one() - k_inhibit * gi_active);
    let new = camp + (ac_eff - k_pde * camp) * dt;
    new.max(S::zero())
}

// ═══════════════════════════════════════════════════════════════
//  Channel Modulation — PKA, PKC, cAMP effects
// ═══════════════════════════════════════════════════════════════

/// PKA modulation of KM current (M-type K⁺ channel).
///
/// PKA → KM ↓ → ↓адаптация → ↑возбуждение.
/// Returns: модулятор [0..1], где 1 = без изменений, 0 = полное ингибирование.
#[inline(always)]
pub fn pka_modulate_km<S: SimdLane>(camp: S, kd_pka: S, max_inhibition: S) -> S {
    let pka_active = camp / (camp + kd_pka);
    S::one() - max_inhibition * pka_active
}

/// cAMP modulation of HCN (Ih) channel.
///
/// ↑cAMP → ↑Ih (сдвиг V_half в деполяризованную сторону).
/// Returns: сдвиг V_half в mV (положительный = деполяризация).
#[inline(always)]
pub fn camp_modulate_hcn<S: SimdLane>(camp: S, kd_camp: S, max_shift_mv: S) -> S {
    let bound = camp / (camp + kd_camp);
    max_shift_mv * bound
}

/// PKC modulation of TRP channels.
///
/// Gq → DAG → PKC → TRP ↑.
/// Returns: множитель g_TRP [1 .. 1+max_potentiation].
#[inline(always)]
pub fn pkc_modulate_trp<S: SimdLane>(gq_active: S, kd_pkc: S, max_potentiation: S) -> S {
    let pkc_active = gq_active / (gq_active + kd_pkc);
    S::one() + max_potentiation * pkc_active
}

// ═══════════════════════════════════════════════════════════════
//  Temperature dependence (Q10)
// ═══════════════════════════════════════════════════════════════

/// Q₁₀ temperature coefficient.
///
/// `celsius` — температура в °C (скаляр, одинакова для всей популяции).
/// `rate_at_ref` — скорость реакции при reference-температуре.
/// `q10` — Q₁₀ коэффициент (≈ 2-3 для GPCR).
/// `ref_temp` — reference температура (22°C для насекомых).
#[inline(always)]
pub fn q10_scale<S: SimdLane>(celsius: f32, rate_at_ref: S, q10: f32, ref_temp: f32) -> S {
    let temp_diff = celsius - ref_temp;
    rate_at_ref * S::splat(q10.powf(temp_diff * 0.1))
}

// ═══════════════════════════════════════════════════════════════
//  Receptor activation: Hill equation
// ═══════════════════════════════════════════════════════════════

/// Hill equation: `r_frac = ligandⁿ / (ligandⁿ + EC₅₀ⁿ)`.
///
/// `n_hill` — коэффициент Хилла (скаляр, константа типа рецептора).
#[inline(always)]
pub fn receptor_activation<S: SimdLane>(ligand: S, ec50: S, n_hill: f32) -> S {
    let lig_n = ligand.powf(n_hill);
    let ec50_n = ec50.powf(n_hill);
    lig_n / (lig_n + ec50_n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    // ─── Gs cascade tests ───────────────────────────────────

    #[test]
    fn test_gs_no_ligand() {
        let camp = f32x8::splat(0.0);
        let ligand = f32x8::splat(0.0);
        let k_ac = f32x8::splat(0.5);
        let k_pde = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = gs_cascade(camp, ligand, k_ac, k_pde, dt);
        assert!(
            (new.to_array()[0]).abs() < 1e-6,
            "no ligand → cAMP stays 0, got {:.6}",
            new.to_array()[0]
        );
    }

    #[test]
    fn test_gs_with_ligand() {
        let camp = f32x8::splat(0.0);
        let ligand = f32x8::splat(1.0);
        let k_ac = f32x8::splat(0.5);
        let k_pde = f32x8::splat(0.1);
        let dt = f32x8::splat(1.0);
        let new = gs_cascade(camp, ligand, k_ac, k_pde, dt);
        assert!(
            (new.to_array()[0] - 0.5).abs() < 1e-6,
            "cAMP should rise to 0.5, got {:.6}",
            new.to_array()[0]
        );
    }

    #[test]
    fn test_gs_steady_state() {
        let k_ac = f32x8::splat(0.5);
        let k_pde = f32x8::splat(0.1);
        let ligand = f32x8::splat(1.0);
        let dt = f32x8::splat(0.1);
        let mut camp = f32x8::splat(0.0);
        // Forward Euler converges: camp(t) = 5·(1 - exp(-k_pde·t))
        // At t = 500·0.1 = 50: 5·(1 - exp(-5)) = 5·0.993 = 4.966 ≈ 5.0
        for _ in 0..500 {
            camp = gs_cascade(camp, ligand, k_ac, k_pde, dt);
        }
        let expected = k_ac.to_array()[0] * ligand.to_array()[0] / k_pde.to_array()[0];
        assert!(
            (camp.to_array()[0] - expected).abs() < 0.05,
            "steady state cAMP should be ~{:.1}, got {:.4}",
            expected,
            camp.to_array()[0]
        );
    }

    // ─── Gq cascade tests ───────────────────────────────────

    #[test]
    fn test_gq_ip3_production() {
        let ip3 = f32x8::splat(0.0);
        let ligand = f32x8::splat(1.0);
        let k_plc = f32x8::splat(0.3);
        let k_deg = f32x8::splat(0.1);
        let k_rel = f32x8::splat(1.0);
        let ca_er = f32x8::splat(1.0);
        let ca = f32x8::splat(0.0);
        let dt = f32x8::splat(1.0);
        let (new_ip3, ca_flux) = gq_ip3_step(ip3, ligand, k_plc, k_deg, k_rel, ca_er, ca, dt);
        assert!(
            (new_ip3.to_array()[0] - 0.3).abs() < 1e-6,
            "IP₃ should rise to 0.3, got {:.6}",
            new_ip3.to_array()[0]
        );
        assert!(
            (ca_flux.to_array()[0] - 0.3).abs() < 1e-6,
            "Ca flux should be 0.3, got {:.6}",
            ca_flux.to_array()[0]
        );
    }

    #[test]
    fn test_gq_simple_ca_release() {
        let ligand = f32x8::splat(1.0);
        let ca_er = f32x8::splat(1.0);
        let ca = f32x8::splat(0.0);
        let k_q = f32x8::splat(1.0);
        let dt = f32x8::splat(1.0);
        let release = gq_ca_release_simple(ligand, ca_er, ca, k_q, dt);
        assert!(
            release.to_array()[0] > 0.0,
            "Ca release should be positive with empty cytosol, got {:.6}",
            release.to_array()[0]
        );
        assert!(
            release.to_array()[0].is_finite(),
            "should be finite, got {:.6}",
            release.to_array()[0]
        );
    }

    // ─── Gi cascade tests ───────────────────────────────────

    #[test]
    fn test_gi_inhibits_camp() {
        let camp = f32x8::splat(5.0);
        let ligand = f32x8::splat(1.0);
        let k_inh = f32x8::splat(0.8);
        let kd_inh = f32x8::splat(0.5);
        let k_pde = f32x8::splat(0.1);
        let basal_prod = f32x8::splat(0.5);
        let dt = f32x8::splat(1.0);
        let new = gi_inhibit_camp(camp, ligand, k_inh, kd_inh, k_pde, basal_prod, dt);
        assert!(
            new.to_array()[0] < 5.0,
            "Gi should reduce cAMP below baseline"
        );
    }

    // ─── Channel modulation tests ───────────────────────────

    #[test]
    fn test_pka_modulate_km_no_camp() {
        let camp = f32x8::splat(0.0);
        let kd = f32x8::splat(0.5);
        let max_inh = f32x8::splat(0.5);
        let mod_f = pka_modulate_km(camp, kd, max_inh);
        assert!(
            (mod_f.to_array()[0] - 1.0).abs() < 1e-6,
            "no cAMP → factor = 1, got {:.4}",
            mod_f.to_array()[0]
        );
    }

    #[test]
    fn test_pka_modulate_km_max_camp() {
        let camp = f32x8::splat(10.0);
        let kd = f32x8::splat(0.5);
        let max_inh = f32x8::splat(0.5);
        let mod_f = pka_modulate_km(camp, kd, max_inh);
        // PKA saturation = 10/(10+0.5) = 0.952, not 100%!
        // mod_f = 1 - 0.5 * 0.952 = 0.524
        let expected = 1.0 - 0.5 * (10.0 / (10.0 + 0.5));
        assert!(
            (mod_f.to_array()[0] - expected).abs() < 0.001,
            "expected {:.4}, got {:.4}",
            expected,
            mod_f.to_array()[0]
        );
    }

    #[test]
    fn test_camp_modulate_hcn() {
        let camp = f32x8::splat(0.0);
        let kd = f32x8::splat(0.5);
        let max_shift = f32x8::splat(10.0);
        let shift = camp_modulate_hcn(camp, kd, max_shift);
        assert!(
            (shift.to_array()[0]).abs() < 1e-6,
            "no cAMP → no shift, got {:.4}",
            shift.to_array()[0]
        );

        let camp_high = f32x8::splat(5.0);
        let shift_high = camp_modulate_hcn(camp_high, kd, max_shift);
        let expected = 10.0 * 5.0 / (5.0 + 0.5);
        assert!(
            (shift_high.to_array()[0] - expected).abs() < 0.01,
            "HCN shift should be ~9.09, got {:.4}",
            shift_high.to_array()[0]
        );
    }

    #[test]
    fn test_pkc_modulate_trp() {
        let gq = f32x8::splat(0.5);
        let kd = f32x8::splat(0.3);
        let max_pot = f32x8::splat(2.0);
        let mod_f = pkc_modulate_trp(gq, kd, max_pot);
        assert!(
            mod_f.to_array()[0] > 1.0,
            "PKC should potentiate TRP: got {:.4}",
            mod_f.to_array()[0]
        );
        assert!(
            mod_f.to_array()[0] <= 3.0,
            "max potentiation 2× → factor ≤ 3, got {:.4}",
            mod_f.to_array()[0]
        );
    }

    // ─── Receptor activation test ───────────────────────────

    #[test]
    fn test_receptor_hill() {
        let lig = f32x8::splat(0.5);
        let ec50 = f32x8::splat(0.5);
        let n = 1.0;
        let act = receptor_activation(lig, ec50, n);
        assert!(
            (act.to_array()[0] - 0.5).abs() < 1e-6,
            "Hill n=1 at EC50 → 0.5, got {:.4}",
            act.to_array()[0]
        );

        let act_steep = receptor_activation(lig, ec50, 4.0);
        assert!(
            (act_steep.to_array()[0] - 0.5).abs() < 1e-6,
            "Hill at EC50 → 0.5 regardless of n, got {:.4}",
            act_steep.to_array()[0]
        );
    }

    #[test]
    fn test_q10_scale() {
        let rate = f32x8::splat(1.0);
        let scaled = q10_scale(22.0, rate, 2.0, 22.0);
        assert!(
            (scaled.to_array()[0] - 1.0).abs() < 1e-6,
            "Q10 at ref temp → 1, got {:.4}",
            scaled.to_array()[0]
        );

        let scaled_hot = q10_scale(32.0, rate, 2.0, 22.0);
        assert!(
            (scaled_hot.to_array()[0] - 2.0).abs() < 0.01,
            "Q10 at +10°C → 2, got {:.4}",
            scaled_hot.to_array()[0]
        );
    }
}
